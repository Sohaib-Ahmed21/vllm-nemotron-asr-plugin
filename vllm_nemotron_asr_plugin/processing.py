# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch
from transformers import (
    BatchFeature,
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrProcessor,
)
from transformers.audio_utils import mel_filter_bank
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import (
    MultiModalDataDict,
)
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import HFMultiModalInputs
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import HfRenderer
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

logger = init_logger(__name__)

_LOG_ZERO_GUARD_VALUE = 2**-24


class NemotronAsrStreamingFeatureExtractor(SequenceFeatureExtractor):
    """Extract Nemotron's log-mel features without a librosa dependency."""

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 512,
        win_length: int = 400,
        preemphasis: float = 0.97,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis

        mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters.T).to(torch.float32)

    def _torch_extract_fbank_features(
        self,
        waveform: torch.Tensor,
        *,
        device: torch.device,
        center: bool,
    ) -> torch.Tensor:
        waveform = waveform.to(device)
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
            center=center,
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)).pow(2)
        mel_spec = self.mel_filters.to(device) @ magnitudes
        return torch.log(mel_spec + _LOG_ZERO_GUARD_VALUE).permute(0, 2, 1)

    @staticmethod
    def _as_mono_tensor(audio: object) -> torch.Tensor:
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim > 1:
            logger.warning(
                "Only mono-channel audio is supported; averaging audio channels."
            )
            waveform = waveform.mean(-1)
        return waveform

    def __call__(
        self,
        raw_speech: object,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool | None = None,
        padding: str | bool | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        device: str | torch.device = "cpu",
        center: bool = True,
        **kwargs,
    ) -> BatchFeature:
        del return_attention_mask, kwargs
        if not center:
            raise ValueError(
                "Nemotron 3.5 ASR feature extraction requires center=True."
            )
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}."
            )
        if sampling_rate is None:
            logger.warning(
                "Audio sampling_rate was not provided; assuming %s.",
                self.sampling_rate,
            )

        if (
            isinstance(raw_speech, (list, tuple))
            and raw_speech
            and np.isscalar(raw_speech[0])
        ):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        if isinstance(raw_speech, (np.ndarray, torch.Tensor)) and raw_speech.ndim <= 1:
            audios = [raw_speech]
        elif isinstance(raw_speech, (Sequence, np.ndarray, torch.Tensor)):
            audios = list(raw_speech)
        else:
            audios = [raw_speech]
        waveforms = [self._as_mono_tensor(audio) for audio in audios]
        if not waveforms:
            raise ValueError("At least one audio waveform is required.")

        lengths = torch.tensor([audio.numel() for audio in waveforms], dtype=torch.long)
        target_length = int(lengths.max().item())
        if isinstance(padding, str) and padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required when padding='max_length'.")
            target_length = max(target_length, max_length)
        elif padding is False or padding is None:
            if len(waveforms) > 1 and len(set(lengths.tolist())) != 1:
                raise ValueError("Variable-length audio requires padding.")
        if truncation and max_length is not None:
            target_length = min(target_length, max_length)
        if pad_to_multiple_of:
            target_length = (
                (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

        target_length = max(target_length, 1)
        padded = torch.full(
            (len(waveforms), target_length),
            self.padding_value,
            dtype=torch.float32,
        )
        for index, waveform in enumerate(waveforms):
            length = min(waveform.numel(), target_length)
            padded[index, :length] = waveform[:length]
            lengths[index] = length

        if self.preemphasis is not None:
            time_mask = torch.arange(target_length)[None, :] < lengths[:, None]
            padded = torch.cat(
                [padded[:, :1], padded[:, 1:] - self.preemphasis * padded[:, :-1]],
                dim=1,
            )
            padded = padded.masked_fill(~time_mask, 0.0)

        extract_device = torch.device(device)
        input_features = self._torch_extract_fbank_features(
            padded,
            device=extract_device,
            center=True,
        )
        feature_lengths = torch.div(
            lengths,
            self.hop_length,
            rounding_mode="floor",
        )
        attention_mask = (
            torch.arange(input_features.shape[1], device=extract_device)[None, :]
            < feature_lengths.to(extract_device)[:, None]
        )
        input_features = input_features.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        return BatchFeature(
            {
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


def _get_subsampling_output_lengths(
    input_lengths: torch.Tensor,
    *,
    subsampling_factor: int,
    subsampling_conv_kernel_size: int,
    subsampling_conv_stride: int,
) -> torch.Tensor:
    num_layers = int(math.log2(subsampling_factor))
    all_paddings = (subsampling_conv_kernel_size - 1) + (subsampling_conv_stride - 1)
    add_pad = all_paddings - subsampling_conv_kernel_size

    output_lengths = input_lengths
    for _ in range(num_layers):
        output_lengths = (
            torch.div(
                output_lengths + add_pad,
                subsampling_conv_stride,
                rounding_mode="floor",
            )
            + 1
        )

    return output_lengths


def _get_max_subsampling_input_length(
    output_length: int,
    *,
    subsampling_factor: int,
    subsampling_conv_kernel_size: int,
    subsampling_conv_stride: int,
) -> int:
    num_layers = int(math.log2(subsampling_factor))
    all_paddings = (subsampling_conv_kernel_size - 1) + (subsampling_conv_stride - 1)
    add_pad = all_paddings - subsampling_conv_kernel_size

    input_length = output_length
    for _ in range(num_layers):
        input_length = subsampling_conv_stride * input_length - add_pad - 1

    return input_length


class Nemotron3_5AsrProcessingInfo(BaseProcessingInfo):
    """vLLM metadata and preprocessing information for Nemotron ASR."""

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_hf_config(self) -> Nemotron3_5AsrConfig:
        return self.ctx.get_hf_config(Nemotron3_5AsrConfig)

    def get_hf_processor(self, **kwargs: object) -> Nemotron3_5AsrProcessor:
        del kwargs
        if not hasattr(self, "_cached_hf_processor"):
            revision = self.ctx.model_config.revision
            processor_config = get_hf_file_to_dict(
                "processor_config.json",
                self.model_id,
                revision=revision,
            )
            if processor_config is None:
                raise ValueError(
                    f"processor_config.json was not found for {self.model_id}."
                )
            processor_config = dict(processor_config)
            feature_config = dict(processor_config.pop("feature_extractor"))
            feature_config.pop("feature_extractor_type", None)
            processor_config.pop("processor_class", None)
            feature_extractor = NemotronAsrStreamingFeatureExtractor(**feature_config)
            self._cached_hf_processor = Nemotron3_5AsrProcessor(
                feature_extractor,
                self.get_tokenizer(),
                **processor_config,
            )
        return self._cached_hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
        )

    def get_feature_extractor(
        self, **kwargs: object
    ) -> NemotronAsrStreamingFeatureExtractor:
        processor = self.get_hf_processor(**kwargs)
        return processor.feature_extractor

    def get_max_audio_samples(self) -> int:
        feature_extractor = self.get_feature_extractor()
        encoder_config = self.get_hf_config().encoder_config
        max_mel_frames = _get_max_subsampling_input_length(
            min(
                encoder_config.max_position_embeddings,
                self.ctx.model_config.max_model_len,
            ),
            subsampling_factor=encoder_config.subsampling_factor,
            subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
            subsampling_conv_stride=encoder_config.subsampling_conv_stride,
        )
        # Centered STFT adds one physical frame beyond floor(samples / hop).
        return max_mel_frames * feature_extractor.hop_length - 1

    @property
    def skip_prompt_length_check(self) -> bool:
        return True


class Nemotron3_5AsrDummyInputsBuilder(
    BaseDummyInputsBuilder[Nemotron3_5AsrProcessingInfo]
):
    """Build a bounded audio input for multimodal profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_len = self.info.get_max_audio_samples()
        audio_overrides = mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class Nemotron3_5AsrMultiModalProcessor(
    EncDecMultiModalProcessor[Nemotron3_5AsrProcessingInfo]
):
    """Convert vLLM audio items into Nemotron encoder inputs."""

    skip_decoder_start_token: bool = True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        return [self.info.get_hf_config().blank_token_id]

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_processor_mm_kwargs)
        if hf_processor_mm_kwargs.get("is_streaming"):
            raise ValueError(
                "Nemotron 3.5 ASR requires a complete audio clip per request."
            )
        feature_extractor = self.info.get_feature_extractor(**hf_processor_mm_kwargs)

        max_samples = self.info.get_max_audio_samples()
        audio_kwargs = cast(
            Mapping[str, object], hf_processor_mm_kwargs.get("audio_kwargs", {})
        )
        for key in ("max_length", "pad_to_multiple_of"):
            value = cast(
                int | None, hf_processor_mm_kwargs.get(key, audio_kwargs.get(key))
            )
            if value is not None and value > max_samples:
                raise ValueError(
                    "Padded audio exceeds the profiled Nemotron audio length."
                )
        audios = mm_items.get_items("audio", AudioProcessorItems)
        for index in range(len(audios)):
            if (
                not feature_extractor.hop_length
                <= audios.get_audio_length(index)
                <= max_samples
            ):
                raise ValueError(
                    "Nemotron 3.5 ASR expects between "
                    f"{feature_extractor.hop_length} and {max_samples} audio samples "
                    f"at {feature_extractor.sampling_rate} Hz."
                )
        hf_processor_mm_kwargs = dict(hf_processor_mm_kwargs)
        hf_processor_mm_kwargs.setdefault(
            "sampling_rate", feature_extractor.sampling_rate
        )

        return hf_inputs._replace(hf_kwargs=hf_processor_mm_kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        max_mel_frames = (
            self.info.get_max_audio_samples()
            // self.info.get_feature_extractor().hop_length
            + 1
        )
        if hf_inputs["input_features"].shape[1] > max_mel_frames:
            raise ValueError("Padded audio exceeds the profiled Nemotron audio length.")
        num_audios = hf_inputs["input_features"].shape[0]
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "attention_mask": MultiModalFieldConfig.batched("audio"),
            "prompt_ids": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            "num_lookahead_tokens": MultiModalFieldConfig.shared(
                "audio", num_audios, keep_on_cpu=True
            ),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        attention_mask = out_mm_kwargs.get_data()["attention_mask"]
        assert isinstance(attention_mask, torch.Tensor)

        encoder_config = self.info.get_hf_config().encoder_config
        output_lengths = _get_subsampling_output_lengths(
            attention_mask.sum(-1),
            subsampling_factor=encoder_config.subsampling_factor,
            subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
            subsampling_conv_stride=encoder_config.subsampling_conv_stride,
        ).tolist()

        def replacement(item_idx: int) -> list[int]:
            return [0] * output_lengths[item_idx]

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=replacement,
            )
        ]


class Nemotron3_5AsrRenderer(HfRenderer):
    def get_dec_start_token_id(self) -> int:
        return self.model_config.hf_config.blank_token_id

    def get_eos_token_id(self) -> int:
        # Acoustic blanks are consumed inside the RNNT loop. The model only
        # exposes blank to the engine once all encoder frames are exhausted.
        return self.model_config.hf_config.blank_token_id
