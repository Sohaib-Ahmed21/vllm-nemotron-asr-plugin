# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Nemotron 3.5 ASR transcription."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    Nemotron3_5AsrConfig,
    NemotronAsrStreamingEncoder,
)
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from transformers.models.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrPromptProjector,
    Nemotron3_5AsrRNNTDecoder,
    Nemotron3_5AsrRNNTJointNetwork,
)
from transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr import (
    DEFAULT_PROMPT_DICTIONARY,
)
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    PromptType,
    TextPrompt,
    TokensPrompt,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY

from .processing import (
    Nemotron3_5AsrDummyInputsBuilder,
    Nemotron3_5AsrMultiModalProcessor,
    Nemotron3_5AsrProcessingInfo,
)

if TYPE_CHECKING:
    from vllm_nemotron_asr_plugin.model_state import (
        Nemotron3_5AsrModelState,
    )


class Nemotron3_5AsrAudioEncoder(nn.Module):
    """Encode mel features and apply language-prompt conditioning."""

    def __init__(self, config: Nemotron3_5AsrConfig):
        super().__init__()
        self.config = config
        self.encoder = NemotronAsrStreamingEncoder(config.encoder_config)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size,
            config.decoder_hidden_size,
        )
        self.prompt_projector = Nemotron3_5AsrPromptProjector(config)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_ids: torch.LongTensor | None = None,
        num_lookahead_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            num_lookahead_tokens=num_lookahead_tokens,
            use_cache=False,
        )
        hidden_states = encoder_outputs.last_hidden_state
        output_mask = encoder_outputs.attention_mask
        if output_mask is not None:
            output_mask = output_mask.bool()

        if prompt_ids is None:
            prompt_ids = torch.full(
                (hidden_states.shape[0],),
                self.config.default_prompt_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
        prompt_ids = prompt_ids.to(hidden_states.device)
        prompt = F.one_hot(
            prompt_ids,
            num_classes=self.config.num_prompts,
        ).to(hidden_states.dtype)
        prompt = prompt[:, None, :].expand(-1, hidden_states.shape[1], -1)
        hidden_states = self.prompt_projector(
            torch.cat((hidden_states, prompt), dim=-1)
        )
        return self.encoder_projector(hidden_states), output_mask


@dataclass
class Nemotron3_5AsrDecodeState:
    encoder_frames: torch.Tensor
    decoder_cache: Nemotron3_5AsrRNNTDecoderCache
    last_token_id: int
    frame_idx: int = 0
    symbols_at_frame: int = 0
    num_tokens_to_replay: int = 0


def _decode_next_tokens(
    states: list[Nemotron3_5AsrDecodeState],
    decoder: Nemotron3_5AsrRNNTDecoder,
    joint: Nemotron3_5AsrRNNTJointNetwork,
    *,
    blank_token_id: int,
    max_symbols_per_step: int,
) -> list[int | None]:
    """Advance active requests to their next transcript token or exhaustion."""
    selected: list[int | None] = [None] * len(states)
    pending = list(enumerate(states))
    while pending:
        active = []
        for index, state in pending:
            if state.frame_idx < state.encoder_frames.shape[0]:
                active.append((index, state))
            elif state.num_tokens_to_replay:
                raise RuntimeError("RNNT exhausted before replaying the prefix.")
        if not active:
            break

        # HF initializes every fresh row, including its blank seed. Once the
        # cache is initialized, blank rows instead preserve the previous state.
        active.sort(key=lambda item: item[1].decoder_cache.is_initialized)
        groups = [
            [s for _, s in active if s.decoder_cache.is_initialized == initialized]
            for initialized in (False, True)
        ]
        decoder_outputs = []
        for initialized, group in zip((False, True), groups):
            if not group:
                continue
            cache = Nemotron3_5AsrRNNTDecoderCache(group[0].decoder_cache.config)
            if initialized:
                cache.cache = torch.cat([s.decoder_cache.cache for s in group])
                cache.hidden_state = torch.cat(
                    [s.decoder_cache.hidden_state for s in group], dim=1
                )
                cache.cell_state = torch.cat(
                    [s.decoder_cache.cell_state for s in group], dim=1
                )
                cache.is_initialized = True
            input_ids = torch.tensor(
                [[s.last_token_id] for s in group],
                dtype=torch.long,
                device=group[0].encoder_frames.device,
            )
            decoder_outputs.append(decoder(input_ids, cache=cache)[:, -1, :])
            for row, state in enumerate(group):
                state.decoder_cache.update(
                    cache.cache[row : row + 1],
                    cache.hidden_state[:, row : row + 1],
                    cache.cell_state[:, row : row + 1],
                )

        encoder_frames = torch.stack([s.encoder_frames[s.frame_idx] for _, s in active])
        logits = joint(torch.cat(decoder_outputs), encoder_frames)
        token_ids = logits.argmax(dim=-1).tolist()
        pending = []
        for (index, state), token_id in zip(active, token_ids):
            state.last_token_id = token_id
            if token_id == blank_token_id:
                state.frame_idx += 1
                state.symbols_at_frame = 0
                pending.append((index, state))
                continue

            state.symbols_at_frame += 1
            if state.symbols_at_frame >= max_symbols_per_step:
                state.frame_idx += 1
                state.symbols_at_frame = 0
            if state.num_tokens_to_replay:
                state.num_tokens_to_replay -= 1
                pending.append((index, state))
            else:
                selected[index] = token_id
    return selected


@MULTIMODAL_REGISTRY.register_processor(
    Nemotron3_5AsrMultiModalProcessor,
    info=Nemotron3_5AsrProcessingInfo,
    dummy_inputs=Nemotron3_5AsrDummyInputsBuilder,
)
class Nemotron3_5AsrForRNNT(nn.Module, SupportsTranscription, SupportsMultiModal):
    supports_transcription_only = True
    supported_languages = {
        code: name
        for code, name in ISO639_1_SUPPORTED_LANGS.items()
        if code in DEFAULT_PROMPT_DICTIONARY
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "audio_encoder.encoder.",
            "encoder_projector.": "audio_encoder.encoder_projector.",
            "prompt_projector.": "audio_encoder.prompt_projector.",
        }
    )

    @staticmethod
    def get_model_state_cls() -> type["Nemotron3_5AsrModelState"]:
        from vllm_nemotron_asr_plugin.model_state import (
            Nemotron3_5AsrModelState,
        )

        return Nemotron3_5AsrModelState

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        del model_config, task_type
        return SpeechToTextConfig(
            sample_rate=16_000,
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError("Nemotron 3.5 ASR supports transcription only.")
        if stt_params.request_prompt or stt_params.hotwords or stt_params.to_language:
            raise ValueError("Text prompts, hotwords and translation are unsupported.")
        language = stt_params.language or "auto"
        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={
                    "audio": (stt_params.audio, stt_params.stt_config.sample_rate)
                },
                mm_processor_kwargs={"language": language},
            ),
            decoder_prompt=TokensPrompt(
                prompt_token_ids=[stt_params.model_config.hf_config.blank_token_id]
            ),
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: Nemotron3_5AsrConfig = vllm_config.model_config.hf_config
        self.audio_encoder = Nemotron3_5AsrAudioEncoder(self.config)
        self.decoder = Nemotron3_5AsrRNNTDecoder(self.config)
        self.joint = Nemotron3_5AsrRNNTJointNetwork(self.config)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = cast(
            torch.Tensor | list[torch.Tensor], kwargs["input_features"]
        )
        attention_mask = cast(
            torch.Tensor | list[torch.Tensor] | None, kwargs.get("attention_mask")
        )
        prompt_ids = cast(torch.Tensor | None, kwargs.get("prompt_ids"))
        lookahead = cast(int | torch.Tensor | None, kwargs.get("num_lookahead_tokens"))
        dtype = self.audio_encoder.encoder_projector.weight.dtype
        if isinstance(input_features, torch.Tensor):
            hidden_states, output_mask = self.audio_encoder(
                input_features.to(dtype),
                attention_mask,
                prompt_ids,
                int(lookahead) if lookahead is not None else None,
            )
            if output_mask is None:
                return list(hidden_states.unbind(0))
            return [states[mask] for states, mask in zip(hidden_states, output_mask)]

        output: list[torch.Tensor] = []
        for i, features in enumerate(input_features):
            mask = attention_mask[i] if attention_mask is not None else None
            prompt = prompt_ids[i : i + 1] if prompt_ids is not None else None
            states, valid = self.audio_encoder(
                features.unsqueeze(0).to(dtype),
                mask.unsqueeze(0) if mask is not None else None,
                prompt,
                int(lookahead) if lookahead is not None else None,
            )
            output.append(states[0] if valid is None else states[0][valid[0]])
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        decode_states: list[Nemotron3_5AsrDecodeState | None] | None = None,
        query_end_positions: list[int] | None = None,
        decode_ready: list[bool] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del positions, kwargs
        selected_ids = torch.full(
            (input_ids.shape[0], 1),
            self.config.blank_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        if decode_states is not None:
            assert query_end_positions is not None and decode_ready is not None
            active = [
                (state, end)
                for state, end, ready in zip(
                    decode_states, query_end_positions, decode_ready
                )
                if state is not None and ready
            ]
            token_ids = _decode_next_tokens(
                [state for state, _ in active],
                self.decoder,
                self.joint,
                blank_token_id=self.config.blank_token_id,
                max_symbols_per_step=self.config.max_symbols_per_step,
            )
            for (_, end), token_id in zip(active, token_ids):
                if token_id is not None:
                    selected_ids[end - 1, 0] = token_id
        return selected_ids

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # RNNT already consumed blanks and selected the next transcript token.
        # Preserve that choice in the sampler; these are not RNNT probabilities.
        logits = torch.full(
            (hidden_states.shape[0], self.config.vocab_size),
            float("-inf"),
            dtype=self.audio_encoder.encoder_projector.weight.dtype,
            device=hidden_states.device,
        )
        return logits.scatter_(1, hidden_states.long(), 0.0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
