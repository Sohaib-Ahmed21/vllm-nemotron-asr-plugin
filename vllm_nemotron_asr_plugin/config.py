# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local execution requirements."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)


class Nemotron3_5AsrForRNNTConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        model_config.model_arch_config.hidden_size = (
            model_config.hf_config.decoder_hidden_size
        )
        if model_config.tokenizer_mode == "auto":
            model_config.tokenizer_mode = "nemotron3_5_asr"
        if model_config.tokenizer_mode != "nemotron3_5_asr":
            raise ValueError(
                "Nemotron 3.5 ASR requires tokenizer mode nemotron3_5_asr."
            )
        # The greedy RNNT adapter does not expose meaningful alternative-token
        # scores, so limit the existing logprob request interface to zero.
        if model_config.max_logprobs != 0:
            logger.info("Limiting Nemotron RNNT alternative-token logprobs to zero.")
            model_config.max_logprobs = 0

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        if not vllm_config.use_v2_model_runner:
            raise ValueError("Nemotron 3.5 ASR requires Model Runner V2.")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("Nemotron 3.5 ASR currently requires --enforce-eager.")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Nemotron 3.5 ASR currently requires TP=1 and PP=1.")
        if vllm_config.model_config.quantization is not None:
            raise ValueError("Nemotron 3.5 ASR does not support vLLM quantization.")
        if vllm_config.speculative_config is not None:
            raise ValueError("Nemotron 3.5 ASR does not support speculative decoding.")
