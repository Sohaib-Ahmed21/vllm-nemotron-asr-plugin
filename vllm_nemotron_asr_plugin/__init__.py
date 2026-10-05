# SPDX-License-Identifier: Apache-2.0
"""Nemotron 3.5 ASR model plugin for vLLM."""


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    from .config import Nemotron3_5AsrForRNNTConfig

    MODELS_CONFIG_MAP["Nemotron3_5AsrForRNNT"] = Nemotron3_5AsrForRNNTConfig
    TokenizerRegistry.register(
        "nemotron3_5_asr", "vllm.tokenizers.hf", "CachedHfTokenizer"
    )
    RENDERER_REGISTRY.register(
        "nemotron3_5_asr",
        "vllm_nemotron_asr_plugin.processing",
        "Nemotron3_5AsrRenderer",
    )
    ModelRegistry.register_model(
        "Nemotron3_5AsrForRNNT",
        "vllm_nemotron_asr_plugin.model:Nemotron3_5AsrForRNNT",
    )
