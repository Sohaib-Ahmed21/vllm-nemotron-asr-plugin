# Nemotron 3.5 ASR Plugin for vLLM

Out-of-tree vLLM model plugin for complete-audio transcription with [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

The plugin integrates `Nemotron3_5AsrForRNNT` with vLLM using batched greedy RNN-T decoding and the standard OpenAI-compatible `/v1/audio/transcriptions` endpoint.

> Despite the checkpoint name containing `streaming`, this plugin currently supports **complete-audio transcription only**. Native cache-aware realtime streaming is not supported.

## Installation

### Prerequisites

This plugin requires:

- a compatible vLLM installation with Model Runner V2 support
- Transformers `>=5.13`
- CUDA
- [`uv`](https://docs.astral.sh/uv/) for package management

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone the repository:

```bash
git clone <repository-url>
cd vllm-nemotron-asr-plugin
```

2. Install the plugin in development mode:

```bash
uv pip install -e .
```

Or install it directly:

```bash
uv pip install .
```

The plugin is registered through `vllm.general_plugins` and is discovered automatically by vLLM.

If `VLLM_PLUGINS` is used as an allowlist, include:

```bash
export VLLM_PLUGINS=nemotron_asr
```

## Verify Installation

After installation, verify that vLLM discovers the plugin and registers `Nemotron3_5AsrForRNNT`:

```bash
python verify_plugin.py
```

A successful check confirms that the installed entry point is available and the Nemotron ASR model is registered with vLLM.

## Usage

### Offline example

Run transcription directly through `vllm.LLM`:

```bash
python example_transcription.py
```

Or provide your own audio:

```bash
python example_transcription.py /path/to/speech.wav --language en
```

### OpenAI-compatible server

Start vLLM:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --enforce-eager \
    --dtype float16 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.5
```

Then transcribe an audio file:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
    -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
    -F file=@/path/to/speech.wav \
    -F language=en \
    -F response_format=json
```

If the checkpoint is already cached, Hugging Face network checks can be disabled with:

```bash
HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b --enforce-eager
```

## Tests

Install test dependencies:

```bash
uv pip install -e '.[test]'
```

Run the test suite:

```bash
pytest -q
```

Run CPU/non-GPU tests only:

```bash
pytest -q -m 'not gpu'
```

Run the transcription end-to-end tests:

```bash
pytest -q tests/test_transcription.py
```

The tests cover plugin registration, multimodal audio processing, RNN-T decoding, batching and request isolation, predictor state, replay/preemption, weight loading, Transformers parity, configuration validation, and `/v1/audio/transcriptions`.

## Evaluation

Install evaluation dependencies:

```bash
uv pip install -e '.[eval]'
```

Run a small FLEURS evaluation:

```bash
python scripts/eval_fleurs.py \
    --language en_us \
    --num-samples 10
```

For another supported language:

```bash
python scripts/eval_fleurs.py \
    --language de_de \
    --num-samples 10
```

See [`scripts/README.md`](scripts/README.md) for evaluation details.

## Supported Configuration

- `nvidia/nemotron-3.5-asr-streaming-0.6b`
- complete-audio transcription
- Model Runner V2
- eager execution
- CUDA
- TP=1
- PP=1
- one audio item per request
- batched independent requests
- greedy RNN-T decoding
- replay after preemption
- OpenAI-compatible `/v1/audio/transcriptions`

## Limitations

The following are currently unsupported:

- native cache-aware realtime streaming
- beam search
- speculative decoding
- quantization
- tensor or pipeline parallelism
- translation
- text prompts and hotwords
- structured outputs
- confidence scoring

Selected-token logprobs exposed by the current implementation are synthetic and are not model probabilities.

## License

Apache-2.0.

The NVIDIA checkpoint is distributed under its own license. See the [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for details.