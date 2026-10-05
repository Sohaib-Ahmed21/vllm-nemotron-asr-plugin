# SPDX-License-Identifier: Apache-2.0
"""Evaluate complete-audio transcription on FLEURS test samples."""

import argparse
import io
import os
import unicodedata

import jiwer
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem
from transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr import (
    DEFAULT_PROMPT_DICTIONARY,
)
from vllm import LLM, SamplingParams


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        char for char in text if not unicodedata.category(char).startswith("P")
    )
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en_us", help="FLEURS language config")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    locale = args.language.replace("_", "-")
    language = next(
        (key for key in DEFAULT_PROMPT_DICTIONARY if key.lower() == locale.lower()),
        "auto",
    )

    revision = (
        HfApi().dataset_info("google/fleurs", revision="refs/convert/parquet").sha
    )
    fs = HfFileSystem()
    paths = sorted(
        fs.glob(f"datasets/google/fleurs@{revision}/{args.language}/test/*.parquet")
    )
    if not paths:
        parser.error(f"No FLEURS test data for {args.language!r}")
    prompts, references = [], []
    for path in paths:
        with fs.open(path, "rb") as file:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=10):
                for row in batch.to_pylist():
                    audio, sample_rate = sf.read(
                        io.BytesIO(row["audio"]["bytes"]), dtype="float32"
                    )
                    prompts.append(
                        {
                            "prompt": "",
                            "multi_modal_data": {"audio": (audio, sample_rate)},
                            "mm_processor_kwargs": {"language": language},
                        }
                    )
                    references.append(normalize(row["transcription"]))
                    if len(prompts) == args.num_samples:
                        break
                if len(prompts) == args.num_samples:
                    break
        if len(prompts) == args.num_samples:
            break
    if not prompts:
        raise RuntimeError("FLEURS returned no audio samples.")

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    llm = LLM(
        model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        enforce_eager=True,
        dtype="float16",
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
    )
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=511))
    if any(output.outputs[0].finish_reason == "length" for output in outputs):
        raise RuntimeError("A transcript reached the token limit; evaluation aborted.")
    hypotheses = [normalize(output.outputs[0].text) for output in outputs]
    print(f"FLEURS revision: {revision}")
    print(f"Language: {args.language}; prompt: {language}; samples: {len(references)}")
    print(f"WER: {jiwer.wer(references, hypotheses):.2%}")
    print(f"CER: {jiwer.cer(references, hypotheses):.2%}")


if __name__ == "__main__":
    main()
