# Evaluation Scripts

## FLEURS evaluation

Install evaluation dependencies and run:

```bash
uv pip install -e '.[eval]'
python scripts/eval_fleurs.py --language en_us --num-samples 10
```

Example for another language:

```bash
python scripts/eval_fleurs.py --language de_de --num-samples 10
```

The script evaluates complete-audio transcription on the FLEURS test split using the installed vLLM plugin and reports WER and CER.

## Package version

Preview or update the package version:

```bash
python scripts/bump_version.py --dry-run
python scripts/bump_version.py patch
python scripts/bump_version.py minor
python scripts/bump_version.py major
python scripts/bump_version.py --set 0.2.0
```

The script updates `[project].version` in `pyproject.toml`. It does not create commits, tags, or releases.