#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify vLLM Nemotron ASR plugin installation and registration."""

import sys
from importlib import import_module
from importlib.metadata import entry_points, version


def check_plugin_installed() -> bool:
    """Check if the plugin package is installed."""
    print("Checking plugin installation...")
    try:
        import_module("vllm_nemotron_asr_plugin")
        package_version = version("vllm-nemotron-asr-plugin")
        print(f"✓ Plugin package found (version {package_version})")
        return True
    except ImportError as exc:
        print(f"✗ Plugin package not found: {exc}")
        print("  Run: python -m pip install -e . --no-deps")
        return False


def check_registration_function() -> bool:
    """Check that the installed entry point loads the registration function."""
    print("\nChecking registration function...")
    try:
        from vllm_nemotron_asr_plugin import register

        entries = list(entry_points(group="vllm.general_plugins", name="nemotron_asr"))
        if len(entries) != 1 or entries[0].load() is not register:
            raise RuntimeError(
                "Expected one installed nemotron_asr plugin entry point."
            )
        print("✓ Registration function loaded successfully")
        return True
    except (ImportError, RuntimeError) as exc:
        print(f"✗ Cannot load registration function: {exc}")
        return False


def check_model_class() -> bool:
    """Check if the Nemotron model class can be imported."""
    print("\nChecking Nemotron model class...")
    try:
        from vllm_nemotron_asr_plugin.model import Nemotron3_5AsrForRNNT

        print(f"✓ {Nemotron3_5AsrForRNNT.__name__} class imported successfully")
        return True
    except ImportError as exc:
        print(f"✗ Cannot import Nemotron model class: {exc}")
        return False


def check_vllm_dependencies() -> bool:
    """Check if vLLM and its dependencies are available."""
    print("\nChecking vLLM dependencies...")
    dependencies = {
        "vllm": "vLLM",
        "torch": "PyTorch",
        "transformers": "Transformers",
    }
    all_ok = True
    for module_name, display_name in dependencies.items():
        try:
            module = import_module(module_name)
            module_version = getattr(module, "__version__", "unknown")
            print(f"✓ {display_name} {module_version}")
        except ImportError as exc:
            print(f"✗ Cannot import {display_name}: {exc}")
            all_ok = False
    return all_ok


def test_model_registration() -> bool:
    """Test that vLLM discovers and registers the plugin model."""
    print("\nTesting model registration...")
    try:
        from vllm import ModelRegistry, envs
        from vllm.plugins import load_general_plugins

        from vllm_nemotron_asr_plugin.model import Nemotron3_5AsrForRNNT

        if envs.VLLM_PLUGINS is not None and "nemotron_asr" not in envs.VLLM_PLUGINS:
            raise RuntimeError("VLLM_PLUGINS must include nemotron_asr or be unset.")
        load_general_plugins()
        model_cls = ModelRegistry.models["Nemotron3_5AsrForRNNT"].load_model_cls()
        if model_cls is not Nemotron3_5AsrForRNNT:
            raise RuntimeError("ModelRegistry did not resolve the plugin model class.")
        print(f"✓ Model registered successfully: {model_cls}")
        return True
    except Exception as exc:
        print(f"✗ Model registration failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def check_cuda_availability() -> bool:
    """Report CUDA availability without requiring it for installation checks."""
    print("\nChecking CUDA availability (optional)...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available (version {torch.version.cuda})")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
        else:
            print("⚠ CUDA unavailable; this plugin requires CUDA for inference.")
        return True
    except Exception as exc:
        print(f"⚠ Cannot check CUDA: {exc}")
        return True


def main() -> int:
    """Run all verification checks."""
    print("=" * 80)
    print("vLLM Nemotron ASR Plugin - Verification Script")
    print("=" * 80)

    checks = [
        ("Plugin Installation", check_plugin_installed),
        ("Registration Function", check_registration_function),
        ("Model Class Import", check_model_class),
        ("vLLM Dependencies", check_vllm_dependencies),
        ("Model Registration", test_model_registration),
        ("CUDA Availability (informational)", check_cuda_availability),
    ]
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as exc:
            print(f"\n✗ Unexpected error in {name}: {exc}")
            results.append((name, False))

    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("\nAll installation checks passed. Inference was not tested.")
        print("\nNext steps:")
        print("  1. Run python example_transcription.py to test transcription")
        print("  2. Read README.md for usage instructions")
        return 0

    print("\n⚠ Some checks failed. Please review the errors above.")
    print("\nTroubleshooting:")
    print("  1. Activate the environment where vLLM and the plugin are installed")
    print("  2. Install the plugin: python -m pip install -e . --no-deps")
    print("  3. Read README.md for requirements and usage instructions")
    return 1


if __name__ == "__main__":
    sys.exit(main())
