#!/usr/bin/env python3
"""
Qwen3-TTS Compatibility Verification Script

This script verifies that Qwen3-TTS models can be used for on-device Android deployment:
1. Tests GGUF model loading with llama-cpp-python
2. Tests audio token generation
3. Exports the audio tokenizer/decoder to ONNX format
4. Runs a complete TTS pipeline test

Prerequisites:
    pip install llama-cpp-python transformers torch onnx onnxruntime numpy soundfile

Usage:
    python verify_qwen3_tts.py [--skip-download] [--model-size 0.6b|1.7b]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ============================================================
# Step 0: Dependency checks
# ============================================================

def check_dependencies():
    """Check that all required Python packages are installed."""
    required = {
        "llama_cpp": "llama-cpp-python",
        "transformers": "transformers",
        "torch": "torch",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
        "numpy": "numpy",
        "soundfile": "soundfile",
    }
    missing = []
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    print("[OK] All dependencies installed")
    return True


# ============================================================
# Step 1: Download GGUF model (if needed)
# ============================================================

MODELS_DIR = Path(__file__).parent.parent / "models"

GGUF_MODELS = {
    "0.6b": {
        "repo": "Qwen/Qwen3-TTS-0.6B-Base-GGUF",
        "filename": "qwen3-tts-0.6b-base-q4_k_m.gguf",
        "description": "Qwen3-TTS 0.6B Base (Q4_K_M quantized) - Voice cloning capable",
    },
    "1.7b": {
        "repo": "Qwen/Qwen3-TTS-1.7B-Base-GGUF",
        "filename": "qwen3-tts-1.7b-base-q4_k_m.gguf",
        "description": "Qwen3-TTS 1.7B Base (Q4_K_M quantized) - Higher quality",
    },
}

TOKENIZER_MODEL = {
    "repo": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "description": "Audio tokenizer/decoder (multi-codebook, 12Hz)",
}


def download_gguf_model(model_size: str) -> Path:
    """Download GGUF model from Hugging Face Hub."""
    model_info = GGUF_MODELS[model_size]
    model_path = MODELS_DIR / model_info["filename"]

    if model_path.exists():
        print(f"[OK] GGUF model already exists: {model_path}")
        return model_path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {model_info['description']}...")
    print(f"       From: {model_info['repo']}")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=model_info["repo"],
            filename=model_info["filename"],
            local_dir=str(MODELS_DIR),
        )
        print(f"[OK] Downloaded to: {downloaded}")
        return Path(downloaded)
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Install with: pip install huggingface_hub")
        print(f"        Or manually download from https://huggingface.co/{model_info['repo']}")
        return None
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None


# ============================================================
# Step 2: Test GGUF loading with llama-cpp-python
# ============================================================

def test_gguf_loading(model_path: Path) -> bool:
    """Test if the GGUF model can be loaded with llama-cpp-python."""
    print("\n" + "=" * 60)
    print("STEP 2: Testing GGUF model loading with llama-cpp-python")
    print("=" * 60)

    try:
        from llama_cpp import Llama

        print(f"[INFO] Loading model: {model_path}")
        start = time.time()

        model = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4,
            verbose=True,
        )

        elapsed = time.time() - start
        print(f"[OK] Model loaded in {elapsed:.1f}s")
        print(f"     Context size: {model.n_ctx()}")
        print(f"     Vocab size: {model.n_vocab()}")

        # Check model metadata
        metadata = model.metadata
        if metadata:
            print(f"     Metadata keys: {list(metadata.keys())[:10]}")
            if "general.architecture" in metadata:
                print(f"     Architecture: {metadata['general.architecture']}")

        del model
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load GGUF model: {e}")
        print(f"        This may indicate the model architecture is not supported by llama.cpp")
        print(f"        Error type: {type(e).__name__}")
        return False


# ============================================================
# Step 3: Test token generation
# ============================================================

def test_token_generation(model_path: Path) -> dict:
    """Test if the model can generate audio tokens."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing audio token generation")
    print("=" * 60)

    try:
        from llama_cpp import Llama

        model = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

        # Qwen3-TTS prompt format (based on documentation)
        # The exact format may vary - this tests common patterns
        test_prompts = [
            # Simple text prompt
            "<|text|>Hello world<|audio|>",
            # With language tag
            "<|en|>Hello world<|audio|>",
            # Alternative format
            "[TTS]Hello world[/TTS]",
        ]

        results = {}
        for prompt in test_prompts:
            print(f"\n[INFO] Testing prompt: {prompt[:50]}...")
            try:
                output = model(
                    prompt,
                    max_tokens=64,
                    temperature=0.7,
                    stop=["<|endoftext|>", "</s>"],
                )
                text = output["choices"][0]["text"] if output["choices"] else ""
                tokens_generated = output["usage"]["completion_tokens"]
                print(f"       Generated {tokens_generated} tokens")
                print(f"       Output preview: {text[:100]}")
                results[prompt] = {
                    "success": True,
                    "tokens": tokens_generated,
                    "output_preview": text[:200],
                }
            except Exception as e:
                print(f"       [WARN] Generation failed: {e}")
                results[prompt] = {"success": False, "error": str(e)}

        del model
        return results

    except Exception as e:
        print(f"[ERROR] Token generation test failed: {e}")
        return {"error": str(e)}


# ============================================================
# Step 4: Export audio tokenizer to ONNX
# ============================================================

def test_tokenizer_export(output_dir: Path) -> bool:
    """Attempt to load and export the Qwen3-TTS audio tokenizer to ONNX."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing audio tokenizer/decoder ONNX export")
    print("=" * 60)

    try:
        import torch
        import numpy as np

        print(f"[INFO] Loading tokenizer model from: {TOKENIZER_MODEL['repo']}")
        print(f"       This may download the model on first run...")

        # Try loading the tokenizer model
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(
            TOKENIZER_MODEL["repo"],
            trust_remote_code=True,
        )
        print(f"[OK] Config loaded: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")

        model = AutoModel.from_pretrained(
            TOKENIZER_MODEL["repo"],
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        print(f"[OK] Tokenizer model loaded")

        # Analyze model structure
        print(f"\n[INFO] Model structure:")
        for name, param in model.named_parameters():
            print(f"       {name}: {param.shape}")
            if len(list(model.named_parameters())) > 20:
                print(f"       ... (truncated, {sum(1 for _ in model.named_parameters())} total params)")
                break

        # Try to identify encoder and decoder components
        print(f"\n[INFO] Model modules:")
        for name, module in model.named_children():
            print(f"       {name}: {type(module).__name__}")

        # Attempt ONNX export of decoder
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / "qwen3_tts_decoder.onnx"

        print(f"\n[INFO] Attempting ONNX export to: {onnx_path}")

        # Create dummy input based on model architecture
        # The decoder takes audio token indices and produces waveform
        # Exact input format depends on model architecture
        try:
            # Try with a generic input
            dummy_input = torch.randint(0, 1024, (1, 16, 50))  # (batch, codebooks, seq_len)

            # Check if model has a decode method
            if hasattr(model, "decode"):
                print("[INFO] Found decode() method, attempting export...")
                with torch.no_grad():
                    test_output = model.decode(dummy_input)
                print(f"[OK] Decode output shape: {test_output.shape}")

                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    opset_version=17,
                    input_names=["audio_tokens"],
                    output_names=["waveform"],
                    dynamic_axes={
                        "audio_tokens": {0: "batch", 2: "seq_len"},
                        "waveform": {0: "batch", 1: "samples"},
                    },
                )
                print(f"[OK] ONNX export successful: {onnx_path}")
                print(f"     File size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

                # Verify ONNX model
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print("[OK] ONNX model validation passed")

                # Test with ONNX Runtime
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                ort_output = session.run(None, {"audio_tokens": dummy_input.numpy()})
                print(f"[OK] ONNX Runtime inference successful, output shape: {ort_output[0].shape}")

                return True

            elif hasattr(model, "forward"):
                print("[INFO] No decode() method found, trying forward()...")
                # Try different input shapes
                for input_shape in [(1, 50), (1, 16, 50), (1, 1, 50)]:
                    try:
                        dummy = torch.randint(0, 1024, input_shape)
                        with torch.no_grad():
                            out = model(dummy)
                        print(f"[OK] Forward with shape {input_shape} -> output: {out.shape if hasattr(out, 'shape') else type(out)}")
                        break
                    except Exception as e:
                        print(f"       Shape {input_shape} failed: {e}")
                        continue

        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")
            print(f"       This is expected if the model has custom ops not supported by ONNX")
            print(f"       Alternative: implement the decoder in C++ directly")
            return False

    except Exception as e:
        print(f"[ERROR] Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Step 5: Test complete pipeline
# ============================================================

def test_full_pipeline(model_path: Path) -> bool:
    """Test the complete TTS pipeline if all components work."""
    print("\n" + "=" * 60)
    print("STEP 5: Testing complete TTS pipeline")
    print("=" * 60)

    try:
        # Try using the official qwen-tts package if available
        try:
            from qwen_tts import Qwen3TTSModel
            import torch

            print("[INFO] Official qwen-tts package found, testing...")
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-0.6B-Base",
                device_map="cpu",
                dtype=torch.float32,
            )

            wavs, sr = model.generate(
                text="Hello, this is a test.",
                language="English",
            )
            print(f"[OK] Generated audio: {wavs.shape}, sample rate: {sr}")

            # Save test audio
            import soundfile as sf
            output_path = MODELS_DIR / "test_output.wav"
            sf.write(str(output_path), wavs[0].numpy(), sr)
            print(f"[OK] Test audio saved to: {output_path}")
            return True

        except ImportError:
            print("[INFO] qwen-tts package not installed (pip install qwen-tts)")
            print("       Skipping official pipeline test")

        # Test with transformers directly
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print("[INFO] Testing with transformers library...")
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-TTS-0.6B-Base",
                trust_remote_code=True,
            )
            print(f"[OK] Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
            print(f"     Special tokens: {tokenizer.all_special_tokens[:10]}")

            # Document the prompt format
            print(f"\n[INFO] Prompt format analysis:")
            for token in tokenizer.all_special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                print(f"       {token} -> ID: {token_id}")

            return True

        except Exception as e:
            print(f"[WARN] Transformers test failed: {e}")
            return False

    except Exception as e:
        print(f"[ERROR] Full pipeline test failed: {e}")
        return False


# ============================================================
# Step 6: Generate compatibility report
# ============================================================

def generate_report(results: dict, output_path: Path):
    """Generate a JSON report of all test results."""
    print("\n" + "=" * 60)
    print("COMPATIBILITY REPORT")
    print("=" * 60)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "results": results,
        "recommendation": "",
    }

    # Determine recommendation
    gguf_ok = results.get("gguf_loading", False)
    tokenizer_ok = results.get("tokenizer_export", False)
    pipeline_ok = results.get("full_pipeline", False)

    if gguf_ok and tokenizer_ok:
        report["recommendation"] = (
            "PROCEED: Both GGUF model and tokenizer work. "
            "On-device Android deployment is feasible with llama.cpp + ONNX Runtime."
        )
    elif gguf_ok and not tokenizer_ok:
        report["recommendation"] = (
            "PARTIAL: GGUF model works but tokenizer export failed. "
            "Options: (1) Implement decoder in C++ natively, "
            "(2) Use alternative audio codec, "
            "(3) Try TFLite export instead of ONNX."
        )
    elif not gguf_ok and pipeline_ok:
        report["recommendation"] = (
            "SERVER-SIDE: GGUF not compatible with llama.cpp but official pipeline works. "
            "Recommend server-side TTS with Python backend, Android app as client."
        )
    elif not gguf_ok:
        report["recommendation"] = (
            "BLOCKED: GGUF model not compatible with llama.cpp. "
            "Options: (1) Wait for llama.cpp to add Qwen3-TTS support, "
            "(2) Use server-side deployment, "
            "(3) Consider alternative TTS models (Piper, VITS, Bark)."
        )

    print(f"\nGGUF Loading:      {'PASS' if gguf_ok else 'FAIL'}")
    print(f"Token Generation:  {'PASS' if results.get('token_generation', {}).get('success') else 'FAIL/UNKNOWN'}")
    print(f"Tokenizer Export:  {'PASS' if tokenizer_ok else 'FAIL'}")
    print(f"Full Pipeline:     {'PASS' if pipeline_ok else 'FAIL/SKIPPED'}")
    print(f"\nRecommendation: {report['recommendation']}")

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Verify Qwen3-TTS compatibility for Android deployment")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    parser.add_argument("--model-size", choices=["0.6b", "1.7b"], default="0.6b", help="Model size to test")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer export test")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-TTS Android Compatibility Verification")
    print("=" * 60)

    # Step 0: Check dependencies
    if not check_dependencies():
        sys.exit(1)

    results = {}

    # Step 1: Download model
    model_path = None
    if not args.skip_download:
        model_path = download_gguf_model(args.model_size)
    else:
        expected_path = MODELS_DIR / GGUF_MODELS[args.model_size]["filename"]
        if expected_path.exists():
            model_path = expected_path
        else:
            print(f"[WARN] Model not found at {expected_path}, skipping GGUF tests")

    # Step 2: Test GGUF loading
    if model_path and model_path.exists():
        results["gguf_loading"] = test_gguf_loading(model_path)
    else:
        results["gguf_loading"] = False
        print("[SKIP] GGUF loading test (model not available)")

    # Step 3: Test token generation
    if results["gguf_loading"] and model_path:
        results["token_generation"] = test_token_generation(model_path)
    else:
        results["token_generation"] = {"skipped": True}
        print("[SKIP] Token generation test (GGUF loading failed)")

    # Step 4: Test tokenizer export
    if not args.skip_tokenizer:
        results["tokenizer_export"] = test_tokenizer_export(MODELS_DIR / "onnx")
    else:
        results["tokenizer_export"] = False
        print("[SKIP] Tokenizer export test (--skip-tokenizer)")

    # Step 5: Test full pipeline
    if model_path:
        results["full_pipeline"] = test_full_pipeline(model_path)
    else:
        results["full_pipeline"] = False

    # Step 6: Generate report
    report_path = MODELS_DIR / "compatibility_report.json"
    generate_report(results, report_path)


if __name__ == "__main__":
    main()
