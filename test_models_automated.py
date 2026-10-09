#!/usr/bin/env python3
"""
Automated model testing script - tests all working MLX models.
No user interaction required - runs automatically.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from lokahi import Inference


# Test paragraph
TEST_INPUT = """Machine learning has revolutionized artificial intelligence.
Neural networks can perform complex tasks like image recognition and natural language processing.
These models learn patterns from data, improving accuracy over time."""


def test_model(model_id):
    """Test a single model with basic inference."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_id}")
    print(f"{'='*70}")

    result = {
        "model": model_id,
        "success": False,
        "error": None,
        "load_time": 0,
        "gen_time": 0,
        "tokens_per_sec": 0
    }

    try:
        # Load model
        print("Loading model...", end=" ", flush=True)
        load_start = time.time()

        inference = Inference(model_id, device="mlx", logging=False)
        inference.ini_model()

        result["load_time"] = time.time() - load_start
        print(f"✅ ({result['load_time']:.1f}s)")

        # Tokenize input
        inputs = inference.tokenizer(TEST_INPUT, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])

        print(f"Input tokens: {input_ids.shape[1]}")

        # Generate tokens
        num_tokens = 20
        print(f"Generating {num_tokens} tokens...", end=" ", flush=True)

        gen_start = time.time()
        outputs = inference.model.generate(input_ids, max_new_tokens=num_tokens)
        mx.eval(outputs)

        result["gen_time"] = time.time() - gen_start
        result["tokens_per_sec"] = num_tokens / result["gen_time"]

        print(f"✅ ({result['tokens_per_sec']:.1f} tok/s)")

        # Decode output
        response = inference.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(TEST_INPUT):].strip()

        print(f"\n💬 Output preview: {generated[:100]}...")

        result["success"] = True

        # Clean up
        del inference

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        result["error"] = str(e)

        import traceback
        traceback.print_exc()

    return result


def main():
    print("\n" + "="*70)
    print("AUTOMATED MODEL TEST SUITE")
    print("="*70 + "\n")

    # Models that should work with MLX backend
    working_models = [
        "deepseek-coder-1.3b",
    ]

    # Models that work but take more memory (optional to test)
    larger_models = [
        "deepseek-coder-6.7b",
    ]

    # Gated models (require HuggingFace auth)
    gated_models = [
        "llama3-1B-chat",
        "llama3-3B-chat",
        "llama3-8B-chat",
        "gemma3-12B",
    ]

    # Models not yet implemented
    not_implemented = [
        "qwen3-next-80B",
        "gpt-oss-20B",
    ]

    print("Testing working models:")
    for m in working_models:
        print(f"  • {m}")

    print("\nLarger models (skipped for speed):")
    for m in larger_models:
        print(f"  • {m}")

    print("\nGated models (require HF auth, skipped):")
    for m in gated_models:
        print(f"  • {m}")

    print("\nNot yet implemented (skipped):")
    for m in not_implemented:
        print(f"  • {m}")

    print("\n" + "="*70)

    results = []
    start_time = time.time()

    for model in working_models:
        result = test_model(model)
        results.append(result)
        time.sleep(2)  # Brief pause between models

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}\n")

    if successful:
        print(f"{'Model':<25} {'Load Time':>10} {'Speed':>12}")
        print("-"*70)
        for r in successful:
            print(f"{r['model']:<25} {r['load_time']:>9.1f}s {r['tokens_per_sec']:>10.1f} t/s")

    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  • {r['model']}: {r['error'][:60]}...")

    print(f"\n⏱️  Total time: {total_time:.1f}s")
    print("\n" + "="*70 + "\n")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
