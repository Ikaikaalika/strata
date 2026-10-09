#!/usr/bin/env python3
"""Quick test for a single model."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from lokahi import Inference

model_id = sys.argv[1] if len(sys.argv) > 1 else "deepseek-coder-6.7b"

print(f"\n{'='*70}")
print(f"Testing: {model_id}")
print(f"{'='*70}\n")

# Load model
print("Loading model...")
start = time.time()
inference = Inference(model_id, device="mlx", logging=False)
inference.ini_model()
print(f"✅ Loaded in {time.time() - start:.1f}s\n")

# Test inference
test_input = "def fibonacci(n):"
print(f"Input: {test_input}")

inputs = inference.tokenizer(test_input, return_tensors="np")
input_ids = mx.array(inputs["input_ids"])

print(f"Generating 30 tokens...\n")
start = time.time()
outputs = inference.model.generate(input_ids, max_new_tokens=30)
mx.eval(outputs)
elapsed = time.time() - start

response = inference.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Output:\n{response}\n")
print(f"✅ Generated in {elapsed:.1f}s ({30/elapsed:.1f} tok/s)\n")
