#!/usr/bin/env python3
"""Quick demo showing MLX-only inference working."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from ollm import Inference

print("\n" + "="*70)
print("MLX-ONLY INFERENCE DEMO")
print("="*70 + "\n")

# Load model
print("Loading deepseek-coder-1.3b...")
inference = Inference("deepseek-coder-1.3b", device="mlx", logging=False)
inference.ini_model()
print("✅ Model loaded!\n")

# Generate code
prompt = "def quicksort(arr):"
print(f"Prompt: {prompt}\n")

inputs = inference.tokenizer(prompt, return_tensors="np")
input_ids = mx.array(inputs["input_ids"])

print("Generating...")
outputs = inference.model.generate(input_ids, max_new_tokens=50)
mx.eval(outputs)

response = inference.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated:\n{response}\n")

print("="*70)
print("✅ SUCCESS - MLX-only inference working!")
print("="*70 + "\n")
