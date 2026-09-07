"""Offline, pinned-artifact MLX-LM baseline; never a Strata speedup receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path("/Users/tylergee/Library/Application Support/Strata/models")
MODELS = {
    "Llama-3.2-3B-Instruct-4bit": ("mlx-community/Llama-3.2-3B-Instruct-4bit", "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"),
    "Qwen3-0.6B-4bit": ("mlx-community/Qwen3-0.6B-4bit", "73e3e38d981303bc594367cd910ea6eb48349da8"),
    "Qwen3.5-0.8B-OptiQ-4bit": ("mlx-community/Qwen3.5-0.8B-OptiQ-4bit", "ef60586933bd2cc02b763f77eb8839a5114bbec1"),
}


def artifact(name):
    repo, revision = MODELS[name]
    path = ROOT / name
    if ROOT.is_symlink() or path.is_symlink() or not path.is_dir():
        raise ValueError("model must be a real directory on the approved internal SSD")
    files = sorted(p for p in path.iterdir() if p.name.endswith((".json", ".safetensors", ".jinja", ".txt")))
    records = []
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("missing weights")
    for file in files:
        if file.is_symlink() or not file.is_file():
            raise ValueError("symlink or nonregular model file")
        provenance = path / ".cache/huggingface/download" / (file.name + ".metadata")
        lines = provenance.read_text().splitlines()
        if len(lines) < 2 or lines[0] != revision:
            raise ValueError(f"unpinned local file: {file.name}")
        digest = hashlib.sha256()
        git_digest = hashlib.sha1(f"blob {file.stat().st_size}\0".encode())
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
                git_digest.update(block)
        expected = lines[1].strip('"')
        actual = digest.hexdigest() if len(expected) == 64 else git_digest.hexdigest()
        if expected != actual:
            raise ValueError(f"artifact hash mismatch: {file.name}")
        records.append({"name": file.name, "sha256": digest.hexdigest(), "bytes": file.stat().st_size})
    return path, {"repository": repo, "revision": revision, "files": records}


def prompt_tokens(tokenizer, length):
    seed = tokenizer.encode("Explain how a computer processes a language model prompt, step by step. ", add_special_tokens=False)
    seed = [int(t) for t in seed if t not in tokenizer.eos_token_ids]
    if not seed or length < 1:
        raise ValueError("invalid prompt fixture")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--prefill-step-size", type=int, default=2048, choices=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--repetitions", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    if args.output.exists() or any(t not in (64, 512, 2048) for t in args.prompt_tokens):
        parser.error("existing receipt or unbounded workload")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".hf")
    from benchmark_mlx_lm import _hardware
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    report = {"schema_version": 1, "benchmark": "recovered-mlx-lm-baseline", "success": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "backend": "mlx_lm",
              "promotion_eligible": False, "ssd_offload": "disabled", "ane_dispatches": 0,
              "batch_size": 1, "generation_tokens": 64, "warmups": 1,
              "repetitions": args.repetitions, "prefill_step_size": args.prefill_step_size,
              "software": {n: version(n) for n in ("mlx", "mlx-lm", "transformers")},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "claim_boundary": "Warm local API baseline; synthetic repeated-text prompts, no chat/network, no Strata candidate, no energy measurement; MLX prefill rate includes first-token processing",
              "records": [], "summaries": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        path, report["artifact"] = artifact(args.model)
        report["hardware_before"] = _hardware()
        begin = time.perf_counter()
        model, tokenizer = load(str(path), lazy=False, tokenizer_config={"trust_remote_code": False})
        report["load_seconds"] = time.perf_counter() - begin
        for length in args.prompt_tokens:
            prompt = prompt_tokens(tokenizer, length)
            tokenizer._eos_token_ids = set()  # Fixed work, exactly 64 generated tokens.
            for repetition in range(-1, args.repetitions):
                mx.random.seed(730)
                mx.reset_peak_memory()
                started = time.perf_counter()
                tokens, timestamps = [], []
                for response in stream_generate(model, tokenizer, prompt, max_tokens=64,
                                                sampler=make_sampler(temp=0), prefill_step_size=args.prefill_step_size):
                    timestamps.append(time.perf_counter())
                    tokens.append(int(response.token))
                ended = time.perf_counter()
                if len(tokens) != 64:
                    raise ValueError(f"unexpected token count {len(tokens)}")
                record = {"prompt_tokens": length, "prompt_token_ids": prompt, "repetition": repetition,
                          "warmup": repetition < 0, "output_token_ids": tokens,
                          "ttft_ms": (timestamps[0] - started) * 1000,
                          "inter_token_ms": [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])],
                          "prefill_tps": response.prompt_tps, "decode_tps": response.generation_tps,
                          "peak_memory_gb": response.peak_memory, "wall_seconds": ended - started,
                          "goodput_tps": len(tokens) / (ended - started)}
                report["records"].append(record)
                print(json.dumps({k: v for k, v in record.items() if k not in ("prompt_token_ids", "output_token_ids", "inter_token_ms")}), flush=True)
            rows = [r for r in report["records"] if r["prompt_tokens"] == length and not r["warmup"]]
            if any(r["output_token_ids"] != rows[0]["output_token_ids"] for r in rows):
                raise ValueError("greedy repetitions did not agree")
            report["summaries"].append({"prompt_tokens": length, **{
                k: statistics.median(r[k] for r in rows) for k in
                ("ttft_ms", "prefill_tps", "decode_tps", "peak_memory_gb", "wall_seconds", "goodput_tps")}})
        report["hardware_after"] = _hardware()
        report["success"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
