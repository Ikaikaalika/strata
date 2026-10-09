# Lōkahi agent instructions

Lōkahi (formerly Strata) is an Apple-Silicon LLM inference runtime and learning
laboratory owned by Common Compute LLC. The Python package is `lokahi`; the old
`ollm` namespace was removed without a compatibility alias.

## Engineering boundaries

- The primary integrator freezes shared contracts before parallel work.
- Specialist agents receive exclusive writable paths. Do not edit outside the
  assigned paths or commit from a specialist lane.
- Preserve unrelated dirty work. Do not reset, clean, stash, or rebase it.
- Do not download packages, repositories, models, or large data until the user
  chooses the destination.
- Use generated small fixtures before real model weights.
- Treat `/Volumes/Tyler HDD` as source-checkout storage only. The user has
  confirmed that it is an HDD. Never use it as an SSD offload target, place
  model weights or caches there to represent SSD behavior, or run storage
  qualification benchmarks against it. Obtain a user-approved SSD destination
  before SSD experiments or large model placement.

## Evidence rules

Label results as correctness, synthetic, or hardware evidence.

- Correctness evidence compares deterministic outputs to an independent oracle.
- Synthetic evidence cannot support hardware throughput, energy, or SSD claims.
- Hardware evidence records chip, macOS build, shapes, dtype, warmup, iterations,
  units, numerical error, and timing boundaries.
- Core ML configuration or private-framework discovery does not prove ANE
  execution. Require compile, dispatch, readback, and numerical verification.
- An on-disk ANE compile cache is not live weight paging.
- Apple Silicon storage remains SSD to unified memory to the compute engine;
  never claim direct SSD execution without measured proof.

## Local gates

The current known MLX-capable test environment is:

```sh
PYTHONPATH=src /usr/bin/python3 -m pytest -q
```

The default Homebrew Python may lack `pytest`; report that as an environment
gap. Also run:

```sh
PYTHONPATH=src python3 -m compileall -q src tests benchmarks
git diff --check
```

Native probes must be isolated executables and fail closed. Hardware performance
runs are serialized; agents may code in parallel but must not run competing GPU,
ANE, or storage benchmarks concurrently.
