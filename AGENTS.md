# Strata agent instructions

Strata is an Apple-Silicon LLM inference runtime and learning laboratory. Keep
the Python import namespace `ollm` until a dedicated compatibility migration.

## Engineering boundaries

- The primary integrator freezes shared contracts before parallel work.
- Specialist agents receive exclusive writable paths. Do not edit outside the
  assigned paths or commit from a specialist lane.
- Preserve unrelated dirty work. Do not reset, clean, stash, or rebase it.
- Strata is built in public at `https://github.com/Ikaikaalika/strata`.
  Publish reviewed source, tests, designs and non-sensitive experiment evidence.
  Keep private fleet records, credentials, model weights, environments and
  unaudited recovery candidates out of Git. See `docs/BUILD_IN_PUBLIC.md`.
- Do not download packages, repositories, models, or large data until the user
  chooses the destination.
- Use generated small fixtures before real model weights.
- The external HDD failed. Do not access `/Volumes/Tyler HDD`, including through
  preserved symlinks. The approved source checkout is
  `/Users/tylergee/Documents/strata-recovered`; the approved model root is
  `/Users/tylergee/Library/Application Support/Strata/models`, both on the
  internal SSD. Other contributors must choose their own SSD locations.
  Keep a 40 GiB free-space reserve for local experiments. SSD placement does
  not by itself qualify production model paging.

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
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
```

The recovered checkout's `.venv` is the locally validated environment, not a
portable installation requirement. Do not install dependencies silently when
it is absent; report the gap and obtain a destination first. Also run:

```sh
PYTHONPATH=src .venv/bin/python -m compileall -q src tests benchmarks
git diff --check
```

Native probes must be isolated executables and fail closed. Hardware performance
runs are serialized; agents may code in parallel but must not run competing GPU,
ANE, or storage benchmarks concurrently.
