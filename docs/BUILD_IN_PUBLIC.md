# Building Strata in public

Decision: 2026-09-07. Develop Strata in the existing public repository,
[Ikaikaalika/strata](https://github.com/Ikaikaalika/strata). CommonCompute is the
intended first integration and testing partner, not a reason to keep runtime
development private until launch.

## What the public checkpoint contains

- Native Objective-C private-ANE experiments, C/ARM64 NEON tensor-layout code,
  and a standalone C++ memory-admission planner with a C ABI.
- Offline tests, reproducible workload contracts, public model-artifact pins,
  measured component receipts, numerical failures and noisy runs.
- The [target runtime architecture](design/STRATA_NATIVE_RUNTIME_V2.md),
  [bounded experiment process](design/BOUNDED_ARCHITECTURE_SEARCH.md), and
  [latest experiment report](experiments/FLEET_NATIVE_OPTIMIZATION_20260907.md).
- A [provider announcement draft](release/COMMONCOMPUTE_PROVIDER_ANNOUNCEMENT_DRAFT.md)
  that is explicitly not sent, integrated or evidence of a released pilot.

These are research checkpoints. MLX-LM remains the full-model oracle/fallback
for supported artifacts. Current ANE and CPU-copy wins do not establish a
full-model Strata speedup, a Darkbloom win, production SSD paging, complete
native Metal/ANE model execution, or an integrated CommonCompute worker.
The planner's 16–512 GiB tests are synthetic arithmetic, not fleet benchmarks.

## Public evidence, private operational data

Keep public: reviewed implementation, tests, architecture decisions, generated
fixture definitions, public model metadata, and non-sensitive benchmark receipts.
Keep local: credentials, provider/device IDs and telemetry, customer prompts and
outputs, downloaded weights, interpreter environments, caches and binaries.

`.local/` contains private operational evidence. `recovery/` contains unaudited
historical candidates and their local provenance inventory. Both are ignored
and preserved locally, not shipped in a public clone. A recovered file becomes
active source only after individual review and validation; never promote the
whole quarantine directory. Retain existing third-party license notices and
review provenance before importing additional reverse-engineered code.

Historical benchmark receipts retain their original source/binary hashes and
host-local paths. Their recorded prompts are controlled fixtures, not customer
requests. A public commit may contain later source than an older receipt:
match hashes before reproducing or attributing results. Never rewrite an old
receipt to make a new candidate appear measured.

## Git workflow

1. Use `origin` at `https://github.com/Ikaikaalika/strata.git` for Strata work.
   The historical `upstream` remote preserves the project's ancestry; it is
   not this project's push destination.
2. Work on a bounded feature branch. Review the diff, data provenance and
   validation evidence before committing an explicit file allowlist.
3. Push the branch to `origin` and record the commit SHA. Review and merge to
   `main` as a separate decision; do not force-push or erase concurrent work.
4. Keep package releases, signed Mac app builds, provider communications and
   production promotion separate from source publication. This checkpoint does
   not change CommonCompute or send announcements.

No model weights or environment binaries belong in these commits. Public CI is
useful for portable contract checks; a CI pass alone is not Apple-Silicon
hardware evidence. Local native validation remains explicitly recorded.

## Validation and reproduction

On the approved recovery host, the existing `.venv` provides the recorded MLX
environment. On another host, choose package/model storage locations before
installing or downloading anything. The dated
[environment snapshot](../benchmarks/requirements-recovery-20260907.txt) records
what was used, not a universal cross-platform lockfile. The recovery baseline
script currently uses that host's explicit model root; porting it is a new
source candidate and must produce a new receipt.

The following tests do not download models or dispatch valid ANE inference:

```sh
make -C native/ane -f Makefile.experiments test \
  BUILD_DIR=build/public-scalar
make -C native/ane -f Makefile.experiments test \
  BUILD_DIR=build/public-neon LAYOUT_FLAGS=-DSTRATA_USE_NEON_TRANSPOSE
make -C native/planning test BUILD_DIR=build/public-validation
STRATA_ANE_EXPERIMENT_BUILD="$PWD/native/ane/build/public-neon" \
  PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_recovery_experiment_contracts.py tests/test_native_ane_experiment_cli.py \
  tests/test_ane_backend.py tests/test_ane_executor.py tests/test_ane_segment.py \
  tests/test_benchmark_target.py tests/test_fleet_target_contracts.py
PYTHONPATH=src .venv/bin/python -m compileall -q src tests benchmarks
git diff --check
```

Use separate build directories for compiler-flag variants. Real ANE/MLX timing
runs follow the experiment reports, use fresh output names and run serially.
Undocumented ANE APIs remain experimental and OS-dependent. Verify distribution
and signed-worker compatibility separately; do not weaken provider isolation
to make an experimental probe run.

Checkpoint validation on 2026-09-07: 117 selected Python tests plus 12 subtests
passed; 400 transpose cases passed in each scalar/NEON UBSan build; 1,837 native
admission cases passed under UBSan. Compilation and Git whitespace checks passed.
This is scoped regression validation, not a claim that the entire legacy suite
or a production integration was tested. No real-model performance runs were
repeated as part of publishing this checkpoint; the dated receipts remain the
hardware evidence.
