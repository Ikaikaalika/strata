# CommonCompute Mac app: Strata pilot announcement draft

Status: **prepared, not sent, not wired into the release pipeline**. Release
version and date are deliberately unset. This draft accompanies an opt-in
experimental benchmark feature only after that feature has been implemented
and verified in the signed provider build. It does not announce a replacement
production engine or a measured fleet-wide speed/earnings improvement.

## Provider email / release announcement

Subject: Help shape faster, more efficient LLM inference on your Mac

CommonCompute [VERSION] introduces an optional Strata testing pilot for selected
Mac providers.

Strata is our experimental Apple-Silicon LLM runtime. We are building it to reduce
model response times, use memory more effectively, and find the right combination
of CPU, GPU and Neural Engine work for each Mac. Tests across smaller Macs and
high-memory systems will help us determine which optimizations and models are
actually useful for CommonCompute providers.

Participation is optional. Regular customer LLM jobs continue using the existing
MLX engine while the pilot runs controlled benchmark workloads. Tests have
defined time, memory and storage limits, and you can stop participating. Model
downloads require your approval; SSD model offload is disabled in this pilot.

Benchmarks may temporarily increase power consumption, temperature and fan
activity. We will explain the test window, diagnostic data collected, download
size and participation terms before you opt in.

Our aim is to expand useful model coverage and improve inference efficiency.
Actual speed, energy savings, job availability and earnings improvements are
not guaranteed; that is what the pilot will help us evaluate.

Update to [VERSION]. Selected providers can join at [VERIFIED OPT-IN LOCATION].
We will share what the tests establish before enabling Strata for customer work.

## Short Mac release-note entry

### Strata experimental testing

- Added an optional, limited Strata benchmark pilot for selected Apple-Silicon
  providers. The research targets faster LLM responses, more efficient memory
  use and hardware-specific CPU/GPU/Neural Engine execution.
- Existing customer LLM inference remains on MLX. The pilot does not claim
  faster full-model inference or support for every listed research target.
- Participation is opt-in, with test/resource limits, cancellation and download
  approval. SSD model offload is disabled. Benchmarks can increase power use,
  device temperature and fan activity while running.

## Required gates before using this copy

1. Implement and test the native worker bridge, opt-in UI, device allowlist,
   cancellation, fail-closed remote disable control and customer-work priority.
   These are proposed release features, not all present in the current app.
2. Prove that the signed/hardened/sandboxed build can execute each allowed native
   test. Keep private-ANE experiments separately configurable and verify the
   applicable distribution terms/third-party notices. Do not weaken the normal
   worker's isolation merely to make a research probe run.
3. Pin test binaries, model revisions and complete artifact hashes. Bound
   benchmark duration and resource use; serialize tests against other workloads.
   Obtain provider approval for model download location and size.
4. State diagnostic fields, retention/access and participation/compensation
   terms before opt-in. Use synthetic test prompts; do not collect customer
   prompts or outputs for this pilot. Validate those promises in implementation.
5. Replace [VERSION] and [VERIFIED OPT-IN LOCATION] only with verified release
   facts. Reconcile app/project version, release notes, signed DMG and appcast.
6. Preview the rendered provider message and exact intended recipient count,
   respect suppression/unsubscribe preferences, and prevent duplicate sends.
   Send only alongside the authorized, available release, never now as a preview.

## CommonCompute integration handoff

Inspected local source on 2026-09-07:

- `CHANGELOG.md` is the source of truth for generated Sparkle release notes;
  add the final short entry under the confirmed release, then regenerate HTML.
- `scripts/release.sh` has a provider announcement step after deployment. Its
  current request passes template/version/download URL, not these custom
  benefit paragraphs. A changelog edit alone does not wire this email copy.
- `apps/api-v2/src/email/templates.ts` defines `appUpdateAnnouncementEmail`;
  `apps/api-v2/src/routes/v1/admin.ts` invokes it. Add a reviewed, version-bound
  content path or a dedicated pilot announcement, with preview/suppression/
  deduplication gates. Do not turn every later app-update email into a stale
  Strata pilot announcement.
- The local CommonCompute checkout has extensive unrelated edits. Integrate on
  an isolated branch/worktree; do not stage or overwrite that checkout wholesale.

## Claims to keep out of the release announcement

Do not say "Strata is 4x faster," "beats MLX/Darkbloom," "uses every accelerator
for every model," "guarantees more earnings," "reduces your power bill," or
"runs any model up to 512 GB." Current evidence includes one-Mac CPU-copy and
ANE FFN microbenchmarks, not those product claims. The new C++ admission planner
has synthetic budget tests; it is not evidence of 512 GB model execution.

Engineering evidence: [fleet architecture](../design/COMMONCOMPUTE_FLEET_OPTIMIZATION.md)
and [native experiment report](../experiments/FLEET_NATIVE_OPTIMIZATION_20260907.md).
