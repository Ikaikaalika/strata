# Common Compute integration playbook

## Current repository mapping

The Common Compute checkout inspected on 2026-08-25 has:

| Existing surface | Strata integration |
|---|---|
| `apps/provider/CommonCompute/Execution/ComputeServiceProtocol.swift` | Add dedicated Strata start/control RPCs and the restricted event-sink interface; do not reuse generic `execute` |
| `apps/provider/CommonComputeComputeService/ComputeService.swift` | Add bounded decode/validation and a persistent Strata engine owner |
| `apps/provider/CommonCompute/Runners/WorkloadRunner.swift` | Keep the host runner interface; adapt Strata events into existing progress and token streaming |
| `apps/provider/CommonCompute/Runners/MLXInferenceRunner.swift` | Preserve as `mlx_llm` oracle and fallback |
| `apps/provider/CommonCompute/Runners/RunnerRegistry.swift` | Register internal `strata_llm` behind a disabled-by-default feature flag |
| `apps/provider/CommonCompute/Models/MLXModelCatalog.swift` | Resolve catalog model ID to immutable revision/digest before building a Strata request |
| `apps/provider/CommonComputeComputeService/RuntimeProfiles.swift` | Do not send Strata through this generic entrypoint/environment contract |

The Common Compute worktree was dirty during inspection. Preserve those
unrelated changes when integrating this package.

## Copy destination

Recommended layout after unpacking the bundle:

```text
docs/integrations/strata-wire-v1/
apps/provider/CommonCompute/Execution/StrataWireV1.swift
apps/provider/CommonCompute/Execution/StrataXPCV1.swift
apps/provider/CommonCompute/Runners/XPCStrataRunner.swift
apps/provider/CommonComputeComputeService/StrataEngineService.swift
apps/provider/Tests/StrataWireV1Tests/
```

Add the two Swift reference sources to both the host and XPC-service targets or
move them into a small local Swift package linked by both targets. Keep one
source of truth; do not fork host and service copies.

## Gate A: contract bridge

1. Copy `StrataWireV1.swift`, `StrataXPCV1.swift`, the schema, and every golden
   fixture.
2. Add Swift tests that decode all `fixtures/valid` documents and reject every
   `fixtures/invalid` document.
3. Encode the valid Swift values and compare their normalized JSON objects to
   the fixture objects.
4. Keep `additionalProperties: false` behavior through the provided strict
   decoders; synthesized `JSONDecoder` alone silently ignores unknown keys.
5. Verify the packaged SHA-256 manifest before changing a copied file.

Gate A passes only when Strata's Python verifier and Common Compute's Swift
tests agree on every fixture.

## Gate B: deterministic XPC fixture

Implement the new RPCs with a tiny deterministic worker before linking MLX:

- accept one valid request and emit sequence `0` accepted;
- emit coalesced text events at a bounded cadence, never one RPC per token;
- emit exactly one terminal receipt;
- cancel before start, during generation, and after terminal completion;
- reject duplicate IDs, non-monotonic events, oversized messages, unknown
  fields, and event-sink invalidation;
- enforce a bounded ingress queue and terminate/restart cleanly;
- keep the XPC service without network, keychain, subprocess, package-install,
  dynamic-extension, or arbitrary filesystem authority.

The event sink exposes only `receiveStrataEnvelope(Data)`. The host validates
size, message type, request identity, sequence, and terminal state before
relaying anything to customer streaming or metering.

## Gate C: one pinned model

Use one exact Llama 3.2 1B artifact first. The host resolves its catalog ID,
revision, and digest. The XPC service independently re-verifies the artifact
manifest and owns the persistent model slot.

Compare Strata to `mlx_llm` for:

- identical chat-template input tokens;
- exact greedy output tokens and seeded-sampling repeatability;
- stop and maximum-output behavior;
- prompt, completion, and total-token accounting;
- cancellation before output and after partial output;
- cold load, warm model, unload, crash, and restart.

Before the first visible delta, Common Compute may retry through `mlx_llm`.
After partial output, it must surface the structured terminal failure and must
not blindly replay the prompt.

## Gate D: serving behavior

Run two requests against the same resident model and prove:

- independent, ordered streams with no token or KV mixing;
- bounded prefill chunks and continuous decode batching;
- fairness and deadline handling;
- cancellation of one request without disturbing the other;
- reserved KV/activation/temporary memory before admission;
- retryable overload rejection rather than unbounded queueing;
- exact results against separately executed MLX references.

Only after this gate should the provider advertise a shared `strata_llm` lane.

## Gates E through G

1. Record serialized cold/warm TTFT, inter-token latency, prompt/decode/server
   throughput, queue time, peak memory, batch occupancy, thermals, and errors.
2. Run at least 100 streamed lifecycle cases plus cancellation and fault
   injection.
3. Complete a 24-hour bounded-memory single-Mac soak.
4. Canary behind a feature flag and provider allowlist with automatic
   `mlx_llm` fallback before output.
5. Require the same signed runtime revision and accounting behavior on several
   Macs before the router prefers Strata.

## Initial feature policy

| Feature | v1 action |
|---|---|
| Plain system/user/assistant messages | Strata candidate |
| Greedy or bounded temperature/top-p sampling | Strata candidate after parity |
| Tool schemas/tool calls | Route to `mlx_llm` |
| Images, audio, files, arbitrary paths | Reject or use another bounded runner |
| SSD offload | Disabled in maximum-speed v1 |
| Private ANE | Research canary only; never implied by this contract |
| Unknown model/runtime revision | Route to `mlx_llm` |

## Definition of done for the first merge

- Contract and fixture tests pass in both repositories.
- The deterministic XPC fixture streams, cancels, restarts, and rejects invalid
  requests without MLX linked.
- `mlx_llm` behavior is unchanged.
- `strata_llm` is disabled by default and not advertised.
- No production deployment, catalog-wide enablement, or performance claim is
  included in the contract merge.
