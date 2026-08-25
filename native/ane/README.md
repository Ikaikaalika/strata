# Strata ANE capability worker

This native probe dynamically loads the private Apple Neural Engine frameworks
and inventories their Objective-C surface. Its default mode performs discovery
only. The optional execution mode generates a tiny projection, compiles it,
passes deterministic tensors through IOSurface, and compares the result with a
CPU reference.

Build and run without third-party dependencies:

```sh
make -C native/ane
native/ane/build/strata-ane-probe
native/ane/build/strata-ane-probe --execute-projection
native/ane/build/strata-ane-probe --benchmark-projection
```

The worker also exposes one bounded callable request protocol:

```sh
native/ane/build/strata-ane-probe --execute-linear-request /absolute/path/request.json
```

This protocol is not a general ANE backend. Schema v1 accepts only contiguous
FP16 logical input/output tensors shaped `[tokens=64, width=256]` and a
contiguous FP16 weight shaped `[out=256, in=256]` in `out_in` order. All tensor
files must be distinct direct children of the request directory. Input and
weight files must be nonsymlink regular files with exact byte sizes; the output
is published by an atomic same-directory rename and refuses nonregular targets.

The proven IOSurface layout is physical `[1, C=256, 1, S=64]`, so the worker
transposes logical token-major input into channel-major surface storage and
transposes output back before publication. A CPU-only round-trip invariant
checks these copies before private execution. Reported `dispatch_ms` measures
only `evaluateWithQoS:options:request:error:`. It excludes request/file I/O,
layout copies, MIL generation, compilation, model load/unload, readback, and
atomic output publication.

`--benchmark-projection` compiles and loads the fixed program once, reuses the
request and IOSurfaces, performs five warmups, and records fifty serialized
`evaluateWithQoS` calls. It remains a projection dispatch benchmark, not a
transformer or full-phase LLM benchmark, and is never planner eligible.

The probe emits one JSON document to stdout in either mode. A failure at any
private lifecycle stage is evidence, not availability: the JSON records the
last successful stage and keeps `execution_verified=false`. Private APIs can
change between macOS builds, so every new build must be probed again.

The private lifecycle and IOSurface request pattern were informed by the
MIT-licensed `maderix/ANE` research project. The M1-compatible MIL dialect and
weight-blob conventions were also informed by the MIT-licensed
`skyfallsin/ane.cpp` project, forked from John Mai's ANE-LM. These are community
reverse-engineering references, not Apple-supported interfaces or evidence of
Apple endorsement. See `THIRD_PARTY_NOTICES.md`.
