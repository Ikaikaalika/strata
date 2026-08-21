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
```

The probe emits one JSON document to stdout in either mode. A failure at any
private lifecycle stage is evidence, not availability: the JSON records the
last successful stage and keeps `execution_verified=false`. Private APIs can
change between macOS builds, so every new build must be probed again.

The private lifecycle, MIL weight-blob layout, and IOSurface request pattern
were informed by the MIT-licensed `maderix/ANE` research project. See
`THIRD_PARTY_NOTICES.md`.
