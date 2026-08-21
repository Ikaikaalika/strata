# Strata ANE capability worker

This native probe performs runtime discovery only. It dynamically loads the
private Apple Neural Engine frameworks and inventories the Objective-C surface
required by a future isolated ANE execution worker. It does not compile or
dispatch an ANE graph, so `execution_verified` and `numeric_verified` are
always `false`.

Build and run without third-party dependencies:

```sh
make -C native/ane
native/ane/build/strata-ane-probe
```

The probe emits one JSON document to stdout. Private APIs can change between
macOS builds; discovery must be treated as a capability observation rather
than a compatibility guarantee.
