# Lōkahi wire v1 integration kit

This directory is the portable contract boundary between Lōkahi and the
Common Compute macOS provider. Copy the complete directory into the Common
Compute repository; do not copy individual fixture or source files without the
manifest produced by the bundle builder.

Version: `1.0.0`

## What this kit freezes

- A bounded `llm.generate` request for one pinned text-model revision.
- Cancel and drain controls.
- Ordered accepted/progress/text/queue/warning events.
- Exactly one terminal completed/failed/cancelled/rejected receipt.
- Swift `Codable` reference types, strict decoders, and stream state checks.
- Cross-language valid and invalid golden fixtures.
- An offline verifier and deterministic ZIP builder using only Python's
  standard library.

The v1 request intentionally has no executable path, arbitrary arguments,
environment variables, tools, URLs, package installation, network flag,
filesystem path, or customer-supplied code. Requests needing unsupported v1
features remain on Common Compute's `mlx_llm` compatibility lane.

## Verify and build

From the Lōkahi repository root:

```sh
/usr/bin/python3 integrations/commoncompute/lokahi-wire-v1/tools/verify_kit.py
/usr/bin/xcrun swiftc -typecheck \
  integrations/commoncompute/lokahi-wire-v1/swift/LokahiWireV1.swift \
  integrations/commoncompute/lokahi-wire-v1/swift/LokahiXPCV1.swift
/usr/bin/python3 integrations/commoncompute/lokahi-wire-v1/tools/build_bundle.py \
  --output-dir integrations/commoncompute/dist
```

The ZIP contains a SHA-256 manifest for every transferred source, schema,
fixture, and document. Re-run `tools/verify_kit.py` after copying the unpacked
directory into Common Compute.

## Contents

| Path | Purpose |
|---|---|
| `schemas/lokahi-wire-v1.schema.json` | Draft 2020-12 machine-readable contract |
| `fixtures/valid/` | Messages that must decode and validate |
| `fixtures/invalid/` | Messages that must fail closed |
| `swift/LokahiWireV1.swift` | Reference `Codable` types and strict validation |
| `swift/LokahiXPCV1.swift` | Narrow XPC and event-sink protocol reference |
| `tools/verify_kit.py` | Offline cross-language fixture verifier |
| `tools/build_bundle.py` | Deterministic bundle and checksum builder |
| `INTEGRATION.md` | Common Compute path map and gated implementation order |

## Claim boundary

This kit completes a portable Gate-A contract candidate. It does not implement
the Common Compute XPC worker, prove sandbox execution, or make Lōkahi a
production runner. Those claims require the acceptance gates in
`INTEGRATION.md`.
