from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).parents[1]
KIT = ROOT / "integrations/commoncompute/strata-wire-v1"
VERIFY = KIT / "tools/verify_kit.py"
BUILD = KIT / "tools/build_bundle.py"


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("strata_wire_verify", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_commoncompute_wire_fixtures_fail_closed() -> None:
    module = _load_verify_module()

    counts = module.verify_fixture_tree(KIT)

    assert counts == {"valid": 7, "invalid": 9}


def test_commoncompute_schema_freezes_four_message_families() -> None:
    schema = json.loads(
        (KIT / "schemas/strata-wire-v1.schema.json").read_text(encoding="utf-8")
    )

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(schema["$defs"]) >= {"start", "control", "event", "receipt"}
    assert schema["$defs"]["start"]["properties"]["operation"] == {
        "const": "llm.generate"
    }
    assert schema["$defs"]["memory"]["properties"]["ssd_offload"] == {
        "const": "disabled"
    }
    assert schema["$defs"]["start"]["additionalProperties"] is False


def test_commoncompute_bundle_is_deterministic_and_manifested(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(BUILD),
        "--root",
        str(KIT),
        "--output-dir",
        str(tmp_path),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    first_report = json.loads(first.stdout)
    bundle = Path(first_report["output"])
    first_bytes = bundle.read_bytes()

    second = subprocess.run(command, check=True, capture_output=True, text=True)
    second_report = json.loads(second.stdout)

    assert bundle.read_bytes() == first_bytes
    assert second_report["sha256"] == hashlib.sha256(first_bytes).hexdigest()
    with zipfile.ZipFile(bundle) as archive:
        root = "strata-commoncompute-integration-v1/"
        manifest = json.loads(archive.read(root + "BUNDLE_MANIFEST.json"))
        assert manifest["version"] == "1.0.0"
        for entry in manifest["files"]:
            content = archive.read(root + entry["path"])
            assert len(content) == entry["bytes"]
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]


def test_commoncompute_kit_does_not_ship_executable_request_fields() -> None:
    schema_text = (KIT / "schemas/strata-wire-v1.schema.json").read_text(
        encoding="utf-8"
    )
    start_properties = json.loads(schema_text)["$defs"]["start"]["properties"]

    assert not {
        "entrypoint",
        "args",
        "environment",
        "network",
        "install_packages",
        "task_directory",
        "tools",
    } & set(start_properties)
