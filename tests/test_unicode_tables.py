"""The committed Unicode tables are exactly what the pinned reference yields.

``native/engine/src/unicode_data.inc`` is extracted from Hugging Face
``tokenizers`` by ``native/engine/tools/gen_unicode_tables.py``. Regenerating
it must reproduce the committed tables, so they can be audited and updated
deliberately when the reference changes its Unicode data.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "native" / "engine"
TABLES = ENGINE / "src" / "unicode_data.inc"
GENERATOR = ENGINE / "tools" / "gen_unicode_tables.py"


def _tables(text: str) -> list[str]:
    # Comment lines name the Python that generated the file; the data must not depend on it.
    return [line for line in text.splitlines() if not line.startswith("//")]


def test_tables_regenerate_from_the_pinned_reference():
    tokenizers = pytest.importorskip("tokenizers")
    committed = TABLES.read_text(encoding="utf-8")
    pinned = re.search(r'#define LOKAHI_UNICODE_REFERENCE "tokenizers ([^"]+)"', committed).group(1)
    if tokenizers.__version__ != pinned:
        pytest.skip(f"tables come from tokenizers {pinned}; {tokenizers.__version__} is installed")
    generated = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True, text=True, check=True).stdout
    assert _tables(generated) == _tables(committed)
