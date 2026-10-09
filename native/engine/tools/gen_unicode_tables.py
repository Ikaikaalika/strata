#!/usr/bin/env python3
"""Generate native/engine/src/unicode_data.inc from the reference tokenizer.

Lokahi must split and normalize text exactly as Hugging Face ``tokenizers``
does, and the reference does not use a single Unicode version: its regex
engine (Oniguruma) classifies code points with recent Unicode data, while its
NFC normalizer (unicode-normalization-alignments) uses older data. The tables
are therefore extracted from the installed ``tokenizers`` package:

* general categories are probed through ``\\p{..}`` classes in the reference
  regex engine, five bit-sliced probes over every scalar value;
* canonical decompositions, combining classes and compositions come from
  Python's ``unicodedata`` (which must be at least as new as the reference
  normalizer), restricted to the code points the reference normalizer knows
  and verified against it.

Regenerate after upgrading the pinned ``tokenizers`` and commit the result:

    python3 native/engine/tools/gen_unicode_tables.py > native/engine/src/unicode_data.inc
"""
from __future__ import annotations

import sys
import unicodedata

import tokenizers
from tokenizers import Regex, normalizers, pre_tokenizers

CATEGORIES = [
    "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No",
    "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So",
    "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
]
MAX = 0x110000
SURROGATES = range(0xD800, 0xE000)
SBASE, SCOUNT = 0xAC00, 11172


def scalar_values() -> list[int]:
    return [cp for cp in range(MAX) if cp not in SURROGATES]


def probe_categories() -> list[int]:
    """Category index per code point, as the reference regex engine sees it."""
    cps = scalar_values()
    text = "".join(map(chr, cps))
    index = [0] * MAX
    for bit in range(5):
        members = [name for i, name in enumerate(CATEGORIES) if i >> bit & 1 and name != "Cs"]
        pattern = "[" + "".join(r"\p{%s}" % name for name in members) + "]+"
        split = pre_tokenizers.Split(Regex(pattern), "removed", invert=True)
        for _, (begin, end) in split.pre_tokenize_str(text):
            for i in range(begin, end):
                index[cps[i]] |= 1 << bit
    for cp in SURROGATES:
        index[cp] = CATEGORIES.index("Cs")
    assert max(index) < len(CATEGORIES), "reference reports an unknown category"
    return index


def normalize_each(normalizer, items: list[str]) -> list[str]:
    """Normalizes every item with one call; NUL separates items and is inert."""
    assert not any("\0" in item for item in items)
    out = normalizer.normalize_str("\0".join(items)).split("\0")
    assert len(out) == len(items)
    return out


def reference_normalization() -> tuple[list, list, list]:
    nfd, nfc = normalizers.NFD(), normalizers.NFC()
    cps = [cp for cp in scalar_values() if cp != 0]
    decomposed = normalize_each(nfd, [chr(cp) for cp in cps])
    known = set()
    for cp, ref in zip(cps, decomposed):
        if ref != chr(cp):
            assert ref == unicodedata.normalize("NFD", chr(cp)), f"U+{cp:04X}: unicodedata is older than the reference"
            known.add(cp)

    decompositions = []
    for cp in sorted(known):
        if SBASE <= cp < SBASE + SCOUNT:
            continue  # Hangul syllables are algorithmic.
        parts = [int(p, 16) for p in unicodedata.decomposition(chr(cp)).split()]
        assert 1 <= len(parts) <= 2, hex(cp)
        decompositions.append((cp, parts[0], parts[1] if len(parts) == 2 else 0))

    # A mark the reference knows reorders around a probe mark exactly as
    # unicodedata says; one it does not know keeps class 0 there.
    marks = [cp for cp in cps if unicodedata.combining(chr(cp))]
    probes = ["á" + chr(m) if unicodedata.combining(chr(m)) == 1 else "a" + chr(m) + "̴" for m in marks]
    reordered = normalize_each(nfd, probes)
    known_marks = {m for m, probe, ref in zip(marks, probes, reordered) if ref == unicodedata.normalize("NFD", probe)}
    combining = []
    start = None
    for cp in range(MAX + 1):
        ccc = unicodedata.combining(chr(cp)) if cp in known_marks else 0
        if start is not None and ccc != current:
            combining.append((start, cp - 1, current))
            start = None
        if start is None and ccc:
            start, current = cp, ccc

    compositions = []
    for cp, first, second in decompositions:
        if second and unicodedata.normalize("NFC", chr(first) + chr(second)) == chr(cp):
            compositions.append((first, second, cp))
    composed = normalize_each(nfc, [chr(first) + chr(second) for first, second, _ in compositions])
    assert all(ref == chr(cp) for ref, (_, _, cp) in zip(composed, compositions))
    return decompositions, combining, sorted(compositions)


def category_ranges(index: list[int]) -> list[tuple[int, int, int]]:
    ranges = []
    start = 0
    for cp in range(1, MAX + 1):
        if cp == MAX or index[cp] != index[start]:
            ranges.append((start, cp - 1, index[start]))
            start = cp
    return ranges


def emit(name: str, ctype: str, rows: list[tuple[int, ...]], per_line: int = 4) -> None:
    print(f"static const {ctype} {name}[] = {{")
    for i in range(0, len(rows), per_line):
        chunk = rows[i : i + per_line]
        print("    " + " ".join("{" + ", ".join(f"0x{v:X}" for v in row) + "}," for row in chunk))
    print("};")


def main() -> int:
    categories = category_ranges(probe_categories())
    decompositions, combining, compositions = reference_normalization()
    reference = f"tokenizers {tokenizers.__version__}"
    print("// Generated by native/engine/tools/gen_unicode_tables.py; do not edit.")
    print(f"// Reference: {reference}; values from unicodedata {unicodedata.unidata_version}")
    print(f"// (Python {sys.version.split()[0]}) restricted to what the reference knows.")
    print("// clang-format off")
    print(f'#define LOKAHI_UNICODE_REFERENCE "{reference}"')
    print("// General category order: " + " ".join(CATEGORIES))
    emit("kCategoryRanges", "CategoryRange", categories)
    emit("kCombiningClasses", "CombiningRange", combining)
    emit("kDecompositions", "Decomposition", decompositions, per_line=3)
    emit("kCompositions", "Composition", compositions, per_line=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
