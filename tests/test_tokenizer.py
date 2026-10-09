"""Correctness evidence: the native tokenizer against Hugging Face ``tokenizers``.

Small BPE tokenizers are trained at test time with the reference library and
saved in the three SentencePiece-style shapes the native tokenizer supports:
Gemma (Replace normalizer, Split pre-tokenizer, byte fallback), Llama 2
(Prepend + Replace normalizer, Strip decoder), and Metaspace with unknown-token
fusion. Every string is encoded and decoded by both implementations and must
match exactly. No model files are downloaded.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import (  # noqa: E402
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>"]
CORPUS = [
    "Hello world, this is a test of the tokenizer.",
    "The quick brown fox jumps over the lazy dog 1234567890 times.",
    "Aloha kākou! Lōkahi means unity; ʻŌlelo Hawaiʻi.",
    "数字と漢字のテスト、そして한국어 문장도 있어요.",
    "Emoji 🙂🌊🦓 and symbols €£¥ © ® ™ ∑ ∫ √",
    "  leading spaces,\ttabs,\nnewlines\r\nand   runs   of   spaces  ",
    "def main():\n    return {'key': [1, 2, 3]}  # code",
]
EDGE_TEXTS = [
    "",
    " ",
    "   ",
    "\n",
    "Hello world",
    "Zebra 🦓 über naïve café",
    "<start_of_turn>user\nHi there<end_of_turn>\n<start_of_turn>model\n",
    "<start><start_of_turn><start_of_tur",
    "a<unused0>b<unused0><unused0>",
    "<bos>text with a literal <bos> inside",
    "tab\there and\rcarriage",
    "▁already▁has▁metaspace",
    "x" * 300,
    "🙂" * 40,
    "混合 mixed テキスト with spaces  and  doubles",
    "é combining mark and ​ zero width",
]
POOL = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.;:!?'\"-_()[]{}\n\t") + list(
    "āēīōūʻ漢字テスト한국🙂🌊€√é́"
)


def _random_texts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        length = rng.randint(0, 60)
        pieces = [rng.choice(POOL) for _ in range(length)]
        if rng.random() < 0.2:
            pieces.insert(rng.randint(0, len(pieces)), rng.choice(["<start_of_turn>", "<end_of_turn>", "<unused0>", "<bos>"]))
        texts.append("".join(pieces))
    return texts


TEXTS = EDGE_TEXTS + CORPUS + _random_texts(400, seed=7)


def _train(normalizer, pre_tokenizer, byte_fallback: bool) -> tuple[dict, list]:
    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=byte_fallback, fuse_unk=True))
    if normalizer is not None:
        tok.normalizer = normalizer
    if pre_tokenizer is not None:
        tok.pre_tokenizer = pre_tokenizer
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS, show_progress=False)
    tok.train_from_iterator(CORPUS * 20, trainer)
    data = json.loads(tok.to_str())
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in data["model"]["merges"]]
    return data["model"]["vocab"], merges


def _vocab_with_bytes(trained: dict) -> dict:
    vocab = {token: index for index, token in enumerate(SPECIALS)}
    for value in range(256):
        vocab[f"<0x{value:02X}>"] = len(vocab)
    for token, _ in sorted(trained.items(), key=lambda item: item[1]):
        vocab.setdefault(token, len(vocab))
    return vocab


def _add_tokens(tok: Tokenizer) -> None:
    tok.add_special_tokens(
        [AddedToken(s, normalized=False, special=True) for s in SPECIALS + ["<start_of_turn>", "<end_of_turn>"]]
    )
    tok.add_tokens([AddedToken(s, normalized=False, special=False) for s in ["<unused0>", "<start>"]])


def gemma_style() -> Tokenizer:
    normalizer = normalizers.Replace(" ", "▁")
    pre = pre_tokenizers.Split(" ", "merged_with_previous")
    trained, merges = _train(normalizer, pre, byte_fallback=True)
    tok = Tokenizer(
        models.BPE(vocab=_vocab_with_bytes(trained), merges=merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True)
    )
    tok.normalizer = normalizer
    tok.pre_tokenizer = pre
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    _add_tokens(tok)
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    return tok


def llama2_style() -> Tokenizer:
    normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    trained, merges = _train(normalizer, None, byte_fallback=True)
    tok = Tokenizer(
        models.BPE(vocab=_vocab_with_bytes(trained), merges=merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True)
    )
    tok.normalizer = normalizer
    tok.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    _add_tokens(tok)
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 1)])
    return tok


def metaspace_unk() -> Tokenizer:
    pre = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=True)
    trained, merges = _train(None, pre, byte_fallback=False)
    tok = Tokenizer(models.BPE(vocab=trained, merges=merges, unk_token="<unk>", byte_fallback=False, fuse_unk=True))
    tok.pre_tokenizer = pre
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="first")
    _add_tokens(tok)
    return tok


FACTORIES = {"gemma": gemma_style, "llama2": llama2_style, "metaspace-unk": metaspace_unk}


@pytest.fixture(scope="module", params=sorted(FACTORIES))
def saved_tokenizer(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"tok-{request.param}")
    tok = FACTORIES[request.param]()
    tok.save(str(directory / "tokenizer.json"))
    return directory, Tokenizer.from_file(str(directory / "tokenizer.json"))


def _cli(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("add_special", [True, False])
def test_encode_matches_reference(native_cli, saved_tokenizer, tmp_path, add_special):
    directory, reference = saved_tokenizer
    texts_file = tmp_path / "texts.json"
    texts_file.write_text(json.dumps(TEXTS), encoding="utf-8")
    args = ["tokenize", "--model", str(directory), "--texts-file", str(texts_file)]
    if not add_special:
        args.append("--no-special")
    ours = _cli(native_cli, *args)["ids"]
    expected = [reference.encode(text, add_special_tokens=add_special).ids for text in TEXTS]
    mismatches = [(text, a, b) for text, a, b in zip(TEXTS, ours, expected) if a != b]
    assert not mismatches, mismatches[:3]


@pytest.mark.parametrize("skip_special", [True, False])
def test_decode_matches_reference(native_cli, saved_tokenizer, tmp_path, skip_special):
    directory, reference = saved_tokenizer
    id_lists = [reference.encode(text).ids for text in TEXTS]
    # Arbitrary id sequences too: decoding must not assume encoder output.
    rng = random.Random(11)
    size = reference.get_vocab_size(with_added_tokens=True)
    id_lists += [[rng.randrange(size) for _ in range(rng.randint(0, 30))] for _ in range(200)]
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(id_lists), encoding="utf-8")
    args = ["detokenize", "--model", str(directory), "--ids-file", str(ids_file)]
    if not skip_special:
        args.append("--keep-special")
    ours = _cli(native_cli, *args)["texts"]
    expected = [reference.decode(ids, skip_special_tokens=skip_special) for ids in id_lists]
    mismatches = [(ids, a, b) for ids, a, b in zip(id_lists, ours, expected) if a != b]
    assert not mismatches, mismatches[:3]


def test_legacy_string_merges_load_identically(native_cli, tmp_path):
    tok = gemma_style()
    data = json.loads(tok.to_str())
    data["model"]["merges"] = [" ".join(pair) for pair in data["model"]["merges"]]
    (tmp_path / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    texts_file = tmp_path / "texts.json"
    texts_file.write_text(json.dumps(TEXTS[:50]), encoding="utf-8")
    ours = _cli(native_cli, "tokenize", "--model", str(tmp_path), "--texts-file", str(texts_file))["ids"]
    assert ours == [tok.encode(text).ids for text in TEXTS[:50]]


def test_added_token_ids_disagreeing_with_reference_fail_closed(native_cli, tmp_path):
    data = json.loads(gemma_style().to_str())
    data["added_tokens"][-1]["id"] += 3  # tokenizers would silently renumber this token
    (tmp_path / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [str(native_cli), "tokenize", "--model", str(tmp_path), "--text", "x"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "but tokenizers would assign" in result.stderr


def test_unsupported_components_fail_closed(native_cli, tmp_path):
    tok = gemma_style()
    data = json.loads(tok.to_str())
    data["normalizer"] = {"type": "NFKC"}
    (tmp_path / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [str(native_cli), "tokenize", "--model", str(tmp_path), "--text", "x"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "unsupported normalizer 'NFKC'" in result.stderr


def test_invalid_utf8_is_rejected(native_cli, saved_tokenizer):
    directory, _ = saved_tokenizer
    result = subprocess.run(
        [str(native_cli).encode(), b"tokenize", b"--model", str(directory).encode(), b"--text", b"bad \xff byte"],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"not valid UTF-8" in result.stderr


@pytest.mark.skipif(not os.environ.get("LOKAHI_MODEL_ROOT"), reason="needs a staged Gemma 3 snapshot")
def test_real_gemma3_tokenizer_matches_reference(native_cli, tmp_path):
    root = Path(os.environ["LOKAHI_MODEL_ROOT"])
    snapshots = sorted(root.glob("models--mlx-community--gemma-3-1b-it-qat-4bit/snapshots/*/tokenizer.json"))
    if not snapshots:
        pytest.skip("Gemma 3 1B snapshot is not staged under LOKAHI_MODEL_ROOT")
    path = snapshots[-1]
    reference = Tokenizer.from_file(str(path))
    texts_file = tmp_path / "texts.json"
    texts_file.write_text(json.dumps(TEXTS), encoding="utf-8")
    ours = _cli(native_cli, "tokenize", "--model", str(path.parent), "--texts-file", str(texts_file))["ids"]
    assert ours == [reference.encode(text).ids for text in TEXTS]
