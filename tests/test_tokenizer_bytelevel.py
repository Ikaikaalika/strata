"""Correctness evidence: byte-level BPE against Hugging Face ``tokenizers``.

One byte-level vocabulary is trained at test time with the reference library
and saved in the configurations the popular byte-level checkpoints ship:
Qwen 3 (NFC, regex Split, ByteLevel), Llama 3 and GLM (digit runs of three,
ignore_merges), GPT-OSS o200k (case-aware letter classes), DeepSeek V3 (three
chained regex Splits) and GPT-2 (ByteLevel's built-in regex, with and without
a prefix space). Every text is pre-tokenized, encoded and decoded by both
implementations and must match exactly, including text drawn from every
Unicode plane. No model files are downloaded.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import (  # noqa: E402
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

# Split patterns as they appear in the published tokenizer.json files.
QWEN3 = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
LLAMA3 = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
O200K = (
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)
DEEPSEEK_V3 = [
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    # Contains literal CR and LF characters inside the classes, as published.
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
    "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
]

CORPUS = [
    "Hello world, this is a test of the tokenizer. It's what we'll use; they've said I'd go.",
    "The quick brown fox jumps over the lazy dog 1234567890 times.",
    "Aloha kākou! Lōkahi means unity; ʻŌlelo Hawaiʻi.",
    "数字と漢字のテスト、そして한국어 문장도 있어요. 中文分词测试。",
    "Emoji 🙂🌊🦓 and symbols €£¥ © ® ™ ∑ ∫ √",
    "  leading spaces,\ttabs,\nnewlines\r\nand   runs   of   spaces  ",
    "def main():\n    return {'key': [1, 2, 3]}  # code\n\n\nprint(main())",
    "Привет, мир! Ελληνικά κείμενα. עברית ועוד. العربية ١٢٣",
    "URLs like https://example.com/path?q=1&r=2 and e-mail@example.org",
]
SPECIALS = {
    "qwen3": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    "llama3": ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    "glm": ["<|endoftext|>", "[gMASK]", "<sop>", "<|user|>", "<|assistant|>"],
    "gpt-oss": ["<|startoftext|>", "<|return|>", "<|start|>", "<|message|>", "<|channel|>", "<|end|>"],
    "deepseek-v3": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"],
    "gpt2": ["<|endoftext|>"],
    "gpt2-prefix-space": ["<|endoftext|>"],
}
# Not special, as Qwen 3's reasoning and tool-call markers are.
PLAIN_ADDED = ["<think>", "</think>", "<tool_call>"]
# Matched after normalization: NFC composes "cafe\u0301" into this token.
NORMALIZED_ADDED = ["café"]

EDGE_TEXTS = [
    "", " ", "   ", "\n", "\r\n\r\n", "\n\n\n  x", "Hello world", "hello  world   ", "a\u00a0b\u3000c",
    "I'M HERE 'S 'Ll 'ſ 'K", "x'sy 'S'T'RE'VE'M'LL'D'", "12345678901 ½ ²³ ١٢٣٤ ⅫⅠ 12/34/5678",
    "ABCdef DEFghi ǅungla ʻokina ʹ Kamehameha Pōhaku 4567 34567", "e\u0301 cafe\u0301 a\u0316\u0301 \u1100\u1161\u11a8 \u00c5 \u212b",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n</think>",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
    "<|start|>assistant<|channel|>final<|message|>Hi<|end|>",
    "<｜begin▁of▁sentence｜><｜User｜>你好<｜Assistant｜>",
    "x" * 300, "🙂" * 40, " " * 50 + "x", "\n" * 20, "1" * 40, "!!!???...///\n\n", "path/to/file.txt\r\n",
    "日本語のテキストとカタカナ、ひらがな一龥", "混合 mixed テキスト with spaces  and  doubles",
    "\U00013460\U00010d40 Unicode 16 letters", "\U00011935\U00011930 not composed by the reference",
]
POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    " \t\n\r\x0b\x0c\x85\xa0\u2028\u2029\u3000\u2003  ",
    "'sStTreREveVEmMllLLdDſK",
    ".,;:!?\"#$%&()*+-/<=>@[\\]^_`{|}~",
    "āēīōūʻ漢字テスト한국ひらがなカタカナ一龥中文",
    "\u0301\u0308\u0316\u0334\u20dd\u0903é\u1100\u1161\u11a8",
    "١٢٣४५६ⅫⅠ½²",
    "🙂🌊🦓€£¥©®™∑∫√",
]


def _random_text(rng: random.Random, specials: list[str]) -> str:
    out = []
    for _ in range(rng.randint(0, 50)):
        r = rng.random()
        if r < 0.82:
            out.append(rng.choice(rng.choice(POOLS)))
        elif r < 0.97:
            while True:
                cp = rng.randrange(1, 0x110000)
                if not 0xD800 <= cp < 0xE000:
                    break
            out.append(chr(cp))
        else:
            out.append(rng.choice(specials + PLAIN_ADDED + NORMALIZED_ADDED))
    return "".join(out)


def _texts(name: str) -> list[str]:
    rng = random.Random(f"bytelevel-{name}")
    return EDGE_TEXTS + CORPUS + [_random_text(rng, SPECIALS[name]) for _ in range(500)]


_TRAINED: dict = {}
# Words absent from the corpus, added to the vocabulary without merges.
WHOLE_WORDS = [" Pōhaku", "Kamehameha", " 4567"]


def _trained() -> tuple[dict, list]:
    """A byte-level vocabulary and merges, plus words only ignore_merges reaches."""
    if not _TRAINED:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=900, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
        )
        tok.train_from_iterator(CORPUS * 20, trainer)
        data = json.loads(tok.to_str())
        vocab = data["model"]["vocab"]
        mapped = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        for word in WHOLE_WORDS:
            token = "".join(piece for piece, _ in mapped.pre_tokenize_str(word))
            assert token not in vocab
            vocab[token] = len(vocab)
        merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in data["model"]["merges"]]
        _TRAINED.update(vocab=vocab, merges=merges)
    return dict(_TRAINED["vocab"]), list(_TRAINED["merges"])


def _split(pattern: str):
    return pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)


def _byte_level_tokenizer(name: str) -> Tokenizer:
    vocab, merges = _trained()
    ignore_merges = name in {"llama3", "glm", "gpt-oss"}
    tok = Tokenizer(
        models.BPE(vocab=vocab, merges=merges, continuing_subword_prefix="", end_of_word_suffix="",
                   ignore_merges=ignore_merges)
    )
    bare = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    if name == "qwen3":
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.Sequence([_split(QWEN3), bare])
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
    elif name in {"llama3", "glm"}:
        tok.pre_tokenizer = pre_tokenizers.Sequence([_split(LLAMA3), bare])
        if name == "llama3":
            tok.post_processor = processors.Sequence([
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(single="<|begin_of_text|> $A",
                                              special_tokens=[("<|begin_of_text|>", len(vocab))]),
            ])
        else:
            tok.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=False)])
    elif name == "gpt-oss":
        tok.pre_tokenizer = pre_tokenizers.Sequence([_split(O200K), bare])
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
    elif name == "deepseek-v3":
        tok.normalizer = normalizers.Sequence([])
        tok.pre_tokenizer = pre_tokenizers.Sequence([_split(p) for p in DEEPSEEK_V3] + [bare])
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
    elif name in {"gpt2", "gpt2-prefix-space"}:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=name.endswith("space"), use_regex=True)
        tok.post_processor = processors.ByteLevel(trim_offsets=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(s, normalized=False, special=True) for s in SPECIALS[name]])
    tok.add_tokens([AddedToken(s, normalized=False, special=False) for s in PLAIN_ADDED])
    tok.add_tokens([AddedToken(s, normalized=True, special=False) for s in NORMALIZED_ADDED])
    return tok


NAMES = sorted(SPECIALS)


@pytest.fixture(scope="module", params=NAMES)
def saved(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"bytelevel-{request.param}")
    _byte_level_tokenizer(request.param).save(str(directory / "tokenizer.json"))
    reference = Tokenizer.from_file(str(directory / "tokenizer.json"))
    return request.param, directory, reference


def _cli(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_fixture_keeps_the_published_configuration(saved):
    name, directory, _ = saved
    data = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    assert data["model"]["continuing_subword_prefix"] == ""
    assert data["decoder"]["type"] == "ByteLevel"
    if name == "qwen3":
        assert data["normalizer"]["type"] == "NFC"
        assert data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == QWEN3
    if name == "deepseek-v3":
        assert [p["pattern"]["Regex"] for p in data["pre_tokenizer"]["pretokenizers"][:3]] == DEEPSEEK_V3
    normalized = [t for t in data["added_tokens"] if t["normalized"]]
    assert [t["content"] for t in normalized] == NORMALIZED_ADDED


def test_words_match_reference(native_cli, saved, tmp_path):
    name, directory, reference = saved
    texts = _texts(name)
    ours = _cli(native_cli, "pretokenize", "--model", str(directory),
                "--texts-file", str(_write(tmp_path / "texts.json", texts)))["words"]
    normalizer = reference.normalizer
    expected = []
    for text in texts:
        normalized = normalizer.normalize_str(text) if normalizer is not None else text
        expected.append([word for word, _ in reference.pre_tokenizer.pre_tokenize_str(normalized)])
    mismatches = [(text, a, b) for text, a, b in zip(texts, ours, expected) if a != b]
    assert not mismatches, mismatches[:3]


@pytest.mark.parametrize("add_special", [True, False])
def test_encode_matches_reference(native_cli, saved, tmp_path, add_special):
    name, directory, reference = saved
    texts = _texts(name)
    args = ["tokenize", "--model", str(directory), "--texts-file", str(_write(tmp_path / "texts.json", texts))]
    if not add_special:
        args.append("--no-special")
    ours = _cli(native_cli, *args)["ids"]
    expected = [reference.encode(text, add_special_tokens=add_special).ids for text in texts]
    mismatches = [(text, a, b) for text, a, b in zip(texts, ours, expected) if a != b]
    assert not mismatches, mismatches[:3]


@pytest.mark.parametrize("skip_special", [True, False])
def test_decode_matches_reference(native_cli, saved, tmp_path, skip_special):
    name, directory, reference = saved
    id_lists = [reference.encode(text).ids for text in _texts(name)]
    # Arbitrary ids too: single-byte tokens in any order exercise lossy UTF-8.
    rng = random.Random(f"decode-{name}")
    size = reference.get_vocab_size(with_added_tokens=True)
    id_lists += [[rng.randrange(size) for _ in range(rng.randint(0, 30))] for _ in range(300)]
    byte_ids = [reference.token_to_id(c) for c in pre_tokenizers.ByteLevel.alphabet()]
    id_lists += [[rng.choice(byte_ids) for _ in range(rng.randint(1, 12))] for _ in range(300)]
    args = ["detokenize", "--model", str(directory), "--ids-file", str(_write(tmp_path / "ids.json", id_lists))]
    if not skip_special:
        args.append("--keep-special")
    ours = _cli(native_cli, *args)["texts"]
    expected = [reference.decode(ids, skip_special_tokens=skip_special) for ids in id_lists]
    mismatches = [(ids, a, b) for ids, a, b in zip(id_lists, ours, expected) if a != b]
    assert not mismatches, mismatches[:3]


def test_normalized_added_token_matches_after_nfc(native_cli, tmp_path):
    reference = _byte_level_tokenizer("qwen3")
    reference.save(str(tmp_path / "tokenizer.json"))
    text = "un cafe\u0301 noir"
    ids = _cli(native_cli, "tokenize", "--model", str(tmp_path), "--text", text, "--no-special")["ids"]
    assert reference.token_to_id("café") in ids
    assert ids == reference.encode(text, add_special_tokens=False).ids


def test_ignore_merges_takes_whole_words_from_the_vocabulary(native_cli, saved):
    name, directory, reference = saved
    whole = reference.token_to_id("ĠPÅįhaku")
    ids = _cli(native_cli, "tokenize", "--model", str(directory), "--text", "Aloha Pōhaku", "--no-special")["ids"]
    assert ids == reference.encode("Aloha Pōhaku", add_special_tokens=False).ids
    assert (whole in ids) == (name in {"llama3", "glm", "gpt-oss"})


def test_unsupported_regex_fails_closed(native_cli, tmp_path):
    data = json.loads(_byte_level_tokenizer("qwen3").to_str())
    data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\w+|\s+"
    _write(tmp_path / "tokenizer.json", data)
    result = subprocess.run(
        [str(native_cli), "tokenize", "--model", str(tmp_path), "--text", "x"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "unsupported escape \\w" in result.stderr


def _model_with_tokenizer(directory: Path, tok: Tokenizer, eos: str) -> Path:
    from lokahi.fixtures import TinyGemma3Spec, write_tiny_gemma3

    size = tok.get_vocab_size(with_added_tokens=True)
    write_tiny_gemma3(directory, TinyGemma3Spec(vocab_size=size, extra={"eos_token_id": [tok.token_to_id(eos)]}))
    tok.save(str(directory / "tokenizer.json"))
    return directory


@pytest.mark.parametrize("seed_prompt", ["Aloha kākou", "数字と漢字", "🙂🌊 emoji"])
def test_run_streams_byte_level_text_exactly(native_cli, tmp_path, seed_prompt):
    # A random tiny model emits arbitrary byte tokens, so the stream has to
    # hold back bytes that end inside a UTF-8 sequence.
    from lokahi.oracles import Gemma3Reference

    tok = _byte_level_tokenizer("qwen3")
    directory = _model_with_tokenizer(tmp_path / "model", tok, "<|im_end|>")
    result = subprocess.run(
        [str(native_cli), "run", "--model", str(directory), "--backend", "cpu", "--prompt", seed_prompt,
         "--max-new", "40"],
        capture_output=True, check=True,
    )
    expected = Gemma3Reference.load(directory).greedy(tok.encode(seed_prompt).ids, 40)
    eos = tok.token_to_id("<|im_end|>")
    if eos in expected:
        expected = expected[: expected.index(eos) + 1]
    assert result.stdout.decode() == tok.decode(expected, skip_special_tokens=True) + "\n"


FAMILIES = re.compile(r"llama-3|qwen|gpt-oss|deepseek|glm", re.IGNORECASE)


def _staged_byte_level_tokenizers() -> list[Path]:
    root = os.environ.get("LOKAHI_MODEL_ROOT")
    if not root:
        return []
    found = []
    for path in sorted(Path(root).glob("models--*/snapshots/*/tokenizer.json")):
        if FAMILIES.search(path.parts[-4]):
            found.append(path)
    return found


@pytest.mark.skipif(not os.environ.get("LOKAHI_MODEL_ROOT"), reason="needs staged model snapshots")
@pytest.mark.parametrize("path", _staged_byte_level_tokenizers() or [None], ids=lambda p: p.parts[-4] if p else "none")
def test_real_byte_level_tokenizers_match_reference(native_cli, tmp_path, path):
    if path is None:
        pytest.skip("no Llama 3, Qwen, GPT-OSS, DeepSeek or GLM snapshot is staged under LOKAHI_MODEL_ROOT")
    reference = Tokenizer.from_file(str(path))
    texts = EDGE_TEXTS + CORPUS + [_random_text(random.Random(3), []) for _ in range(300)]
    ours = _cli(native_cli, "tokenize", "--model", str(path.parent),
                "--texts-file", str(_write(tmp_path / "texts.json", texts)))["ids"]
    expected = [reference.encode(text).ids for text in texts]
    mismatches = [(text, a, b) for text, a, b in zip(texts, ours, expected) if a != b]
    assert not mismatches, mismatches[:3]
