"""tools/reference_check.py - DEV ONLY: produce golden data with Microsoft's code.

This is the only file in the project that imports third-party ML libraries
(torch, transformers, llmlingua, tiktoken). Runtime code never imports it.

It runs the official llmlingua PromptCompressor (pinned commit, see
tools/requirements-ref.txt) on the files in examples/ and writes CSV/TXT
golden data into tests/golden/:

    tokenizer_cases.csv          case,text                 tricky tokenizer inputs
    tokenizer_tokens.csv         case,position,token,token_id
    <case>.chunks.csv            chunk,text                 output of __chunk_context
    <case>.tokens.csv            chunk,position,token,token_id,p_keep
    <case>.words.csv             chunk,word_index,word,p_word,tiktoken_tokens,label
    <case>.compressed.txt        the compressed prompt
    model_card.hidden_layerNN.csv  hidden states (NN=00 is the embedding output)
    meta.csv                     versions and commits

Run (inside the reference environment):
    .venv-ref/Scripts/python tools/reference_check.py
"""
import sys

sys.dont_write_bytecode = True

import csv
import os
import platform
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
import transformers
from llmlingua import PromptCompressor
from llmlingua.utils import TokenClfDataset
from llmlingua.version import VERSION as LLMLINGUA_VERSION

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "downloads"
GOLDEN = ROOT / "tests" / "golden"
EXAMPLES = ROOT / "examples"
LLMLINGUA_COMMIT = "5a4c78ae18ab17a98cf997e8259354e546081d64"

# llmlingua picks its word-merging rules from the model *name*, so the local
# folder must contain "bert-base-multilingual-cased". Hard links avoid a copy.
MODEL_DIR = DOWNLOADS / "llmlingua-2-bert-base-multilingual-cased-meetingbank"
MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "model.safetensors"]

MODEL_CARD_SETTINGS = dict(
    rate=0.6,
    force_tokens=["\n", ".", "!", "?", ","],
    chunk_end_tokens=[".", "\n"],
    drop_consecutive=True,
)
# Opt-in Traditional Chinese settings used by csv-lingua's --lang zh-hant.
ZH_HANT_SETTINGS = dict(
    rate=0.6,
    force_tokens=["\n", ".", "!", "?", ",", "。", "，", "！", "？", "、", "：", "；"],
    chunk_end_tokens=[".", "\n", "。", "！", "？"],
    drop_consecutive=True,
)

TOKENIZER_CASES = {
    "model_card": None,  # filled from examples/model_card.txt
    "punctuation_numbers": "Q3 revenue rose 12.5% to $4,300,000 (vs. $3.9M); EBITDA: -0.7pp... "
    "e-mail a.b@c.io at 10:30am - #1!!! 3/4 ½ x² ① 1,000.00€ ~/path_to/file.txt [x]{y}<z>",
    "accents": "Crème brûlée à la française, naïve café, Ångström, Øresund, São Paulo, "
    "Dvořák, Łódź, İstanbul, Straße, ǅ, Ｆｕｌｌｗｉｄｔｈ ｔｅｘｔ",
    "quotes_dashes": "“Hello,” she said — ‘it’s fine’ – really… «bonjour» „Guten Tag“ don't it's",
    "long_word": "short " + "a" * 100 + " " + "b" * 101 + " " + "antidisestablishmentarianism" * 4 + " end",
    "blank_lines": "First line.\n\n\nSecond line after blank lines.\n  \n\tTabbed\r\nWindows line\n",
    "added_tokens": "tight.[NEW0]Sarah said [NEW12] and [SEP] [CLS]x [MASK][PAD][UNK] [new0] [NEW100] [NEW99]",
    "invisible": "zero\u200bwidth soft\u00adhyphen \ufeffbom \x01ctrl\x7fdel \ufffdrepl non\u00a0breaking",
    "traditional_chinese": "我們今天討論「產品規劃」，預算是三百五十萬元。臺灣、高雄；龜鑑驫麤𠀀，Python 3.12 很好用！",
    "cjk_mixed": "日本語のテキスト、한국어 텍스트。ｶﾀｶﾅ 〇〆 㐀 ⺀",
    "emoji_scripts": "Great job 👍🏽 🎉 👨‍👩‍👧 مرحبا بالعالم नमस्ते दुनिया สวัสดีชาวโลก",
}


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_text(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def prepare_model_dir():
    MODEL_DIR.mkdir(exist_ok=True)
    for name in MODEL_FILES:
        target = MODEL_DIR / name
        if not target.exists():
            try:
                os.link(DOWNLOADS / name, target)
            except OSError:
                target.write_bytes((DOWNLOADS / name).read_bytes())


def token_map_for(compressor, force_tokens):
    """Same as the first lines of compress_prompt_llmlingua2."""
    return {t: compressor.added_tokens[i] for i, t in enumerate(force_tokens)
            if len(compressor.tokenizer.tokenize(t)) != 1}


def chunk_like_reference(compressor, text, settings):
    token_map = token_map_for(compressor, settings["force_tokens"])
    chunk_end = list(settings["chunk_end_tokens"])
    chunk_end += [token_map[c] for c in settings["chunk_end_tokens"] if c in token_map]
    for original, new in token_map.items():
        text = text.replace(original, new)
    chunks = compressor._PromptCompressor__chunk_context(text, chunk_end_tokens=set(chunk_end))
    return chunks, token_map


def run_chunks(compressor, chunks, hidden_states=False):
    """Run the reference model exactly like __compress: padded 512 batch with mask."""
    dataset = TokenClfDataset(chunks, tokenizer=compressor.tokenizer, max_len=compressor.max_seq_len)
    results = []
    with torch.no_grad():
        for item in (dataset[i] for i in range(len(dataset))):
            ids = item["ids"].unsqueeze(0)
            mask = item["mask"].unsqueeze(0) == 1
            out = compressor.model(input_ids=ids, attention_mask=mask, output_hidden_states=hidden_states)
            probs = F.softmax(out.logits, dim=-1)[0, :, 1]
            active_ids = torch.masked_select(ids[0], mask[0]).tolist()
            active_probs = torch.masked_select(probs, mask[0]).numpy()
            tokens = compressor.tokenizer.convert_ids_to_tokens(active_ids)
            n = len(active_ids)
            layers = [h[0, :n].numpy() for h in out.hidden_states] if hidden_states else None
            results.append((tokens, active_ids, active_probs, layers))
    return results


def golden_case(compressor, name, text, settings, hidden_states=False):
    chunks, token_map = chunk_like_reference(compressor, text, settings)
    write_csv(GOLDEN / f"{name}.chunks.csv", ["chunk", "text"], list(enumerate(chunks)))

    runs = run_chunks(compressor, chunks, hidden_states)
    token_rows, word_rows = [], []
    for c, (tokens, ids, probs, layers) in enumerate(runs):
        token_rows += [[c, i, t, tid, f"{p:.8f}"] for i, (t, tid, p) in enumerate(zip(tokens, ids, probs))]
        words, word_probs, _ = compressor._PromptCompressor__merge_token_to_word(
            tokens, list(probs), force_tokens=settings["force_tokens"],
            token_map=token_map, force_reserve_digit=False)
        p_words = compressor._PromptCompressor__token_prob_to_word_prob(word_probs, convert_mode="mean")
        word_rows += [[c, i, w, f"{float(p):.8f}", len(compressor.oai_tokenizer.encode(w))]
                      for i, (w, p) in enumerate(zip(words, p_words))]
        if layers:
            for k, layer in enumerate(layers):
                np.savetxt(GOLDEN / f"{name}.hidden_layer{k:02d}.csv", layer, fmt="%.8f", delimiter=",")
    write_csv(GOLDEN / f"{name}.tokens.csv", ["chunk", "position", "token", "token_id", "p_keep"], token_rows)

    result = compressor.compress_prompt_llmlingua2(text, return_word_label=True, **settings)
    pairs = result["fn_labeled_original_prompt"].split("\t\t|\t\t")
    labels = [pair.rsplit(" ", 1) for pair in pairs]
    assert len(labels) == len(word_rows), f"{name}: {len(labels)} labels vs {len(word_rows)} words"
    for row, (word, label) in zip(word_rows, labels):
        assert row[2] == word, (row[2], word)
        row.append(int(label))
    write_csv(GOLDEN / f"{name}.words.csv",
              ["chunk", "word_index", "word", "p_word", "tiktoken_tokens", "label"], word_rows)
    with open(GOLDEN / f"{name}.compressed.txt", "w", encoding="utf-8", newline="") as f:
        f.write(result["compressed_prompt"])
    kept = sum(r[-1] for r in word_rows)
    print(f"  {name}: {len(chunks)} chunks, {len(token_rows)} tokens, {len(word_rows)} words, {kept} kept")


def golden_unicode_classes():
    """How HF's BertNormalizer + BertPreTokenizer classify every code point.

    Classes: removed (dropped by clean_text), space, chinese (padded with
    spaces), punct (isolated by the pre-tokenizer). Everything else is
    "other" and not listed. Written as ranges: first,last,class.
    """
    from tokenizers import normalizers, pre_tokenizers

    norm = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True,
                                      strip_accents=None, lowercase=False)
    pre = pre_tokenizers.BertPreTokenizer()

    def hf_class(ch):
        normalized = norm.normalize_str(ch)
        if normalized == "":
            return "removed"
        if normalized == " ":
            return "space"
        if normalized == f" {ch} ":
            return "chinese"
        parts = [p for p, _ in pre.pre_tokenize_str(f"a{ch}b")]
        if parts == ["a", ch, "b"]:
            return "punct"
        if parts == ["a", "b"]:
            return "space"
        assert parts == [f"a{ch}b"], (hex(ord(ch)), parts)
        return "other"

    ranges = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:  # surrogates cannot appear in a Python str passed to Rust
            continue
        cls = hf_class(chr(cp))
        if cls == "other":
            continue
        if ranges and ranges[-1][2] == cls and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp, cls])
    write_csv(GOLDEN / "unicode_classes.csv", ["first", "last", "class"],
              [[f"{a:04X}", f"{b:04X}", c] for a, b, c in ranges])
    print(f"  unicode classes: {len(ranges)} ranges")


def main():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    golden_unicode_classes()
    prepare_model_dir()
    torch.set_num_threads(1)
    compressor = PromptCompressor(model_name=str(MODEL_DIR), use_llmlingua2=True, device_map="cpu")
    tok = compressor.tokenizer
    print(f"  added tokens: [NEW0]={tok.convert_tokens_to_ids('[NEW0]')} "
          f"[NEW99]={tok.convert_tokens_to_ids('[NEW99]')}, len(tokenizer)={len(tok)}")
    print(f"  special tokens skipped when merging words: {sorted(compressor.special_tokens)}")

    # 1. tokenizer
    TOKENIZER_CASES["model_card"] = read_text(EXAMPLES / "model_card.txt")
    write_csv(GOLDEN / "tokenizer_cases.csv", ["case", "text"], TOKENIZER_CASES.items())
    rows = []
    for case, text in TOKENIZER_CASES.items():
        tokens = tok.tokenize(text)
        rows += [[case, i, t, tid] for i, (t, tid) in enumerate(zip(tokens, tok.convert_tokens_to_ids(tokens)))]
    write_csv(GOLDEN / "tokenizer_tokens.csv", ["case", "position", "token", "token_id"], rows)
    print(f"  tokenizer: {len(TOKENIZER_CASES)} cases, {len(rows)} tokens")

    # 2. model + compression
    meeting = read_text(EXAMPLES / "meeting.txt")
    zh = read_text(EXAMPLES / "meeting.zh-hant.txt")
    golden_case(compressor, "model_card", TOKENIZER_CASES["model_card"], MODEL_CARD_SETTINGS, hidden_states=True)
    golden_case(compressor, "meeting", meeting, MODEL_CARD_SETTINGS)
    golden_case(compressor, "meeting_rate033", meeting, dict(MODEL_CARD_SETTINGS, rate=0.33))
    # No '.' and no newline: forces the 511-token chunk + truncation path.
    no_chunk_end = meeting.replace(".", ";").replace("\n", " ")
    golden_case(compressor, "no_chunk_end", no_chunk_end, MODEL_CARD_SETTINGS)
    golden_case(compressor, "mixed", read_text(EXAMPLES / "mixed.txt"), MODEL_CARD_SETTINGS)
    golden_case(compressor, "zh_hant_default", zh, MODEL_CARD_SETTINGS)
    golden_case(compressor, "zh_hant", zh, ZH_HANT_SETTINGS)

    write_csv(GOLDEN / "meta.csv", ["key", "value"], [
        ("python", platform.python_version()),
        ("torch", torch.__version__),
        ("transformers", transformers.__version__),
        ("tiktoken", tiktoken.__version__),
        ("llmlingua", LLMLINGUA_VERSION),
        ("llmlingua_commit", LLMLINGUA_COMMIT),
        ("attn_implementation", compressor.model.config._attn_implementation),
        ("torch_threads", torch.get_num_threads()),
    ])


if __name__ == "__main__":
    main()
