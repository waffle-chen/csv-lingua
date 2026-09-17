"""csvlingua.py - LLMLingua-2 prompt compression in plain Python + numpy.

Quick use:

    from csvlingua import compress_text
    short = compress_text(long_text, rate=0.6)            # keep about 60% of the words
    short = compress_text(chinese_text, lang="zh-hant")   # Traditional Chinese settings

The whole inference path, top to bottom:

    Part 1  Tokenizer        text -> WordPiece tokens -> ids
    Part 2  CSV model        model_csv_int8/*.csv -> numpy arrays
    Part 3  The math         linear, layer_norm, gelu, softmax
    Part 4  BERT             ids -> p_keep for every token
    Part 5  Compression      p_keep -> words -> threshold -> compressed text
    Part 6  One-call API     compress_text()

Faithful to Microsoft's reference implementation:
    https://github.com/microsoft/LLMLingua  (commit 5a4c78ae, llmlingua 0.2.2)
    https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
"""
import csv
import functools
import math
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_csv_int8"   # the default: 8-bit CSV model (0.59 GB, shipped)
FULL_MODEL_DIR = ROOT / "model_csv"   # full-precision CSV model (1.87 GB, built by convert.py)


# =============================================================================
# Part 1  Tokenizer
#
# Hugging Face's BertTokenizerFast, rebuilt step by step:
#   1. split out added tokens such as [CLS] or [NEW0] (they are never split further)
#   2. normalize: drop control characters, turn whitespace into ' ',
#      put spaces around every Chinese character
#   3. pre-tokenize: split on spaces, and isolate every punctuation character
#   4. WordPiece: cut each word into the longest pieces found in the vocabulary;
#      pieces after the first start with '##'
# Settings for this model: no lower-casing, no accent stripping, 100-char word limit.
# =============================================================================

WORDPIECE_PREFIX = "##"
MAX_CHARS_PER_WORD = 100
UNK = "[UNK]"

# Unicode's White_Space property (what Rust's char::is_whitespace uses).
WHITESPACE = set(
    "\x09\x0a\x0b\x0c\x0d \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
ASCII_PUNCTUATION = set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")

# The "CJK Unified Ideographs" blocks that BERT treats as one-character words.
CHINESE_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F), (0x2B920, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
]

# Hugging Face's Rust tokenizer uses an older Unicode table than Python's
# unicodedata. These code points (rare-script punctuation and format signs)
# are classified differently there; we follow Hugging Face. Measured over all
# 1.1M code points by tests/test_tokenizer.py against tests/golden/unicode_classes.csv.
HF_UNICODE_EXCEPTIONS = [
    (0x061D, 0x061D, "other"), (0x0890, 0x0891, "other"), (0x08E2, 0x08E2, "other"),
    (0x09FD, 0x09FD, "other"), (0x0A76, 0x0A76, "other"), (0x0C77, 0x0C77, "other"),
    (0x0C84, 0x0C84, "other"), (0x166D, 0x166D, "punct"), (0x1B4E, 0x1B4F, "other"),
    (0x1B7D, 0x1B7F, "other"), (0x2E43, 0x2E4F, "other"), (0x2E52, 0x2E5D, "other"),
    (0x10D6E, 0x10D6E, "other"), (0x10EAD, 0x10EAD, "other"), (0x10F55, 0x10F59, "other"),
    (0x10F86, 0x10F89, "other"), (0x110CD, 0x110CD, "other"), (0x111C9, 0x111C9, "punct"),
    (0x113D4, 0x113D5, "other"), (0x113D7, 0x113D8, "other"), (0x1144B, 0x1144F, "other"),
    (0x1145A, 0x1145B, "other"), (0x1145D, 0x1145D, "other"), (0x11660, 0x1166C, "other"),
    (0x116B9, 0x116B9, "other"), (0x1183B, 0x1183B, "other"), (0x11944, 0x11946, "other"),
    (0x119E2, 0x119E2, "other"), (0x11A3F, 0x11A46, "other"), (0x11A9A, 0x11A9C, "other"),
    (0x11A9E, 0x11AA2, "other"), (0x11B00, 0x11B09, "other"), (0x11BE1, 0x11BE1, "other"),
    (0x11C41, 0x11C45, "other"), (0x11C70, 0x11C71, "other"), (0x11EF7, 0x11EF8, "other"),
    (0x11F43, 0x11F4F, "other"), (0x11FFF, 0x11FFF, "other"), (0x12FF1, 0x12FF2, "other"),
    (0x13430, 0x1343F, "other"), (0x16D6D, 0x16D6F, "other"), (0x16E97, 0x16E9A, "other"),
    (0x16FE2, 0x16FE2, "other"), (0x1E5FF, 0x1E5FF, "other"), (0x1E95E, 0x1E95F, "other"),
]
HF_EXCEPTION_CLASS = {cp: cls for first, last, cls in HF_UNICODE_EXCEPTIONS for cp in range(first, last + 1)}


def is_chinese_char(ch):
    return any(first <= ord(ch) <= last for first, last in CHINESE_RANGES)


@functools.cache
def char_class(ch):
    """How BERT's normalizer and pre-tokenizer treat one character.

    removed  control/format/private-use characters, U+0000 and U+FFFD
    space    whitespace (tab, newline and carriage return included)
    chinese  CJK ideograph: becomes a word of its own
    punct    punctuation: becomes a word of its own
    other    part of a normal word
    """
    cp = ord(ch)
    if cp in HF_EXCEPTION_CLASS:
        return HF_EXCEPTION_CLASS[cp]
    category = unicodedata.category(ch)
    if ch in "\t\n\r":
        return "space"
    if cp in (0, 0xFFFD) or category in ("Cc", "Cf", "Co", "Cs"):
        return "removed"
    if ch in WHITESPACE:
        return "space"
    if is_chinese_char(ch):
        return "chinese"
    if ch in ASCII_PUNCTUATION or category.startswith("P"):
        return "punct"
    return "other"


def normalize(text):
    """BertNormalizer(clean_text, handle_chinese_chars, no lowercase, no accent strip)."""
    out = []
    for ch in text:
        cls = char_class(ch)
        if cls == "removed":
            continue
        elif cls == "space":
            out.append(" ")
        elif cls == "chinese":
            out.append(f" {ch} ")
        else:
            out.append(ch)
    return "".join(out)


def pre_tokenize(text):
    """BertPreTokenizer: split on whitespace, then isolate each punctuation char.

    'Hi, you!' -> ['Hi', ',', 'you', '!']
    """
    words = []
    for chunk in text.split(" "):  # after normalize() the only whitespace left is ' '
        current = ""
        for ch in chunk:
            if char_class(ch) == "punct":
                if current:
                    words.append(current)
                words.append(ch)
                current = ""
            else:
                current += ch
        if current:
            words.append(current)
    return words


def wordpiece(word, vocab):
    """Greedy longest-match-first WordPiece.

    'unaffable' -> ['un', '##aff', '##able'] (if those pieces are in vocab).
    A word longer than 100 characters, or one that cannot be fully covered by
    vocabulary pieces, becomes the single token [UNK].
    """
    if len(word) > MAX_CHARS_PER_WORD:
        return [UNK]
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        while end > start:
            piece = word[start:end] if start == 0 else WORDPIECE_PREFIX + word[start:end]
            if piece in vocab:
                break
            end -= 1
        else:
            return [UNK]
        pieces.append(piece)
        start = end
    return pieces


def load_tokenizer(model_dir=MODEL_DIR):
    """Read vocab.csv -> tokenizer dict.

    tokenizer["vocab"]     {token: id}
    tokenizer["tokens"]    [token for id 0, 1, 2, ...]
    tokenizer["added"]     special tokens that are cut out before normalizing
    tokenizer["added_re"]  regex that finds them (longest first = leftmost-longest match)
    """
    tokens, added = [], []
    with open(Path(model_dir) / "vocab.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            assert int(row["id"]) == len(tokens)
            tokens.append(row["token"])
            if row["special"] == "1":
                added.append(row["token"])
    by_length = sorted(added, key=len, reverse=True)
    return {
        "tokens": tokens,
        "vocab": {token: i for i, token in enumerate(tokens)},
        "added": set(added),
        "added_re": re.compile("(" + "|".join(re.escape(t) for t in by_length) + ")"),
    }


def tokenize(text, tokenizer):
    """Text -> WordPiece tokens (no [CLS]/[SEP] added), like HF tokenizer.tokenize()."""
    tokens = []
    # re.split with a capture group: even positions are normal text, odd positions added tokens
    for i, piece in enumerate(tokenizer["added_re"].split(text)):
        if i % 2 == 1:
            tokens.append(piece)
            continue
        for word in pre_tokenize(normalize(piece)):
            tokens.extend(wordpiece(word, tokenizer["vocab"]))
    return tokens


def tokens_to_ids(tokens, tokenizer):
    vocab = tokenizer["vocab"]
    return [vocab.get(token, vocab[UNK]) for token in tokens]


# The WordPiece decoder's "cleanup" rules, applied to each token separately.
DECODER_CLEANUP = [
    (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"),
    (" 'm", "'m"), (" do not", " don't"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"),
]


def tokens_to_string(tokens):
    """HF convert_tokens_to_string: join tokens back into text.

    Every token after the first gets a leading space, unless it starts with '##'
    (then the '##' is removed). Then the cleanup rules run on that single token,
    e.g. ' ,' -> ','. So ['Hello', ',', 'wor', '##ld'] -> 'Hello, world'.
    """
    out = []
    for i, token in enumerate(tokens):
        if i > 0:
            token = token[len(WORDPIECE_PREFIX):] if token.startswith(WORDPIECE_PREFIX) else " " + token
        for dirty, clean in DECODER_CLEANUP:
            token = token.replace(dirty, clean)
        out.append(token)
    return "".join(out)


# =============================================================================
# Part 2  Reading the CSV model
#
# config.csv     key,value: architecture, tokenizer settings, how the CSVs were written
# vocab.csv      id,token,special for all 119,647 tokens
# manifest.csv   tensor,file,first_row,rows,cols,ndim,storage (one line per file)
# weights/*.csv  one CSV row = one matrix row; biases and LayerNorm γ/β are one row
# =============================================================================

WORD_EMBEDDINGS = "bert.embeddings.word_embeddings.weight"


def read_key_value_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {row["key"]: row["value"] for row in csv.DictReader(f)}


def read_manifest(model_dir=MODEL_DIR):
    """{tensor name: [file entries in row order]}"""
    manifest = {}
    with open(Path(model_dir) / "manifest.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            entry = {"file": row["file"], "storage": row.get("storage") or "float",
                     **{k: int(row[k]) for k in ("first_row", "rows", "cols", "ndim")}}
            manifest.setdefault(row["tensor"], []).append(entry)
    return manifest


POWERS_OF_TEN = 10.0 ** np.arange(19)


def parse_csv_block(raw):
    """Whole CSV lines (ASCII bytes) -> (float64 values, number of columns).

    The files only contain digits, '-', '.', ',' and newlines, so the whole block
    can be turned into numbers with array operations instead of parsing number by
    number. Reading "12.75" works like this:

        place   digits from this character to the end of its number: 4 3 . 2 1
        mantissa  sum of digit·10^(digits after it)  =  1275   (exact, integers)
        value     mantissa / 10^(digits after the '.')  =  1275 / 100

    The mantissa is exact (at most 13 digits, so below 2^53) and 10^k is exact, so
    the single division at the end is correctly rounded: the values are identical
    to what a C parser such as np.loadtxt returns, only several times faster, and
    numpy releases the GIL, so blocks can be parsed on several threads.
    """
    b = np.frombuffer(raw, dtype=np.uint8)
    separator = (b == 44) | (b == 10)  # ',' or '\n'
    digit = (b >= 48) & (b <= 57)
    digits_to_buffer_end = np.cumsum(digit[::-1], dtype=np.int32)[::-1]
    at_number_end = np.where(separator, digits_to_buffer_end, np.int32(-1))
    at_number_end = np.maximum.accumulate(at_number_end[::-1])[::-1]  # value at the next separator
    place = digits_to_buffer_end - at_number_end  # digits up to the end of this number

    ends = np.flatnonzero(separator)
    starts = np.empty(ends.size, np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    mantissa = np.add.reduceat((b - 48) * digit * POWERS_OF_TEN[place - digit], starts)

    decimals = np.zeros(ends.size, np.int64)
    dots = np.flatnonzero(b == 46)
    if dots.size:
        decimals[np.searchsorted(ends, dots)] = place[dots]
    values = mantissa / POWERS_OF_TEN[decimals]
    minus = np.flatnonzero(b == 45)
    if minus.size:
        negative = np.searchsorted(ends, minus)
        values[negative] = -values[negative]

    columns = int(np.count_nonzero(separator[:int(np.flatnonzero(b == 10)[0]) + 1]))
    return values, columns


BLOCK_BYTES = 1 << 20  # parse about a megabyte at a time: big enough to be fast, small enough to stay in cache


def split_rows(raw, block_bytes=BLOCK_BYTES):
    """Cut CSV bytes into pieces of about block_bytes that each end on a line break."""
    cuts = [0]
    while cuts[-1] + block_bytes < len(raw):
        newline = raw.find(b"\n", cuts[-1] + block_bytes)
        if newline < 0:
            break
        cuts.append(newline + 1)
    cuts.append(len(raw))
    return [raw[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]


def parse_csv_bytes(raw, parallel=False):
    """CSV bytes -> rows×columns float32 array.

    The pieces are independent, so parsing them on several threads (parallel=True)
    produces exactly the same numbers as parsing them one after another.
    """
    pieces = split_rows(raw)
    if parallel and len(pieces) > 1:
        parsed = list(thread_pool().map(parse_csv_block, pieces))
    else:
        parsed = [parse_csv_block(piece) for piece in pieces]
    values = np.concatenate([v for v, _ in parsed]).astype(np.float32)
    return values.reshape(-1, parsed[0][1])


def dequantize(matrix, storage):
    """int8 files store  scale, q₁, …, q_cols  per row; the weights are W_r = s_r·q_r."""
    return matrix[:, 1:] * matrix[:, :1] if storage == "int8" else matrix


def load_csv_matrix(path, storage="float", parallel=False):
    """One weight CSV file -> rows×cols float32 array."""
    with open(path, "rb") as f:
        return dequantize(parse_csv_bytes(f.read(), parallel), storage)


def load_csv_matrices(paths, storages):
    """Load several weight CSVs at once: one thread per file, all CPU cores busy."""
    return list(thread_pool().map(load_csv_matrix, paths, storages))


def load_csv_rows(path, rows, storage="float"):
    """Only the given row numbers of a weight CSV -> len(rows)×cols float32 array.

    The word-embedding table has 119,647 rows but a text uses only a few thousand
    of them, so the file is read (fast) and only the interesting lines are parsed.
    """
    with open(path, "rb") as f:
        raw = f.read()
    line_ends = np.flatnonzero(np.frombuffer(raw, dtype=np.uint8) == 10)
    line_starts = np.concatenate([[0], line_ends[:-1] + 1])
    wanted = b"".join(raw[line_starts[r]:line_ends[r] + 1] for r in rows)
    return dequantize(parse_csv_bytes(wanted, parallel=True), storage)


def make_model(config, weights, word_embeddings):
    """Bundle everything the forward pass needs.

    config           {key: str} from config.csv
    weights          {tensor name: float32 array}, word embeddings excluded
    word_embeddings  {token id: 768-vector} for the ids that were loaded,
                     or the whole table as one array indexed by token id
    """
    return {
        "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_hidden_layers"]),
        "num_heads": int(config["num_attention_heads"]),
        "eps": np.float32(config["layer_norm_eps"]),
        "weights": weights,
        "word_embeddings": word_embeddings,
    }


def load_word_embeddings(model_dir, manifest, token_ids=None):
    """Load the word-embedding rows for token_ids (the whole table when None)."""
    entries = manifest[WORD_EMBEDDINGS]
    if token_ids is None:
        return np.concatenate(load_csv_matrices([Path(model_dir) / e["file"] for e in entries],
                                                [e["storage"] for e in entries]))
    vectors = {}
    for entry in entries:
        first, last = entry["first_row"], entry["first_row"] + entry["rows"]
        ids = sorted(i for i in set(token_ids) if first <= i < last)
        if ids:
            rows = load_csv_rows(Path(model_dir) / entry["file"], [i - first for i in ids], entry["storage"])
            vectors.update(zip(ids, rows))
    return vectors


def load_model(model_dir=MODEL_DIR, token_ids=None):
    """Read the CSV model into numpy arrays.

    token_ids: load only those rows of the word-embedding table (half the model);
    None loads the whole table.
    """
    model_dir = Path(model_dir)
    config = read_key_value_csv(model_dir / "config.csv")
    manifest = read_manifest(model_dir)

    wanted = [(name, e) for name, entries in manifest.items() if name != WORD_EMBEDDINGS for e in entries]
    matrices = load_csv_matrices([model_dir / e["file"] for _, e in wanted], [e["storage"] for _, e in wanted])
    parts = {}
    for (name, _), matrix in zip(wanted, matrices):
        parts.setdefault(name, []).append(matrix)
    weights = {}
    for name, pieces in parts.items():
        matrix = np.concatenate(pieces)
        weights[name] = matrix[0] if manifest[name][0]["ndim"] == 1 else matrix

    embeddings = load_word_embeddings(model_dir, manifest, token_ids)
    model = make_model(config, weights, embeddings)
    model["embedding_rows_loaded"] = len(embeddings) if isinstance(embeddings, dict) else len(embeddings)
    return model


def lookup_word_embeddings(ids, model):
    """E_word[ids]: the embedding vector of every token.   ids: n -> n×768"""
    table = model["word_embeddings"]
    if not isinstance(table, dict):
        return table[ids]
    try:
        return np.stack([table[i] for i in ids])
    except KeyError as missing:
        raise KeyError(f"token id {missing.args[0]} was not loaded; pass its id to load_model(token_ids=...)") from None


# =============================================================================
# Part 3  The math
#
# Every activation stays float32, like the reference (PyTorch float32).
# =============================================================================


def linear(x, W, b):
    """y = x·Wᵀ + b

    x: n×d_in, W: d_out×d_in, b: d_out -> n×d_out
    Row r of W holds the weights feeding output unit r.
    """
    return x @ W.T + b


def layer_norm(x, gamma, beta, eps):
    """LayerNorm(x) = γ·(x − mean) / sqrt(var + eps) + β, over the last axis.

    x: n×768, γ: 768, β: 768 -> n×768
    """
    centered = x - x.mean(axis=-1, keepdims=True)
    var = (centered ** 2).mean(axis=-1, keepdims=True)
    return gamma * centered / np.sqrt(var + eps) + beta


erf_math = np.vectorize(math.erf, otypes=[np.float64])
erf_math.__doc__ = "erf(x) = 2/√π ∫₀ˣ exp(−t²) dt via Python's math.erf, one element at a time (slow)."

# Coefficients of the C library's erf (Sun fdlibm s_erf.c, as used by FreeBSD and musl).
# Polynomials are listed from the constant term upward.
ERF_ERX = 8.45062911510467529297e-01
ERF_PP = [1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02,
          -5.77027029648944159157e-03, -2.37630166566501626084e-05]
ERF_QQ = [1.0, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03,
          1.32494738004321644526e-04, -3.96022827877536812320e-06]
ERF_PA = [-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01,
          3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02,
          -2.16637559486879084300e-03]
ERF_QA = [1.0, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02,
          1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02]
ERF_RA = [-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01,
          -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02,
          -8.12874355063065934246e+01, -9.81432934416914548592e+00]
ERF_SA = [1.0, 1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02,
          6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02,
          6.57024977031928170135e+00, -6.04244152148580987438e-02]
ERF_RB = [-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01,
          -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03,
          -4.83519191608651397019e+02]
ERF_SB = [1.0, 3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03,
          3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02,
          -2.24409524465858183362e+01]
ERF_ONE_OVER_035 = 2.8571414947509766  # 1/0.35 as the C code compares it (high 32 bits 0x4006db6d)


def polynomial(coefficients, s):
    """c₀ + c₁s + c₂s² + …, evaluated with Horner's rule.

    The steps are done in place: the arrays here are millions of numbers wide,
    and every temporary costs a full pass through memory.
    """
    result = np.full_like(s, coefficients[-1])
    for c in reversed(coefficients[:-1]):
        np.multiply(result, s, out=result)
        np.add(result, c, out=result)
    return result


def erf_exact(x):
    """erf(x) = 2/√π ∫₀ˣ exp(−t²) dt, the C library's algorithm on whole arrays.

    |x| < 0.84375          erf = x + x·PP(x²)/QQ(x²)
    0.84375 ≤ |x| < 1.25   erf = ±(erx + PA(s)/QA(s)),  s = |x| − 1
    1.25 ≤ |x| < 6         erf = ±(1 − exp(−z² − 0.5625)·exp((z−|x|)(z+|x|) + R(1/x²)/S(1/x²)) / |x|)
                           (z = |x| with its low 32 bits cleared; R, S switch at |x| = 1/0.35)
    |x| ≥ 6                erf = ±1
    Within one or two float64 ulps of math.erf; after the float32 cast in gelu() the
    results are identical (checked on millions of values by tests/test_math.py).
    x: any shape -> same shape (float64)
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)

    # |x| < 0.84375 holds for most inputs, so it is computed for every element
    # (in place, no index gathers); the rarer branches then overwrite their parts.
    z = ax * ax
    out = polynomial(ERF_PP, z)
    np.divide(out, polynomial(ERF_QQ, z), out=out)
    np.multiply(out, x, out=out)
    np.add(out, x, out=out)

    middle = (ax >= 0.84375) & (ax < 1.25)
    if middle.any():
        s = ax[middle] - 1
        out[middle] = np.copysign(1 - (1 - ERF_ERX - polynomial(ERF_PA, s) / polynomial(ERF_QA, s)), x[middle])

    for low, high, R, S in [(1.25, ERF_ONE_OVER_035, ERF_RA, ERF_SA), (ERF_ONE_OVER_035, 6.0, ERF_RB, ERF_SB)]:
        tail = (ax >= low) & (ax < high)
        if tail.any():
            a = ax[tail]
            s = 1 / (a * a)
            z = (a.view(np.uint64) & np.uint64(0xFFFFFFFF00000000)).view(np.float64)
            erfc = np.exp(-z * z - 0.5625) * np.exp((z - a) * (z + a) + polynomial(R, s) / polynomial(S, s)) / a
            out[tail] = np.copysign(1 - erfc, x[tail])

    large = ax >= 6
    if large.any():
        out[large] = np.copysign(1.0, x[large])
    return out


def erf_abramowitz_stegun(x):
    """erf approximation, Abramowitz & Stegun formula 7.1.26 (|error| ≤ 1.5×10⁻⁷).

    erf(x) ≈ sign(x)·(1 − (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)·exp(−x²)),  t = 1/(1 + p·|x|)
    x: any shape -> same shape (float64)
    """
    p, a1, a2, a3, a4, a5 = 0.3275911, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    ax = np.abs(x.astype(np.float64))
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return np.sign(x) * (1.0 - poly * np.exp(-ax * ax))


@functools.cache
def thread_pool():
    return ThreadPoolExecutor(os.cpu_count())


def on_all_cores(function, x, blocks=24):
    """function(x) computed block of rows by block of rows on all CPU cores.

    numpy releases Python's GIL inside element-wise operations, so the blocks
    really run at the same time. Every element sees exactly the same arithmetic,
    so the result is identical to function(x).
    """
    if len(x) < 2:
        return function(x)
    return np.concatenate(list(thread_pool().map(function, np.array_split(x, min(blocks, len(x))))))


def gelu(x, erf=erf_exact):
    """GELU(x) = 0.5·x·(1 + erf(x/√2))       x: n×3072 (any shape) -> same shape"""
    def formula(part):
        return 0.5 * part * (1 + erf(part / np.float32(math.sqrt(2))).astype(np.float32))
    return on_all_cores(formula, x)


def softmax(x, axis=-1):
    """softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)    (the max is subtracted first so exp cannot overflow)"""
    e = np.subtract(x, x.max(axis=axis, keepdims=True))  # one new array, then everything in place
    np.exp(e, out=e)
    return np.divide(e, e.sum(axis=axis, keepdims=True), out=e)


# =============================================================================
# Part 4  BERT
#
# BertForTokenClassification, post-LayerNorm, inference only (no dropout).
# Each chunk runs at its exact length n, so no padding and no attention mask.
# =============================================================================


def embed(ids, model):
    """h₀ = LayerNorm(E_word[ids] + E_pos[0..n−1] + E_type[0])     ids: n -> n×768"""
    w = model["weights"]
    n = len(ids)
    x = (lookup_word_embeddings(ids, model)
         + w["bert.embeddings.position_embeddings.weight"][:n]
         + w["bert.embeddings.token_type_embeddings.weight"][0])
    return layer_norm(x, w["bert.embeddings.LayerNorm.weight"], w["bert.embeddings.LayerNorm.bias"], model["eps"])


def self_attention(h, model, layer):
    """MultiHeadSelfAttention(h) = concat over 12 heads of softmax(Q·Kᵀ/√64)·V

    h: n×768
    Q, K, V = linear(h): n×768 -> split into heads: 12×n×64
    scores  = Q·Kᵀ/√64: 12×n×n  (row i: how much token i looks at each token)
    output  = softmax(scores)·V: 12×n×64 -> joined back: n×768
    """
    w = model["weights"]
    p = f"bert.encoder.layer.{layer}.attention.self."
    n, d = h.shape
    heads = model["num_heads"]
    size = d // heads  # 64

    def split_heads(x):  # n×768 -> 12×n×64
        return x.reshape(n, heads, size).transpose(1, 0, 2)

    Q = split_heads(linear(h, w[p + "query.weight"], w[p + "query.bias"]))
    K = split_heads(linear(h, w[p + "key.weight"], w[p + "key.bias"]))
    V = split_heads(linear(h, w[p + "value.weight"], w[p + "value.bias"]))
    # dividing by √64 = 8 before the product touches 8× fewer numbers, and since 8
    # is a power of two the division is exact, so the scores are bit for bit the same
    scores = (Q / np.float32(math.sqrt(size))) @ K.transpose(0, 2, 1)
    context = softmax(scores, axis=-1) @ V
    return context.transpose(1, 0, 2).reshape(n, d)


def encoder_layer(h, model, layer, erf=erf_exact):
    """One transformer layer (post-LayerNorm):

    a = LayerNorm(h + W_o·MultiHeadSelfAttention(h))
    h' = LayerNorm(a + W_2·GELU(W_1·a))          h: n×768 -> n×768 (W_1: 3072×768, W_2: 768×3072)
    """
    w = model["weights"]
    p = f"bert.encoder.layer.{layer}."
    attention = linear(self_attention(h, model, layer),
                       w[p + "attention.output.dense.weight"], w[p + "attention.output.dense.bias"])
    a = layer_norm(h + attention,
                   w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"], model["eps"])
    inner = gelu(linear(a, w[p + "intermediate.dense.weight"], w[p + "intermediate.dense.bias"]), erf)
    out = linear(inner, w[p + "output.dense.weight"], w[p + "output.dense.bias"])
    return layer_norm(a + out, w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"], model["eps"])


def bert_forward(ids, model, erf=erf_exact, hidden_states=None):
    """Token ids -> keep-probability for every token.

    ids: n (starting with [CLS], ending with [SEP]) -> p_keep: n
    logits = W_c·h₁₂ (n×2);  p_keep = softmax(logits)[:, 1]
    If a list is passed as hidden_states, h₀ (embedding output) .. h₁₂ are appended to it.
    """
    h = embed(ids, model)
    if hidden_states is not None:
        hidden_states.append(h)
    for layer in range(model["num_layers"]):
        h = encoder_layer(h, model, layer, erf)
        if hidden_states is not None:
            hidden_states.append(h)
    w = model["weights"]
    logits = linear(h, w["classifier.weight"], w["classifier.bias"])
    return softmax(logits, axis=-1)[:, 1]


# =============================================================================
# Part 5  Compression (LLMLingua-2)
#
# llmlingua/prompt_compressor.py: compress_prompt_llmlingua2, __chunk_context,
# __merge_token_to_word, __token_prob_to_word_prob, __compress.
#   1. force tokens that are not one WordPiece token (e.g. '\n') -> [NEWi]
#   2. tokenize the whole text, cut into chunks of <= 510 tokens at a chunk-end token
#   3. every chunk: [CLS] + tokens + [SEP] -> BERT -> p_keep per token
#   4. tokens -> words (mean p_keep; force tokens get 1.0)
#   5. drop repeated punctuation, percentile threshold, keep words above it
#   6. join kept words back into text, chunks joined with ""
# =============================================================================

MAX_SEQ_LEN = 512
CLS, SEP = "[CLS]", "[SEP]"
# The reference skips these when building words. [PAD] is missing on purpose:
# llmlingua's load_model sets pad_token_id to None, which removes [PAD] from
# tokenizer.special_tokens_map (padding never reaches this step anyway).
SKIPPED_TOKENS = {"[CLS]", "[SEP]", "[UNK]", "[MASK]"}

SETTINGS = {
    # the model card's settings, byte-for-byte the reference behaviour
    "default": {
        "force_tokens": ["\n", ".", "!", "?", ","],
        "chunk_end_tokens": [".", "\n"],
    },
    # opt-in Traditional Chinese: Chinese punctuation is protected, 。！？ end chunks,
    # punctuation left side by side is tidied, and no spaces go between Chinese characters
    "zh-hant": {
        "force_tokens": ["\n", ".", "!", "?", ",", "。", "，", "！", "？", "、", "：", "；"],
        "chunk_end_tokens": [".", "\n", "。", "！", "？"],
    },
}


def force_token_map(force_tokens, tokenizer):
    """{force token: [NEWi]} for every force token that is not exactly one WordPiece token.

    '\\n' tokenizes to nothing (it is whitespace), so with the model-card list it becomes [NEW0].
    i is the token's position in force_tokens.
    """
    return {t: f"[NEW{i}]" for i, t in enumerate(force_tokens) if len(tokenize(t, tokenizer)) != 1}


def split_into_chunks(text, chunk_end_tokens, tokenizer):
    """__chunk_context: cut the text into pieces the model can read (510 tokens + [CLS] + [SEP]).

    From the 511th token of the window, walk back to the last chunk-end token and cut
    after it. If the window has none, the chunk is 511 tokens long (the 512 cap later
    drops its [SEP]). Each chunk is turned back into a string with tokens_to_string.
    """
    max_len = MAX_SEQ_LEN - 2
    tokens = tokenize(text, tokenizer)
    n = len(tokens)
    chunks, start = [], 0
    while start < n:
        if start + max_len > n - 1:
            chunks.append(tokens_to_string(tokens[start:n]))
            break
        end = start + max_len
        for j in range(end - start):
            if tokens[end - j] in chunk_end_tokens:
                end -= j
                break
        chunks.append(tokens_to_string(tokens[start:end + 1]))
        start = end + 1
    return chunks


def prepare_chunks(text, tokenizer, force_tokens, chunk_end_tokens):
    """Steps 1-3 without the model: returns (chunks, token_map).

    chunk = {"text": chunk string, "tokens": [CLS]+tokens+[SEP] (max 512), "ids": token ids}
    """
    token_map = force_token_map(force_tokens, tokenizer)
    chunk_end = set(chunk_end_tokens) | {token_map[t] for t in chunk_end_tokens if t in token_map}
    for original, new in token_map.items():
        text = text.replace(original, new)
    chunks = []
    for chunk_text in split_into_chunks(text, chunk_end, tokenizer):
        tokens = ([CLS] + tokenize(chunk_text, tokenizer) + [SEP])[:MAX_SEQ_LEN]
        chunks.append({"text": chunk_text, "tokens": tokens, "ids": tokens_to_ids(tokens, tokenizer)})
    return chunks, token_map


def restore_force_tokens(text, token_map):
    """replace_added_token: [NEWi] -> the original character."""
    for original, new in token_map.items():
        text = text.replace(new, original)
    return text


def merge_tokens_to_words(tokens, p_keep, force_tokens, token_map):
    """Tokens -> words with one probability each (__merge_token_to_word, mean mode).

    - [CLS], [SEP], [UNK], [MASK] are skipped
    - a token starting with '##' continues the previous word, unless it is a force token
    - a force token word gets probability 1.0
    - word probability = mean of its token probabilities (float32, like the reference)
    Note: the reference strips with str.lstrip("##"), which removes *every* leading '#'.
    Returns (words, word_probs, token_word) where token_word[i] is the word index of
    token i, or -1 for a skipped token.
    """
    new_tokens = set(token_map.values())
    words, token_probs, token_word = [], [], []
    for token, p in zip(tokens, p_keep):
        pure = token.lstrip("#")
        if token in SKIPPED_TOKENS:
            token_word.append(-1)
            continue
        if pure in force_tokens or pure in new_tokens or not token.startswith("##"):
            if pure in force_tokens or pure in new_tokens:
                p = 1.0
            words.append(restore_force_tokens(token, token_map))
            token_probs.append([p])
        else:
            words[-1] += pure
            token_probs[-1].append(p)
        token_word.append(len(words) - 1)
    word_probs = [sum(ps) / len(ps) for ps in token_probs]
    return words, word_probs, token_word


def drop_repeated_force_tokens(words, word_probs, force_tokens, reduce_rate):
    """drop_consecutive, part 1: a force token that repeats the previous force token
    with no kept-looking word in between gets probability 0.

    'looking kept' uses a first, unweighted percentile: q = int(100·reduce_rate).
    word_probs is changed in place.
    """
    threshold = np.percentile(word_probs, int(100 * reduce_rate))
    between, previous = False, None
    for i, (word, p) in enumerate(zip(words, word_probs)):
        if word in force_tokens:
            if between:
                between = False
            elif word == previous:
                word_probs[i] = 0.0
            previous = word
        else:
            between |= p > threshold


def keep_threshold(words, word_probs, reduce_rate, word_weight=None):
    """The percentile threshold: q = int(100·reduce_rate + 1), linear interpolation.

    The reference repeats every word probability once per GPT-3.5 (tiktoken) token of
    the word. tiktoken is not allowed here, so by default every word counts once
    (word_weight=None). Pass word_weight(word) -> int to use other counts.
    """
    if word_weight is None:
        values = word_probs
    else:
        values = [p for word, p in zip(words, word_probs) for _ in range(word_weight(word))]
    return np.percentile(values, int(100 * reduce_rate + 1))


def is_cjk_punctuation(ch):
    """Full-width punctuation: 。，！？、：；「」（）… (CJK symbols and full-width forms)."""
    cp = ord(ch)
    return 0x3000 <= cp <= 0x303F or 0xFF01 <= cp <= 0xFF65


def words_to_text(words, lang="default"):
    """Join kept words. 'default' is exactly the reference (tokens_to_string).

    'zh-hant' drops the space between two Chinese characters, next to full-width
    punctuation and next to a newline, but keeps the one between Latin and Chinese
    words: ['我', '們', '。', '\\n', 'OK', '很', '好'] -> '我們。\\nOK 很好'.
    """
    if lang != "zh-hant":
        return tokens_to_string(words)
    text = ""
    for i, word in enumerate(words):
        if i > 0:
            if word.startswith(WORDPIECE_PREFIX):
                word = word[len(WORDPIECE_PREFIX):]
            elif not (text.endswith("\n") or word.startswith("\n") or not text or not word
                      or is_cjk_punctuation(text[-1]) or is_cjk_punctuation(word[0])
                      or (is_chinese_char(text[-1]) and is_chinese_char(word[0]))):
                word = " " + word
        for dirty, clean in DECODER_CLEANUP:
            word = word.replace(dirty, clean)
        text += word
    return text


PUNCTUATION_STRENGTH = {
    "。": 3, "？": 3, "！": 3, ".": 3, "?": 3, "!": 3,
    "；": 2, "：": 2, ";": 2, ":": 2,
    "，": 1, "、": 1, ",": 1,
    # brackets and quotes: weakest, so a bracket left next to a real punctuation
    # mark (its content was dropped) disappears instead of the mark
    "「": 0, "」": 0, "『": 0, "』": 0, "（": 0, "）": 0, "《": 0, "》": 0,
}


def tidy_punctuation(words, labels):
    """zh-hant only: clean up punctuation that compression left in awkward places.

    Dropping the words between two punctuation marks often leaves them side by side
    ('方便。，抱怨'), or leaves a mark at the start of a line. So, among kept words:
      - of several punctuation marks in a row, only the strongest stays
        (。？！ > ；： > ，、; the first one wins a tie)
      - a punctuation mark at the start of a line or chunk is dropped
    labels is changed in place.
    """
    run = []  # indices of adjacent kept punctuation marks
    at_line_start = True
    for i in [i for i, label in enumerate(labels) if label] + [None]:
        word = words[i] if i is not None else None
        if word in PUNCTUATION_STRENGTH:
            if at_line_start:
                labels[i] = 0
            else:
                run.append(i)
            continue
        if run:
            strongest = max(run, key=lambda j: (PUNCTUATION_STRENGTH[words[j]], -j))
            for j in run:
                labels[j] = int(j == strongest)
            run = []
        at_line_start = word == "\n"


def compress_chunk(chunk, p_keep, rate, force_tokens, token_map,
                   drop_consecutive=True, lang="default", word_weight=None):
    """Steps 4-6 for one chunk (__compress). Adds words, word_probs, threshold, labels, kept_text."""
    reduce_rate = max(0, 1 - rate)
    words, word_probs, token_word = merge_tokens_to_words(chunk["tokens"], list(p_keep), force_tokens, token_map)
    labels, kept = [], []
    threshold = None
    if words:  # (the reference crashes on a chunk without words, e.g. only "[UNK]")
        if drop_consecutive:
            drop_repeated_force_tokens(words, word_probs, force_tokens, reduce_rate)
        threshold = keep_threshold(words, word_probs, reduce_rate, word_weight)
        for word, p in zip(words, word_probs):
            if p > threshold or (threshold == 1.0 and p == threshold):
                if drop_consecutive and word in force_tokens and kept and kept[-1] == word:
                    labels.append(0)
                else:
                    kept.append(word)
                    labels.append(1)
            else:
                labels.append(0)
        if lang == "zh-hant":
            tidy_punctuation(words, labels)
            kept = [word for word, label in zip(words, labels) if label]
    chunk.update(p_keep=p_keep, words=words, word_probs=word_probs, token_word=token_word,
                 threshold=threshold, labels=labels, kept_text=words_to_text(kept, lang))
    return chunk


def compute_p_keep(chunks, model, erf=erf_exact, on_chunk=None):
    """Step 3 for every chunk: chunk["p_keep"] = BERT's keep-probabilities.

    on_chunk(index, chunk) is called after each chunk, e.g. to report timing.
    """
    for i, chunk in enumerate(chunks):
        chunk["p_keep"] = bert_forward(chunk["ids"], model, erf)
        if on_chunk:
            on_chunk(i, chunk)


def apply_rate(chunks, token_map, rate, lang="default", drop_consecutive=True, word_weight=None):
    """Steps 4-6 for every chunk, using the stored p_keep -> compressed text.

    The model's probabilities do not depend on the rate, so this can be re-run
    for another rate without touching the model.
    """
    if rate >= 1:  # reduce_rate 0: the reference returns the chunk strings untouched
        return restore_force_tokens("".join(c["text"] for c in chunks), token_map)
    force_tokens = SETTINGS[lang]["force_tokens"]
    for chunk in chunks:
        compress_chunk(chunk, chunk["p_keep"], rate, force_tokens, token_map, drop_consecutive, lang, word_weight)
    return "".join(c["kept_text"] for c in chunks)


# Fenced blocks (``` or ~~~) and `inline code`: compressing these would break them,
# so they are copied to the output untouched.
CODE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]+`", re.S)
CHINESE = re.compile("[" + "".join(f"{chr(first)}-{chr(last)}" for first, last in CHINESE_RANGES) + "]")


def choose_language(text, lang="auto"):
    """'auto' uses the Chinese settings as soon as the text contains Chinese characters."""
    if lang != "auto":
        return lang
    return "zh-hant" if CHINESE.search(text) else "default"


def split_code(text, protect_code=True):
    """Text -> [(is_code, piece)], keeping the pieces in order."""
    if not protect_code:
        return [(False, text)]
    pieces, position = [], 0
    for match in CODE.finditer(text):
        if match.start() > position:
            pieces.append((False, text[position:match.start()]))
        pieces.append((True, match.group()))
        position = match.end()
    pieces.append((False, text[position:]))
    return [(is_code, piece) for is_code, piece in pieces if piece]


def plan(text, tokenizer, lang="auto", protect_code=True):
    """Everything that happens before the model runs.

    Returns {"lang", "parts", "token_ids"}. A part is either
        {"code": "..."}                      copied to the output unchanged, or
        {"chunks": [...], "token_map": {...}} text to compress.
    Keeping the plan lets a caller run the model once and then render several
    rates (a slider, a search for a token budget) without running BERT again.
    """
    lang = choose_language(text, lang)
    settings = SETTINGS[lang]
    parts = []
    for is_code, piece in split_code(text, protect_code):
        if is_code:
            parts.append({"code": piece})
        else:
            chunks, token_map = prepare_chunks(piece, tokenizer, settings["force_tokens"],
                                               settings["chunk_end_tokens"])
            parts.append({"chunks": chunks, "token_map": token_map})
    token_ids = [i for part in parts for chunk in part.get("chunks", []) for i in chunk["ids"]]
    return {"lang": lang, "parts": parts, "token_ids": token_ids}


def plan_chunks(compression_plan):
    """Every chunk of the plan, in order."""
    return [chunk for part in compression_plan["parts"] for chunk in part.get("chunks", [])]


def run_model(compression_plan, model, erf=erf_exact, on_chunk=None):
    """Step 3 for the whole plan: BERT's keep-probability for every token."""
    compute_p_keep(plan_chunks(compression_plan), model, erf, on_chunk)
    return compression_plan


def render(compression_plan, rate=0.6, drop_consecutive=True, word_weight=None):
    """Steps 4-6 for the whole plan -> the compressed text.

    Cheap: the model's probabilities are already known, so another rate costs
    only the word merging and the threshold.
    """
    lang = compression_plan["lang"]
    return "".join(part["code"] if "code" in part else
                   apply_rate(part["chunks"], part["token_map"], rate, lang, drop_consecutive, word_weight)
                   for part in compression_plan["parts"])


def compress(text, tokenizer, model, rate=0.6, lang="auto", drop_consecutive=True,
             erf=erf_exact, word_weight=None, on_chunk=None, protect_code=True):
    """compress_prompt_llmlingua2 for one text. Returns (compressed text, plan).

    model must hold the embedding rows of the plan's token ids
    (see plan() + load_model(token_ids=...)).
    """
    compression_plan = plan(text, tokenizer, lang, protect_code)
    if rate < 1:
        run_model(compression_plan, model, erf, on_chunk)
    return render(compression_plan, rate, drop_consecutive, word_weight), compression_plan


# =============================================================================
# Part 6  One-call API
# =============================================================================

_loaded = {}


def get_tokenizer_and_model(model_dir=MODEL_DIR):
    """Load the tokenizer and the whole model once; later calls reuse them."""
    model_dir = Path(model_dir)
    if model_dir not in _loaded:
        if not (model_dir / "manifest.csv").exists():
            raise FileNotFoundError(f"{model_dir} has no manifest.csv (full model: run convert.py; int8: quantize.py)")
        _loaded[model_dir] = (load_tokenizer(model_dir), load_model(model_dir))
    return _loaded[model_dir]


def compress_text(text, rate=0.6, lang="auto", model_dir=MODEL_DIR, protect_code=True):
    """Compress a string with LLMLingua-2 and return the shorter string.

    rate         share of words to keep, e.g. 0.6 keeps about 60% (the model card's setting)
    lang         "auto" (Chinese rules when the text has Chinese characters),
                 "default" (exactly Microsoft's settings) or "zh-hant"
    model_dir    model_csv_int8/ (default) or model_csv/ (full precision, if built)
    protect_code ``` blocks and `inline code` are copied through untouched
    The first call loads the model (a few seconds); later calls are fast.
    """
    tokenizer, model = get_tokenizer_and_model(model_dir)
    compressed, _ = compress(text, tokenizer, model, rate, lang, protect_code=protect_code)
    return compressed


def compress_file(input_path, output_path=None, rate=0.6, lang="auto", model_dir=MODEL_DIR, protect_code=True):
    """Compress a UTF-8 text file. Writes output_path if given and returns the compressed text."""
    with open(input_path, encoding="utf-8", newline="") as f:
        compressed = compress_text(f.read(), rate, lang, model_dir, protect_code)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(compressed)
    return compressed
