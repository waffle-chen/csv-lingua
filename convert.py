"""convert.py - turn Microsoft's LLMLingua-2 BERT model into plain CSV files.

    downloads/model.safetensors  ─┐
    downloads/vocab.txt           ├─>  model_csv/config.csv
    downloads/tokenizer.json      │    model_csv/vocab.csv
    downloads/config.json        ─┘    model_csv/manifest.csv
                                       model_csv/weights/*.csv

Steps
  1. download the original files from Hugging Face (pinned commit, sha256 check)
  2. parse model.safetensors with struct + json + np.frombuffer
  3. measure the exact CSV size of every tensor and project the total
  4. write config.csv and vocab.csv
  5. write the weights, one CSV row per matrix row, sharded below 45 MB
  6. report sizes

This is the only file that ever reads the binary safetensors file.
Run:  python convert.py
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import shutil
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
MODEL_COMMIT = "5f0c82792b7ea14c6484e015b6a072009496b7f2"
LLMLINGUA_COMMIT = "5a4c78ae18ab17a98cf997e8259354e546081d64"  # llmlingua 0.2.2
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "README.md",
    "model.safetensors",
]
NUM_ADDED_TOKENS = 100  # [NEW0]..[NEW99], added at runtime by llmlingua's init_llmlingua2

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "downloads"
MODEL_CSV = ROOT / "model_csv"
WEIGHTS = MODEL_CSV / "weights"

MAX_FILE_BYTES = 45_000_000  # shard target; the hard limit is 50 MB
STOP_ABOVE_BYTES = 1_900_000_000  # ask the human before writing more than this
MB = 1_000_000


# ---------------------------------------------------------------- 1. download


def hub_sha256s():
    """Ask the Hub API for the sha256 of each LFS file at the pinned commit.

    Returns {filename: sha256}. Only LFS files (model.safetensors) have one.
    Returns {} if the API cannot be reached.
    """
    url = f"https://huggingface.co/api/models/{REPO}/revision/{MODEL_COMMIT}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            info = json.load(response)
    except OSError as error:
        print(f"  (could not read hashes from the Hub API: {error})")
        return {}
    return {s["rfilename"]: s["lfs"]["sha256"] for s in info.get("siblings", []) if s.get("lfs")}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 * MB):
            digest.update(block)
    return digest.hexdigest()


def download(name):
    """Download one file into downloads/, printing progress."""
    url = f"https://huggingface.co/{REPO}/resolve/{MODEL_COMMIT}/{name}"
    target = DOWNLOADS / name
    partial = target.with_name(name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done, last_shown = 0, -1
        while block := response.read(MB):
            out.write(block)
            done += len(block)
            percent = 100 * done // total if total else 0
            if total > 10 * MB and percent != last_shown:
                print(f"\r  {name}: {done / MB:7.1f} / {total / MB:.1f} MB ({percent}%)", end="", flush=True)
                last_shown = percent
    if total > 10 * MB:
        print()
    partial.replace(target)


def fetch_sources():
    """Step 1: make sure every source file is in downloads/ and verified."""
    DOWNLOADS.mkdir(exist_ok=True)
    missing = [name for name in FILES if not (DOWNLOADS / name).exists()]

    # Free space: the download (~0.71 GB) plus the CSV model (~1.8 GB) plus margin.
    needed = (0.75 if "model.safetensors" in missing else 0) + 2.0
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"  free disk space: {free:.1f} GB (need about {needed:.1f} GB)")
    if free < needed:
        sys.exit("Not enough free disk space. Free some space and run again.")

    for name in missing:
        try:
            download(name)
        except OSError as error:
            sys.exit(
                f"Download of {name} failed: {error}\n"
                f"Please download it by hand from https://huggingface.co/{REPO}/tree/{MODEL_COMMIT}\n"
                f"into {DOWNLOADS} and run convert.py again."
            )

    source_sha = sha256_of(DOWNLOADS / "model.safetensors")
    expected = hub_sha256s().get("model.safetensors")
    if expected is None:
        print(f"  model.safetensors sha256 {source_sha} (no Hub hash available to compare)")
    elif expected != source_sha:
        sys.exit(f"sha256 mismatch for model.safetensors: got {source_sha}, Hub says {expected}")
    else:
        print(f"  model.safetensors sha256 {source_sha} matches the Hub")
    return source_sha


# ---------------------------------------------------- 2. parse safetensors


def read_safetensors(path):
    """Parse a .safetensors file without the safetensors library.

    Layout of the file:
        8 bytes      little-endian uint64 N = length of the JSON header
        N bytes      JSON: {name: {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}}
        rest         raw little-endian tensor bytes; offsets count from the end of the header

    Returns {name: float32 array} in the file's order.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data = f.read()

    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        assert entry["dtype"] == "F32", f"{name}: only float32 is expected, got {entry['dtype']}"
        start, end = entry["data_offsets"]
        array = np.frombuffer(data, dtype="<f4", count=(end - start) // 4, offset=start)
        tensors[name] = array.reshape(entry["shape"])
    return tensors


# ------------------------------------------------------ 3. measure the size


def rounded(tensor, decimals):
    """float32 tensor -> float64 values rounded to `decimals` places.

    Adding 0.0 turns -0.0 into 0.0, so tiny negatives are written "0.000000".
    """
    return np.round(tensor.astype(np.float64), decimals) + 0.0


def csv_bytes(tensor, decimals):
    """Exact number of bytes the tensor will take as CSV text.

    Every number is written as  [-]<integer digits>.<decimals digits>
    and followed by exactly one separator (',' or '\\n').
    """
    values = rounded(tensor, decimals)
    size = np.abs(values)
    integer_digits = np.where(size < 1, 1, np.floor(np.log10(np.maximum(size, 1))) + 1)
    minus_signs = values < 0
    return int(integer_digits.sum() + minus_signs.sum() + values.size * (decimals + 2))


def as_matrix(tensor):
    """A 1-D tensor (bias, LayerNorm γ/β) is stored as a single CSV row."""
    return tensor.reshape(1, -1) if tensor.ndim == 1 else tensor


# ------------------------------------------------- 4. config.csv, vocab.csv


def write_config(source_sha, decimals, total_numbers):
    config = json.loads((DOWNLOADS / "config.json").read_text(encoding="utf-8"))
    tok = json.loads((DOWNLOADS / "tokenizer.json").read_text(encoding="utf-8"))
    tok_config = json.loads((DOWNLOADS / "tokenizer_config.json").read_text(encoding="utf-8"))

    normalizer, model, decoder = tok["normalizer"], tok["model"], tok["decoder"]
    assert normalizer["type"] == "BertNormalizer" and tok["pre_tokenizer"]["type"] == "BertPreTokenizer"
    assert model["type"] == "WordPiece" and decoder["type"] == "WordPiece"

    rows = [
        ("model_repo", REPO),
        ("model_commit", MODEL_COMMIT),
        ("source_file", "model.safetensors"),
        ("source_sha256", source_sha),
        ("llmlingua_commit", LLMLINGUA_COMMIT),
        ("csv_decimals", decimals),
        ("csv_total_numbers", total_numbers),
        # architecture
        ("hidden_size", config["hidden_size"]),
        ("num_hidden_layers", config["num_hidden_layers"]),
        ("num_attention_heads", config["num_attention_heads"]),
        ("intermediate_size", config["intermediate_size"]),
        ("max_position_embeddings", config["max_position_embeddings"]),
        ("type_vocab_size", config["type_vocab_size"]),
        ("vocab_size", config["vocab_size"]),
        ("layer_norm_eps", repr(config["layer_norm_eps"])),
        ("hidden_act", config["hidden_act"]),
        ("keep_label_index", 1),
        # tokenizer
        ("model_max_length", tok_config["model_max_length"]),
        ("normalizer_clean_text", normalizer["clean_text"]),
        ("normalizer_handle_chinese_chars", normalizer["handle_chinese_chars"]),
        ("normalizer_strip_accents", normalizer["strip_accents"]),
        ("normalizer_lowercase", normalizer["lowercase"]),
        ("pre_tokenizer", tok["pre_tokenizer"]["type"]),
        ("wordpiece_prefix", model["continuing_subword_prefix"]),
        ("wordpiece_max_input_chars_per_word", model["max_input_chars_per_word"]),
        ("unk_token", model["unk_token"]),
        ("cls_token", tok_config["cls_token"]),
        ("sep_token", tok_config["sep_token"]),
        ("pad_token", tok_config["pad_token"]),
        ("mask_token", tok_config["mask_token"]),
        ("decoder_prefix", decoder["prefix"]),
        ("decoder_cleanup", decoder["cleanup"]),
        ("added_tokens_first_id", len(model["vocab"])),
        ("added_tokens_count", NUM_ADDED_TOKENS),
        ("added_tokens_pattern", "[NEW{i}]"),
    ]
    with open(MODEL_CSV / "config.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["key", "value"])
        writer.writerows(rows)
    return config


def write_vocab(vocab_size):
    """vocab.csv: id,token,special for every row of the embedding table.

    vocab.txt is split on '\\n' only: str.splitlines() would also split on
    rare characters such as U+2028 that are valid inside tokens.
    """
    lines = (DOWNLOADS / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    tok = json.loads((DOWNLOADS / "tokenizer.json").read_text(encoding="utf-8"))
    assert all(tok["model"]["vocab"][t] == i for i, t in enumerate(lines)), "vocab.txt and tokenizer.json disagree"
    assert len(tok["model"]["vocab"]) == len(lines)

    special_ids = {t["id"] for t in tok["added_tokens"] if t["special"]}
    tokens = lines + [f"[NEW{i}]" for i in range(NUM_ADDED_TOKENS)]
    assert len(tokens) == vocab_size, f"{len(tokens)} tokens but vocab_size is {vocab_size}"

    with open(MODEL_CSV / "vocab.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "token", "special"])
        for i, token in enumerate(tokens):
            writer.writerow([i, token, int(i in special_ids or i >= len(lines))])
    return len(lines)


# ----------------------------------------------------------- 5. the weights


def format_rows(matrix, decimals):
    """rows×cols floats -> one text line per row: 'x,x,...,x\\n'."""
    template = ",".join([f"%.{decimals}f"] * matrix.shape[1]) + "\n"
    return [template % tuple(row) for row in matrix.tolist()]


def write_tensor(name, tensor, projected_bytes, decimals):
    """Write one tensor as CSV; split it into shards below MAX_FILE_BYTES.

    One CSV row is one matrix row, so shard k simply holds rows
    first_row .. first_row+rows-1. Returns manifest lines.
    """
    matrix = as_matrix(tensor)
    n_rows, n_cols = matrix.shape
    sharded = projected_bytes > MAX_FILE_BYTES
    manifest, shard, out, used, first_row = [], 0, None, 0, 0

    def open_shard(row):
        nonlocal out, used, first_row
        file_name = f"{name}.part{shard:02d}.csv" if sharded else f"{name}.csv"
        out = open(WEIGHTS / file_name, "w", encoding="ascii", newline="\n")
        used, first_row = 0, row
        manifest.append([name, f"weights/{file_name}", row, 0, n_cols, tensor.ndim])

    open_shard(0)
    block_rows = 2048
    for start in range(0, n_rows, block_rows):
        lines = format_rows(rounded(matrix[start : start + block_rows], decimals), decimals)
        for offset, line in enumerate(lines):
            if used + len(line) > MAX_FILE_BYTES and used > 0:
                out.close()
                manifest[-1][3] = start + offset - first_row
                shard += 1
                open_shard(start + offset)
            out.write(line)
            used += len(line)
    out.close()
    manifest[-1][3] = n_rows - first_row
    return manifest


# -------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decimals", type=int, default=7, help="decimal places per weight (default 7; 6 is too coarse for 1e-4 parity)")
    args = parser.parse_args()
    t_start = time.time()

    print("1. Source files")
    source_sha = fetch_sources()

    print("\n2. Tensor inventory (model.safetensors)")
    tensors = read_safetensors(DOWNLOADS / "model.safetensors")
    total_numbers = 0
    for name, tensor in tensors.items():
        total_numbers += tensor.size
        if ".layer." not in name or ".layer.0." in name:
            print(f"  {name:55s} {str(list(tensor.shape)):>14s} {tensor.size:>12,d}")
    print(f"  ... layers 1-11 repeat layer 0 ...")
    print(f"  {len(tensors)} tensors, {total_numbers:,} numbers")

    print(f"\n3. Measured CSV size at {args.decimals} decimals")
    projected = {name: csv_bytes(t, args.decimals) for name, t in tensors.items()}
    groups = {"word embeddings": ["bert.embeddings.word_embeddings.weight"]}
    groups["everything else"] = [n for n in tensors if n not in groups["word embeddings"]]
    for label, names in groups.items():
        size = sum(projected[n] for n in names)
        count = sum(tensors[n].size for n in names)
        print(f"  {label:16s} {count:>12,d} numbers  {size / 1e9:6.3f} GB  {size / count:.3f} bytes/number")
    projected_total = sum(projected.values())
    print(f"  projected weights total: {projected_total / 1e9:.3f} GB ({projected_total / total_numbers:.3f} bytes/number)")
    if projected_total > STOP_ABOVE_BYTES:
        sys.exit(f"Projected size is above {STOP_ABOVE_BYTES / 1e9} GB. Stopping so the human can decide.")

    print("\n4. config.csv and vocab.csv")
    MODEL_CSV.mkdir(exist_ok=True)
    config = write_config(source_sha, args.decimals, total_numbers)
    n_wordpiece = write_vocab(config["vocab_size"])
    print(f"  vocab: {n_wordpiece:,} WordPiece tokens + {NUM_ADDED_TOKENS} added tokens "
          f"[NEW0]={n_wordpiece}..[NEW{NUM_ADDED_TOKENS - 1}]={n_wordpiece + NUM_ADDED_TOKENS - 1}")

    print("\n5. Weights")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for old in WEIGHTS.glob("*.csv"):
        old.unlink()
    manifest = []
    for i, (name, tensor) in enumerate(tensors.items(), 1):
        manifest += write_tensor(name, tensor, projected[name], args.decimals)
        print(f"\r  {i}/{len(tensors)} tensors written", end="", flush=True)
    print()
    with open(MODEL_CSV / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["tensor", "file", "first_row", "rows", "cols", "ndim"])
        writer.writerows(manifest)

    print("\n6. Result")
    files = sorted(MODEL_CSV.rglob("*.csv"), key=lambda p: p.stat().st_size)
    total = sum(p.stat().st_size for p in files)
    weights_total = sum(p.stat().st_size for p in WEIGHTS.glob("*.csv"))
    print(f"  files: {len(files)} CSV ({len(list(WEIGHTS.glob('*.csv')))} weight files)")
    print(f"  total: {total / 1e9:.3f} GB (weights {weights_total / 1e9:.3f} GB; projected {projected_total / 1e9:.3f} GB)")
    print(f"  largest file: {files[-1].relative_to(ROOT)} {files[-1].stat().st_size / MB:.1f} MB")
    print(f"  time: {time.time() - t_start:.0f} s")
    print("\n  downloads/model.safetensors is kept for the precision and reference checks;"
          "\n  it can be deleted afterwards (the CSVs do not need it).")


if __name__ == "__main__":
    main()
