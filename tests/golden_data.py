"""Helpers to read the golden reference data in tests/golden/ (made by tools/reference_check.py)."""
import csv
import functools
from collections import defaultdict
from pathlib import Path

import csvlingua

GOLDEN = Path(__file__).resolve().parent / "golden"
COMPRESSION_CASES = ["model_card", "meeting", "meeting_rate033", "no_chunk_end", "zh_hant_default", "zh_hant"]


def read_rows(name):
    with open(GOLDEN / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_text(name):
    with open(GOLDEN / name, encoding="utf-8", newline="") as f:
        return f.read()


def by_chunk(rows):
    chunks = defaultdict(list)
    for row in rows:
        chunks[int(row["chunk"])].append(row)
    return [chunks[i] for i in sorted(chunks)]


@functools.cache
def tokenizer():
    return csvlingua.load_tokenizer()


HAS_FULL_MODEL = (csvlingua.FULL_MODEL_DIR / "manifest.csv").exists()
HAS_INT8_MODEL = (csvlingua.MODEL_DIR / "manifest.csv").exists()


@functools.cache
def model(model_dir=csvlingua.FULL_MODEL_DIR):
    """A CSV model (all embedding shards), loaded once for all tests. Default: full precision."""
    return csvlingua.load_model(model_dir)


@functools.cache
def our_p_keep(case, model_dir=csvlingua.FULL_MODEL_DIR):
    """Run our BERT on the reference's token ids of every chunk -> [p_keep array per chunk]."""
    return [csvlingua.bert_forward([int(r["token_id"]) for r in rows], model(model_dir))
            for rows in by_chunk(read_rows(f"{case}.tokens.csv"))]
