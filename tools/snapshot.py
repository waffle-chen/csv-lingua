"""tools/snapshot.py - DEV ONLY: record what the shipped model currently produces.

Writes tests/output_snapshots.csv: for every example and several rates, a hash of
the compressed text produced by model_csv_int8/. tests/test_snapshots.py compares
against it, so any change that moves an output shows up immediately.

Run it again (and read the diff) when an output is *meant* to change:

    python tools/snapshot.py
"""
import sys

sys.dont_write_bytecode = True

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import csvlingua  # noqa: E402
from tests.snapshots import SNAPSHOT_FILE, snapshot_rows  # noqa: E402


def main():
    tokenizer, model = csvlingua.get_tokenizer_and_model()
    rows = snapshot_rows(tokenizer, model)
    with open(SNAPSHOT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["example", "lang", "rate", "words_kept", "characters", "sha256_16"])
        writer.writerows(rows)
    print(f"wrote {SNAPSHOT_FILE.relative_to(ROOT)} ({len(rows)} rows)")
    for row in rows:
        print("  " + "  ".join(str(cell) for cell in row))


if __name__ == "__main__":
    main()
