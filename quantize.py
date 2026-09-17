"""quantize.py - build an 8-bit version of the CSV model: model_csv/ -> model_csv_int8/.

Symmetric per-row int8 quantization of every weight matrix:

    s_r = max_j |W_rj| / 127                  one scale per matrix row
    q_rj = round(W_rj / s_r)  in [-127, 127]  one small integer per weight
    W_rj ≈ s_r · q_rj                         what csvlingua.load_csv_matrix computes

Each CSV row is written as  s_r, q_r1, q_r2, ...  (the scale first), so the file still
has one CSV row per matrix row. Biases, LayerNorm γ/β and the 2×768 classifier stay
as full decimals (they are tiny). The files are about a third of the size, which makes
loading faster; the arithmetic after loading is the same float32 math as before.

Run:  python quantize.py
"""
import sys

sys.dont_write_bytecode = True

import csv
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import csvlingua

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model_csv"
TARGET = ROOT / "model_csv_int8"
LEVELS = 127
SCALE_DECIMALS = 12
KEEP_FLOAT = {"classifier.weight"}


def quantize_rows(W):
    """W: rows×cols float -> (s: rows, q: rows×cols integers in [-127, 127])"""
    s = np.round(np.abs(W).max(axis=1) / LEVELS, SCALE_DECIMALS)
    safe = np.where(s == 0, 1, s)
    q = np.clip(np.round(W / safe[:, None]), -LEVELS, LEVELS).astype(np.int32)
    return s, q


def quantize_file(source_file, target_file):
    """One weight CSV -> one int8 CSV with the same rows. Returns the file size in bytes."""
    W = np.loadtxt(source_file, delimiter=",", dtype=np.float64, ndmin=2)
    s, q = quantize_rows(W)
    template = f"%.{SCALE_DECIMALS}f," + ",".join(["%d"] * W.shape[1]) + "\n"
    with open(target_file, "w", encoding="ascii", newline="\n") as out:
        for scale, row in zip(s.tolist(), q.tolist()):
            out.write(template % (scale, *row))
    return target_file.stat().st_size


def main():
    start = time.time()
    if not (SOURCE / "manifest.csv").exists():
        sys.exit("model_csv/ not found; run convert.py first.")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    (TARGET / "weights").mkdir(parents=True)
    shutil.copy(SOURCE / "vocab.csv", TARGET / "vocab.csv")

    config = csvlingua.read_key_value_csv(SOURCE / "config.csv")
    config.update(quantization="int8, symmetric, one scale per matrix row",
                  quantization_levels=LEVELS, quantization_scale_decimals=SCALE_DECIMALS,
                  quantization_source="model_csv", quantization_kept_float="1-D tensors, classifier.weight")
    with open(TARGET / "config.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["key", "value"])
        writer.writerows(config.items())

    manifest_rows, jobs = [], []
    for tensor, entries in csvlingua.read_manifest(SOURCE).items():
        for e in entries:
            quantized = e["ndim"] == 2 and tensor not in KEEP_FLOAT
            manifest_rows.append([tensor, e["file"], e["first_row"], e["rows"], e["cols"], e["ndim"],
                                  "int8" if quantized else "float"])
            if quantized:
                jobs.append((SOURCE / e["file"], TARGET / e["file"]))
            else:
                shutil.copy(SOURCE / e["file"], TARGET / e["file"])
    with ProcessPoolExecutor() as pool:
        list(pool.map(quantize_file, *zip(*jobs)))
    with open(TARGET / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["tensor", "file", "first_row", "rows", "cols", "ndim", "storage"])
        writer.writerows(manifest_rows)

    files = sorted(TARGET.rglob("*.csv"), key=lambda p: p.stat().st_size)
    source_total = sum(p.stat().st_size for p in SOURCE.rglob("*.csv"))
    total = sum(p.stat().st_size for p in files)
    print(f"model_csv_int8/: {len(files)} CSV files, {total / 1e9:.3f} GB "
          f"({100 * total / source_total:.0f}% of model_csv/), largest {files[-1].stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
