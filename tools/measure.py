"""tools/measure.py - DEV ONLY: measure every number used in the README.

Uses only numpy. Rows that need the full-precision model are skipped when
model_csv/ was not built (python convert.py). Writes examples/results.csv and
refreshes the compressed examples and their traces.

    python tools/measure.py
"""
import sys

sys.dont_write_bytecode = True

import csv
import difflib
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import csvlingua  # noqa: E402
from tests import golden_data  # noqa: E402
from tests.test_compression import (case_input, label_agreement, reference_p_keep,  # noqa: E402
                                    run_case, tiktoken_counts)

CASES = golden_data.COMPRESSION_CASES
RATES = (0.3, 0.5, 0.6, 0.8)
results = []


def record(metric, value, note=""):
    results.append((metric, value, note))
    print(f"  {metric:46s} {value:>18s}  {note}", flush=True)


def folder_size(folder):
    files = sorted(Path(folder).rglob("*.csv"), key=lambda p: p.stat().st_size)
    return len(files), sum(p.stat().st_size for p in files), files[-1]


def worst_difference(a_chunks, b_chunks):
    return max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(a_chunks, b_chunks))


def previous_gelu(x, erf=None):
    """The first implementation, kept to measure the speed-up: math.erf per element, one thread."""
    return 0.5 * x * (1 + csvlingua.erf_math(x / np.float32(math.sqrt(2))).astype(np.float32))


def measure_sizes():
    print("CSV models")
    count, size, largest = folder_size(csvlingua.MODEL_DIR)
    record("int8_total_gb", f"{size / 1e9:.3f}", f"model_csv_int8/, {count} files")
    record("int8_largest_file_mb", f"{largest.stat().st_size / 1e6:.1f}", largest.name)
    if golden_data.HAS_FULL_MODEL:
        count, size, largest = folder_size(csvlingua.FULL_MODEL_DIR)
        record("full_total_gb", f"{size / 1e9:.3f}", f"model_csv/, {count} files, 7 decimals")


def measure_loading_and_speed():
    print("Loading and speed")
    tokenizer = csvlingua.load_tokenizer()
    meeting = (ROOT / "examples" / "meeting.txt").read_text(encoding="utf-8")
    ids = csvlingua.plan(meeting, tokenizer)["token_ids"]
    start = time.perf_counter()
    csvlingua.load_model(csvlingua.MODEL_DIR, token_ids=ids)
    record("load_int8_for_one_text_s", f"{time.perf_counter() - start:.1f}", f"{len(set(ids))} embedding rows")
    start = time.perf_counter()
    model = csvlingua.load_model(csvlingua.MODEL_DIR)
    record("load_int8_whole_model_s", f"{time.perf_counter() - start:.1f}", "all 119,647 embedding rows")

    raw = (csvlingua.MODEL_DIR / "weights" / "bert.encoder.layer.0.intermediate.dense.weight.csv").read_bytes()
    for label, parallel in [("one_thread", False), ("all_cores", True)]:
        csvlingua.parse_csv_bytes(raw, parallel)
        start = time.perf_counter()
        csvlingua.parse_csv_bytes(raw, parallel)
        record(f"csv_parser_mb_per_s_{label}", f"{len(raw) / 1e6 / (time.perf_counter() - start):.0f}")

    long_text = meeting * 20
    start = time.perf_counter()
    tokens = csvlingua.tokenize(long_text, tokenizer)
    record("tokenizer_k_chars_per_s", f"{len(long_text) / 1000 / (time.perf_counter() - start):.0f}",
           f"{len(tokens)} tokens from {len(long_text) // 1000} k chars")

    chunk_ids = [int(r["token_id"]) for r in golden_data.by_chunk(golden_data.read_rows("no_chunk_end.tokens.csv"))[0]]
    assert len(chunk_ids) == 512
    for label, erf in [("exact", csvlingua.erf_exact), ("fast_gelu", csvlingua.erf_abramowitz_stegun)]:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            csvlingua.bert_forward(chunk_ids, model, erf)
            times.append(time.perf_counter() - start)
        record(f"seconds_per_512_token_chunk_{label}", f"{sorted(times)[1]:.2f}")
    fast, csvlingua.gelu = csvlingua.gelu, previous_gelu
    start = time.perf_counter()
    first_version = csvlingua.bert_forward(chunk_ids, model)
    record("seconds_per_512_token_chunk_first_version", f"{time.perf_counter() - start:.2f}",
           "np.vectorize(math.erf), one thread")
    csvlingua.gelu = fast
    record("speed_up_is_bit_identical", str(np.array_equal(first_version, csvlingua.bert_forward(chunk_ids, model))))
    return model


def measure_parity(int8_model):
    print("Parity with Microsoft's implementation")
    int8 = {case: golden_data.our_p_keep(case, csvlingua.MODEL_DIR) for case in CASES}
    tokens = sum(len(p) for chunks in int8.values() for p in chunks)
    record("int8_max_abs_dp_keep_vs_reference",
           f"{max(worst_difference(int8[c], reference_p_keep(c)) for c in CASES):.1e}", f"{tokens} tokens")
    if golden_data.HAS_FULL_MODEL:
        full = {case: golden_data.our_p_keep(case) for case in CASES}
        record("full_max_abs_dp_keep_vs_reference",
               f"{max(worst_difference(full[c], reference_p_keep(c)) for c in CASES):.1e}", "7-decimal CSV weights")
        states = []
        rows = golden_data.read_rows("model_card.tokens.csv")
        csvlingua.bert_forward([int(r["token_id"]) for r in rows], golden_data.model(), hidden_states=states)
        worst = max(float(np.abs(s - np.loadtxt(golden_data.GOLDEN / f"model_card.hidden_layer{k:02d}.csv",
                                                delimiter=",")).max()) for k, s in enumerate(states))
        record("full_max_abs_hidden_state_all_layers", f"{worst:.1e}", "model_card, embeddings + 12 layers")

    print("Word labels")
    estimated = one_per_word = total = 0
    for case in CASES:
        reference = [x for c in run_case(case, reference_p_keep(case), tiktoken_counts(case).__getitem__)
                     for x in c["labels"]]
        for rule, name in [(None, "estimated"), (lambda word: 1, "one")]:
            ours = [x for c in run_case(case, reference_p_keep(case), rule) for x in c["labels"]]
            changed = sum(x != y for x, y in zip(ours, reference))
            if name == "estimated":
                estimated += changed
            else:
                one_per_word += changed
        total += len(reference)
    record("labels_changed_by_token_estimate", f"{estimated}/{total}",
           "reference p_keep; estimate_token_count vs the real GPT-3.5 counts")
    record("labels_changed_by_one_per_word", f"{one_per_word}/{total}", "the simpler rule, for comparison")
    agreement = [label_agreement(case, run_case(case, int8[case], None))[0] for case in CASES]
    record("int8_label_agreement_with_reference", f"{100 * min(agreement):.1f}-{100 * max(agreement):.1f}%",
           "8-bit model and the token-count estimate vs llmlingua")
    agreement = [label_agreement(case, run_case(case, int8[case], tiktoken_counts(case).__getitem__))[0]
                 for case in CASES]
    record("int8_label_agreement_with_token_counts", f"{100 * min(agreement):.1f}-{100 * max(agreement):.1f}%",
           "same, but with the reference's GPT-3.5 token counts")

    if golden_data.HAS_FULL_MODEL:
        print("8-bit vs full precision")
        tokenizer = golden_data.tokenizer()
        labels, sequences = [], []
        for case in ["model_card", "meeting", "no_chunk_end", "zh_hant"]:
            text, lang, _ = case_input(case)
            full_plan = csvlingua.plan(text, tokenizer, lang)
            int8_plan = csvlingua.plan(text, tokenizer, lang)
            for chunk, p_keep in zip(csvlingua.plan_chunks(full_plan), golden_data.our_p_keep(case)):
                chunk["p_keep"] = p_keep
            csvlingua.run_model(int8_plan, int8_model)
            for rate in RATES:
                csvlingua.render(full_plan, rate)
                csvlingua.render(int8_plan, rate)
                a = [x for c in csvlingua.plan_chunks(full_plan) for x in c["labels"]]
                b = [x for c in csvlingua.plan_chunks(int8_plan) for x in c["labels"]]
                kept_a = [w for c in csvlingua.plan_chunks(full_plan) for w, x in zip(c["words"], c["labels"]) if x]
                kept_b = [w for c in csvlingua.plan_chunks(int8_plan) for w, x in zip(c["words"], c["labels"]) if x]
                labels.append(sum(x == y for x, y in zip(a, b)) / len(a))
                sequences.append(difflib.SequenceMatcher(None, kept_a, kept_b, autojunk=False).ratio())
        record("int8_vs_full_label_agreement", f"{100 * min(labels):.1f}-{100 * max(labels):.1f}%",
               f"4 texts, rates {RATES}")
        record("int8_vs_full_kept_word_similarity", f"{100 * min(sequences):.1f}-{100 * max(sequences):.1f}%")


def measure_command_line():
    print("Command line (fresh process, writes the examples)")
    for name in ["meeting", "meeting.zh-hant", "mixed"]:
        output = ROOT / "examples" / f"{name}.compressed.txt"
        start = time.perf_counter()
        done = subprocess.run([sys.executable, "-B", str(ROOT / "compress.py"), str(ROOT / "examples" / f"{name}.txt"),
                               "-o", str(output), "--trace", str(ROOT / "examples" / f"{name}.trace.csv")],
                              capture_output=True, text=True, encoding="utf-8", check=True)
        key = name.replace(".", "_").replace("-", "_")
        record(f"cli_{key}_wall_s", f"{time.perf_counter() - start:.1f}",
               re.search(r"words:\s+(.*)", done.stderr).group(1).strip())
        record(f"cli_{key}_peak_ram_mb", re.search(r"peak RAM (\d+) MB", done.stderr).group(1))


def main():
    measure_sizes()
    int8_model = measure_loading_and_speed()
    measure_parity(int8_model)
    measure_command_line()
    with open(ROOT / "examples" / "results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["metric", "value", "note"])
        writer.writerows(results)
    print(f"wrote examples/results.csv ({len(results)} rows)")


if __name__ == "__main__":
    main()
