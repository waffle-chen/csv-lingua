"""tools/measure.py - DEV ONLY: measure every number in the README results table.

Uses only numpy (plus convert.py's safetensors parser for the precision check,
which needs downloads/model.safetensors). Writes examples/results.csv.

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

import convert  # noqa: E402
import csvlingua  # noqa: E402
from tests import golden_data  # noqa: E402
from tests.test_compression import (case_input, label_agreement, reference_p_keep, run_case,  # noqa: E402
                                    tiktoken_counts)

INT8_DIR = ROOT / "model_csv_int8"


def previous_gelu(x, erf=None):
    """The first implementation, kept to measure the speed-up: math.erf per element, one thread."""
    return 0.5 * x * (1 + csvlingua.erf_math(x / np.float32(math.sqrt(2))).astype(np.float32))


def max_over_chunks(a_chunks, b_chunks):
    return max(float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())
               for a, b in zip(a_chunks, b_chunks))


def run_cli(*args):
    """Run compress.py in a fresh process and return (stderr text, wall seconds)."""
    start = time.perf_counter()
    done = subprocess.run([sys.executable, "-B", str(ROOT / "compress.py"), *args],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    return done.stderr, time.perf_counter() - start


def main():
    results = []

    def record(metric, value, note=""):
        results.append((metric, value, note))
        print(f"  {metric:52s} {value:>14s}  {note}")

    print("CSV model")
    files = sorted((ROOT / "model_csv").rglob("*.csv"), key=lambda p: p.stat().st_size)
    record("csv_total_gb", f"{sum(p.stat().st_size for p in files) / 1e9:.3f}", "model_csv/, all CSV files")
    record("csv_file_count", str(len(files)))
    record("csv_largest_file_mb", f"{files[-1].stat().st_size / 1e6:.1f}", files[-1].name)
    decimals = csvlingua.read_key_value_csv(csvlingua.MODEL_DIR / "config.csv")["csv_decimals"]
    record("csv_decimals", decimals)

    print("Loading")
    start = time.perf_counter()
    csv_model = golden_data.model()
    record("load_full_model_s", f"{time.perf_counter() - start:.1f}", "all 22 embedding shards, process pool")
    start = time.perf_counter()
    int8_model = csvlingua.load_model(INT8_DIR)
    record("load_int8_model_s", f"{time.perf_counter() - start:.1f}", "model_csv_int8/, process pool")
    start = time.perf_counter()
    tensors = convert.read_safetensors(convert.DOWNLOADS / "model.safetensors")
    config = csvlingua.read_key_value_csv(csvlingua.MODEL_DIR / "config.csv")
    float32_model = csvlingua.make_model(
        config, {k: v for k, v in tensors.items() if k != csvlingua.WORD_EMBEDDINGS},
        [(0, tensors[csvlingua.WORD_EMBEDDINGS])])
    record("load_safetensors_s", f"{time.perf_counter() - start:.1f}", "for comparison: binary float32")

    print("Speed (one 512-token chunk, median of 3)")
    ids = [int(r["token_id"]) for r in golden_data.by_chunk(golden_data.read_rows("no_chunk_end.tokens.csv"))[0]]
    assert len(ids) == 512
    for label, erf in [("exact_erf", csvlingua.erf_exact), ("abramowitz_stegun", csvlingua.erf_abramowitz_stegun)]:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            csvlingua.bert_forward(ids, csv_model, erf)
            times.append(time.perf_counter() - start)
        record(f"seconds_per_512_token_chunk_{label}", f"{sorted(times)[1]:.2f}")
    fast_gelu = csvlingua.gelu
    csvlingua.gelu = previous_gelu
    start = time.perf_counter()
    previous_p = csvlingua.bert_forward(ids, csv_model)
    record("seconds_per_512_token_chunk_previous_math_erf", f"{time.perf_counter() - start:.2f}",
           "np.vectorize(math.erf), one thread")
    csvlingua.gelu = fast_gelu
    record("speedup_is_bit_identical", str(np.array_equal(previous_p, csvlingua.bert_forward(ids, csv_model))))

    print("Precision and parity (all golden chunks)")
    cases = golden_data.COMPRESSION_CASES
    chunks = {case: golden_data.by_chunk(golden_data.read_rows(f"{case}.tokens.csv")) for case in cases}
    ours = {case: golden_data.our_p_keep(case) for case in cases}
    f32 = {case: [csvlingua.bert_forward([int(r["token_id"]) for r in rows], float32_model) for rows in chunks[case]]
           for case in cases}
    fast = {case: [csvlingua.bert_forward([int(r["token_id"]) for r in rows], csv_model, csvlingua.erf_abramowitz_stegun)
                   for rows in chunks[case]] for case in cases}
    n_tokens = sum(len(rows) for case in cases for rows in chunks[case])
    record("csv_precision_max_abs_dp_keep", f"{max(max_over_chunks(ours[c], f32[c]) for c in cases):.1e}",
           f"CSV weights vs original float32 weights, {n_tokens} tokens")
    record("parity_max_abs_dp_keep_csv_vs_reference",
           f"{max(max_over_chunks(ours[c], reference_p_keep(c)) for c in cases):.1e}", "our numpy + CSV vs PyTorch")
    record("parity_max_abs_dp_keep_float32_vs_reference",
           f"{max(max_over_chunks(f32[c], reference_p_keep(c)) for c in cases):.1e}", "our numpy + float32 vs PyTorch")
    record("abramowitz_stegun_max_abs_dp_keep", f"{max(max_over_chunks(fast[c], ours[c]) for c in cases):.1e}",
           "fast GELU vs exact GELU")
    states = []
    rows = golden_data.read_rows("model_card.tokens.csv")
    csvlingua.bert_forward([int(r["token_id"]) for r in rows], csv_model, hidden_states=states)
    worst = max(float(np.abs(s - np.loadtxt(golden_data.GOLDEN / f"model_card.hidden_layer{k:02d}.csv",
                                            delimiter=",")).max()) for k, s in enumerate(states))
    record("parity_max_abs_hidden_state_all_layers", f"{worst:.1e}", "model_card, embedding output + 12 layers")

    print("Labels (tiktoken approximation)")
    total_words = changed_by_approx = end_to_end_diff = exact_diff = 0
    for case in cases:
        counts = tiktoken_counts(case).__getitem__
        with_counts = run_case(case, reference_p_keep(case), counts)
        weight_one = run_case(case, reference_p_keep(case), None)
        a = [x for c in with_counts for x in c["labels"]]
        b = [x for c in weight_one for x in c["labels"]]
        changed = sum(x != y for x, y in zip(a, b))
        total_words += len(a)
        changed_by_approx += changed
        exact_diff += label_agreement(case, run_case(case, ours[case], counts))[1]
        end_to_end_diff += label_agreement(case, run_case(case, ours[case], None))[1]
        record(f"labels_changed_by_weight_1_{case}", f"{changed}/{len(a)}")
    record("labels_changed_by_weight_1_total", f"{changed_by_approx}/{total_words}",
           "reference p_keep; tiktoken counts vs 1 per word")
    record("labels_differing_ours_with_tiktoken_counts", f"{exact_diff}/{total_words}", "only the model differs")
    record("labels_differing_ours_end_to_end", f"{end_to_end_diff}/{total_words}", "our model + weight 1 vs llmlingua")

    print("zh-hant punctuation tidy")
    text, _, _ = case_input("zh_hant")
    for tidy in (False, True):
        settings = csvlingua.SETTINGS["zh-hant"]
        zh_chunks, token_map = csvlingua.prepare_chunks(text, golden_data.tokenizer(), settings["force_tokens"],
                                                        settings["chunk_end_tokens"])
        for chunk, p_keep in zip(zh_chunks, ours["zh_hant"]):
            chunk["p_keep"] = p_keep
        csvlingua.apply_rate(zh_chunks, token_map, 0.6, "zh-hant" if tidy else "default")
        labels = [x for c in zh_chunks for x in c["labels"]]
        if tidy:
            changed = sum(a != b for a, b in zip(labels, untidy))
            record("zh_hant_labels_changed_by_punctuation_tidy", f"{changed}/{len(labels)}", "rate 0.6")
        untidy = labels

    print("int8 model vs full CSV model")
    int8_files = sorted(INT8_DIR.rglob("*.csv"), key=lambda p: p.stat().st_size)
    record("int8_total_gb", f"{sum(p.stat().st_size for p in int8_files) / 1e9:.3f}", "model_csv_int8/")
    record("int8_largest_file_mb", f"{int8_files[-1].stat().st_size / 1e6:.1f}")
    worst_labels = worst_words = 1.0
    worst_dp = 0.0
    tokenizer = golden_data.tokenizer()
    for case in ["model_card", "meeting", "no_chunk_end", "zh_hant"]:
        text, lang, _ = case_input(case)
        settings = csvlingua.SETTINGS[lang]
        full_chunks, token_map = csvlingua.prepare_chunks(text, tokenizer, settings["force_tokens"], settings["chunk_end_tokens"])
        int8_chunks, _ = csvlingua.prepare_chunks(text, tokenizer, settings["force_tokens"], settings["chunk_end_tokens"])
        for chunk, p_keep in zip(full_chunks, ours[case]):
            chunk["p_keep"] = p_keep
        csvlingua.compute_p_keep(int8_chunks, int8_model)
        worst_dp = max(worst_dp, max_over_chunks([c["p_keep"] for c in int8_chunks], ours[case]))
        cells = []
        for rate in (0.3, 0.5, 0.6, 0.8):
            csvlingua.apply_rate(full_chunks, token_map, rate, lang)
            csvlingua.apply_rate(int8_chunks, token_map, rate, lang)
            a = [x for c in full_chunks for x in c["labels"]]
            b = [x for c in int8_chunks for x in c["labels"]]
            kept_a = [w for c in full_chunks for w, x in zip(c["words"], c["labels"]) if x]
            kept_b = [w for c in int8_chunks for w, x in zip(c["words"], c["labels"]) if x]
            labels_agree = sum(x == y for x, y in zip(a, b)) / len(a)
            words_agree = difflib.SequenceMatcher(None, kept_a, kept_b, autojunk=False).ratio()
            worst_labels, worst_words = min(worst_labels, labels_agree), min(worst_words, words_agree)
            cells.append(f"{rate}: {100 * labels_agree:.1f}%/{100 * words_agree:.1f}%")
        record(f"int8_agreement_{case}", "; ".join(cells), "rate: word labels / kept-word sequence")
    record("int8_min_label_agreement", f"{100 * worst_labels:.1f}%", "requirement: at least 95%")
    record("int8_min_kept_sequence_similarity", f"{100 * worst_words:.1f}%")
    record("int8_max_abs_dp_keep_vs_full", f"{worst_dp:.2f}")

    print("Command line (fresh process)")
    for name, source, extra in [("meeting", "meeting.txt", []),
                                 ("meeting_zh_hant", "meeting.zh-hant.txt", ["--lang", "zh-hant"]),
                                 ("meeting_int8", "meeting.txt", ["--model", "int8"]),
                                 ("meeting_zh_hant_int8", "meeting.zh-hant.txt", ["--lang", "zh-hant", "--model", "int8"])]:
        output = ROOT / "examples" / f"{Path(source).stem}{'.int8' if 'int8' in extra else ''}.compressed.txt"
        stderr, wall = run_cli(str(ROOT / "examples" / source), "-o", str(output), *extra)
        load = re.search(r"model:\s+([\d.]+) s", stderr).group(1)
        ram = re.search(r"peak RAM (\d+) MB", stderr).group(1)
        record(f"cli_{name}_wall_s", f"{wall:.1f}")
        record(f"cli_{name}_load_s", load)
        record(f"cli_{name}_peak_ram_mb", ram)

    with open(ROOT / "examples" / "results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["metric", "value", "note"])
        writer.writerows(results)
    print("wrote examples/results.csv")


if __name__ == "__main__":
    main()
