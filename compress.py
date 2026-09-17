"""compress.py - compress a text file with LLMLingua-2, using the CSV model.

    python compress.py examples/meeting.txt -o examples/meeting.compressed.txt --rate 0.6
    python compress.py examples/meeting.txt --trace examples/meeting.trace.csv
    python compress.py examples/meeting.zh-hant.txt --lang zh-hant

Without -o the compressed text goes to stdout. Statistics always go to stderr.
--trace writes every decision as CSV: chunk,position,token,token_id,word,p_keep,kept
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import ctypes
import time
from pathlib import Path

import numpy as np

import csvlingua

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "tests" / "golden"
MODELS = {"int8": csvlingua.MODEL_DIR, "full": csvlingua.FULL_MODEL_DIR}
# Inputs for which tests/golden/ holds Microsoft's own results: (file name, lang, rate) -> case
GOLDEN_CASES = {
    ("model_card.txt", "default", 0.6): "model_card",
    ("meeting.txt", "default", 0.6): "meeting",
    ("meeting.txt", "default", 0.33): "meeting_rate033",
    ("meeting.zh-hant.txt", "default", 0.6): "zh_hant_default",
    ("meeting.zh-hant.txt", "zh-hant", 0.6): "zh_hant",
}


def log(message=""):
    print(message, file=sys.stderr, flush=True)


def peak_memory_mb():
    """Peak resident memory of this process in MB (standard library only)."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters(cb=ctypes.sizeof(Counters))
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        info(process(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1e6
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes on macOS, KB on Linux


def read_golden(name):
    with open(GOLDEN / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def reference_parity(input_path, lang, rate, chunks):
    """Compare this run with Microsoft's results in tests/golden/, when they exist for this input."""
    case = GOLDEN_CASES.get((Path(input_path).name, lang, rate))
    if case is None or not (GOLDEN / f"{case}.tokens.csv").exists():
        return "n/a (no reference data for this input, language and rate)"
    if [c["text"] for c in chunks] != [r["text"] for r in read_golden(f"{case}.chunks.csv")]:
        return f"n/a (input differs from the text behind tests/golden/{case})"
    reference_p = [float(r["p_keep"]) for r in read_golden(f"{case}.tokens.csv")]
    our_p = np.concatenate([c["p_keep"] for c in chunks])
    reference_labels = [int(r["label"]) for r in read_golden(f"{case}.words.csv")]
    our_labels = [label for c in chunks for label in c["labels"]]
    same = sum(a == b for a, b in zip(our_labels, reference_labels))
    return (f"max |p_keep - reference| = {np.abs(our_p - reference_p).max():.1e} over {len(our_p)} tokens; "
            f"word labels agree {same}/{len(reference_labels)} ({100 * same / len(reference_labels):.1f}%)")


def write_trace(path, chunks):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["chunk", "position", "token", "token_id", "word", "p_keep", "kept"])
        for c, chunk in enumerate(chunks):
            for i, (token, token_id, p) in enumerate(zip(chunk["tokens"], chunk["ids"], chunk["p_keep"])):
                w = chunk["token_word"][i]
                word, kept = ("", "") if w < 0 else (chunk["words"][w], chunk["labels"][w])
                writer.writerow([c, i, token, token_id, word, f"{p:.6f}", kept])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="text file to compress (UTF-8)")
    parser.add_argument("-o", "--output", help="where to write the compressed text (default: stdout)")
    parser.add_argument("--rate", type=float, default=0.6, help="share of words to keep (default 0.6)")
    parser.add_argument("--model", choices=sorted(MODELS), default="int8",
                        help="'int8' = model_csv_int8/ (default, shipped); 'full' = model_csv/ (build it with convert.py)")
    parser.add_argument("--lang", choices=sorted(csvlingua.SETTINGS), default="default",
                        help="'default' = model card settings; 'zh-hant' = Traditional Chinese punctuation and joining")
    parser.add_argument("--trace", help="write a CSV with every token's p_keep and keep decision")
    parser.add_argument("--fast-gelu", action="store_true",
                        help="use the Abramowitz-Stegun erf approximation instead of exact math.erf")
    parser.add_argument("--no-drop-consecutive", dest="drop_consecutive", action="store_false",
                        help="keep repeated punctuation (the model card uses drop_consecutive=True)")
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate must be above 0")

    with open(args.input, encoding="utf-8", newline="") as f:
        text = f.read()

    t0 = time.perf_counter()
    tokenizer = csvlingua.load_tokenizer()
    settings = csvlingua.SETTINGS[args.lang]
    chunks, _ = csvlingua.prepare_chunks(text, tokenizer, settings["force_tokens"], settings["chunk_end_tokens"])
    t1 = time.perf_counter()
    log(f"tokenizer: {t1 - t0:.1f} s, {sum(len(c['tokens']) for c in chunks)} tokens in {len(chunks)} chunks")

    model = None
    if args.rate < 1 and chunks:
        if not (MODELS[args.model] / "manifest.csv").exists():
            parser.error(f"{MODELS[args.model].name}/ not found (for --model full run: python convert.py)")
        model = csvlingua.load_model(MODELS[args.model], token_ids=[i for c in chunks for i in c["ids"]])
        log(f"model:     {time.perf_counter() - t1:.1f} s to load CSVs "
            f"({model['embedding_rows_loaded']} word-embedding rows)")

    def on_chunk(i, chunk):
        now = time.perf_counter()
        log(f"chunk {i + 1}/{len(chunks)}: {len(chunk['tokens'])} tokens, BERT {now - on_chunk.last:.2f} s")
        on_chunk.last = now

    on_chunk.last = time.perf_counter()
    erf = csvlingua.erf_abramowitz_stegun if args.fast_gelu else csvlingua.erf_exact
    compressed, chunks = csvlingua.compress(text, tokenizer, model, args.rate, args.lang,
                                            args.drop_consecutive, erf, on_chunk=on_chunk)

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(compressed)
    else:
        sys.stdout.reconfigure(encoding="utf-8", newline="")
        sys.stdout.write(compressed)

    if model is not None:
        words = sum(len(c["words"]) for c in chunks)
        kept = sum(sum(c["labels"]) for c in chunks)
        log(f"words:     {words} -> {kept} ({100 * kept / max(words, 1):.1f}% kept, target rate {args.rate})")
        log(f"chars:     {len(text)} -> {len(compressed)}")
        if args.trace:
            write_trace(args.trace, chunks)
            log(f"trace:     {args.trace}")
        parity = (reference_parity(args.input, args.lang, args.rate, chunks) if args.drop_consecutive
                  else "n/a (the reference data uses drop_consecutive)")
        log(f"parity:    {parity}")
    log(f"total:     {time.perf_counter() - t0:.1f} s, peak RAM {peak_memory_mb():.0f} MB")


if __name__ == "__main__":
    main()
