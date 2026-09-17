"""tools/walkthrough.py - show what happens to one sentence, step by step.

For demos and for reading along with csvlingua.py: every step prints the real
numbers it produced, in Markdown.

    python tools/walkthrough.py
    python tools/walkthrough.py "your own sentence here" --rate 0.5
    python tools/walkthrough.py > examples/walkthrough.md
"""
import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import csvlingua  # noqa: E402

DEFAULT_TEXT = ("John: So, um, I've been thinking about the project, you know, "
                "and I believe we need to make some changes.")


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    print()


def show(text, rate, model_dir):
    print("# One sentence, step by step\n")
    print(f"Input ({len(text)} characters):\n\n> {text}\n")

    print("## 1. Plan: language, protected characters, chunks\n")
    tokenizer = csvlingua.load_tokenizer(model_dir)
    compression_plan = csvlingua.plan(text, tokenizer)
    chunks = csvlingua.plan_chunks(compression_plan)
    settings = csvlingua.SETTINGS[compression_plan["lang"]]
    token_map = compression_plan["parts"][0]["token_map"]
    print(f"- language rules: `{compression_plan['lang']}`")
    print(f"- protected characters: {' '.join(repr(t) for t in settings['force_tokens'])}")
    print(f"- replaced before tokenizing: {token_map or 'nothing'} (a newline is not a WordPiece token)")
    print(f"- code parts kept as they are: {sum(1 for p in compression_plan['parts'] if 'code' in p)}")
    print(f"- chunks: {len(chunks)} (at most 512 tokens each)\n")

    print("## 2. Tokenizer\n")
    chunk = chunks[0]
    print(f"`[CLS]` and `[SEP]` are added, so this chunk has {len(chunk['tokens'])} tokens.\n")
    table(["position", "token", "token id"],
          [(i, f"`{t}`", i_) for i, (t, i_) in enumerate(zip(chunk["tokens"][:12], chunk["ids"][:12]))] + [("…", "…", "…")])

    print("## 3. The CSV model\n")
    model = csvlingua.load_model(model_dir, token_ids=compression_plan["token_ids"])
    manifest = csvlingua.read_manifest(model_dir)
    entry = manifest[csvlingua.WORD_EMBEDDINGS][0]
    vector = csvlingua.lookup_word_embeddings([chunk["ids"][1]], model)[0]
    print(f"- {sum(len(v) for v in manifest.values())} CSV files; only the "
          f"{model['embedding_rows_loaded']} word-embedding rows this text needs were read")
    print(f"- `{entry['file']}` stores row = token id, so token {chunk['ids'][1]} "
          f"(`{chunk['tokens'][1]}`) is one line of that file")
    print(f"- its first numbers: {np.array2string(vector[:6], precision=5, separator=', ')} … (768 in total)\n")

    print("## 4. BERT\n")
    states = []
    p_keep = chunk["p_keep"] = csvlingua.bert_forward(chunk["ids"], model, hidden_states=states)
    for other in chunks[1:]:
        other["p_keep"] = csvlingua.bert_forward(other["ids"], model)
    print(f"- embeddings: {states[0].shape[0]} tokens × {states[0].shape[1]} numbers")
    print(f"- 12 layers, each: attention over {len(chunk['ids'])}×{len(chunk['ids'])} token pairs "
          f"per head (12 heads of 64), then 768 → 3072 → 768")
    print(f"- hidden state after layer 1, first numbers: "
          f"{np.array2string(states[1][0][:4], precision=4, separator=', ')}")
    print(f"- hidden state after layer 12, first numbers: "
          f"{np.array2string(states[12][0][:4], precision=4, separator=', ')}")
    print(f"- classifier: 768 → 2 → softmax → keep-probability per token\n")
    table(["token", "p_keep"],
          [(f"`{t}`", f"{p:.4f}") for t, p in list(zip(chunk["tokens"], p_keep))[:12]] + [("…", "…")])

    print("## 5. Tokens become words\n")
    csvlingua.render(compression_plan, rate)
    print("`##` pieces join the word before them, a word's probability is the mean of its "
          "tokens, and protected punctuation gets 1.0.\n")
    table(["word", "p_word", "kept"],
          [(f"`{w}`".replace("\n", "\\n"), f"{p:.4f}", "yes" if k else "no")
           for w, p, k in list(zip(chunk["words"], chunk["word_probs"], chunk["labels"]))[:14]] + [("…", "…", "…")])

    print("## 6. Threshold\n")
    reduce_rate = max(0, 1 - rate)
    print(f"- rate {rate} → q = int(100·(1 − {rate}) + 1) = {int(100 * reduce_rate + 1)}")
    print(f"- the {int(100 * reduce_rate + 1)}-th percentile of this chunk's word probabilities "
          f"is **{chunk['threshold']:.4f}**")
    print(f"- {sum(chunk['labels'])} of {len(chunk['words'])} words are above it\n")

    print("## 7. Result\n")
    compressed = csvlingua.render(compression_plan, rate)
    print(f"> {compressed.strip()}\n")
    print(f"{len(text)} characters → {len(compressed)} characters "
          f"({100 * len(compressed) / max(len(text), 1):.0f}%).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("--rate", type=float, default=0.6)
    parser.add_argument("--model-dir", default=csvlingua.MODEL_DIR)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    show(args.text, args.rate, Path(args.model_dir))


if __name__ == "__main__":
    main()
