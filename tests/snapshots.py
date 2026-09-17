"""What the shipped model produces for the examples: shared by the test and the tool."""
import hashlib
from pathlib import Path

import csvlingua

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_FILE = ROOT / "tests" / "output_snapshots.csv"
EXAMPLES = ["model_card", "meeting", "meeting.zh-hant", "mixed", "interview"]
RATES = (0.3, 0.6, 0.9)


def snapshot_rows(tokenizer, model):
    """[example, language, rate, kept words, characters, hash of the compressed text]"""
    rows = []
    for name in EXAMPLES:
        with open(ROOT / "examples" / f"{name}.txt", encoding="utf-8", newline="") as f:
            text = f.read()
        plan = csvlingua.plan(text, tokenizer)
        csvlingua.run_model(plan, model)
        for rate in RATES:
            compressed = csvlingua.render(plan, rate)
            kept = sum(sum(chunk["labels"]) for chunk in csvlingua.plan_chunks(plan))
            rows.append([name, plan["lang"], rate, kept, len(compressed),
                         hashlib.sha256(compressed.encode("utf-8")).hexdigest()[:16]])
    return rows
