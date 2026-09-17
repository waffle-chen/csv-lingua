"""The shipped model must keep producing the same output for the examples.

If this fails, an output changed. That is fine when it was meant to: look at the
difference, then run `python tools/snapshot.py` to record the new values.
"""
import csv
import unittest

import csvlingua
from tests import golden_data
from tests.snapshots import SNAPSHOT_FILE, snapshot_rows


@unittest.skipUnless(golden_data.HAS_INT8_MODEL, "model_csv_int8/ missing")
class Snapshots(unittest.TestCase):
    def test_examples_produce_the_recorded_output(self):
        with open(SNAPSHOT_FILE, encoding="utf-8", newline="") as f:
            expected = [list(row.values()) for row in csv.DictReader(f)]
        tokenizer = golden_data.tokenizer()
        model = csvlingua.load_model(csvlingua.MODEL_DIR, token_ids=None)
        ours = [[str(cell) for cell in row] for row in snapshot_rows(tokenizer, model)]
        self.assertEqual(ours, expected)


if __name__ == "__main__":
    unittest.main()
