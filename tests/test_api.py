"""The one-call API: compress_text() and compress_file()."""
import tempfile
import unittest
from pathlib import Path

import csvlingua
from tests import golden_data

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@unittest.skipUnless(golden_data.HAS_INT8_MODEL, "model_csv_int8/ missing")
class OneCallApi(unittest.TestCase):
    def test_compress_text_matches_the_step_by_step_pipeline(self):
        text = (EXAMPLES / "model_card.txt").read_text(encoding="utf-8")
        short = csvlingua.compress_text(text, rate=0.6, workers=None)
        tokenizer, model = csvlingua.get_tokenizer_and_model()
        expected, _ = csvlingua.compress(text, tokenizer, model, 0.6)
        self.assertEqual(short, expected)
        self.assertLess(len(short), len(text))

    def test_rate_one_keeps_the_text(self):
        self.assertEqual(csvlingua.compress_text("Hello, world.", rate=1.0, workers=None), "Hello, world.")

    def test_compress_file_writes_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out.txt"
            text = csvlingua.compress_file(EXAMPLES / "model_card.txt", output, rate=0.5, workers=None)
            self.assertEqual(output.read_text(encoding="utf-8"), text)

    def test_chinese(self):
        short = csvlingua.compress_text("主持人：好，我們開始吧。今天的議程有三個部分。", lang="zh-hant", workers=None)
        self.assertNotIn(" ", short)


if __name__ == "__main__":
    unittest.main()
