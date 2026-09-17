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
        short = csvlingua.compress_text(text, rate=0.6)
        tokenizer, model = csvlingua.get_tokenizer_and_model()
        expected, _ = csvlingua.compress(text, tokenizer, model, 0.6)
        self.assertEqual(short, expected)
        self.assertLess(len(short), len(text))

    def test_rate_one_keeps_the_text(self):
        self.assertEqual(csvlingua.compress_text("Hello, world.", rate=1.0), "Hello, world.")

    def test_compress_file_writes_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out.txt"
            text = csvlingua.compress_file(EXAMPLES / "model_card.txt", output, rate=0.5)
            self.assertEqual(output.read_text(encoding="utf-8"), text)

    def test_chinese(self):
        short = csvlingua.compress_text("主持人：好，我們開始吧。今天的議程有三個部分。", lang="zh-hant")
        self.assertNotIn(" ", short)

    def test_auto_language_picks_chinese_rules(self):
        self.assertEqual(csvlingua.choose_language("only English here."), "default")
        self.assertEqual(csvlingua.choose_language("mixed English and 中文"), "zh-hant")
        self.assertEqual(csvlingua.choose_language("中文", lang="default"), "default")

    def test_code_is_kept_exactly(self):
        code = '```python\nfor i in range(10):\n    print("hello, world", i)\n```'
        text = f"Here is, um, the example we talked about in the meeting:\n{code}\nPlease run it later.\n"
        short = csvlingua.compress_text(text, rate=0.5)
        self.assertIn(code, short)
        self.assertLess(len(short), len(text))

    def test_inline_code_is_kept(self):
        text = "You should really call `compress_text(x, rate=0.6)` before you send the prompt, I think."
        self.assertIn("`compress_text(x, rate=0.6)`", csvlingua.compress_text(text, rate=0.4))

    def test_code_can_be_compressed_too(self):
        text = "Some words here.\n```\nfor i in range(10):\n    print(i)\n```\nAnd more words here.\n"
        self.assertNotIn("```", csvlingua.compress_text(text, rate=0.3, protect_code=False))


if __name__ == "__main__":
    unittest.main()
