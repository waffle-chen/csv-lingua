"""The 8-bit model (model_csv_int8/, built by quantize.py) against the full CSV model.

Requirement: the keep/drop decision of at least 95% of the words must match the
full model, for English and Chinese, at several rates.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import csvlingua
import quantize
from tests import golden_data
from tests.test_compression import case_input

INT8_DIR = Path(__file__).resolve().parent.parent / "model_csv_int8"
RATES = [0.3, 0.5, 0.6, 0.8]
MIN_AGREEMENT = 0.95


class Int8Storage(unittest.TestCase):
    def test_row_is_scale_then_integers(self):
        W = np.array([[0.5, -1.27, 0.0], [0.0, 0.0, 0.0]])
        s, q = quantize.quantize_rows(W)
        np.testing.assert_allclose(s, [0.01, 0.0])
        np.testing.assert_array_equal(q, [[50, -127, 0], [0, 0, 0]])
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder, "w.csv"), Path(folder, "q.csv")
            np.savetxt(source, W, fmt="%.7f", delimiter=",")
            quantize.quantize_file(source, target)
            self.assertEqual(target.read_text().splitlines()[0], "0.010000000000,50,-127,0")
            np.testing.assert_allclose(csvlingua.load_csv_matrix(target, "int8"), W, atol=1e-7)


@unittest.skipUnless(golden_data.HAS_FULL_MODEL and golden_data.HAS_INT8_MODEL,
                     "needs model_csv/ (python convert.py) and model_csv_int8/")
class Int8Accuracy(unittest.TestCase):
    def test_labels_agree_with_full_model(self):
        int8 = csvlingua.load_model(INT8_DIR)
        tokenizer = golden_data.tokenizer()
        for case in ["model_card", "meeting", "zh_hant"]:
            text, lang, _ = case_input(case)
            settings = csvlingua.SETTINGS[lang]
            full_chunks, token_map = csvlingua.prepare_chunks(text, tokenizer, settings["force_tokens"],
                                                              settings["chunk_end_tokens"])
            int8_chunks, _ = csvlingua.prepare_chunks(text, tokenizer, settings["force_tokens"],
                                                      settings["chunk_end_tokens"])
            for chunk, p_keep in zip(full_chunks, golden_data.our_p_keep(case)):
                chunk["p_keep"] = p_keep
            csvlingua.compute_p_keep(int8_chunks, int8)
            for rate in RATES:
                with self.subTest(case=case, rate=rate):
                    csvlingua.apply_rate(full_chunks, token_map, rate, lang)
                    csvlingua.apply_rate(int8_chunks, token_map, rate, lang)
                    a = [x for c in full_chunks for x in c["labels"]]
                    b = [x for c in int8_chunks for x in c["labels"]]
                    self.assertGreaterEqual(np.mean(np.array(a) == np.array(b)), MIN_AGREEMENT)


if __name__ == "__main__":
    unittest.main()
