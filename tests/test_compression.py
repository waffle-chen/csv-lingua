"""Compression steps vs Microsoft's llmlingua (golden data).

Two views:
  - exact: feed the reference's own p_keep and GPT-3.5 token counts into our code;
    chunks, words, labels and the compressed text must be identical.
  - ours: our CSV model's p_keep and our default weight of 1 per word; the label
    agreement is measured and must stay high.
"""
import unittest
from pathlib import Path

import numpy as np

import csvlingua
from tests import golden_data

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def read_example(name):
    with open(EXAMPLES / name, encoding="utf-8", newline="") as f:
        return f.read()


def case_input(case):
    """(text, settings name, rate) exactly as tools/reference_check.py used them."""
    meeting = read_example("meeting.txt")
    return {
        "model_card": (read_example("model_card.txt"), "default", 0.6),
        "meeting": (meeting, "default", 0.6),
        "meeting_rate033": (meeting, "default", 0.33),
        "no_chunk_end": (meeting.replace(".", ";").replace("\n", " "), "default", 0.6),
        "zh_hant_default": (read_example("meeting.zh-hant.txt"), "default", 0.6),
        "zh_hant": (read_example("meeting.zh-hant.txt"), "zh-hant", 0.6),
    }[case]


def run_case(case, p_keep_per_chunk, word_weight):
    """Our steps 1-6 with the given p_keep; the join is the reference's for comparison."""
    text, settings_name, rate = case_input(case)
    settings = csvlingua.SETTINGS[settings_name]
    chunks, token_map = csvlingua.prepare_chunks(text, golden_data.tokenizer(),
                                                 settings["force_tokens"], settings["chunk_end_tokens"])
    for chunk, p_keep in zip(chunks, p_keep_per_chunk):
        csvlingua.compress_chunk(chunk, p_keep, rate, settings["force_tokens"], token_map,
                                 drop_consecutive=True, lang="default", word_weight=word_weight)
    return chunks


def reference_p_keep(case):
    return [np.array([np.float32(r["p_keep"]) for r in rows])
            for rows in golden_data.by_chunk(golden_data.read_rows(f"{case}.tokens.csv"))]


def tiktoken_counts(case):
    return {r["word"]: int(r["tiktoken_tokens"]) for r in golden_data.read_rows(f"{case}.words.csv")}


def label_agreement(case, chunks):
    reference = [int(r["label"]) for r in golden_data.read_rows(f"{case}.words.csv")]
    ours = [label for chunk in chunks for label in chunk["labels"]]
    assert len(ours) == len(reference)
    return sum(a == b for a, b in zip(ours, reference)) / len(reference), sum(a != b for a, b in zip(ours, reference))


class Chunking(unittest.TestCase):
    def test_chunks_match_reference(self):
        for case in golden_data.COMPRESSION_CASES:
            with self.subTest(case=case):
                text, settings_name, _ = case_input(case)
                settings = csvlingua.SETTINGS[settings_name]
                chunks, _ = csvlingua.prepare_chunks(text, golden_data.tokenizer(),
                                                     settings["force_tokens"], settings["chunk_end_tokens"])
                expected = [r["text"] for r in golden_data.read_rows(f"{case}.chunks.csv")]
                self.assertEqual([c["text"] for c in chunks], expected)

    def test_newline_becomes_new0(self):
        tok = golden_data.tokenizer()
        self.assertEqual(csvlingua.force_token_map(csvlingua.SETTINGS["default"]["force_tokens"], tok), {"\n": "[NEW0]"})


class ExactReferenceSteps(unittest.TestCase):
    """With the reference's p_keep and tiktoken counts, every decision must be identical."""

    def test_words_labels_and_text(self):
        for case in golden_data.COMPRESSION_CASES:
            with self.subTest(case=case):
                counts = tiktoken_counts(case)
                chunks = run_case(case, reference_p_keep(case), word_weight=counts.__getitem__)
                golden_words = golden_data.read_rows(f"{case}.words.csv")
                self.assertEqual([w for c in chunks for w in c["words"]], [r["word"] for r in golden_words])
                agreement, _ = label_agreement(case, chunks)
                self.assertEqual(agreement, 1.0)
                self.assertEqual("".join(c["kept_text"] for c in chunks),
                                 golden_data.read_text(f"{case}.compressed.txt"))

    def test_word_probabilities(self):
        for case in golden_data.COMPRESSION_CASES:
            with self.subTest(case=case):
                text, settings_name, _ = case_input(case)
                settings = csvlingua.SETTINGS[settings_name]
                chunks, token_map = csvlingua.prepare_chunks(text, golden_data.tokenizer(),
                                                             settings["force_tokens"], settings["chunk_end_tokens"])
                ours = []
                for chunk, p_keep in zip(chunks, reference_p_keep(case)):
                    ours += csvlingua.merge_tokens_to_words(chunk["tokens"], list(p_keep),
                                                            settings["force_tokens"], token_map)[1]
                reference = [float(r["p_word"]) for r in golden_data.read_rows(f"{case}.words.csv")]
                self.assertLess(np.abs(np.array(ours, dtype=np.float64) - reference).max(), 1e-7)


@unittest.skipUnless(golden_data.HAS_INT8_MODEL, "model_csv_int8/ missing")
class Int8ModelEndToEnd(unittest.TestCase):
    """The shipped 8-bit model against llmlingua's own keep/drop labels."""

    def test_label_agreement_with_reference(self):
        for case in golden_data.COMPRESSION_CASES:
            with self.subTest(case=case):
                p_keep = golden_data.our_p_keep(case, csvlingua.MODEL_DIR)
                exact_weights = run_case(case, p_keep, tiktoken_counts(case).__getitem__)
                weight_one = run_case(case, p_keep, None)
                self.assertGreaterEqual(label_agreement(case, exact_weights)[0], 0.98)
                self.assertGreaterEqual(label_agreement(case, weight_one)[0], 0.95)


@unittest.skipUnless(golden_data.HAS_FULL_MODEL, "model_csv/ not built (python convert.py)")
class FullModelEndToEnd(unittest.TestCase):
    def test_label_agreement_with_reference(self):
        for case in golden_data.COMPRESSION_CASES:
            with self.subTest(case=case):
                exact_weights = run_case(case, golden_data.our_p_keep(case), tiktoken_counts(case).__getitem__)
                weight_one = run_case(case, golden_data.our_p_keep(case), None)
                self.assertGreaterEqual(label_agreement(case, exact_weights)[0], 0.995)
                self.assertGreaterEqual(label_agreement(case, weight_one)[0], 0.95)


class ChinesePunctuationTidy(unittest.TestCase):
    def tidy(self, words, labels):
        csvlingua.tidy_punctuation(words, labels)
        return "".join(w for w, label in zip(words, labels) if label)

    def test_adjacent_marks_keep_the_strongest(self):
        words = ["方", "便", "。", "但", "，", "抱", "怨", "、", "；", "好"]
        labels = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(self.tidy(words, labels), "方便。抱怨；好")

    def test_mark_at_line_start_is_dropped(self):
        words = ["，", "好", "。", "\n", "：", "嗯", "對"]
        labels = [1, 1, 1, 1, 1, 0, 1]
        self.assertEqual(self.tidy(words, labels), "好。\n對")

    def test_dropped_words_do_not_count(self):
        words = ["好", "，", "的", "，", "對"]
        labels = [1, 1, 0, 0, 1]
        self.assertEqual(self.tidy(words, labels), "好，對")


class ChineseJoin(unittest.TestCase):
    def test_no_spaces_between_chinese_characters(self):
        words = ["主", "持", "人", "：", "好", "。", "\n", "Python", "很", "好", "!", "OK", "?"]
        self.assertEqual(csvlingua.words_to_text(words, "zh-hant"), "主持人：好。\nPython 很好! OK?")
        self.assertEqual(csvlingua.words_to_text(words, "default"), "主 持 人 ： 好 。 \n Python 很 好! OK?")


if __name__ == "__main__":
    unittest.main()
