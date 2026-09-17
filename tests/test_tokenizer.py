"""Tokenizer tests: our plain-Python WordPiece must match Hugging Face token for token."""
import unittest

import csvlingua
from tests import golden_data


class CharacterClasses(unittest.TestCase):
    def test_every_code_point_matches_hugging_face(self):
        hf = {}
        for row in golden_data.read_rows("unicode_classes.csv"):
            for cp in range(int(row["first"], 16), int(row["last"], 16) + 1):
                hf[cp] = row["class"]
        wrong = [
            f"U+{cp:04X}: ours {csvlingua.char_class(chr(cp))}, HF {hf.get(cp, 'other')}"
            for cp in range(0x110000)
            if not 0xD800 <= cp <= 0xDFFF and csvlingua.char_class(chr(cp)) != hf.get(cp, "other")
        ]
        self.assertEqual(wrong, [])


class SmallExamples(unittest.TestCase):
    def setUp(self):
        self.tok = golden_data.tokenizer()

    def test_added_tokens_are_cut_out_first(self):
        self.assertEqual(csvlingua.tokenize("tight.[NEW0]Sarah", self.tok), ["tight", ".", "[NEW0]", "Sarah"])

    def test_new_token_ids_follow_the_wordpiece_vocabulary(self):
        self.assertEqual(csvlingua.tokens_to_ids(["[NEW0]", "[NEW99]", "[CLS]", "[SEP]"], self.tok),
                         [119547, 119646, 101, 102])

    def test_word_longer_than_100_characters_is_unknown(self):
        self.assertEqual(csvlingua.wordpiece("b" * 101, self.tok["vocab"]), ["[UNK]"])
        self.assertNotIn("[UNK]", csvlingua.wordpiece("a" * 100, self.tok["vocab"]))

    def test_punctuation_is_isolated(self):
        self.assertEqual(csvlingua.pre_tokenize(csvlingua.normalize("Hi,you!\tok")), ["Hi", ",", "you", "!", "ok"])

    def test_chinese_characters_become_single_words(self):
        self.assertEqual(csvlingua.pre_tokenize(csvlingua.normalize("我們ok")), ["我", "們", "ok"])

    def test_tokens_to_string(self):
        self.assertEqual(csvlingua.tokens_to_string(["Hello", ",", "wor", "##ld", "!"]), "Hello, world!")
        self.assertEqual(csvlingua.tokens_to_string(["I", "'", "ve"]), "I ' ve")  # cleanup is per token


class GoldenTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = golden_data.tokenizer()

    def test_tricky_texts_match_hugging_face(self):
        expected = {}
        for row in golden_data.read_rows("tokenizer_tokens.csv"):
            expected.setdefault(row["case"], []).append((row["token"], int(row["token_id"])))
        for row in golden_data.read_rows("tokenizer_cases.csv"):
            with self.subTest(case=row["case"]):
                tokens = csvlingua.tokenize(row["text"], self.tok)
                ours = list(zip(tokens, csvlingua.tokens_to_ids(tokens, self.tok)))
                self.assertEqual(ours, expected.get(row["case"], []))

    def test_every_reference_chunk_tokenizes_identically(self):
        for case in golden_data.COMPRESSION_CASES:
            chunks = golden_data.read_rows(f"{case}.chunks.csv")
            token_chunks = golden_data.by_chunk(golden_data.read_rows(f"{case}.tokens.csv"))
            for chunk, rows in zip(chunks, token_chunks):
                with self.subTest(case=case, chunk=chunk["chunk"]):
                    tokens = (["[CLS]"] + csvlingua.tokenize(chunk["text"], self.tok) + ["[SEP]"])[:512]
                    ids = csvlingua.tokens_to_ids(tokens, self.tok)
                    self.assertEqual(list(zip(tokens, ids)), [(r["token"], int(r["token_id"])) for r in rows])


if __name__ == "__main__":
    unittest.main()
