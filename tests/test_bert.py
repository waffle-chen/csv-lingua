"""BERT forward pass with the CSV weights vs Microsoft's PyTorch model (golden data)."""
import unittest

import numpy as np

import csvlingua
from tests import golden_data

P_KEEP_TOLERANCE = 1e-4


@unittest.skipUnless(golden_data.HAS_FULL_MODEL, "model_csv/ not built (python convert.py)")
class FullModelParity(unittest.TestCase):
    def test_hidden_states_every_layer(self):
        rows = golden_data.read_rows("model_card.tokens.csv")
        states = []
        csvlingua.bert_forward([int(r["token_id"]) for r in rows], golden_data.model(), hidden_states=states)
        for layer, ours in enumerate(states):  # layer 0 is the embedding output
            with self.subTest(layer=layer):
                reference = np.loadtxt(golden_data.GOLDEN / f"model_card.hidden_layer{layer:02d}.csv", delimiter=",")
                self.assertLess(np.abs(ours - reference).max(), 1e-3)

    def test_p_keep_matches_reference_for_every_chunk(self):
        for case in golden_data.COMPRESSION_CASES:
            chunks = golden_data.by_chunk(golden_data.read_rows(f"{case}.tokens.csv"))
            for c, (rows, ours) in enumerate(zip(chunks, golden_data.our_p_keep(case))):
                with self.subTest(case=case, chunk=c):
                    reference = np.array([float(r["p_keep"]) for r in rows])
                    self.assertLess(np.abs(ours - reference).max(), P_KEEP_TOLERANCE)


class ModelFiles(unittest.TestCase):
    def test_only_needed_embedding_shards_are_loaded(self):
        manifest = csvlingua.read_manifest()
        needed = csvlingua.embedding_shards_needed(manifest, [101, 102, 119646])
        shards = manifest[csvlingua.WORD_EMBEDDINGS]
        self.assertEqual(needed, [shards[0], shards[-1]])
        self.assertEqual(sum(e["rows"] for e in shards), 119647)

    def test_missing_shard_is_reported(self):
        model = csvlingua.make_model({"hidden_size": 4, "num_hidden_layers": 0, "num_attention_heads": 1,
                                      "layer_norm_eps": "1e-12"}, {}, [(0, np.zeros((10, 4), np.float32))])
        self.assertEqual(csvlingua.lookup_word_embeddings([9], model).shape, (1, 4))
        with self.assertRaises(KeyError):
            csvlingua.lookup_word_embeddings([10], model)


if __name__ == "__main__":
    unittest.main()
