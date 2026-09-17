"""The building blocks on small, hand-computed inputs."""
import math
import unittest

import numpy as np

import csvlingua as cl


class Math(unittest.TestCase):
    def test_linear(self):
        x = np.array([[1, 2]], dtype=np.float32)          # n=1, d_in=2
        W = np.array([[1, 0], [0, 1], [1, 1]], np.float32)  # d_out=3
        b = np.array([0.5, 0, -1], np.float32)
        np.testing.assert_allclose(cl.linear(x, W, b), [[1.5, 2, 2]])

    def test_layer_norm(self):
        x = np.array([[1, 2, 3, 4]], np.float32)  # mean 2.5, var 1.25
        gamma = np.array([1, 1, 2, 2], np.float32)
        beta = np.array([0, 0, 0, 1], np.float32)
        s = math.sqrt(1.25)
        np.testing.assert_allclose(cl.layer_norm(x, gamma, beta, np.float32(1e-12)),
                                   [[-1.5 / s, -0.5 / s, 2 * 0.5 / s, 2 * 1.5 / s + 1]], rtol=1e-6)

    def test_gelu_known_values(self):
        x = np.array([-3, -1, 0, 1, 3], np.float32)
        expected = [-0.00404951, -0.15865525, 0.0, 0.84134475, 2.99595049]  # 0.5·x·(1+erf(x/√2))
        np.testing.assert_allclose(cl.gelu(x), expected, atol=1e-6)
        self.assertEqual(cl.gelu(x).dtype, np.float32)

    def test_abramowitz_stegun_error_bound(self):
        x = np.linspace(-6, 6, 100001)
        error = np.abs(cl.erf_abramowitz_stegun(x) - cl.erf_math(x))
        self.assertLessEqual(error.max(), 1.5e-7)

    def test_vectorized_erf_equals_math_erf_in_float32(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0, 3, 1_000_000), np.linspace(-7, 7, 400_001),
                            [0.0, -0.0, 0.84375, 1.25, cl.ERF_ONE_OVER_035, 6.0, -6.0, 1e-30, 40.0]])
        x = x.astype(np.float32) / np.float32(math.sqrt(2))
        ours, reference = cl.erf_exact(x), cl.erf_math(x)
        self.assertLessEqual(np.abs(ours - reference).max(), 1e-15)
        np.testing.assert_array_equal(ours.astype(np.float32), reference.astype(np.float32))

    def test_gelu_on_all_cores_is_identical(self):
        x = np.random.default_rng(1).normal(0, 2, (300, 3072)).astype(np.float32)
        single = 0.5 * x * (1 + cl.erf_math(x / np.float32(math.sqrt(2))).astype(np.float32))
        np.testing.assert_array_equal(cl.gelu(x), single)

    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]], np.float32)
        e = np.exp([1.0, 2.0, 3.0])
        np.testing.assert_allclose(cl.softmax(x), [e / e.sum(), [1 / 3] * 3], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
