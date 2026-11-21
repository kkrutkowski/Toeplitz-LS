import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from toeplitz_ls import ihs, ihsdd, ihsf, tls, tlsdd, tlsf


def _make_signal(M=1024, seed=42):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 200.0, size=M))
    y = 1.2 + 0.8 * np.sin(2.0 * np.pi * 0.37 * t) + 0.35 * np.cos(2.0 * np.pi * 0.74 * t) + rng.normal(0.0, 0.05, size=M)
    dy = rng.uniform(0.08, 0.15, size=M)
    return t, y, dy


class VectorizeTests(unittest.TestCase):
    def test_fastchi2_vectorize_modes_match(self):
        t, y, dy = _make_signal(M=1024, seed=11)
        for mod, rtol, atol in ((tlsf, 3e-4, 3e-5), (tls, 1e-10, 1e-11)):
            for backend in ("pswf43", "pswf21", "lra"):
                for nterms in (1, 3):
                    p_depth = mod.power(777, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, vectorize="depth")
                    p_breadth = mod.power(777, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, vectorize="breadth")
                    p_auto = mod.power(777, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, vectorize="auto")
                    p_upper = mod.power(777, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, vectorize=" BREADTH ")
                    np.testing.assert_allclose(p_breadth, p_depth, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(p_auto, p_depth, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(p_upper, p_breadth, rtol=0.0, atol=0.0)

                    f_d, ap_d = mod.autopower(t, y, dy, fmax=2.0, backend=backend, nterms=nterms, vectorize="depth")
                    f_b, ap_b = mod.autopower(t, y, dy, fmax=2.0, backend=backend, nterms=nterms, vectorize="breadth")
                    np.testing.assert_array_equal(f_d, f_b)
                    np.testing.assert_allclose(ap_b, ap_d, rtol=rtol, atol=atol)

    def test_fastchi2_breadth_with_piecewise_and_solvers(self):
        t, y, dy = _make_signal(M=512, seed=12)
        for solver in ("levinson", "zohar", "bareiss", "ldlt", "svd"):
            p_d, c_d = tls.power(333, 0.003, 0.01, t, y, dy, backend="pswf43", solver=solver, nterms=2, spreading="piecewise", vectorize="depth", autonan=False)
            p_b, c_b = tls.power(
                333, 0.003, 0.01, t, y, dy, backend="pswf43", solver=solver, nterms=2, spreading="piecewise", vectorize="breadth", autonan=False
            )
            np.testing.assert_allclose(p_b, p_d, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(c_b, c_d, rtol=1e-7, atol=1e-8)

    def test_ihs_vectorize_modes_match(self):
        t, y, dy = _make_signal(M=1024, seed=13)
        for mod, rtol, atol in ((ihsf, 3e-4, 3e-5), (ihs, 1e-10, 1e-11)):
            for backend in ("pswf43", "pswf21", "lra"):
                for nterms in (1, 3):
                    for norm in ("raw", "nll", "default"):
                        p_depth = mod.power(777, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, normalization=norm, vectorize="depth")
                        p_breadth = mod.power(777, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, normalization=norm, vectorize="breadth")
                        p_auto = mod.power(777, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, normalization=norm, vectorize="auto")
                        np.testing.assert_allclose(p_breadth, p_depth, rtol=rtol, atol=atol)
                        np.testing.assert_allclose(p_auto, p_depth, rtol=rtol, atol=atol)

    def test_vectorize_validation_and_double_double(self):
        t, y, dy = _make_signal(M=64, seed=14)
        for mod in (tlsf, tls, ihsf, ihs):
            with self.assertRaises(ValueError):
                mod.power(32, 0.01, 0.01, t, y, dy, vectorize="invalid")
            with self.assertRaises(ValueError):
                mod.autopower(t, y, dy, fmax=1.0, vectorize="invalid")
            with self.assertRaises(ValueError):
                mod.power(32, 0.01, 0.01, t, y, dy, vectorize=True)

        for mod in (tlsdd, ihsdd):
            for bad in ("depth", " DEPTH ", 1, "invalid", 99):
                with self.assertRaises(ValueError):
                    mod.power(32, 0.01, 0.01, t, y, dy, vectorize=bad)
                with self.assertRaises(ValueError):
                    mod.autopower(t, y, dy, fmax=1.0, vectorize=bad)

        for backend in ("pswf43", "pswf21", "lra"):
            p_auto = tlsdd.power(48, 0.01, 0.01, t, y, dy, backend=backend, nterms=2, vectorize="auto")
            p_breadth = tlsdd.power(48, 0.01, 0.01, t, y, dy, backend=backend, nterms=2, vectorize="breadth")
            p_ref = tls.power(48, 0.01, 0.01, t, y, dy, backend=backend, nterms=2, vectorize="depth")
            self.assertEqual(len(p_auto), 48)
            for a, b, r in zip(p_auto, p_breadth, p_ref):
                self.assertAlmostEqual(float(a), float(b), places=15)
                self.assertAlmostEqual(float(a), float(r), places=11)

        p_ihs_auto = ihsdd.power(48, 0.01, 0.01, t, y, dy, nterms=2, vectorize="auto")
        p_ihs_b = ihsdd.power(48, 0.01, 0.01, t, y, dy, nterms=2, vectorize="breadth")
        p_ihs_ref = ihs.power(48, 0.01, 0.01, t, y, dy, nterms=2, vectorize="depth")
        for a, b, r in zip(p_ihs_auto, p_ihs_b, p_ihs_ref):
            self.assertAlmostEqual(float(a), float(b), places=15)
            self.assertAlmostEqual(float(a), float(r), places=11)

    def test_concurrent_mixed_vectorize_calls(self):
        t, y, dy = _make_signal(M=2048, seed=15)
        ref = tls.power(600, 0.002, 0.01, t, y, dy, backend="pswf43", nterms=3, vectorize="depth")
        modes = [("depth", "minmax"), ("breadth", "minmax"), ("breadth", "piecewise"), ("auto", "auto")] * 3

        def _run(pair):
            vmode, smode = pair
            return tls.power(600, 0.002, 0.01, t, y, dy, backend="pswf43", nterms=3, spreading=smode, vectorize=vmode)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_run, modes))
        for res in results:
            np.testing.assert_allclose(res, ref, rtol=1e-10, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
