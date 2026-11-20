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


class SpreadingTests(unittest.TestCase):
    def test_fastchi2_spreading_modes_match(self):
        t, y, dy = _make_signal(M=1024, seed=1)
        for mod, rtol, atol in ((tlsf, 2e-4, 2e-5), (tls, 1e-10, 1e-11)):
            for backend in ("pswf43", "pswf21"):
                for nterms in (1, 3):
                    p_mm = mod.power(512, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, spreading="minmax")
                    p_pp = mod.power(512, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, spreading="piecewise")
                    p_auto = mod.power(512, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, spreading="auto")
                    p_upper = mod.power(512, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, spreading=" PIECEWISE ")
                    np.testing.assert_allclose(p_pp, p_mm, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(p_auto, p_mm, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(p_upper, p_pp, rtol=0.0, atol=0.0)

                    f_mm, ap_mm = mod.autopower(t, y, dy, fmax=2.0, backend=backend, nterms=nterms, spreading="minmax")
                    f_pp, ap_pp = mod.autopower(t, y, dy, fmax=2.0, backend=backend, nterms=nterms, spreading="piecewise")
                    np.testing.assert_array_equal(f_mm, f_pp)
                    np.testing.assert_allclose(ap_pp, ap_mm, rtol=rtol, atol=atol)

    def test_ihs_spreading_modes_match(self):
        t, y, dy = _make_signal(M=1024, seed=2)
        for mod, rtol, atol in ((ihsf, 2e-4, 2e-5), (ihs, 1e-10, 1e-11)):
            for backend in ("pswf43", "pswf21"):
                for nterms in (1, 3):
                    p_mm = mod.power(512, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, normalization="nll", spreading="minmax")
                    p_pp = mod.power(512, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, normalization="nll", spreading="piecewise")
                    p_auto = mod.power(512, 0.002, 0.01, t, y, dy, backend=backend, nterms=nterms, normalization="nll", spreading="auto")
                    np.testing.assert_allclose(p_pp, p_mm, rtol=rtol, atol=atol)
                    np.testing.assert_allclose(p_auto, p_mm, rtol=rtol, atol=atol)

                    f_mm, ap_mm = mod.autopower(t, y, dy, fmax=2.0, backend=backend, nterms=nterms, spreading="minmax")
                    f_pp, ap_pp = mod.autopower(t, y, dy, fmax=2.0, backend=backend, nterms=nterms, spreading="piecewise")
                    np.testing.assert_array_equal(f_mm, f_pp)
                    np.testing.assert_allclose(ap_pp, ap_mm, rtol=rtol, atol=atol)

    def test_large_m_moderate_n_regime(self):
        t, y, dy = _make_signal(M=8192, seed=3)
        for mod, rtol, atol in ((tlsf, 2e-4, 2e-5), (tls, 1e-10, 1e-11)):
            p_mm = mod.power(1024, 0.002, 0.01, t, y, dy, backend="pswf43", nterms=3, spreading="minmax")
            p_pp = mod.power(1024, 0.002, 0.01, t, y, dy, backend="pswf43", nterms=3, spreading="piecewise")
            p_auto = mod.power(1024, 0.002, 0.01, t, y, dy, backend="pswf43", nterms=3, spreading="auto")
            np.testing.assert_allclose(p_pp, p_mm, rtol=rtol, atol=atol)
            np.testing.assert_allclose(p_auto, p_mm, rtol=rtol, atol=atol)

    def test_spreading_validation_and_double_double(self):
        t, y, dy = _make_signal(M=64, seed=4)
        for mod in (tlsf, tls, ihsf, ihs):
            with self.assertRaises(ValueError):
                mod.power(32, 0.01, 0.01, t, y, dy, spreading="invalid")
            with self.assertRaises(ValueError):
                mod.autopower(t, y, dy, fmax=1.0, spreading="invalid")

        for mod in (tlsdd, ihsdd):
            with self.assertRaises(ValueError):
                mod.power(32, 0.01, 0.01, t, y, dy, spreading="piecewise")
            with self.assertRaises(ValueError):
                mod.autopower(t, y, dy, fmax=1.0, spreading="piecewise")
            with self.assertRaises(ValueError):
                mod.power(32, 0.01, 0.01, t, y, dy, spreading="invalid")

        p_auto = tlsdd.power(32, 0.01, 0.01, t, y, dy, nterms=1, spreading="auto")
        p_mm = tlsdd.power(32, 0.01, 0.01, t, y, dy, nterms=1, spreading="minmax")
        self.assertEqual(len(p_auto), 32)
        for a, b in zip(p_auto, p_mm):
            self.assertAlmostEqual(float(a), float(b), places=14)

        p_ihs_auto = ihsdd.power(32, 0.01, 0.01, t, y, dy, nterms=1, spreading="auto")
        p_ihs_mm = ihsdd.power(32, 0.01, 0.01, t, y, dy, nterms=1, spreading="minmax")
        for a, b in zip(p_ihs_auto, p_ihs_mm):
            self.assertAlmostEqual(float(a), float(b), places=14)

    def test_concurrent_mixed_spreading_calls(self):
        t, y, dy = _make_signal(M=2048, seed=5)
        ref = tls.power(512, 0.002, 0.01, t, y, dy, backend="pswf43", nterms=3, spreading="minmax")
        modes = ["minmax", "piecewise", "auto"] * 4

        def _run(mode):
            return tls.power(512, 0.002, 0.01, t, y, dy, backend="pswf43", nterms=3, spreading=mode)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_run, modes))
        for res in results:
            np.testing.assert_allclose(res, ref, rtol=1e-10, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
