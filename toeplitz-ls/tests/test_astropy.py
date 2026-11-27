"""Integration tests for the Astropy ``LombScargle`` backend."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

astropy = pytest.importorskip("astropy")
from astropy import units as u
from astropy.timeseries import LombScargle
from astropy.timeseries.periodograms.lombscargle import implementations as astropy_ls_impl

import toeplitz_ls  # noqa: F401
from toeplitz_ls import tls, tlsf, tlsh


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    parts = []
    for token in version_str.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


_ASTROPY_VERSION = _parse_version_tuple(astropy.__version__)
_REQUIRES_ASTROPY_8 = pytest.mark.skipif(
    _ASTROPY_VERSION < (8, 0),
    reason=f"Astropy >= 8.0 required for fast/fastchi2 LRA comparison (found {astropy.__version__})",
)


def _make_signal(M=250, seed=123):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 35.0, size=M)).astype(np.float64)
    f_true = 0.43
    y = (
        1.4 * np.sin(2.0 * np.pi * f_true * t + 0.2)
        + 0.6 * np.cos(2.0 * np.pi * 2.0 * f_true * t - 0.5)
        + 0.35 * np.sin(2.0 * np.pi * 3.0 * f_true * t + 0.1)
        + 0.25 * rng.normal(size=M)
    ).astype(np.float64)
    dy = rng.uniform(0.15, 0.35, size=M).astype(np.float64)
    return t, y, dy, f_true


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PACKAGE_ROOT}{os.pathsep}{existing}" if existing else str(PACKAGE_ROOT)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)


class TestAstropyRegistration:
    def test_methods_advertised(self):
        for name in ("tls", "tlsf", "tlsh"):
            assert name in astropy_ls_impl.available_methods()
            assert name in LombScargle.available_methods
        for forbidden in ("tlsdd", "ihs", "ihsf", "ihsh", "ihsdd"):
            assert forbidden not in astropy_ls_impl.available_methods()
            assert forbidden not in LombScargle.available_methods

    def test_import_toeplitz_ls_does_not_import_astropy(self):
        proc = _run_python("import sys, toeplitz_ls; assert 'astropy' not in sys.modules")
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize("order", ["tls_first", "astropy_first"])
    def test_import_order_both_work(self, order):
        if order == "tls_first":
            imports = "import toeplitz_ls\nfrom astropy.timeseries import LombScargle"
        else:
            imports = "from astropy.timeseries import LombScargle\nimport toeplitz_ls"
        code = (
            f"{imports}\n"
            "import numpy as np\n"
            "rng = np.random.default_rng(0)\n"
            "t = np.sort(rng.uniform(0, 20, 80))\n"
            "y = np.sin(2 * np.pi * 0.5 * t) + 0.2 * rng.normal(size=80)\n"
            "for m in ('tls', 'tlsf', 'tlsh'):\n"
            "    assert m in LombScargle.available_methods\n"
            "    freq, power = LombScargle(t, y, nterms=2).autopower(method=m, maximum_frequency=2.0)\n"
            "    assert power.dtype == np.float64\n"
            "    assert freq.shape == power.shape\n"
            "    assert np.all(np.isfinite(power))\n"
        )
        proc = _run_python(code)
        assert proc.returncode == 0, proc.stderr


class TestAstropyInterfaceAndEdgeCases:
    @pytest.mark.parametrize("method", ["tls", "tlsf", "tlsh"])
    def test_dtypes_and_quantity_units(self, method):
        t, y, dy, _ = _make_signal(M=120, seed=7)

        # Plain float32 inputs should still produce float64 outputs.
        freq, power = LombScargle(t.astype(np.float32), y.astype(np.float32), dy.astype(np.float32)).autopower(
            method=method,
            minimum_frequency=0.05,
            maximum_frequency=1.5,
        )
        assert power.dtype == np.float64
        assert freq.shape == power.shape
        assert np.all(np.isfinite(power))

        # Astropy Quantity inputs preserve frequency and power units.
        freq_q, power_q = LombScargle(t * u.day, y * u.mag, dy * u.mag, nterms=2).autopower(
            method=method,
            minimum_frequency=0.05 / u.day,
            maximum_frequency=1.5 / u.day,
        )
        assert isinstance(freq_q, u.Quantity)
        assert isinstance(power_q, u.Quantity)
        assert freq_q.unit == 1 / u.day
        assert power_q.unit == u.dimensionless_unscaled
        assert power_q.value.dtype == np.float64

        # PSD without dy carries y.unit ** 2.
        _, power_psd = LombScargle(t * u.day, y * u.mag, normalization="psd").autopower(
            method=method,
            minimum_frequency=0.05 / u.day,
            maximum_frequency=1.5 / u.day,
        )
        assert power_psd.unit == u.mag**2
        assert power_psd.value.dtype == np.float64

    @pytest.mark.parametrize("method,mod", [("tls", tls), ("tlsf", tlsf), ("tlsh", tlsh)])
    @pytest.mark.parametrize("normalization", ["standard", "model", "log", "psd", "anova", "nll"])
    def test_nan_bins_replaced_with_zero(self, method, mod, normalization):
        # f=0.0 is strictly singular (sin(0) = 0), so native tls/tlsf/tlsh report NaN at bin 0.
        t, y, dy, _ = _make_signal(M=128, seed=13)
        freqs = 0.0 + 0.01 * np.arange(32, dtype=np.float64)

        native = mod.power(32, 0.01, 0.0, t, y, dy, nterms=2, normalization="standard", autonan=True)
        assert np.isnan(native[0])

        power = LombScargle(t, y, dy, nterms=2, normalization=normalization).power(freqs, method=method)
        assert power.dtype == np.float64
        assert np.all(np.isfinite(power))
        assert power[0] == 0.0

    @pytest.mark.parametrize("method", ["tls", "tlsf", "tlsh"])
    def test_irregular_grid_rejected(self, method):
        t, y, dy, _ = _make_signal(M=100, seed=11)
        irregular = np.array([0.1, 0.2, 0.35, 0.5, 0.8])
        with pytest.raises(ValueError, match="regular grid"):
            LombScargle(t, y, dy).power(irregular, method=method)

    @pytest.mark.parametrize("method", ["tls", "tlsf", "tlsh"])
    def test_small_regular_grids_padded_and_sliced(self, method):
        t, y, dy, _ = _make_signal(M=120, seed=19)
        ls = LombScargle(t, y, dy, nterms=2)

        for freqs in (
            np.array([0.43]),
            np.array([0.0]),
            np.linspace(0.2, 0.6, 5),
        ):
            power = ls.power(freqs, method=method)
            assert power.shape == freqs.shape
            assert power.dtype == np.float64
            assert np.all(np.isfinite(power))

        # 2D frequency grid whose flattened form is regular preserves shape.
        freqs_2d = (0.1 + 0.05 * np.arange(12)).reshape(3, 4)
        power_2d = ls.power(freqs_2d, method=method, assume_regular_frequency=True)
        assert power_2d.shape == (3, 4)
        assert power_2d.dtype == np.float64

    @pytest.mark.parametrize("method", ["tls", "tlsf", "tlsh"])
    def test_fit_mean_and_center_data(self, method):
        t, y, dy, _ = _make_signal(M=100, seed=23)
        freqs = np.linspace(0.1, 1.0, 32)

        with pytest.raises(ValueError, match="fit_mean=True"):
            LombScargle(t, y, dy, fit_mean=False).power(freqs, method=method)

        p_centered = LombScargle(t, y, dy, fit_mean=True, center_data=True).power(freqs, method=method)
        p_uncentered = LombScargle(t, y, dy, fit_mean=True, center_data=False).power(freqs, method=method)
        np.testing.assert_array_equal(p_centered, p_uncentered)

    def test_method_kwds_and_invalid_args(self):
        t, y, dy, _ = _make_signal(M=120, seed=29)
        freqs = np.linspace(0.1, 1.0, 40)
        ls = LombScargle(t, y, dy, nterms=2)

        p_default = ls.power(freqs, method="tls")
        p_custom = ls.power(freqs, method="tls", method_kwds={"backend": "pswf21", "solver": "zohar", "spreading": "minmax", "vectorize": "breadth"})
        np.testing.assert_allclose(p_custom, p_default, rtol=1e-10, atol=1e-12)

        with pytest.raises(TypeError, match="unexpected method_kwds"):
            ls.power(freqs, method="tls", method_kwds={"autonan": False})

        with pytest.raises(ValueError, match="normalization"):
            ls.power(freqs, method="tls", normalization="bogus")

    @pytest.mark.parametrize("method,mod", [("tls", tls), ("tlsf", tlsf), ("tlsh", tlsh)])
    @pytest.mark.parametrize("normalization", ["anova", "nll"])
    @pytest.mark.parametrize("nterms", [1, 3])
    def test_tls_extra_normalizations_match_native(self, method, mod, normalization, nterms):
        t, y, dy, _ = _make_signal(M=150, seed=31)
        f0, df, Nf = 0.05, 0.01, 64
        freqs = f0 + df * np.arange(Nf, dtype=np.float64)

        p_astropy = LombScargle(t, y, dy, nterms=nterms, normalization=normalization).power(freqs, method=method, assume_regular_frequency=True)
        grid_df = float(freqs[1] - freqs[0])
        p_native = np.array(mod.power(Nf, grid_df, float(freqs[0]), t, y, dy, nterms=nterms, normalization=normalization, autonan=True), dtype=np.float64)
        p_native[np.isnan(p_native)] = 0.0

        assert p_astropy.dtype == np.float64
        np.testing.assert_array_equal(p_astropy, p_native)


@_REQUIRES_ASTROPY_8
class TestAstropy8Integration:
    """Compare tls, tlsf, and tlsh against Astropy >= 8.0 fast and fastchi2."""

    @pytest.mark.parametrize("normalization", ["standard", "model", "log", "psd"])
    @pytest.mark.parametrize("use_dy", [True, False])
    def test_nterms1_agrees_with_astropy_fast(self, normalization, use_dy):
        t, y, dy, f_true = _make_signal(M=250, seed=101)
        dy_arg = dy if use_dy else None
        ls = LombScargle(t, y, dy_arg, nterms=1, normalization=normalization)

        freq, p_ref = ls.autopower(method="fast", minimum_frequency=0.05, maximum_frequency=1.5)
        df = float(freq[1] - freq[0])

        _, p_tls = ls.autopower(method="tls", minimum_frequency=0.05, maximum_frequency=1.5)
        _, p_tlsf = ls.autopower(method="tlsf", minimum_frequency=0.05, maximum_frequency=1.5)
        _, p_tlsh = ls.autopower(method="tlsh", minimum_frequency=0.05, maximum_frequency=1.5)

        assert p_tls.dtype == np.float64
        assert p_tlsf.dtype == np.float64
        assert p_tlsh.dtype == np.float64

        # tls is ~2 orders of magnitude more accurate than Astropy's LRA fast method,
        # so tolerances reflect Astropy's approximation error rather than tls's.
        np.testing.assert_allclose(p_tls, p_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(p_tlsf, p_ref, rtol=5e-3, atol=2e-3)

        # FP16 (tlsh) has ~3 decimal digits of mantissa; verify peak and bulk agreement.
        assert abs(float(freq[int(np.argmax(p_tlsh))]) - f_true) <= df
        scale = max(float(np.max(np.abs(p_ref))), 1.0)
        assert float(np.median(np.abs(p_tlsh - p_ref))) < 1e-2 * scale
        assert float(np.max(np.abs(p_tlsh - p_ref))) < 5e-2 * scale

    @pytest.mark.parametrize("nterms", [1, 2, 3])
    @pytest.mark.parametrize("normalization", ["standard", "model", "log", "psd"])
    @pytest.mark.parametrize("use_dy", [True, False])
    def test_multiharmonic_agrees_with_astropy_fastchi2(self, nterms, normalization, use_dy):
        t, y, dy, f_true = _make_signal(M=250, seed=202)
        dy_arg = dy if use_dy else None
        ls = LombScargle(t, y, dy_arg, nterms=nterms, normalization=normalization)

        freq, p_ref = ls.autopower(method="fastchi2", minimum_frequency=0.05, maximum_frequency=1.2)
        df = float(freq[1] - freq[0])

        p_tls = ls.power(freq, method="tls", assume_regular_frequency=True)
        p_tlsf = ls.power(freq, method="tlsf", assume_regular_frequency=True)
        p_tlsh = ls.power(freq, method="tlsh", assume_regular_frequency=True)

        assert p_tls.dtype == np.float64
        assert p_tlsf.dtype == np.float64
        assert p_tlsh.dtype == np.float64

        np.testing.assert_allclose(p_tls, p_ref, rtol=2e-4, atol=1e-4)
        np.testing.assert_allclose(p_tlsf, p_ref, rtol=1e-2, atol=5e-3)

        assert abs(float(freq[int(np.argmax(p_tlsh))]) - f_true) <= df
        scale = max(float(np.max(np.abs(p_ref))), 1.0)
        assert float(np.median(np.abs(p_tlsh - p_ref))) < 2e-2 * scale
        assert float(np.max(np.abs(p_tlsh - p_ref))) < 1e-1 * scale
