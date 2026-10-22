"""Round-trip tests for the JNesty FITS I/O and the JAX model serialiser.

These tests answer one question: *does a saved fit come back as the same fit?*
That requires more than checking that a file exists -- the model tree has to
rebuild into a model that produces the same flux, the ties have to survive, the
posterior and the evidence have to be identical, and the saved file must not
disturb the object it came from.

Run with::

    python -m pytest tests/test_jax_io.py -v
    python tests/test_jax_io.py            # standalone, no pytest needed
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("GALSPEC_JAX_X64", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "develop" / "dev_jax"))

from galspec import jax_models as jxm                                          # noqa: E402
from galspec.jax_models.convolution import convolve_lsf, convolve_lsf_var     # noqa: E402
from galspec.jax_models.serialization import (                                # noqa: E402
    deserialize_jax_model_tree,
    serialize_jax_model_tree,
)
from galspec.jnesty_fit import JNesty_Fit                                     # noqa: E402
from galspec.jnesty_fit_io import (                                           # noqa: E402
    load_jnesty_fit_from_fits,
    save_jnesty_fit_to_fits,
)

WAVE = np.linspace(4000.0, 7000.0, 2500)
PROFILE_V = np.linspace(-1500.0, 1500.0, 201)
PROFILE_F = np.exp(-0.5 * (PROFILE_V / 400.0) ** 2)


def _flux_relative_difference(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30))


# --------------------------------------------------------------------------------------
# 1. Model trees
# --------------------------------------------------------------------------------------
def test_leaf_models_roundtrip():
    """Every leaf model class comes back identical, including its constructor data."""
    cases = {
        "PowerLaw1D": jxm.PowerLaw1D(amplitude=1.2, x_0=5500.0, alpha=1.5,
                                     fixed={"x_0": True}, name="Continuum"),
        "Polynomial1D": jxm.Polynomial1D(degree=3, name="poly"),
        "BrokenPowerLaw1D": jxm.BrokenPowerLaw1D(amplitude=1.0, x_break=5000.0,
                                                 alpha_1=1.0, alpha_2=2.0, name="broken"),
        "Line_Gaussian": jxm.Line_Gaussian(amplitude=1.0, dv=50.0, sigma=300.0,
                                           wavec=6563.0, name="Ha"),
        "Line_MultiGauss": jxm.Line_MultiGauss(n_components=3, amp_c=1.0, dv_c=0.0,
                                               sigma_c=300.0, wavec=6563.0, name="Ha"),
        "Line_Exponential": jxm.Line_Exponential(sigma_pix_max=900.0, amplitude=1.0,
                                                 dv=0.0, sigma=300.0, wavec=5000.0,
                                                 name="exp"),
        "Line_GaussHermite": jxm.Line_GaussHermite(clip=False, amplitude=1.0, dv=0.0,
                                                   sigma=300.0, wavec=5000.0, h3=0.05,
                                                   h4=0.02, name="gh"),
        "Line_template": jxm.Line_template(template_velc=PROFILE_V, template_flux=PROFILE_F,
                                           amplitude=1.0, dv=0.0, name="templ"),
        "IronTemplate": jxm.IronTemplate(amplitude=0.2, stddev=2500.0, z=0.0,
                                         template_name="park2022", sigma_max=4000.0,
                                         name="FeII"),
        "StarSpectrum": jxm.StarSpectrum(Star_type="G", velscale=69.0, sigma_max=5000.0,
                                         name="star"),
        "Multi_StarSpectrum": jxm.Multi_StarSpectrum(Star_types=("A", "F", "G", "K", "M"),
                                                     velscale=69.0, sigma_max=5000.0,
                                                     name="stars"),
    }
    for label, model in cases.items():
        model.prepare(WAVE)
        tree, arrays = serialize_jax_model_tree(model)
        restored = deserialize_jax_model_tree(tree, arrays, wave=WAVE)
        assert restored.free_param_names == model.free_param_names, label
        assert restored.param_names == model.param_names, label
        theta = np.asarray(model.get_free_params(), dtype=float)
        difference = _flux_relative_difference(model.flux(WAVE, theta),
                                               restored.flux(WAVE, theta))
        assert difference < 1e-12, f"{label} does not round trip ({difference:.2e})"
        print(f"  {label:20s} rel diff {difference:.1e}, arrays {len(arrays)}")


def test_compound_with_ties_and_convolution_roundtrip():
    """A compound model with ties, a multiplicative part and an LSF rebuilds exactly."""
    continuum = jxm.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=1.0, name="Continuum")
    ha = jxm.Line_MultiGauss(n_components=3, amp_c=1.0, dv_c=0.0, sigma_c=300.0,
                             wavec=6563.0, name="Halpha", amp_w0=0.4, dv_w0=0.0,
                             sigma_w0=2000.0, amp_w1=0.05, dv_w1=0.0, sigma_w1=6000.0)
    hb = jxm.Line_MultiGauss(n_components=3, amp_c=0.3, dv_c=0.0, sigma_c=300.0,
                             wavec=4861.0, name="Hbeta", amp_w0=0.4, dv_w0=0.0,
                             sigma_w0=2000.0, amp_w1=0.05, dv_w1=0.0, sigma_w1=6000.0)
    hb.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("Halpha")
    hb.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("Halpha")
    hb.dv_w0.tied = jxm.tie.tie_MultiGauss_dv_w0("Halpha")

    o3 = jxm.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.0, amp_c1=1.0 / 2.98, dv_c=0.0, sigma_c=300.0,
        wavec0=5007.0, wavec1=4959.0, name="OIII", amp_w0=0.2, dv_w0=0.0, sigma_w0=1500.0)
    o3.amp_c1.tied = jxm.tie.tie_MultiGauss_doublet_ratio("OIII", 2.98)

    model = convolve_lsf(continuum + ha + hb + o3, 5500.0, 2000.0, name="convolved")
    model.prepare(WAVE)
    tree, arrays = serialize_jax_model_tree(model)
    restored = deserialize_jax_model_tree(tree, arrays, wave=WAVE)

    assert restored.free_param_names == model.free_param_names
    # a convolution wrapper has no ``submodel_names`` of its own: look inside
    assert tuple(restored.inner.submodel_names) == tuple(model.inner.submodel_names)
    theta = np.asarray(model.get_free_params(), dtype=float)
    difference = _flux_relative_difference(model.flux(WAVE, theta),
                                           restored.flux(WAVE, theta))
    assert difference < 1e-12, f"compound does not round trip ({difference:.2e})"
    print(f"  tied + convolved compound: rel diff {difference:.1e}")


def test_variable_convolution_roundtrip():
    """The frozen kernel of a wavelength-dependent LSF survives the round trip."""
    inner = jxm.Line_Gaussian(amplitude=1.0, dv=0.0, sigma=400.0, wavec=5000.0, name="line")
    model = convolve_lsf_var(inner, WAVE, (WAVE, np.full(WAVE.size, 2000.0)), name="var")
    model.prepare(WAVE)
    tree, arrays = serialize_jax_model_tree(model)
    assert arrays, "the frozen kernel should be stored as arrays"
    restored = deserialize_jax_model_tree(tree, arrays, wave=WAVE)
    theta = np.asarray(model.get_free_params(), dtype=float)
    difference = _flux_relative_difference(model.flux(WAVE, theta),
                                           restored.flux(WAVE, theta))
    assert difference < 1e-12
    print(f"  convolve_lsf_var: rel diff {difference:.1e}, {len(arrays)} arrays")


# --------------------------------------------------------------------------------------
# 2. FITS round trip
# --------------------------------------------------------------------------------------
def _tiny_fit(device: str = "cpu", nlive: int = 80, max_iterations: int = 300,
              custom_priors=None):
    """A small but real JNesty fit: continuum + one Gaussian line with a tie."""
    rng = np.random.default_rng(0)
    wave = np.linspace(6400.0, 6720.0, 700)
    velocity = (wave - 6563.0) / 6563.0 * 299792.458
    truth = (2.0 * np.exp(-0.5 * ((velocity - 120.0) / 180.0) ** 2)
             + 0.6 * np.exp(-0.5 * (velocity / 120.0) ** 2) + 0.8)
    flux = truth + rng.normal(0.0, 0.01, wave.size)
    error = np.full(wave.size, 0.01)

    narrow = jxm.Line_MultiGauss(n_components=2, amp_c=2.0, dv_c=0.0, sigma_c=200.0,
                                 wavec=6563.0, name="Halpha", amp_w0=0.3, dv_w0=0.0,
                                 sigma_w0=1500.0)
    model = jxm.PowerLaw1D(amplitude=0.8, x_0=5500.0, alpha=1.0, name="Continuum") + narrow
    bounds = {
        "Continuum.amplitude": (0.2, 3.0), "Continuum.alpha": (-1.0, 3.0),
        "Halpha.amp_c": (1e-3, 10.0), "Halpha.amp_w0": (1e-3, 5.0),
        "Halpha.dv_c": (-500.0, 500.0), "Halpha.sigma_c": (50.0, 600.0),
        "Halpha.dv_w0": (-1500.0, 1500.0), "Halpha.sigma_w0": (300.0, 4000.0),
    }
    fit = JNesty_Fit(model, wave, flux, error, bounds_dict=bounds, nlive=nlive,
                     device=device, verbose=False, custom_priors=custom_priors)
    fit.fit(progress=False, dlogz=1.0, max_iterations=max_iterations)
    return fit, wave


def test_fit_roundtrip():
    """A saved and reloaded fit reproduces samples, evidence, model and methods."""
    fit, wave = _tiny_fit()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "tiny.fits")
    save_jnesty_fit_to_fits(fit, path)
    assert os.path.getsize(path) > 10_000

    loaded = load_jnesty_fit_from_fits(path)
    assert list(loaded.param_names) == list(fit.param_names)
    assert list(loaded.full_param_names) == list(fit.full_param_names)
    assert np.allclose(np.array(loaded.param_bounds), np.array(fit.param_bounds))
    assert loaded.log_evidence == fit.log_evidence
    assert loaded.log_evidence_err == fit.log_evidence_err
    assert np.allclose(loaded.samples_equal_weight, fit.samples_equal_weight)
    assert tuple(loaded.model.submodel_names) == tuple(fit.model.submodel_names)

    _, _, theta = fit.get_best_fit()
    expected = np.asarray(fit.model.flux(wave, theta))
    _, _, theta_loaded = loaded.get_best_fit()
    difference = _flux_relative_difference(expected, loaded.model.flux(wave, theta_loaded))
    assert difference < 1e-12, f"reloaded model differs ({difference:.2e})"

    assert np.allclose(loaded.get_quantiles((0.16, 0.5, 0.84))[1],
                       fit.get_quantiles((0.16, 0.5, 0.84))[1])
    assert isinstance(loaded.summary(), dict)
    assert isinstance(loaded.prior_dominated(), dict)
    assert loaded.sampler is None
    print(f"  round trip: logZ = {loaded.log_evidence:.2f}, "
          f"flux rel diff {difference:.1e}")


def test_saving_does_not_change_the_fit():
    """Writing a file must be a pure read of the fit object."""
    fit, _ = _tiny_fit(nlive=60, max_iterations=200)
    before = {
        "model": np.asarray(fit.model.get_free_params(), dtype=float),
        "samples": np.array(fit.samples_equal_weight, copy=True),
        "logz": float(fit.log_evidence),
        "quantiles": np.array(fit.get_quantiles()),
    }
    directory = tempfile.mkdtemp()
    first = os.path.join(directory, "a.fits")
    second = os.path.join(directory, "b.fits")
    save_jnesty_fit_to_fits(fit, first)
    save_jnesty_fit_to_fits(fit, second)

    assert np.array_equal(np.asarray(fit.model.get_free_params(), dtype=float), before["model"])
    assert np.array_equal(np.asarray(fit.samples_equal_weight), before["samples"])
    assert float(fit.log_evidence) == before["logz"]
    assert np.array_equal(np.asarray(fit.get_quantiles()), before["quantiles"])
    assert os.path.getsize(first) == os.path.getsize(second)
    print("  save is read-only and deterministic")


def test_custom_prior_loss_is_reported():
    """A custom prior cannot be stored: the file records it and loading warns."""
    # keyed by qualified name, the same convention as ``bounds_dict``
    custom = {"Halpha.sigma_c": lambda u: 50.0 + 550.0 * u}
    fit, _ = _tiny_fit(nlive=60, max_iterations=200, custom_priors=custom)
    assert fit.custom_priors, "the custom prior should have been resolved"

    path = os.path.join(tempfile.mkdtemp(), "custom.fits")
    save_jnesty_fit_to_fits(fit, path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_jnesty_fit_from_fits(path)
    assert any("custom priors" in str(item.message) for item in caught), \
        "loading a file with custom priors must warn"
    assert loaded.custom_priors == {}
    assert np.isfinite(loaded.log_evidence)
    assert np.allclose(loaded.samples_equal_weight, fit.samples_equal_weight)
    print(f"  custom priors: recorded in the file, callables not restored "
          f"(logZ = {loaded.log_evidence:.2f})")


def test_device_buffers_can_be_released():
    """release_device_buffers() drops the sampler but keeps the results."""
    fit, _ = _tiny_fit(nlive=40, max_iterations=150)
    logz = fit.log_evidence
    samples = np.array(fit.samples_equal_weight, copy=True)
    fit.release_device_buffers()
    assert fit.sampler is None
    assert fit.log_evidence == logz
    assert np.array_equal(np.asarray(fit.samples_equal_weight), samples)
    assert fit.get_quantiles().shape == (3, fit.ndim)
    print("  device buffers released, results intact")


# --------------------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------------------
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        try:
            print(f"[run ] {name}")
            test()
            print(f"[ ok ] {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            import traceback
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
    print()
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())