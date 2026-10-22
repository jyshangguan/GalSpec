"""Tests for the model-level additions: iron parameterisation, ties, labels, plots.

Covers the four things that are easy to break silently:

* ``IronTemplate`` speaks the ``IronTemplate_new`` language (``sigma``, ``dv``,
  fixed ``z``) and is reproduced by the CPU astropy model to machine precision;
* ``IronTemplate_tied`` reduces to ``IronTemplate`` for a one-component kernel, so
  the generalisation cannot quietly change the old model's meaning;
* parameter labels become mathtext (``BH$\\alpha$:$\\sigma_{w0}$``) and the fitters
  expose them;
* ``plot_posterior_hist`` draws one panel per requested parameter and annotates
  the median and the 16-84 band.

Run with::

    python -m pytest tests/test_model_features.py -v
    python tests/test_model_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax                                                          # noqa: E402
jax.config.update("jax_enable_x64", True)

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

import galspec                                                      # noqa: E402
from galspec.jnesty_fit import JNesty_Fit                            # noqa: E402
from galspec import jax_models as jxm                               # noqa: E402
from galspec.param_labels import parameter_labels, parameter_symbol  # noqa: E402

WAVE = np.linspace(4000.0, 6000.0, 4000)


def _values(model, overrides=None):
    """Free parameter vector of a model, with optional overrides by name."""
    values = np.asarray(model.get_free_params(), dtype=float)
    if overrides:
        for name, value in overrides.items():
            values[model.free_param_names.index(name)] = value
    return values


# --------------------------------------------------------------------------------------
# 1. IronTemplate speaks the IronTemplate_new language
# --------------------------------------------------------------------------------------
def test_iron_template_matches_cpu_iron_template_new():
    cpu = galspec.IronTemplate_new(amplitude=1.3, sigma=2500.0, z=0.0, dv=250.0,
                                   template_name="park2022")
    jax_model = jxm.IronTemplate(amplitude=1.3, sigma=2500.0, z=0.0, dv=250.0,
                                 template_name="park2022")
    expected = np.asarray(cpu(WAVE))
    got = np.asarray(jax_model.flux(WAVE, _values(jax_model)))
    relative = np.max(np.abs(expected - got)) / np.max(np.abs(expected))
    assert relative < 1e-12, relative
    print(f"  CPU/JAX iron parity {relative:.1e} of the peak")


def test_iron_template_uses_sigma_dv_and_fixed_z():
    model = jxm.IronTemplate(amplitude=1.0, sigma=2000.0)
    assert model.param_names == ("amplitude", "sigma", "z", "dv")
    assert model.free_param_names == ("amplitude", "sigma", "dv"), "z must default to fixed"
    # sigma changes the width, dv moves the whole template in wavelength
    narrow = np.asarray(model.flux(WAVE, _values(model, {"sigma": 1200.0})))
    wide = np.asarray(model.flux(WAVE, _values(model, {"sigma": 4000.0})))
    assert not np.allclose(narrow, wide)
    shifted = np.asarray(model.flux(WAVE, _values(model, {"dv": 800.0})))
    assert np.argmax(shifted) > np.argmax(np.asarray(model.flux(WAVE, _values(model))))
    print("  sigma broadens, dv shifts, z is fixed by default")


def test_iron_template_accepts_the_old_stddev_name():
    import warnings as python_warnings

    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always")
        model = jxm.IronTemplate(amplitude=1.0, stddev=2500.0)
    assert "sigma" in model.param_names and "stddev" not in model.param_names
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    new = jxm.IronTemplate(amplitude=1.0, sigma=2500.0)
    assert np.allclose(np.asarray(model.flux(WAVE, _values(model))),
                       np.asarray(new.flux(WAVE, _values(new))))
    print("  stddev is accepted, deprecated, and means sigma")


# --------------------------------------------------------------------------------------
# 2. IronTemplate_tied
# --------------------------------------------------------------------------------------
def test_tied_iron_reduces_to_single_gaussian():
    plain = jxm.IronTemplate(amplitude=1.0, sigma=2600.0, dv=180.0, template_name="park2022")
    tied = jxm.IronTemplate_tied(n_components=1, amplitude=1.0, sigma_0=2600.0, dv_0=180.0,
                                 w_0=1.0, template_name="park2022")
    expected = np.asarray(plain.flux(WAVE, _values(plain)))
    got = np.asarray(tied.flux(WAVE, _values(tied)))
    relative = np.max(np.abs(expected - got)) / np.max(np.abs(expected))
    assert relative < 1e-12, relative
    print(f"  tied(1 component) == IronTemplate: {relative:.1e}")


def test_tied_iron_weights_are_normalised_and_ties_resolve():
    tied = jxm.IronTemplate_tied(n_components=2, amplitude=1.0, sigma_0=1800.0,
                                 sigma_1=3000.0, w_0=2.0, w_1=3.0, name="FeII")
    base = np.asarray(tied.flux(WAVE, _values(tied)))
    double = np.asarray(tied.flux(WAVE, _values(tied, {"w_0": 4.0, "w_1": 6.0})))
    assert np.allclose(base, double), "weights must be normalised to unit sum"

    line = jxm.Line_MultiGauss(n_components=3, amp_c=1.0, sigma_c=700.0, wavec=6563.0,
                               name="Halpha", amp_w0=0.3, sigma_w0=2400.0,
                               amp_w1=0.05, sigma_w1=6000.0)
    tied.dv_0.tied = jxm.tie.tie_MultiGauss_dv_w0("Halpha")
    tied.sigma_0.tied = jxm.tie.tie_MultiGauss_sigma_w0("Halpha")
    tied.w_0.tied = jxm.tie.tie_MultiGauss_amp_w0("Halpha")
    compound = line + tied
    resolved = compound.param_dict()
    flat = dict(zip(compound.qualified_param_names, compound.param_names))
    assert np.isclose(resolved[flat["FeII.sigma_0"]], line.sigma_w0.value)
    assert np.isclose(resolved[flat["FeII.w_0"]], line.amp_w0.value)
    assert flat["FeII.sigma_0"] not in compound.free_param_names, \
        "a tied parameter must not be free"
    print("  weights normalised; ties resolve onto the line model")


def test_tied_iron_broadens_with_the_kernel():
    """A wider kernel must produce a wider Fe II profile, monotonic in sigma."""
    widths = []
    for sigma in (1200.0, 2500.0, 5000.0):
        tied = jxm.IronTemplate_tied(n_components=1, amplitude=1.0, sigma_0=sigma,
                                     w_0=1.0, template_name="park2022")
        flux = np.asarray(tied.flux(WAVE, _values(tied)))
        # second moment about the centroid, as a width proxy
        weight = np.clip(flux, 0.0, None)
        centre = np.sum(WAVE * weight) / np.sum(weight)
        var = np.sum(weight * (WAVE - centre) ** 2) / np.sum(weight)
        widths.append(np.sqrt(var))
    assert widths[0] < widths[1] < widths[2], widths
    print(f"  width proxy grows with sigma: {np.round(widths, 1)}")


# --------------------------------------------------------------------------------------
# 3. Parameter labels
# --------------------------------------------------------------------------------------
def test_parameter_symbols():
    assert parameter_symbol("amplitude") == "A"
    assert parameter_symbol("amp_c") == "A_{c}"
    assert parameter_symbol("amp_c0") == "A_{c0}"
    assert parameter_symbol("sigma_w1") == r"\sigma_{w1}"
    assert parameter_symbol("dv_w0") == r"\Delta v_{w0}"
    assert parameter_symbol("alpha") == r"\alpha"
    assert parameter_symbol("wavec") == r"\lambda"
    assert parameter_symbol("c2") == "c_{2}"
    print("  amp_c -> A_{c}, sigma_w1 -> \\sigma_{w1}, alpha -> \\alpha")


def test_parameter_labels_prefix_the_model():
    line = jxm.Line_MultiGauss(n_components=3, amp_c=1.0, sigma_c=700.0, wavec=6563.0,
                               name="Halpha")
    line.label = r"BH$\alpha$"
    model = line + jxm.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=1.0, name="Continuum")
    labels = dict(zip(model.param_names, parameter_labels(model)))
    for name in model.param_names:
        if name.startswith("sigma_w1"):
            assert labels[name] == r"BH$\alpha$:$\sigma_{w1}$", labels[name]
    print(f"  example label: {labels[[n for n in model.param_names if n.startswith('sigma_w1')][0]]}")


def test_models_can_declare_their_own_symbols():
    """A model class may override the symbol table for its parameters."""
    from galspec.param_labels import qualified_parameter_labels

    class Line_Exponential(jxm.Line_Gaussian):
        PARAM_SYMBOLS = {"sigma": r"\varsigma", "amplitude": r"\mathcal{A}"}

    leaf = jxm.Line_Gaussian(amplitude=1.0, wavec=6563.0, dv=0.0, sigma=300.0, name="L")
    custom = Line_Exponential(amplitude=1.0, wavec=6563.0, dv=0.0, sigma=300.0, name="L2")
    assert parameter_labels(leaf)[0] == r"$A$"
    assert parameter_labels(custom)[0] == r"$\mathcal{A}$"
    assert r"\varsigma" in parameter_labels(custom)[2]
    compound = jxm.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=1.0, name="C") + custom
    labels = qualified_parameter_labels(
        compound, [q for q in compound.qualified_param_names if q.startswith("L2")])
    assert labels[0] == r"L2:$\mathcal{A}$"
    print(f"  model-declared symbol: {labels[0]}")


def test_fit_exposes_labels():
    model = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, sigma_c=700.0, wavec=6563.0,
                                name="Halpha")
    model.label = r"BH$\alpha$"
    model += jxm.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=1.0, name="Continuum")
    fit = JNesty_Fit(model, WAVE[:200].copy(), np.ones(200), np.ones(200),
                             bounds_dict={"Halpha.sigma_c": (100.0, 2000.0),
                                          "Halpha.amp_c": (0.1, 10.0),
                                          "Continuum.amplitude": (0.1, 10.0),
                                          "Continuum.alpha": (-2.0, 2.0)})
    assert len(fit.param_labels) == len(fit.param_names)
    assert any(r"BH$\alpha$:$\sigma" in label for label in fit.param_labels)
    print(f"  fit labels: {fit.param_labels[:3]}")


# --------------------------------------------------------------------------------------
# 4. Posterior histograms
# --------------------------------------------------------------------------------------
def test_plot_posterior_hist_shapes_and_annotations():
    rng = np.random.default_rng(7)
    samples = np.column_stack([rng.normal(2.0, 0.5, 4000), rng.normal(-1.0, 0.2, 4000),
                               rng.normal(0.0, 1.0, 4000)])
    fig, axes = galspec.plot.plot_posterior_hist(samples, params=[0, 2], ncols=2)
    assert axes.shape == (1, 2)
    for ax in axes.ravel():
        assert len(ax.lines) >= 1, "median line missing"
        assert ax.get_title(), "median/percentiles not annotated"
    # the median of the first parameter must be reproduced in the title
    centre = np.percentile(samples[:, 0], 50)
    assert f"{centre:.4g}" in axes.ravel()[0].get_title()
    plt.close(fig)
    print(f"  median recovered in the panel title '{axes.ravel()[0].get_title()}'")


def test_plot_posterior_hist_labels_and_file(tmp_path=None):
    rng = np.random.default_rng(11)
    samples = rng.normal(size=(2000, 2))
    out = Path(tmp_path or "/tmp") / "posterior_hist_test.png"
    fig, axes = galspec.plot.plot_posterior_hist(
        samples, params=[0, 1], labels=[r"$A_{c}$", r"$\sigma_{w0}$"], ncols=2,
        filename=out)
    assert axes.ravel()[0].get_xlabel() == r"$A_{c}$"
    assert out.is_file() and out.stat().st_size > 5000
    plt.close(fig)
    print(f"  wrote {out} ({out.stat().st_size // 1024} kB)")


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