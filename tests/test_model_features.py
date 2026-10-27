"""Tests for the model-level additions: iron parameterisation, ties, labels, plots.

Covers the things that are easy to break silently:

* ``IronTemplate`` speaks the ``IronTemplate_new`` language (``sigma``, ``dv``,
  fixed ``z``) and is reproduced by the CPU astropy model to machine precision;
* the iron templates are resampled at load onto a grid even in ``ln(wavelength)``
  and zero-padded, so one Gaussian is the same number of channels everywhere and
  the tabulated edge does not leak into the fitted range;
* ``IronTemplate_tied`` reduces to ``IronTemplate`` for a one-component kernel --
  the generalisation cannot quietly change the old model's meaning -- and the CPU
  class is now the same function as the JAX one: an exact convolution in velocity
  space, which is what the old FFT-deconvolution version was not;
* a two-component ``Line_MultiGauss`` is exchangeable (why the wind ratio is
  bounded to [0, 1]);
* parameter labels become mathtext (``BH$\\alpha$:$\\sigma_{w0}$``) and the fitters
  expose them;
* ``plot_posterior_hist`` draws one panel per requested parameter and annotates
  the median and the 16-84 band;
* the generated ``tie_MultiGauss_*`` table follows the parameter it names (the
  hand-written table had three winds tied to one shift);
* the toggle prior puts a *finite* slice of the unit cube on exactly zero, so a
  fixed-dimension sampler can explore how many components a profile needs, and
  a toggled-off component is bit-for-bit the smaller model.

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
import jax.numpy as jnp                                             # noqa: E402
jax.config.update("jax_enable_x64", True)

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

import galspec                                                      # noqa: E402
from galspec.jnesty_fit import JNesty_Fit                            # noqa: E402
from galspec import jax_models as jxm                               # noqa: E402
from galspec.param_labels import parameter_labels, parameter_symbol  # noqa: E402
from galspec.constants import ls_km as LS_KM                               # noqa: E402

import pytest                                                              # noqa: E402

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
    tied = jxm.IronTemplate_tied(n_components=1, amplitude=1.0, sigma_c=2600.0,
                                 dv_c=180.0, template_name="park2022")
    expected = np.asarray(plain.flux(WAVE, _values(plain)))
    got = np.asarray(tied.flux(WAVE, _values(tied)))
    relative = np.max(np.abs(expected - got)) / np.max(np.abs(expected))
    assert relative < 1e-12, relative
    print(f"  tied(1 component) == IronTemplate: {relative:.1e}")


def test_tied_iron_parameters_follow_the_line_model():
    """The kernel has the same parameters as the Line_MultiGauss, and no weights."""
    tied = jxm.IronTemplate_tied(n_components=2, amplitude=1.0, name="FeII")
    assert tied.param_names == ("amplitude", "z", "dv_c", "sigma_c",
                                "amp_w0", "dv_w0", "sigma_w0")
    assert tied.kernel_param_names() == ("dv_c", "sigma_c", "amp_w0",
                                         "dv_w0", "sigma_w0")
    assert not [name for name in tied.param_names if name.startswith("w_")], \
        "the kernel must have no weights of its own"
    two = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, sigma_c=1000.0, wavec=6563.0)
    assert set(tied.kernel_param_names()) <= set(two.param_names), \
        "every kernel parameter must exist on the line model too"
    print("  kernel parameters mirror Line_MultiGauss and carry no weights")


def test_tied_iron_scales_with_the_line_peak_only():
    """The kernel is normalised: scaling the line's peak leaves the iron untouched."""
    line = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, sigma_c=1500.0, wavec=6563.0,
                               name="b_halpha", amp_w0=0.3, sigma_w0=6000.0)
    tied = jxm.IronTemplate_tied(n_components=2, amplitude=1.0, name="FeII")
    compound = line + tied
    flat = dict(zip(compound.qualified_param_names, compound.param_names))
    iron = []
    for factor in (1.0, 5.0):
        model = compound.copy()
        model[flat["b_halpha.amp_c"]].value = factor
        total = np.asarray(model.flux(WAVE, model.get_free_params()))
        # the line on its own, at the scaled peak
        theta = np.array([factor, 0.0, 1500.0, 0.3, 0.0, 6000.0])
        iron.append(total - np.asarray(model["b_halpha"].flux(WAVE, theta)))
    relative = np.max(np.abs(iron[0] - iron[1])) / np.max(np.abs(iron[0]))
    assert relative < 1e-12, relative
    print(f"  iron independent of the line's peak: {relative:.1e}")


def test_tied_iron_ties_resolve_onto_core_and_wind():
    tied = jxm.IronTemplate_tied(n_components=2, amplitude=1.0, name="FeII")
    line = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, sigma_c=1500.0, wavec=6563.0,
                               name="b_halpha", amp_w0=0.3, sigma_w0=6000.0)
    tied.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("b_halpha")
    tied.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("b_halpha")
    tied.amp_w0.tied = jxm.tie.tie_MultiGauss_amp_w0("b_halpha")
    tied.dv_w0.tied = jxm.tie.tie_MultiGauss_dv_w0("b_halpha")
    tied.sigma_w0.tied = jxm.tie.tie_MultiGauss_sigma_w0("b_halpha")
    compound = line + tied
    resolved = compound.param_dict()
    flat = dict(zip(compound.qualified_param_names, compound.param_names))
    for kernel_param, line_param in (("dv_c", "dv_c"), ("sigma_c", "sigma_c"),
                                     ("amp_w0", "amp_w0"), ("dv_w0", "dv_w0"),
                                     ("sigma_w0", "sigma_w0")):
        assert np.isclose(resolved[flat[f"FeII.{kernel_param}"]],
                          resolved[flat[f"b_halpha.{line_param}"]]), kernel_param
        assert flat[f"FeII.{kernel_param}"] not in compound.free_param_names, \
            "a tied parameter must not be free"
    free_qualified = [q for q, f in flat.items() if f in set(compound.free_param_names)]
    iron_free = [q for q in free_qualified if q.startswith("FeII.")]
    assert iron_free == ["FeII.amplitude"], iron_free
    print("  both the core and the wind resolve onto the line; only A is free")


def test_tied_iron_broadens_with_the_kernel():
    """A wider kernel must produce a wider Fe II profile, monotonic in sigma."""
    widths = []
    for sigma in (1200.0, 2500.0, 5000.0):
        tied = jxm.IronTemplate_tied(n_components=1, amplitude=1.0, sigma_c=sigma,
                                     template_name="park2022")
        flux = np.asarray(tied.flux(WAVE, _values(tied)))
        # second moment about the centroid, as a width proxy
        weight = np.clip(flux, 0.0, None)
        centre = np.sum(WAVE * weight) / np.sum(weight)
        var = np.sum(weight * (WAVE - centre) ** 2) / np.sum(weight)
        widths.append(np.sqrt(var))
    assert widths[0] < widths[1] < widths[2], widths
    print(f"  width proxy grows with sigma: {np.round(widths, 1)}")


def test_tied_iron_kernel_radius_covers_wide_widths():
    """Evaluating outside a fit must not truncate the kernel early.

    The kernel radius is static (it has to be, under jit), so the model sizes it
    from the widest width it was told about.  A 4-sigma truncation at the wrong
    width is a small but real error, and this is the regression test for it.
    """
    narrow = jxm.IronTemplate_tied(n_components=1, amplitude=1.0, sigma_c=1200.0)
    wide = jxm.IronTemplate_tied(n_components=1, amplitude=1.0, sigma_c=9000.0)
    assert wide.sigma_max >= 9000.0 > wide._radius * 0 + 1200.0
    assert wide._radius > narrow._radius
    # a profile built with a big radius must match an explicitly wide one
    exact = jxm.IronTemplate(amplitude=1.0, sigma=9000.0, sigma_max=9000.0)
    from_wide = np.asarray(wide.flux(WAVE, _values(wide)))
    from_exact = np.asarray(exact.flux(WAVE, _values(exact)))
    relative = np.max(np.abs(from_wide - from_exact)) / np.max(np.abs(from_exact))
    assert relative < 1e-12, relative
    print(f"  kernel radius tracks the widest width (max radius {wide._radius} px)")


def test_cpu_tied_iron_is_a_velocity_space_convolution():
    """The CPU tied iron is an exact convolution on a uniform velocity grid.

    The reference shares no code with either backend: the template is placed on a
    uniform velocity grid, convolved there, and interpolated back.  The CPU class
    used to be 8-50% off this reference depending on its internal settings.
    """
    from galspec.iron_template import flux_park2022, wave_park2022

    wave = np.arange(4200.0, 6800.0, 1.0)
    window = (wave_park2022 > 4500) & (wave_park2022 < 5500)
    flux = flux_park2022 / np.max(flux_park2022[window])
    for sigma in (700.0, 1500.0, 6900.0):
        kernel = galspec.Line_MultiGauss(n_components=1, amp_c=1.0, dv_c=0.0,
                                         sigma_c=sigma, wavec=6563.0)
        got = galspec.IronTemplate_tied(amplitude=1.0, z=0.0, kernel=kernel)(wave)
        want = np.interp(wave, wave_park2022,
                         _smooth_in_velocity_space(wave_park2022, flux, sigma, 800 / 2.3548))
        relative = np.max(np.abs(got - want)) / np.max(want)
        # 2e-3 rather than 1e-3: at 700 km/s the 126 km/s template grid resolves the
        # narrowest kernel with only ~6 channels, and that sampling is the floor.
        assert relative < 2e-3, (sigma, relative)
    print("  CPU tied iron is within 0.2% of an exact velocity-space convolution")


def test_multigauss_components_are_exchangeable():
    """The core and the wind are the same two numbers under a label swap.

    This is a property of the model, not of the fit: it is why the worked example
    bounds the wind ratio to [0, 1] instead of leaving it free.  The swap maps
    ``amp_w0`` to ``1/amp_w0``, which is outside [0, 1] whenever the wind is
    fainter than the core -- so that single bound removes the ambiguity.
    """
    line = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, dv_c=0.0, sigma_c=1500.0,
                               amp_w0=0.3, dv_w0=-500.0, sigma_w0=6000.0,
                               wavec=6563.0)
    base = np.asarray(line.flux(WAVE, _values(line)))

    # the exchange map: amp_c*A, sigma_c<->sigma_w0, dv_c+dv_w0<->dv_c,
    # amp_w0 -> 1/amp_w0, dv_w0 -> -dv_w0
    swapped = jxm.Line_MultiGauss(
        n_components=2, amp_c=1.0 * 0.3, dv_c=0.0 + (-500.0), sigma_c=6000.0,
        amp_w0=1.0 / 0.3, dv_w0=500.0, sigma_w0=1500.0, wavec=6563.0)
    other = np.asarray(swapped.flux(WAVE, _values(swapped)))

    relative = np.max(np.abs(base - other)) / np.max(np.abs(base))
    assert relative < 1e-12, relative
    assert swapped.amp_w0.value > 1.0, "the mirror solution needs amp_w0 > 1"
    print(f"  swap is exact ({relative:.1e}) and leaves [0, 1]")


def _smooth_in_velocity_space(wave, flux, sigma, sigma_intr, step=5.0):
    """The iron as it is meant to be: a Gaussian convolution in velocity space.

    The template is sampled on a uniform velocity grid, convolved there -- so the
    requested width is the requested width at every wavelength -- and returned on
    the template's own grid.  Zero padding outside the tabulated range.
    """
    v = np.log(wave / wave[0]) * LS_KM
    v_grid = np.arange(-80000.0, v[-1] + 80000.0 + step, step)
    flux_v = np.interp(v_grid, v, flux, left=0.0, right=0.0)
    extra = np.sqrt(max(sigma ** 2 - sigma_intr ** 2, 0.0))
    half = int(5.0 * extra / step)
    offsets = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (offsets / extra) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(np.pad(flux_v, half, mode="edge"), kernel,
                           mode="same")[half:half + len(flux_v)]
    return np.interp(v, v_grid, smoothed)


def test_iron_templates_are_even_in_log_wavelength():
    """Both templates are resampled at load, so a channel is one velocity everywhere.

    Before this the grid was even in wavelength (2.00 A), so one channel spanned
    150 km/s at the blue end and 107 km/s at the red end and a velocity width was
    a different number of channels at each end.
    """
    from galspec.iron_template import (flux_boroson1992, flux_park2022,
                                       load_iron_template, resample_log_even,
                                       wave_boroson1992, wave_park2022)

    for name, wave in (("park2022", wave_park2022), ("boroson1992", wave_boroson1992)):
        spacing = np.diff(np.log(wave))
        assert np.ptp(spacing) < 1e-9 * np.mean(spacing), name
        velscale = np.mean(spacing) * LS_KM
        assert 60.0 < velscale < 200.0, (name, velscale)

    # the tabulated range keeps its samples; beyond it the grid continues at the
    # same spacing, carrying zero flux, so the smoothing has nothing to mirror
    assert np.isclose(flux_park2022.max(), 1.0) and np.isclose(flux_boroson1992.max(), 1.0)
    filled = wave_park2022[flux_park2022 > 0.0]
    assert 3995.0 < filled[0] < 4005.0 and 5595.0 < filled[-1] < 5605.0
    assert wave_park2022[0] < 3200.0 and wave_park2022[-1] > 6600.0
    assert flux_park2022[0] == 0.0 and flux_park2022[-1] == 0.0
    # 100 000 km/s of padding either side, on the 126 km/s grid
    unpadded, _ = load_iron_template("park2022", pad_velocity=0.0)
    padding = len(wave_park2022) - len(unpadded)
    assert 2 * 790 <= padding <= 2 * 800, padding

    # an unknown template is refused by name, not with a KeyError from a dict lookup
    with pytest.raises(KeyError):
        load_iron_template("boroson1993")
    # oversampling halves the channel width and is available if ever needed
    wave, _ = resample_log_even(wave_park2022, flux_park2022, oversample=2)
    coarse = np.mean(np.diff(np.log(wave_park2022)))
    assert len(wave) == 2 * len(wave_park2022)
    assert np.isclose(np.mean(np.diff(np.log(wave))), 0.5 * coarse, rtol=0.5 * coarse)


def test_iron_template_broadens_in_velocity_space():
    """A Gaussian width must be the same at both ends of the template.

    Compares :class:`galspec.IronTemplate_new` against an independent convolution
    on a uniform velocity grid.  On a wavelength-even grid this test fails by
    several per cent; on the log-even grid that both backends now load it agrees
    to well under one per cent.
    """
    from galspec.iron_template import flux_park2022, wave_park2022

    window = (wave_park2022 > 4500) & (wave_park2022 < 5500)
    flux = flux_park2022 / np.max(flux_park2022[window])
    sigma_intr = 800.0 / 2.3548
    wavelengths = np.arange(4200.0, 5400.0, 1.0)

    for sigma in (700.0, 1500.0, 4000.0):
        model = galspec.IronTemplate_new(amplitude=1.0, sigma=sigma, z=0.0, dv=0.0)
        got = np.asarray(model(wavelengths))
        want = np.interp(wavelengths, wave_park2022,
                         _smooth_in_velocity_space(wave_park2022, flux, sigma, sigma_intr))
        error = np.max(np.abs(got - want)) / np.max(want)
        assert error < 0.01, (sigma, error)


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


def _qualified_theta(model, **overrides):
    """Free parameter vector addressed by ``Submodel.parameter`` names."""
    theta = np.asarray(model.get_free_params(), dtype=float).copy()
    flat = dict(zip(model.qualified_param_names, model.param_names))
    for qualified, value in overrides.items():
        theta[model.free_param_names.index(flat[qualified])] = value
    return theta


# --------------------------------------------------------------------------------------
# 5. Setting the free parameters refreshes the tied ones
# --------------------------------------------------------------------------------------
def test_set_free_params_refreshes_tied_values_on_the_leaves():
    """After a fit a submodel shows the fitted value, as an astropy submodel does.

    Ties are resolved during evaluation, so the *total* model was always right; the
    leaves kept their construction defaults.  ``plot_fit`` draws a component by
    evaluating the leaf, so the published figure showed [N II] with the default
    700 km/s width instead of the 298 km/s it is tied to.
    """
    def doublet(name, amp_c0):
        return jxm.Line_MultiGauss_doublet(n_components=1, amp_c0=amp_c0,
                                           amp_c1=amp_c0 / 2.96, dv_c=0.0,
                                           sigma_c=700.0, wavec0=6583.0,
                                           wavec1=6548.0, name=name)

    o3 = doublet("OIII", 2.0)
    n2 = doublet("NII", 0.8)
    n2.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("OIII")
    n2.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("OIII")
    n2.amp_c1.tied = jxm.tie.tie_MultiGauss_doublet_ratio("NII", 2.96)
    compound = o3 + n2

    theta = _qualified_theta(compound, **{"OIII.amp_c0": 2.0, "OIII.dv_c": 70.0,
                                          "OIII.sigma_c": 300.0, "NII.amp_c0": 0.8})
    compound.set_free_params(theta)

    assert np.isclose(float(n2.sigma_c.value), 300.0), n2.sigma_c.value
    assert np.isclose(float(n2.dv_c.value), 70.0), n2.dv_c.value
    assert np.isclose(float(n2.amp_c1.value), 0.8 / 2.96), n2.amp_c1.value

    wave = np.linspace(6500.0, 6700.0, 2001)
    total = np.asarray(compound(wave))
    parts = sum(np.asarray(compound[name](wave)) for name in compound.submodel_names)
    assert np.allclose(total, parts, rtol=1e-12, atol=1e-12), \
        f"components differ from the total by {np.abs(total - parts).max():.3g}"

    # the drawn doublet is the tied one: a 300 km/s Gaussian, not a 700 km/s one
    y = np.asarray(compound["NII"](wave))
    above = np.where(y > y.max() / 2.0)[0]
    fwhm_kms = (wave[above[-1]] - wave[above[0]]) / 6583.0 * LS_KM
    assert np.isclose(fwhm_kms, 2.3548 * 300.0, rtol=0.05), fwhm_kms
    print(f"  [N II] drawn at {fwhm_kms:.0f} km/s FWHM, components sum to the total")


def test_sync_tied_values_is_idempotent_and_leaves_free_params_alone():
    line = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, sigma_c=1500.0,
                               wavec=6563.0, name="b_halpha")
    other = jxm.Line_MultiGauss(n_components=2, amp_c=0.3, sigma_c=1200.0,
                                wavec=4862.0, name="b_hbeta")
    other.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("b_halpha")
    compound = line + other
    theta = _qualified_theta(compound, **{"b_halpha.sigma_c": 1234.0})
    compound.set_free_params(theta)
    first = dict(compound.param_dict())
    compound.sync_tied_values()
    assert compound.param_dict() == first
    assert np.isclose(float(other.sigma_c.value), 1234.0)
    assert np.allclose(compound.get_free_params(), theta), "free parameters moved"
    print("  sync is idempotent and does not touch the free parameters")


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


def test_plot_posterior_hist_uses_the_posterior_for_the_view():
    """A wide prior must not squeeze the posterior into a spike.

    The axis has to be set from the samples, not from the prior bounds: with the
    axis stretched to the prior, a parameter measured to 0.1% is a one-pixel line
    and the shape that the panel exists to show is invisible.
    """
    rng = np.random.default_rng(3)
    samples = np.column_stack([rng.normal(2.0, 0.01, 4000),
                               rng.normal(1000.0, 1.0, 4000)])
    bounds = [(0.0, 100.0), (300.0, 8000.0)]      # priors far wider than the posterior

    class _FakeFit:
        samples_equal_weight = samples
        param_names = ("amplitude", "sigma")
        full_param_names = ("A.amplitude", "A.sigma")
        param_bounds = bounds
        param_labels = ("$A$.$A$", "$A$:$\\sigma$")

    fig, axes = galspec.plot.plot_posterior_hist(_FakeFit(), params=[0, 1], ncols=2)
    for ax, column in zip(axes.ravel(), (0, 1)):
        lo, hi = ax.get_xlim()
        inside = samples[:, column]
        # the view covers the posterior, not the prior
        assert (inside > lo).mean() > 0.99 and (inside < hi).mean() > 0.99, ax.get_xlim()
        span = hi - lo
        assert span < 0.25 * (bounds[column][1] - bounds[column][0]), "axis still prior-wide"
        # the prior is not lost: it is written in the corner
        assert any("prior" in text.get_text() for text in ax.texts), "prior not annotated"
    plt.close(fig)
    print(f"  posterior-only view: first panel {axes.ravel()[0].get_xlim()}")


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


# --------------------------------------------------------------------------------------
# 6. Broad lines with more components: the ties and the toggle prior
# --------------------------------------------------------------------------------------
def test_multigauss_tie_classes_follow_their_own_parameter():
    """Every ``tie_MultiGauss_<param>_w<i>`` follows ``<param>_w<i>``, not a neighbour.

    The hand-written tie table had ``dv_w2``, ``dv_w3`` and ``dv_w4`` all returning
    ``dv_w1``, so a four-component fit would have tied three wings to one shift and
    reported a perfect -- and meaningless -- agreement.  The classes are generated
    now; this test pins the generated table down parameter by parameter.
    """
    line = galspec.Line_MultiGauss(
        n_components=5, amp_c=1.0, dv_c=10.0, sigma_c=1100.0, wavec=6563.0,
        par_w={"amp_w0": 0.31, "dv_w0": 120.0, "sigma_w0": 2200.0,
               "amp_w1": 0.22, "dv_w1": 230.0, "sigma_w1": 3300.0,
               "amp_w2": 0.13, "dv_w2": 340.0, "sigma_w2": 4400.0,
               "amp_w3": 0.04, "dv_w3": 450.0, "sigma_w3": 5500.0})
    model = {"ref": line}

    names = ["amp_c", "dv_c", "sigma_c"] + [
        f"{stem}_w{i}" for i in range(4) for stem in ("amp", "dv", "sigma")]
    for name in names:
        tie_class = getattr(galspec, f"tie_MultiGauss_{name}")
        tie = tie_class(ref_name="ref")
        expected = float(getattr(line, name).value)
        # astropy ties return the Parameter itself (or a float, after arithmetic)
        result = tie(model)
        got = float(result.value if hasattr(result, "value") else result)
        assert got == pytest.approx(expected), f"{name}: {got} != {expected}"
        assert tie.tied_param == name, tie.tied_param

    # the parameters really are distinct, so the assertion above is not vacuous
    shifts = [float(getattr(line, f"dv_w{i}").value) for i in range(4)]
    assert len(set(shifts)) == 4, shifts
    print(f"  {len(names)} tie classes follow their own parameter")


def test_component_toggle_gives_off_a_finite_slice_of_the_cube():
    """``p_off`` of the unit interval maps to exactly zero, the rest onto the slab.

    A component that can be switched off needs the "off" state to own real prior
    volume -- a delta function at zero is a state the sampler never visits and a
    nested sampling run cannot see.  The linear slab is the recommended one: it
    does not concentrate density at the tiny amplitudes where an extra Gaussian
    hides in the noise.
    """
    u = np.linspace(0.0, 1.0, 20001)
    for p_off in (0.0, 0.2, 0.5, 0.9):
        prior = galspec.uniform_toggle(p_off=p_off)
        values = np.asarray(jax.vmap(prior)(jnp.asarray(u)), dtype=float)
        assert np.all(values >= 0.0)
        assert np.isclose(np.mean(values == 0.0), p_off, atol=1.5e-4), (
            p_off, np.mean(values == 0.0))
        on = u >= p_off
        slab = (u[on] - p_off) / (1.0 - p_off)
        assert np.allclose(values[on], slab, atol=1e-12), "slab is not linear"
        assert np.isclose(values[-1], 1.0), values[-1]

    # the log slab keeps its floor: "on" means "at least lower"
    log_prior = galspec.log_uniform_toggle(p_off=0.5, lower=0.02)
    values = np.asarray(jax.vmap(log_prior)(jnp.asarray(u)), dtype=float)
    on = values[values > 0.0]
    assert on.min() == pytest.approx(0.02, rel=1e-9), on.min()
    assert on.max() == pytest.approx(1.0, rel=1e-9), on.max()
    # ... and it is uniform in log: the middle of the slab is the geometric mean
    assert on[on.size // 2] == pytest.approx(np.sqrt(0.02 * 1.0), rel=1e-3), \
        on[on.size // 2]

    with pytest.raises(ValueError):
        galspec.log_uniform_toggle(lower=0.0)
    with pytest.raises(ValueError):
        galspec.component_toggle(p_off=1.0)
    with pytest.raises(ValueError):
        galspec.component_toggle(slab="cauchy")
    print(f"  toggle puts a finite slice of the cube on zero for "
          f"p_off = 0.0 .. 0.9; the log slab floors at its lower bound")


def test_component_toggle_survives_jit_and_vmap():
    """The transform is traced by the samplers, so it must not branch in Python."""
    prior = galspec.uniform_toggle(p_off=0.3)
    u = jnp.asarray([0.0, 0.1, 0.29, 0.3, 0.65, 1.0])
    traced = np.asarray(jax.jit(jax.vmap(prior))(u), dtype=float)
    plain = np.array([float(prior(float(x))) for x in np.asarray(u)])
    assert np.allclose(traced, plain, atol=1e-15), (traced, plain)
    assert traced[2] == 0.0, traced          # u <  p_off -> off
    assert traced[4] == pytest.approx(0.5), traced   # slab, linear
    assert traced[5] == 1.0, traced
    print(f"  jit/vmap agree with the plain call: {traced}")


def test_a_toggled_off_component_is_one_fewer_component():
    """The semantic anchor of the whole scheme: ``amp = 0`` *is* the smaller model.

    If switching a component off left any trace -- a normalisation, a padding, a
    kernel radius -- then "the minimum number of components" would not mean what
    the selection rule says it means.
    """
    wave = np.linspace(4300.0, 6800.0, 2000)
    core = dict(amp_c=1.0, dv_c=50.0, sigma_c=1200.0, wavec=6563.0)
    wind0 = dict(amp_w0=0.40, dv_w0=-400.0, sigma_w0=3000.0)
    off_wind = dict(amp_w1=0.0, dv_w1=900.0, sigma_w1=7000.0)

    two = galspec.Line_MultiGauss(n_components=2, par_w=dict(wind0), **core)
    three = galspec.Line_MultiGauss(n_components=3, par_w=dict(wind0, **off_wind), **core)
    peak = np.max(np.abs(np.asarray(two(wave))))
    assert np.max(np.abs(np.asarray(three(wave)) - np.asarray(two(wave)))) < 1e-12 * peak

    jax_two = jxm.Line_MultiGauss(n_components=2, par_w=dict(wind0), **core)
    jax_three = jxm.Line_MultiGauss(n_components=3, par_w=dict(wind0, **off_wind), **core)
    jax_peak = np.max(np.abs(np.asarray(jax_two.flux(wave, _values(jax_two)))))
    diff = np.max(np.abs(np.asarray(jax_three.flux(wave, _values(jax_three)))
                         - np.asarray(jax_two.flux(wave, _values(jax_two)))))
    assert diff < 1e-12 * jax_peak, (diff, jax_peak)

    # the iron follows: an off wing must not change the broadened template either
    iron_two = galspec.IronTemplate_tied(amplitude=0.2, z=0.0, kernel=two)
    iron_three = galspec.IronTemplate_tied(amplitude=0.2, z=0.0, kernel=three)
    iron_peak = np.max(np.abs(np.asarray(iron_two(wave))))
    iron_diff = np.max(np.abs(np.asarray(iron_three(wave)) - np.asarray(iron_two(wave))))
    assert iron_diff < 1e-10 * iron_peak, (iron_diff, iron_peak)
    print(f"  off wing: line {diff / jax_peak:.1e}, iron {iron_diff / iron_peak:.1e} "
          f"relative")


# --------------------------------------------------------------------------------------
# 7. The worked example's component-count plumbing
# --------------------------------------------------------------------------------------
def _load_example():
    """``example/fit_iras07546_jax.py`` as a module (it is a script, not a package)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "example" / "fit_iras07546_jax.py"
    spec = importlib.util.spec_from_file_location("_example_iras", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_grows_the_broad_lines_and_pins_down_every_prior():
    """``n_wings`` adds components, nothing else, and every one of them has a prior.

    The example puts the narrow lines on a measured template, so it has 27 free
    parameters -- three fewer than the published fit's 30, because the four doublets
    lose their width and [O III] loses its two wind components.  Asking for four
    components must still add exactly 5 parameters per extra wing -- three on broad
    H-alpha (an amplitude, a shift and a width) and one amplitude per remaining
    Balmer line, whose shifts and widths are tied to H-alpha -- and every free
    parameter must carry a prior, or the fit refuses to start.
    """
    example = _load_example()
    flux = np.linspace(0.4, 2.2, 500)

    default = example.build_model()
    assert len(default.free_param_names) == 27, len(default.free_param_names)
    assert default["b_halpha"].n_components == 2
    # every narrow line is the template: an amplitude for a singlet, or the atomic
    # ratio for the second peak of a doublet, and no width anywhere
    narrow = [name for name in default.submodel_names
              if type(default[name]).__name__ in ("Line_template",
                                                  "Line_template_doublet")]
    assert len(narrow) == 10, narrow
    # six singlets, four doublets -- the ten narrow lines of the published fit
    assert sum(type(default[name]).__name__ == "Line_template"
               for name in narrow) == 6, narrow
    assert default["OIII"].param_names == ("amp_c0", "amp_c1", "dv_c", "wavec0",
                                           "wavec1")

    wide = example.build_model(n_wings=3)
    assert len(wide.free_param_names) == 37, len(wide.free_param_names)
    for line in ("b_halpha", "b_hbeta", "b_hgamma"):
        assert wide[line].n_components == 4, (line, wide[line].n_components)
    # He I 5876 and He II 4686 keep a single broad Gaussian
    assert wide["b_hei"].n_components == 1
    assert wide["b_heii"].n_components == 1
    # the extra wings are tied to broad H-alpha, component by component (the
    # amplitudes are *not* tied between lines: each keeps its own Balmer decrement.
    # Only the shifts and widths are shared.)
    for line in ("b_hbeta", "b_hgamma"):
        for stem in ("dv_w", "sigma_w"):
            for i in (1, 2):
                tie = getattr(wide[line], f"{stem}{i}").tied
                assert tie is not None, (line, stem, i)
                got = tie(wide)
                got = float(got.value if hasattr(got, "value") else got)
                expected = float(getattr(wide["b_halpha"], f"{stem}{i}").value)
                assert got == expected, (line, stem, i, got, expected)

    for count in (0, 1, 3):
        model = example.build_model(n_wings=count)
        example.set_priors(model, flux)
        missing = [q for q in example.free_qualified_names(model)
                   if model.get_prior(q) is None]
        assert not missing, (count, missing)
        # each wing costs five free parameters: three on broad H-alpha and one
        # amplitude per remaining Balmer line (their shifts and widths are tied)
        assert len(model.free_param_names) == 22 + 5 * count, count

    # releasing the broad ties frees exactly the eight shared parameters
    released = example.build_model(tie_broad=False)
    assert len(released.free_param_names) == 35, len(released.free_param_names)

    print(f"  example: {len(default.free_param_names)} -> {len(wide.free_param_names)} "
          f"free parameters for 1 -> 3 wings (5 per extra wing)")


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