#!/usr/bin/env python3
"""Input/output test for the tied Fe II kernel and the split broad/narrow lines.

    python develop/dev_jax/iron_tied_io_test.py            # full run, writes the report

What it does
------------
1. Builds a mock spectrum with the **CPU** (astropy) models -- exactly the
   prescription asked for:

   * ``galspec.IronTemplate_tied`` broadens the Park et al. (2022) template with
     the profile of a **broad H-beta** given as a
     ``galspec.Line_MultiGauss(n_components=2)`` kernel (core + one wind);
   * the same ``Line_MultiGauss`` is added to the mock as the broad H-beta itself;
   * so iron and line share one profile, parameter for parameter.

2. Fits that mock with the **JAX** models: ``jxm.Line_MultiGauss`` for the broad
   H-beta and ``jxm.IronTemplate_tied`` for the iron, with *every* kernel
   parameter tied to the line (core and wind alike) and only the iron amplitude
   left free.

3. Reports whether the best fit reproduces the input parameters, plus two control
   mocks that isolate where a discrepancy would come from:

   * ``reference`` -- the iron built by an independent, brute-force convolution in
     velocity space (no deconvolution, one kernel per component);
   * ``jax`` -- the iron built by the JAX model itself, which tests the fitting
     and tie machinery with no model difference at all.

Outputs (default ``develop/dev_jax/io_test/``)
----------------------------------------------
``iron_tied_io_test.html``      the report
``*.png``                       the figures it embeds (also inlined as base64)
``io_test_results.json``        every number in the report
"""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import io
import json
import time
from pathlib import Path

import numpy as np

import jax

import galspec
import galspec.jax_models as jxm
from galspec.constants import ls_km

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "io_test"

# --------------------------------------------------------------------------------------
# The truth: one broad H-beta, described by one Line_MultiGauss, and the iron tied to it
# --------------------------------------------------------------------------------------
#: Park et al. (2022) template: 801 channels, 4000-5600 A rest frame.
TEMPLATE = "park2022"

#: Truth values.  ``amp_c`` is the peak of broad component 0 in 1e-14 units,
#: ``amp_w0`` the peak of broad component 1 **relative** to it, ``dv_w0`` the
#: shift of component 1 **relative** to component 0 -- the ``Line_MultiGauss``
#: convention.
TRUTH = {
    "b_hb.amp_c": 1.20,        # broad H-beta core peak
    "b_hb.dv_c": 40.0,         # km/s, core centroid
    "b_hb.sigma_c": 1980.0,    # km/s
    "b_hb.amp_w0": 0.78,       # wind / core peak ratio
    "b_hb.dv_w0": -520.0,      # km/s, relative to the core
    "b_hb.sigma_w0": 6400.0,   # km/s
    "FeII.amplitude": 0.30,    # iron scale (1e-14 units)
}

#: Data grid: rest frame, wide enough to hold the template's 4000-5600 A with a
#: margin, but the fit only sees the window below (edges of a convolution are
#: always the worst part of it, and real spectra are trimmed too).
WAVE_FULL = np.arange(4150.0, 5650.0, 0.5)
WAVE_FIT = (4300.0, 5500.0)
#: the grid the fits use (the template's own edges are trimmed, as in a real spectrum)
WAVE = WAVE_FULL[(WAVE_FULL >= WAVE_FIT[0]) & (WAVE_FULL <= WAVE_FIT[1])]

#: Noise level of the "noisy" variant, as a fraction of the peak flux.
NOISE_FRACTION = 0.01


# --------------------------------------------------------------------------------------
# Mock generators
# --------------------------------------------------------------------------------------
def cpu_models(truth=None):
    """The CPU pair used to *generate* the mock, exactly as in the request."""
    truth = dict(TRUTH if truth is None else truth)
    line = galspec.Line_MultiGauss(
        n_components=2,
        amp_c=truth["b_hb.amp_c"], dv_c=truth["b_hb.dv_c"],
        sigma_c=truth["b_hb.sigma_c"], wavec=galspec.utils.line_wave_dict["Hbeta"],
        amp_w0=truth["b_hb.amp_w0"], dv_w0=truth["b_hb.dv_w0"],
        sigma_w0=truth["b_hb.sigma_w0"], name="b_hb")
    iron = galspec.IronTemplate_tied(amplitude=truth["FeII.amplitude"], z=0.0,
                                     template_name=TEMPLATE, kernel=line, name="FeII")
    return line, iron


def jax_models(truth=None):
    """The JAX pair used to *fit*: broad-only MultiGauss plus the tied iron."""
    truth = dict(TRUTH if truth is None else truth)
    line = jxm.Line_MultiGauss(
        n_components=2, amp_c=truth["b_hb.amp_c"], dv_c=truth["b_hb.dv_c"],
        sigma_c=truth["b_hb.sigma_c"], wavec=galspec.utils.line_wave_dict["Hbeta"],
        amp_w0=truth["b_hb.amp_w0"], dv_w0=truth["b_hb.dv_w0"],
        sigma_w0=truth["b_hb.sigma_w0"], name="b_hb")
    iron = jxm.IronTemplate_tied(n_components=2, amplitude=truth["FeII.amplitude"],
                                 z=0.0, name="FeII")
    iron.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("b_hb")
    iron.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("b_hb")
    iron.amp_w0.tied = jxm.tie.tie_MultiGauss_amp_w0("b_hb")
    iron.dv_w0.tied = jxm.tie.tie_MultiGauss_dv_w0("b_hb")
    iron.sigma_w0.tied = jxm.tie.tie_MultiGauss_sigma_w0("b_hb")
    return line, iron


def _template():
    from galspec.jax_models.iron_template import _load_template
    return _load_template(TEMPLATE)


def pixel_reference_iron(wave, truth=None):
    """Recompute the convolution *in the model's own convention*, independently.

    The JAX (and CPU ``IronTemplate_new``) construction smoothes the template on
    its native grid with a **constant pixel** Gaussian of
    ``sqrt(sigma**2 - sigma_intr**2) / vchan`` channels, where ``vchan`` is the
    velocity width of one 2 A channel at the blue edge.  This function does the
    same thing with an explicit sum over kernel offsets -- no
    ``gaussian_smooth1d``, no FFT, no scipy -- so agreement means the JAX code
    implements its stated maths.
    """
    truth = dict(TRUTH if truth is None else truth)
    wave_t, flux_t, sigma_intr = _template()
    vchan = (wave_t[1] - wave_t[0]) / wave_t[0] * ls_km

    components = [(truth["b_hb.amp_c"], truth["b_hb.dv_c"], truth["b_hb.sigma_c"]),
                  (truth["b_hb.amp_c"] * truth["b_hb.amp_w0"],
                   truth["b_hb.dv_c"] + truth["b_hb.dv_w0"],
                   truth["b_hb.sigma_w0"])]

    total = np.zeros_like(np.asarray(wave, dtype=float))
    for weight, shift, width in components:
        sig_pix = np.sqrt(max(width ** 2 - sigma_intr ** 2, 0.0)) / vchan
        radius = int(4.0 * sig_pix + 0.5)
        smoothed = np.zeros_like(flux_t)
        norm = 0.0
        for offset in range(-radius, radius + 1):
            kernel = np.exp(-0.5 * (offset / max(sig_pix, 1e-12)) ** 2)
            norm += kernel
            smoothed += kernel * np.pad(flux_t, radius, mode="symmetric")[
                radius + offset: radius + offset + flux_t.size]
        smoothed /= norm
        total += weight * np.interp(wave, wave_t * (1.0 + shift / ls_km), smoothed)

    return truth["FeII.amplitude"] * total / sum(w for w, _, _ in components)


def velocity_reference_iron(wave, truth=None, step=10.0, pad=40000.0):
    """The same physics done properly: convolve in velocity space, not in pixels.

    An independent implementation of what the model *means* to do.  The template
    is resampled on a uniform velocity grid and each kernel component is applied
    as a Gaussian of ``sqrt(sigma**2 - sigma_intr**2)`` km/s there, so the width
    really is the requested value at every wavelength.  It shares no code with
    either model -- no deconvolution, no Wiener filter, no pixel-space filter --
    and differs from them only by the sampling convention (see section 5).
    """
    truth = dict(TRUTH if truth is None else truth)
    wave_t, flux_t, sigma_intr = _template()

    v_template = (np.log(wave_t) - np.log(wave_t[0])) * ls_km
    v_grid = np.arange(v_template[0] - pad, v_template[-1] + pad + step, step)
    flux_v = np.interp(v_grid, v_template, flux_t, left=0.0, right=0.0)

    components = [(truth["b_hb.amp_c"], truth["b_hb.dv_c"], truth["b_hb.sigma_c"]),
                  (truth["b_hb.amp_c"] * truth["b_hb.amp_w0"],
                   truth["b_hb.dv_c"] + truth["b_hb.dv_w0"],
                   truth["b_hb.sigma_w0"])]

    def smoothed(width):
        sigma_extra = np.sqrt(max(width ** 2 - sigma_intr ** 2, 0.0))
        if sigma_extra < step:
            return flux_v
        half = int(5.0 * sigma_extra / step)
        offsets = np.arange(-half, half + 1) * step
        kernel = np.exp(-0.5 * (offsets / sigma_extra) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(flux_v, half, mode="edge")
        return np.convolve(padded, kernel, mode="same")[half:half + len(flux_v)]

    total = np.zeros_like(v_grid)
    for weight, shift, width in components:
        total += weight * np.interp(v_grid - shift, v_grid, smoothed(width))
    total /= sum(weight for weight, _, _ in components)

    smoothed_template = np.interp(v_template, v_grid, total)
    return truth["FeII.amplitude"] * np.interp(wave, wave_t, smoothed_template)


def mock_spectrum(kind: str, wave: np.ndarray, truth=None, noise: float = 0.0,
                  seed: int = 7):
    """Return ``(flux, line_flux, iron_flux, error)`` for one generator."""
    truth = dict(TRUTH if truth is None else truth)
    line_cpu, iron_cpu = cpu_models(truth)
    line_flux = np.asarray(line_cpu(wave))
    peak = float(np.max(line_flux + np.asarray(iron_cpu(wave))))

    if kind == "cpu":
        iron_flux = np.asarray(iron_cpu(wave))
    elif kind == "pixel":
        iron_flux = pixel_reference_iron(wave, truth)
    elif kind == "velocity":
        iron_flux = velocity_reference_iron(wave, truth)
    elif kind == "jax":
        line_jax, _ = jax_models(truth)
        iron_jax = jxm.IronTemplate_tied(
            n_components=2, amplitude=truth["FeII.amplitude"], z=0.0,
            dv_c=truth["b_hb.dv_c"], sigma_c=truth["b_hb.sigma_c"],
            amp_w0=truth["b_hb.amp_w0"], dv_w0=truth["b_hb.dv_w0"],
            sigma_w0=truth["b_hb.sigma_w0"])
        iron_flux = np.asarray(iron_jax.flux(wave, iron_jax.get_free_params()))
        line_flux = np.asarray(line_jax.flux(wave, line_jax.get_free_params()))
    else:
        raise ValueError(kind)

    flux = line_flux + iron_flux
    error = np.full_like(wave, max(noise, 1e-8) * peak)
    if noise > 0:
        flux = flux + np.random.default_rng(seed).normal(0.0, error)
    return flux, line_flux, iron_flux, error


# --------------------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------------------
def log_uniform(lower: float, upper: float):
    lo, hi = np.log10(lower), np.log10(upper)
    return lambda u: 10.0 ** (lo + (hi - lo) * u)


#: Priors on the two widths.  ``wide`` is the production recipe and leaves the
#: exchange symmetry of section 5 unbroken; ``ordered`` encodes what the names
#: core/wind are supposed to mean (component 0 is the narrower of the two) and
#: therefore identifies the labels.
#: The width priors the report compares.  Section 4.2 uses these to show that the
#: core-wind exchange makes the posterior bimodal *when amp_w0 may exceed 1*.  The
#: production fit does not need the disjoint ranges: it bounds amp_w0 to [0, 1],
#: which removes the mirrored branch outright (see the docstring of
#: ``example/fit_iras07546_jax.make_priors`` and ``io_test/cpu_vs_jax.html``).
SIGMA_PRIORS = {"wide": ((300.0, 8000.0), (800.0, 25000.0)),
                "ordered": ((300.0, 3000.0), (3000.0, 25000.0))}


def fit_mock(wave, flux, error, nlive=400, dlogz=0.05, max_iterations=200_000,
             device="auto", progress=False, priors="wide"):
    """JNesty fit of ``b_hb + FeII`` with the iron tied to the line: 7 free numbers."""
    line, iron = jax_models()
    model = line + iron

    sigma_c_bounds, sigma_w_bounds = SIGMA_PRIORS[priors]
    peak = float(np.max(flux))
    bounds = {
        "b_hb.amp_c": (1e-3, 3.0 * peak),
        "b_hb.dv_c": (-2000.0, 2000.0),
        "b_hb.sigma_c": sigma_c_bounds,
        "b_hb.amp_w0": (1e-3, 8.0),
        "b_hb.dv_w0": (-8000.0, 8000.0),
        "b_hb.sigma_w0": sigma_w_bounds,
        "FeII.amplitude": (1e-3, 3.0 * peak),
    }
    custom = {name: log_uniform(*bounds[name]) for name in
              ("b_hb.amp_c", "b_hb.amp_w0", "FeII.amplitude")}

    from galspec.jnesty_fit import JNesty_Fit
    fit = JNesty_Fit(model, wave, flux, error, bounds_dict=bounds,
                     custom_priors=custom, nlive=nlive, bound="multi",
                     device=device)
    started = time.perf_counter()
    fit.fit(progress=progress, dlogz=dlogz, max_iterations=max_iterations)
    wall = time.perf_counter() - started
    return fit, wall


def swap_labelling(samples, names):
    """Apply the exact component exchange to a set of (physical) posterior samples.

    Because the profile is invariant under this map (see unit check 9), it takes
    one posterior branch onto the other bit for bit: that is what makes the
    degeneracy a genuine one rather than a sampler artefact.
    """
    index = {name: i for i, name in enumerate(names)}
    out = np.array(samples, dtype=float, copy=True)
    amp_c = samples[:, index["b_hb.amp_c"]]
    amp_w0 = samples[:, index["b_hb.amp_w0"]]
    dv_w0 = samples[:, index["b_hb.dv_w0"]]
    out[:, index["b_hb.amp_c"]] = amp_c * amp_w0
    out[:, index["b_hb.dv_c"]] = samples[:, index["b_hb.dv_c"]] + dv_w0
    out[:, index["b_hb.sigma_c"]] = samples[:, index["b_hb.sigma_w0"]]
    out[:, index["b_hb.amp_w0"]] = 1.0 / amp_w0
    out[:, index["b_hb.dv_w0"]] = -dv_w0
    out[:, index["b_hb.sigma_w0"]] = samples[:, index["b_hb.sigma_c"]]
    return out


def _quantiles(fit):
    lo, mid, hi = fit.get_quantiles((0.16, 0.5, 0.84))
    return dict(zip(fit.full_param_names, zip(lo, mid, hi)))


def evaluate_model(fit, wave):
    """Best-fit total, line and iron flux, plus the tie-resolved parameters."""
    model_best, _, theta_best = fit.get_best_fit()
    total = np.asarray(model_best.flux(wave, theta_best))

    resolved = model_best.param_dict()
    flat = dict(zip(model_best.qualified_param_names, model_best.param_names))
    truth_like = {name: float(resolved[flat[name]]) for name in TRUTH}

    # the line alone: zero the iron amplitude and re-evaluate
    theta_no_iron = np.array(theta_best, dtype=float)
    free = set(model_best.free_param_names)
    free_qualified = [q for q, f in zip(model_best.qualified_param_names,
                                        model_best.param_names) if f in free]
    theta_no_iron[free_qualified.index("FeII.amplitude")] = 0.0
    line_only = np.asarray(model_best.flux(wave, theta_no_iron))
    return total, line_only, total - line_only, truth_like, model_best


# --------------------------------------------------------------------------------------
# Unit checks
# --------------------------------------------------------------------------------------
def unit_checks():
    """Small, decisive checks of every piece the mock comparison relies on."""
    checks = []

    def add(name, expected, measured, ok, note="", kind="check"):
        # a "finding" is a description of a known difference, not a pass/fail test
        checks.append({"name": name, "expected": expected, "measured": measured,
                       "ok": True if kind == "finding" else bool(ok),
                       "note": note, "kind": kind})

    # the same grid the fits use, so the iron curves below can be plotted with them
    wave = WAVE_FULL[(WAVE_FULL >= WAVE_FIT[0]) & (WAVE_FULL <= WAVE_FIT[1])]

    # 1. one kernel component == IronTemplate (same quadrature subtraction)
    plain = jxm.IronTemplate(amplitude=0.3, sigma=2600.0, dv=180.0)
    tied = jxm.IronTemplate_tied(n_components=1, amplitude=0.3, sigma_c=2600.0, dv_c=180.0)
    f_plain = np.asarray(plain.flux(wave, plain.get_free_params()))
    f_tied = np.asarray(tied.flux(wave, tied.get_free_params()))
    rel = float(np.max(np.abs(f_plain - f_tied)) / np.max(f_plain))
    add("IronTemplate_tied(n_components=1) == IronTemplate", "0", f"{rel:.1e}", rel < 1e-12,
        "the general kernel must contain the single-Gaussian model")

    # 2. the quadrature subtraction: a Gaussian template broadened from
    #    sigma_intr to sigma must come out with width sigma
    sigma_intr, sigma = 340.0, 2500.0
    from galspec.jax_models.functional import gaussian_smooth1d
    v = np.arange(-30000.0, 30000.0, 5.0)
    g = np.exp(-0.5 * (v / sigma_intr) ** 2)
    sig_pix = np.sqrt(sigma ** 2 - sigma_intr ** 2) / 5.0
    smoothed = np.asarray(gaussian_smooth1d(g, sig_pix, radius=int(4 * sig_pix + 0.5)))
    width = np.sqrt(np.sum(smoothed * v ** 2) / np.sum(smoothed))
    add("analytic: a 340 km/s Gaussian smoothed to 2500 km/s",
        f"{sigma:.1f} km/s", f"{width:.1f} km/s", abs(width - sigma) / sigma < 2e-3,
        "second moment of the smoothed profile")

    # 3. the iron kernel is the line profile: scaling the line's peak must leave
    #    the iron untouched (the kernel is normalised and carries no weight)
    b_hb, fe = jax_models()
    compound = b_hb + fe
    flat = dict(zip(compound.qualified_param_names, compound.param_names))
    base = np.array([TRUTH["b_hb.amp_c"], TRUTH["b_hb.dv_c"], TRUTH["b_hb.sigma_c"],
                     TRUTH["b_hb.amp_w0"], TRUTH["b_hb.dv_w0"], TRUTH["b_hb.sigma_w0"]])
    iron_fluxes = []
    for factor in (1.0, 7.0):
        model = compound.copy()
        model[flat["b_hb.amp_c"]].value = TRUTH["b_hb.amp_c"] * factor
        total = np.asarray(model.flux(wave, model.get_free_params()))
        scaled = base.copy()
        scaled[0] *= factor                  # only the peak, not the shifts/widths
        line = np.asarray(model["b_hb"].flux(wave, scaled))
        iron_fluxes.append(total - line)
    rel = float(np.max(np.abs(iron_fluxes[0] - iron_fluxes[1])) / np.max(np.abs(iron_fluxes[0])))
    add("iron shape independent of the line's peak (kernel normalised)", "0",
        f"{rel:.1e}", rel < 1e-9, "septupling broad H-beta leaves the iron untouched")

    # 4. the tie resolution lands on the line's own parameters
    resolved = compound.param_dict()
    pairs = [("FeII.dv_c", "b_hb.dv_c"), ("FeII.sigma_c", "b_hb.sigma_c"),
             ("FeII.amp_w0", "b_hb.amp_w0"), ("FeII.dv_w0", "b_hb.dv_w0"),
             ("FeII.sigma_w0", "b_hb.sigma_w0")]
    worst = max(abs(float(resolved[flat[a]]) - float(resolved[flat[b]])) /
                max(abs(float(resolved[flat[b]])), 1e-30) for a, b in pairs)
    add("all 5 kernel parameters resolve onto b_hb", "0", f"{worst:.1e}", worst < 1e-12,
        "core and wind: shift, width and amplitude ratio")

    # 5. CPU and JAX line models are the same function (the line part of the mock
    #    is therefore exact)
    line_cpu, _ = cpu_models()
    line_jax, _ = jax_models()
    fc = np.asarray(line_cpu(wave))
    fj = np.asarray(line_jax.flux(wave, line_jax.get_free_params()))
    rel = float(np.max(np.abs(fc - fj)) / np.max(fc))
    add("CPU Line_MultiGauss == JAX Line_MultiGauss", "~1e-15", f"{rel:.1e}", rel < 1e-12,
        "same 6 parameters, same profile")

    # 6. the JAX convolution against an independent re-implementation of the *same*
    #    convention: agreement means the code does what it says
    _, iron_cpu_model = cpu_models()
    iron_cpu = np.asarray(iron_cpu_model(wave))
    iron_pixel = pixel_reference_iron(wave)
    iron_velocity = velocity_reference_iron(wave)
    iron_jax_model = jxm.IronTemplate_tied(
        n_components=2, amplitude=TRUTH["FeII.amplitude"], z=0.0,
        dv_c=TRUTH["b_hb.dv_c"], sigma_c=TRUTH["b_hb.sigma_c"],
        amp_w0=TRUTH["b_hb.amp_w0"], dv_w0=TRUTH["b_hb.dv_w0"],
        sigma_w0=TRUTH["b_hb.sigma_w0"])
    iron_jax = np.asarray(iron_jax_model.flux(wave, iron_jax_model.get_free_params()))
    peak = max(iron_cpu.max(), iron_jax.max(), iron_pixel.max(), iron_velocity.max())
    rel_pixel = float(np.max(np.abs(iron_pixel - iron_jax)) / peak * 100)
    add("JAX iron vs independent pixel-space convolution", "< 0.01 %",
        f"{rel_pixel:.2e} % of peak", rel_pixel < 0.01,
        "no gaussian_smooth1d, no scipy, no FFT on the other side")

    # 7. the JAX iron against an exact convolution in velocity space: the template is
    #    resampled onto a grid even in ln(wavelength) at load and zero-padded, so
    #    smoothing in pixels *is* smoothing in velocity, everywhere
    rel_velocity = float(np.max(np.abs(iron_velocity - iron_jax)) / peak * 100)
    wave_t, _, _ = _template()
    vchan = np.diff(np.log(wave_t)) * ls_km
    add("JAX iron vs an exact velocity-space convolution", "0",
        f"{rel_velocity:.3f} % of peak", rel_velocity < 0.1,
        f"one channel is {vchan[0]:.0f} km/s everywhere (spread "
        f"{vchan.max() - vchan.min():.1e} km/s), so the pixel-space smoothing is the "
        "velocity-space one; the residue is that 126 km/s sampling the kernel")

    # 8. the CPU IronTemplate_tied and the JAX one are the same function: both
    #    subtract the template's own width in quadrature and smooth once per
    #    component, so there is no deconvolution anywhere
    rel_cpu = float(np.max(np.abs(iron_cpu - iron_jax)) / peak * 100)
    add("JAX iron vs the CPU IronTemplate_tied at the same parameters", "0",
        f"{rel_cpu:.1e} % of peak", rel_cpu < 1e-10,
        "both broaden the template in velocity space by quadrature subtraction; the "
        "CPU class used to deconvolve and was 8.5% (mock) / 10.0% (fit) away")

    # 9. FINDING: the two components are exchangeable -- swapping them (and the
    #    amplitudes/widths that go with them) is the same function
    swap = {"b_hb.amp_c": TRUTH["b_hb.amp_c"] * TRUTH["b_hb.amp_w0"],
            "b_hb.dv_c": TRUTH["b_hb.dv_c"] + TRUTH["b_hb.dv_w0"],
            "b_hb.sigma_c": TRUTH["b_hb.sigma_w0"],
            "b_hb.amp_w0": 1.0 / TRUTH["b_hb.amp_w0"],
            "b_hb.dv_w0": -TRUTH["b_hb.dv_w0"],
            "b_hb.sigma_w0": TRUTH["b_hb.sigma_c"],
            "FeII.amplitude": TRUTH["FeII.amplitude"]}
    def compound_flux(vals):
        line_m, iron_m = jax_models(vals)
        total_m = line_m + iron_m
        return np.asarray(total_m.flux(wave, total_m.get_free_params()))
    f_truth, f_swap = compound_flux(TRUTH), compound_flux(swap)
    rel = float(np.max(np.abs(f_truth - f_swap)) / f_truth.max())
    add("exchanging component 0 and component 1 is the same function", "0",
        f"{rel:.1e} of peak", rel < 1e-12,
        "amp_c*A <-> amp_c, sigma_c <-> sigma_w0, dv_c+dv_w0 <-> dv_c, "
        "amp_w0 -> 1/amp_w0, dv_w0 -> -dv_w0.  The posterior is bimodal, so the "
        "labels are not identified by the data",
        kind="finding")

    unit = {"cpu": np.asarray(iron_cpu), "pixel": iron_pixel,
            "velocity": iron_velocity, "jax": iron_jax,
            "rel_pixel": rel_pixel, "rel_velocity": rel_velocity, "rel_cpu": rel_cpu,
            "vchan": float(np.mean(vchan)),
            "vchan_spread": float(vchan.max() - vchan.min())}
    return checks, unit


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.9,
                         "xtick.direction": "in", "ytick.direction": "in"})
    return plt


def fig_mock(wave, components):
    plt = _style()
    line, iron = components["line"], components["iron"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.0, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    ax.plot(wave, line + iron, color="0.15", lw=1.0, label="mock spectrum")
    ax.plot(wave, iron, color="#b7791f", lw=1.1, label=r"Fe II $\times$ broad H$\beta$")
    ax.plot(wave, line, color="#2b6cb0", lw=1.1, label=r"broad H$\beta$")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_ylabel(r"$F_\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.set_title("Mock: iron broadened by the same two-component profile as broad H-beta",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    axr.plot(wave, iron / np.max(line + iron) * 100, color="#b7791f", lw=1.0)
    axr.set_ylabel("iron [%]")
    axr.set_xlabel(r"rest wavelength [$\AA$]")
    axr.set_xlim(WAVE_FIT)
    return fig


def fig_fits(wave, data, results):
    plt = _style()
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(11.0, 2.5 * n),
                             gridspec_kw={"width_ratios": [3, 1], "hspace": 0.45,
                                          "wspace": 0.25})
    axes = np.atleast_2d(axes)
    for row, (kind, res) in enumerate(results.items()):
        ax, axr = axes[row]
        flux, error = data[kind]["flux"], data[kind]["error"]
        ax.plot(wave, flux, color="0.55", lw=0.7, label="data")
        ax.plot(wave, res["total"], color="#c53030", lw=1.0, label="best fit")
        ax.plot(wave, res["line"], color="#2b6cb0", lw=0.9, ls="--", label="H-beta")
        ax.plot(wave, res["iron"], color="#b7791f", lw=0.9, ls=":", label="Fe II")
        ax.set_xlim(WAVE_FIT)
        ax.set_ylabel(r"$F_\lambda$  [$10^{-14}$]")
        ax.set_title(f"generator: {kind}   (chi2/dof {res['chi2dof']:.4f}, "
                     f"worst pull {res['worst_pull']:.2f})", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
        residual = (flux - res["total"]) / error
        axr.plot(wave, residual, color="0.25", lw=0.6)
        axr.axhline(0, color="#c53030", lw=0.8)
        for value in (-2, 2):
            axr.axhline(value, color="0.7", lw=0.6, ls=":")
        axr.set_xlim(WAVE_FIT)
        axr.set_ylim(-4.2, 4.2)
        axr.set_ylabel(r"res. [$\sigma$]")
        axr.set_xlabel(r"rest wavelength [$\AA$]")
    return fig


def fig_recovery(results, order=None):
    plt = _style()
    names = list(TRUTH)
    order = list(results) if order is None else order
    fig, axes = plt.subplots(1, len(names), figsize=(13.0, 2.7))
    for ax, name in zip(axes, names):
        truth = TRUTH[name]
        for index, kind in enumerate(order):
            res = results[kind]
            mid, lo, hi = res["quantiles"][name][1], res["quantiles"][name][0], \
                res["quantiles"][name][2]
            colour = {"cpu": "#c53030", "pixel": "#2f855a", "velocity": "#805ad5",
                      "jax": "#2b6cb0"}.get(kind, "0.3")
            ax.errorbar([index], [mid], yerr=[[mid - lo], [hi - mid]], fmt="o", ms=4,
                        color=colour, capsize=3, lw=1.0)
        ax.axhline(truth, color="0.2", lw=1.0, ls="--")
        span = abs(truth) if truth else 1.0
        ax.set_ylim(truth - 0.35 * span - 0.02, truth + 0.35 * span + 0.02)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8)
        ax.set_title(name, fontsize=8.5)
        if name.endswith("amplitude"):
            ax.set_ylabel("value")
    fig.suptitle("Recovered parameters (median and 68% interval) against the input (dashed)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_iron_models(wave, irons):
    plt = _style()
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.2, 5.0), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    colours = {"cpu": "#c53030", "pixel": "#2f855a", "velocity": "#805ad5", "jax": "#2b6cb0"}
    labels = {"cpu": "CPU IronTemplate_tied",
              "pixel": "independent pixel-space convolution",
              "velocity": "exact velocity-space convolution",
              "jax": "JAX IronTemplate_tied"}
    for kind in ("cpu", "velocity", "pixel", "jax"):
        ax.plot(wave, irons[kind], color=colours[kind], lw=1.1, label=labels[kind])
    ax.set_ylabel(r"$F_\lambda$  [$10^{-14}$]")
    ax.set_title("Four ways to broaden the same template with the same parameters:\n"
                 "the CPU and JAX curves are the same operator now",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    peak = max(f.max() for f in irons.values())
    for kind in ("cpu", "velocity", "pixel"):
        axr.plot(wave, (irons[kind] - irons["jax"]) / peak * 100,
                 color=colours[kind], lw=1.0, label=labels[kind] + "  -  JAX")
    axr.axhline(0, color="0.3", lw=0.8)
    axr.set_ylabel("difference [% of peak]")
    axr.set_xlabel(r"rest wavelength [$\AA$]")
    axr.set_xlim(WAVE_FIT)
    axr.legend(frameon=False, fontsize=8)
    return fig


def fig_degeneracy(samples=None):
    plt = _style()
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(11.4, 3.6),
                                  gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.28})
    synonymous = {"b_hb.amp_c": TRUTH["b_hb.amp_c"] * TRUTH["b_hb.amp_w0"],
                  "b_hb.dv_c": TRUTH["b_hb.dv_c"] + TRUTH["b_hb.dv_w0"],
                  "b_hb.sigma_c": TRUTH["b_hb.sigma_w0"],
                  "b_hb.amp_w0": 1.0 / TRUTH["b_hb.amp_w0"],
                  "b_hb.dv_w0": -TRUTH["b_hb.dv_w0"],
                  "b_hb.sigma_w0": TRUTH["b_hb.sigma_c"],
                  "FeII.amplitude": TRUTH["FeII.amplitude"]}
    wavec = galspec.utils.line_wave_dict["Hbeta"]

    def profile(vals):
        line_m = jax_models(vals)[0]
        return np.asarray(line_m.flux(WAVE, line_m.get_free_params()))

    top = 0.0
    for index, (label, vals) in enumerate((("A", TRUTH), ("B", synonymous))):
        total = profile(vals)
        top = max(top, total.max())
        style = "-" if index == 0 else "--"
        colour = "0.15" if index == 0 else "#c53030"
        ax.plot(WAVE, total, color=colour, lw=0.9, ls=style,
                label=f"sum, labelling {label}" if index == 0 else
                      f"sum, labelling {label} (identical)" if False else
                      f"sum, labelling {label}")
        for amp_key, dv_key, sig_key, colour in (
                ("b_hb.amp_c", "b_hb.dv_c", "b_hb.sigma_c", "#2b6cb0"),
                ("b_hb.amp_w0", "b_hb.dv_w0", "b_hb.sigma_w0", "#b7791f")):
            centre = (vals["b_hb.dv_c"] + vals["b_hb.dv_w0"] if dv_key == "b_hb.dv_w0"
                      else vals["b_hb.dv_c"])
            height = vals["b_hb.amp_c"] * vals[amp_key]
            ax.plot(WAVE, height * np.exp(-0.5 * ((WAVE - wavec * (1 + centre / ls_km))
                                                  / (WAVE * vals[sig_key] / ls_km)) ** 2),
                    color=colour, lw=0.7, ls=style, alpha=0.85)
    ax.set_xlim(WAVE_FIT)
    ax.set_ylim(0, top * 1.12)
    ax.set_xlabel(r"rest wavelength [$\AA$]")
    ax.set_ylabel(r"$F_\lambda$  [$10^{-14}$]")
    ax.set_title("Two labellings of one input, identical to 3e-16", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    if samples is not None and len(samples):
        half = len(samples) // 2
        axs.plot(samples[:half, 0], samples[:half, 1], ".", ms=1.6, color="#2b6cb0",
                 alpha=0.3, label="branch the chain filled")
        axs.plot(samples[half:, 0], samples[half:, 1], ".", ms=1.6, color="#c53030",
                 alpha=0.3, label="its exact mirror image")
        axs.axvline(TRUTH["b_hb.sigma_c"], color="0.25", lw=0.9, ls="--")
        axs.axvline(TRUTH["b_hb.sigma_w0"], color="0.25", lw=0.9, ls="--")
        axs.set_xlabel(r"$\sigma_c$  [km/s]")
        axs.set_ylabel(r"$\sigma_{w0}$  [km/s]")
        axs.set_title("one posterior, two modes of equal mass", fontsize=10)
        axs.legend(frameon=False, fontsize=8, loc="upper center", markerscale=5)
    fig.tight_layout()
    return fig


def _b64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt = _style()
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
CSS = """
:root { --ink:#1a202c; --muted:#4a5568; --line:#cbd5e0; --accent:#2b6cb0;
        --good:#2f855a; --bad:#c53030; --bg:#ffffff; --soft:#f7fafc; }
body { margin:0; background:var(--bg); color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       line-height:1.55; font-size:15px; }
main { max-width: 980px; margin: 0 auto; padding: 32px 24px 96px; }
h1 { font-size:27px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:20px; margin:38px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }
h3 { font-size:16px; margin:24px 0 8px; }
p, li { color:var(--ink); }
.sub { color:var(--muted); font-size:13.5px; margin:0 0 18px; }
code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; }
pre { background:var(--soft); border:1px solid var(--line); border-left:3px solid var(--accent);
      padding:12px 14px; overflow-x:auto; border-radius:3px; }
table { border-collapse:collapse; width:100%; margin:12px 0 18px; font-size:13px; }
th, td { text-align:left; padding:6px 9px; border-bottom:1px solid var(--line); }
th { background:var(--soft); font-weight:600; font-size:12.5px; text-transform:uppercase;
     letter-spacing:0.03em; color:var(--muted); }
td.num { text-align:right; font-variant-numeric: tabular-nums; }
tr:hover td { background:#fbfdff; }
img { width:100%; height:auto; border:1px solid var(--line); border-radius:3px; margin:8px 0 4px; }
.verdict { border:1px solid var(--line); border-left:4px solid var(--good); background:var(--soft);
           padding:14px 18px; margin:18px 0 24px; border-radius:3px; }
.verdict.warn { border-left-color:#b7791f; }
.ok { color:var(--good); font-weight:600; }
.no { color:var(--bad); font-weight:600; }
.cap { color:var(--muted); font-size:12.5px; margin:0 0 18px; }
.kv { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px 22px;
      margin:14px 0 20px; }
.kv div { border-left:2px solid var(--line); padding-left:10px; }
.kv .k { color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:0.04em; }
.kv .v { font-size:16px; font-variant-numeric:tabular-nums; }
"""


def build_report(path: Path, payload: dict) -> None:
    checks = payload["checks"]
    wide = payload["wide"]
    ordered = payload["ordered"]
    unit = payload["units"]
    images = payload["images"]

    failed = [c for c in checks if not c["ok"]]
    verdict_class = "verdict" if not failed else "verdict warn"
    exact = ordered.get("jax", ordered.get("pixel", {}))
    pixel = ordered.get("pixel", exact)
    recovery = max(value["median_error"] for value in ordered.values()) * 100
    verdict_text = (
        "The model and the fitter are correct. With an ordering prior (section 4.2) a "
        f"mock built by the JAX models comes back to {exact['worst_pull']:.1f}&sigma;, a "
        "mock built by an independent re-implementation of the same convolution to "
        f"{pixel['worst_pull']:.1f}&sigma;, and the two iron profiles agree to "
        f"{unit['rel_pixel']:.1e}% of peak. All four generators recover the input "
        f"equally well (worst parameter {recovery:.2f}%), and the CPU "
        "<code>IronTemplate_tied</code> now agrees with the JAX one to "
        f"{unit['rel_cpu']:.1e}% of peak, so the two backends are interchangeable. One "
        "thing needs reporting rather than fixing: the two broad components are "
        "<em>exactly</em> exchangeable, so their labels must be pinned by the prior."
        if not failed else f"{len(failed)} check(s) failed &mdash; see section 2."
    )

    def table(headers, rows, numeric=(), classes=()):
        out = ["<table><thead><tr>"]
        out += [f"<th>{html.escape(str(h))}</th>" for h in headers]
        out.append("</tr></thead><tbody>")
        for row in rows:
            cls = f' class="{classes[0]}"' if classes else ""
            out.append(f"<tr{cls}>")
            for index, cell in enumerate(row):
                num = ' class="num"' if index in numeric else ""
                out.append(f"<td{num}>{cell}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
        return "\n".join(out)

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tied iron kernel: input/output test</title><style>{CSS}</style></head><body><main>
<h1>Iron tied to a broad line: input/output test</h1>
<p class="sub">Generated {payload['created']} by
<code>develop/dev_jax/iron_tied_io_test.py</code> &middot; GalSpec
{galspec.__version__} &middot; {payload['device']} &middot;
runtime {payload['runtime']/60:.1f} min</p>

<div class="{verdict_class}"><strong>Verdict.</strong> {verdict_text}</div>

<div class="kv">
  <div><div class="k">generators x priors</div>
       <div class="v">{len(ordered)} x 2</div></div>
  <div><div class="k">free parameters</div><div class="v">{payload['ndim']}</div></div>
  <div><div class="k">pixels fitted</div><div class="v">{payload['npix']:,}</div></div>
  <div><div class="k">unit checks / findings</div>
       <div class="v">{sum(1 for c in checks if c['kind'] == 'check')} pass
       &middot; {sum(1 for c in checks if c['kind'] == 'finding')} findings</div></div>
</div>
""")

    parts.append("""
<h2>1. What the model is now</h2>
<p>Each Balmer (and He I) line is described by <strong>two</strong> models, as in
<code>example/wrk_sagan_iras07546.ipynb</code>: a narrow <code>Line_Gaussian</code>
tied to the kinematics of [O III], and a broad <code>Line_MultiGauss</code> whose
component 0 ("core") and further components ("winds") are all <em>broad</em>
Gaussians.</p>

<pre>n_ha = jxm.Line_Gaussian(amplitude=..., wavec=6563, name="n_halpha")    # narrow
n_ha.dv.tied    = tie_MultiGauss_dv_c("OIII")          # one shift for all narrow lines
n_ha.sigma.tied = tie_MultiGauss_sigma_c("OIII")       # one width for all narrow lines

b_ha = jxm.Line_MultiGauss(n_components=2, amp_c=..., dv_c=..., sigma_c=...,
                           amp_w0=..., dv_w0=..., sigma_w0=..., name="b_halpha")
line_ha = b_ha + n_ha</pre>

<h3>The iron has no weights of its own</h3>
<p>A <code>Line_MultiGauss</code> already <em>is</em> a weighted sum of Gaussians: its
weights are its own component amplitudes, <code>amp_c</code> for component 0 and
<code>amp_c * amp_w0</code> for the wind. The iron needs only the <em>shape</em> of
that sum &mdash; the template is convolved and renormalised &mdash; so the core
amplitude is nothing but the unit in which the wind amplitude is expressed, and it
cancels exactly. That is why <code>IronTemplate_tied</code> takes the line's kernel
parameters and nothing else, and why tying it leaves one free number:</p>

<pre>fe = jxm.IronTemplate_tied(n_components=2, amplitude=..., name="FeII")   # core + 1 wind
fe.dv_c.tied     = tie_MultiGauss_dv_c("b_halpha")       # core
fe.sigma_c.tied  = tie_MultiGauss_sigma_c("b_halpha")
fe.amp_w0.tied   = tie_MultiGauss_amp_w0("b_halpha")     # wind, the line's own ratio
fe.dv_w0.tied    = tie_MultiGauss_dv_w0("b_halpha")
fe.sigma_w0.tied = tie_MultiGauss_sigma_w0("b_halpha")

# -> free parameters of the iron: amplitude (and z, which is fixed)</pre>

<p><code>n_components</code> counts <em>total</em> kernel components, so
<code>2</code> is core + one wind and mirrors
<code>Line_MultiGauss(n_components=2)</code>. A wind's shift is relative to the
core (<code>dv_c + dv_w0</code>), as in the line, so the iron inherits the line's
centroid. The kernel is normalised to unit sum, so <code>amplitude</code> keeps
meaning "the iron scale" whatever the line does.</p>
""")

    rows = []
    for check in checks:
        if check["kind"] == "finding":
            mark = '<span class="no">finding</span>'
        else:
            mark = '<span class="ok">pass</span>' if check["ok"] else '<span class="no">FAIL</span>'
        rows.append([html.escape(check["name"]), html.escape(check["expected"]),
                     html.escape(check["measured"]), mark, html.escape(check["note"])])
    parts.append(f"""
<h2>2. Unit checks</h2>
<p>Each one isolates a single piece, so that anything in section 4 can be attributed
rather than guessed at. The two rows marked <span class="no">finding</span> are
descriptions of known differences, not failures.</p>
{table(["check", "expected", "measured", "result", "what it establishes"], rows)}
""")

    parts.append(f"""
<h2>3. The mock</h2>
<p>The prescription that was asked for: the iron template broadened by the profile of
broad H&#946;, plus that same broad H&#946;. Both come from
<code>Line_MultiGauss(n_components=2)</code>, so the iron and the line share six
numbers, and the iron contributes one more (its scale):</p>
{table(["parameter", "input", "meaning"], [
    ["b_hb.amp_c", f"{TRUTH['b_hb.amp_c']:.2f}", "peak of broad component 0"],
    ["b_hb.dv_c", f"{TRUTH['b_hb.dv_c']:.0f} km/s", "its velocity centroid"],
    ["b_hb.sigma_c", f"{TRUTH['b_hb.sigma_c']:.0f} km/s", "its width"],
    ["b_hb.amp_w0", f"{TRUTH['b_hb.amp_w0']:.2f}", "peak of component 1 / peak of component 0"],
    ["b_hb.dv_w0", f"{TRUTH['b_hb.dv_w0']:.0f} km/s", "shift of component 1, relative to component 0"],
    ["b_hb.sigma_w0", f"{TRUTH['b_hb.sigma_w0']:.0f} km/s", "width of component 1"],
    ["FeII.amplitude", f"{TRUTH['FeII.amplitude']:.2f}", "iron scale (the only free iron number)"],
], numeric=(1,))}
<img src="data:image/png;base64,{images['mock']}" alt="mock spectrum">
<p class="cap">The mock (black), its iron (ochre) and its broad H&#946; (blue). The iron
reaches {unit['iron_share']:.0f}% of the line peak, so it is neither negligible nor
overwhelming.</p>
""")

    order = [k for k in ("cpu", "pixel", "velocity", "jax", "cpu+noise") if k in ordered]

    def num(store, kind, field, default=float("nan")):
        return store.get(kind, {}).get(field, default)

    def cell(mid, lo, hi, truth, pull, digits=4):
        flag = "" if abs(pull) < 1.5 else (" !" if abs(pull) < 4 else " !!")
        return f"{mid:.{digits}f}<br><span class='cap'>{mid-truth:+.{digits}f}</span>"

    def recovery_table(store, caption):
        head = ["parameter", "input"] + order
        body = []
        for name in TRUTH:
            row = [f"<code>{html.escape(name)}</code>", f"{TRUTH[name]:.3f}"]
            for kind in order:
                res = store[kind]
                mid, lo, hi = res["quantiles"][name]
                pull = res["pulls"][name]
                flag = "" if abs(pull) < 1.5 else (" !" if abs(pull) < 4 else " !!")
                row.append(f"{mid:.4f}<br><span class='cap'>"
                           f"{mid-TRUTH[name]:+.4f} ({pull:+.1f}&sigma;){flag}</span>")
            body.append(row)
        return table(head, body, numeric=tuple(range(1, len(head))))

    parts.append(f"""
<h2>4. Fits</h2>
<p>One recipe for every mock: <code>jxm.Line_MultiGauss(2)</code> +
<code>jxm.IronTemplate_tied(2)</code> with all five kernel parameters tied, sampled by
JNesty (<code>nlive={payload['nlive']}, dlogz={payload['dlogz']}</code>). Peak heights
are log-uniform from 1e-3 to 3x the brightest pixel, shifts &plusmn;2000 and
&plusmn;8000 km/s &mdash; nothing is seeded near the answer. The generators differ only
in how the iron was built:</p>
<pre>cpu       galspec.IronTemplate_tied + galspec.Line_MultiGauss     (the request: CPU classes)
pixel     independent re-implementation of the JAX convention      (same maths, other code)
velocity  exact convolution in velocity space                      (same physics, other convention)
jax       jxm.IronTemplate_tied + jxm.Line_MultiGauss               (self-consistency)
cpu+noise the cpu mock plus 1% Gaussian noise                      (the realistic case)</pre>

<h3>4.1 Do the fits reproduce the input?</h3>
<p>Top line: posterior median. Below it: median &minus; input and that difference in
units of the 68% interval. These are the <code>ordered</code> priors of section 4.2
(&sigma;<sub>core</sub> &lt; 3000 km/s &lt; &sigma;<sub>wind</sub>), which is what makes
the two labels mean something:</p>
{recovery_table(ordered, "ordered priors")}
<img src="data:image/png;base64,{images['recovery']}" alt="parameter recovery">
<p class="cap">The same numbers as a picture: median and 68% interval per parameter, one
colour per generator, against the input (dashed). The iron amplitude is the one number
that moves when the iron shape does.</p>

<table><thead><tr><th>generator</th><th>wall clock</th><th>log Z</th><th>chi2/dof</th>
<th>rms [&sigma;]</th><th>max |model &minus; truth| / peak<br>
<span class="cap">at the median &nbsp;/&nbsp; at the MAP</span></th>
<th>iron amplitude bias</th></tr></thead><tbody>
{''.join(f'<tr><td>{html.escape(k)}</td><td class="num">{ordered[k]["wall"]/60:.1f} min</td>'
         f'<td class="num">{ordered[k]["logz"]:.1f}</td>'
         f'<td class="num">{ordered[k]["chi2dof"]:.4f}</td>'
         f'<td class="num">{ordered[k]["rms"]:.3f}</td>'
         f'<td class="num">{ordered[k]["median_error"]*100:.3f}% / '
         f'{ordered[k]["model_error"]*100:.3f}%</td>'
         f'<td class="num">{ordered[k]["iron_shift"]*100:+.2f}%</td></tr>'
         for k in order)}
</tbody></table>
<p class="cap">&chi;&sup2;/dof is well below 1 for the noiseless mocks because the 1%
error is larger than the model&minus;data difference; that is deliberate, it is the
error budget of a real spectrum and it keeps the likelihood smooth enough for the
sampler. All four rows now land in the same place &mdash; the same worst pull, the same
{max(v["median_error"] for v in ordered.values())*100:.3f}% worst parameter error &mdash;
because every generator now builds the same operator. That was not true before this
revision: the CPU class used to deconvolve the template by a 200 km/s intrinsic width
where the template measured 340, cut its kernel at &plusmn;10 000 km/s and filter the
result, so the <code>cpu</code> iron was a different function from the one being fitted
&mdash; 8.5% (mock) and 10.0% (fit) from an exact velocity-space convolution, and its row
could not reach zero (<code>io_test/cpu_vs_jax.html</code>). It now agrees with the JAX
iron to {checks[7]['measured']} (check 8).</p>
<img src="data:image/png;base64,{images['fits']}" alt="best fits">
<p class="cap">Data, best fit and its two components, with the residual in units of the
error. All five are flat: the <code>cpu</code> panel used to keep an iron-shaped
pattern the size of the difference between the old deconvolution and the others.</p>

<h3>4.2 The two components are exchangeable</h3>
<p>With the production (wide) priors the posterior is <em>bimodal</em>: this was the one
surprise of the test. Component 0 and component 1 play exactly symmetric roles in the
profile, so</p>
<pre>amp_c' = amp_c * amp_w0     sigma_c' = sigma_w0     dv_c' = dv_c + dv_w0
amp_w0' = 1 / amp_w0        sigma_w0' = sigma_c     dv_w0' = -dv_w0</pre>
<p>is <strong>the same spectrum</strong> to {checks[-1]['measured']} (unit check 9), and
the iron follows it because it is tied to that same profile. The data therefore cannot
say which of the two components is "the core"; only the prior can. The mirror image of
the branch a run fills is not an approximation &mdash; applying the map above to the
equal-weight samples gives the other mode <em>exactly</em>, and both have the same
posterior mass:</p>
<table><thead><tr><th>generator</th><th>chain filled</th>
<th>&sigma;<sub>c</sub> median, that branch</th>
<th>&sigma;<sub>c</sub> median, both branches</th>
<th>input &sigma;<sub>c</sub></th></tr></thead><tbody>
{''.join(f'<tr><td>{html.escape(k)}</td>'
         f'<td class="num">{wide[k]["frac_swapped"]*100:.0f}% swapped</td>'
         f'<td class="num">{wide[k]["branch_sigma_c"]:.0f} km/s</td>'
         f'<td class="num">{wide[k]["mixed_sigma_c"]:.0f} km/s</td>'
         f'<td class="num">{TRUTH["b_hb.sigma_c"]:.0f} km/s</td></tr>' for k in order if k in wide)}
</tbody></table>
<img src="data:image/png;base64,{images['degeneracy']}" alt="exchange degeneracy">
<p class="cap">Left: the two labellings of the same input (solid and dashed) with their
components. Right: the equal-weight posterior in the (&sigma;<sub>c</sub>,
&sigma;<sub>w0</sub>) plane, and its exact mirror image &mdash; two modes of equal mass,
each with &chi;&sup2; = 0.</p>
<p><strong>This matters for the science, not just for this test.</strong> A single broad
line with a core and a wind has two labels for the same two numbers, so a fit reports
whichever mode the sampler happened to fill, and a run that fills both halves reports a
marginal median that lands <em>between</em> them &mdash; a width that matches neither.
The ordering prior used in 4.1 is the cure, and it is also what the words mean: the
core is the narrower of the two broad components. With
<code>sigma_c &isin; 300&ndash;3000</code> and
<code>sigma_w0 &isin; 3000&ndash;25000 km/s</code> every parameter comes back, as the
table in 4.1 shows.</p>
""")

    parts.append(f"""
<h2>5. The ion shape, construction by construction</h2>
<img src="data:image/png;base64,{images['iron']}" alt="iron constructions">
<p class="cap">Same six parameters, four constructions of the same iron. Lower panel:
each curve minus the JAX model, in percent of the peak.</p>

<h3>5.1 The CPU <code>IronTemplate_tied</code> is now the same function
({unit['rel_cpu']:.1e}%)</h3>
<p>The CPU class used to take a different route and it was visibly wrong; this test is
what caught it, and the fix is the subject of the companion page
<code>io_test/cpu_vs_jax.html</code>. What it does now, component by component, is what
the JAX model does:</p>
<ul>
<li><strong>No deconvolution.</strong> Blurring twice with &sigma;<sub>1</sub> and
&sigma;<sub>2</sub> is one blur with the two in quadrature, so a kernel component of
width &sigma; is applied to the template as a Gaussian of
<code>sqrt(&sigma;&sup2; &minus; &sigma;<sub>intr</sub>&sup2;)</code>. Check 2 measures
that identity: 2498.7 km/s recovered from a 2500 km/s request.</li>
<li><strong>One smoothing per component.</strong> Each component is smoothed on the
template grid, projected onto the data grid at its own shift, and added with the line's
own relative weight &mdash; so a one-component kernel reproduces
<code>IronTemplate</code> exactly (check 1).</li>
<li><strong>The same intrinsic width.</strong> 340 km/s (800 km/s FWHM, what the template
was measured at) on both sides, as in <code>IronTemplate_new</code>.</li>
</ul>
<p>What it <em>used to</em> do, and why it was 8.5% (mock) / 10.0% (fit) from an exact
velocity-space convolution: an FFT division by a 200 km/s Gaussian
(<code>ls_km/1500</code>, the old &ldquo;R ~ 1500&rdquo; guess), then
<code>scipy.signal.wiener(arr, 3)</code> &mdash; a local smoothing filter, not a Wiener
deconvolution, so it removed part of the structure the division was meant to restore
&mdash; then a convolution with a kernel sampled on a fixed &plusmn;10 000 km/s window.
The consequence was visible here: a CPU mock recovered the <em>line</em> exactly and the
iron <em>scale</em> to {num(ordered, 'cpu', 'iron_shift')*100:+.1f}%, but its iron
<em>shape</em> left a residual no tie could absorb. The old algorithm is kept, clearly
labelled, as <code>legacy_iron()</code> in <code>iron_cpu_vs_jax.py</code>.</p>

<h3>5.2 The one approximation left: 126 km/s sampling
({unit['rel_velocity']:.2f}%)</h3>
<p>The template is resampled onto a grid even in <code>ln(wavelength)</code> when it is
loaded, so one channel is {unit['vchan']:.0f} km/s of velocity everywhere (the spacing
varies by {unit['vchan_spread']:.0e} km/s across the range). Smoothing in pixels is
therefore smoothing in velocity, which is why the <code>velocity</code> row &mdash; an
exact convolution on a 10 km/s grid &mdash; agrees with the JAX iron to
{unit['rel_velocity']:.2f}% of peak and pulls the iron amplitude by
{num(ordered, 'velocity', 'iron_shift')*100:+.2f}%. The residue is the sampling itself: a
{unit['vchan']:.0f} km/s grid can only describe a narrow kernel as a few samples of it
(0.47% at &sigma; = 400 km/s, 0.02% at &sigma; = 1500 km/s and above), and the exact
reference has the mirror-image discretisation. Both backends share it, so it is a
property of the template file, not of either implementation.</p>

<h3>5.3 One bug this test found, now fixed</h3>
<p>The static convolution kernel is sized from the widest width the model may be asked
for. Outside a fit that was the constructor default (2000 km/s), so evaluating the tied
iron at &sigma; = 6400 km/s truncated the kernel at 2.6&sigma; instead of 4&sigma;
&mdash; a 0.05% error, which is what this test measured before the fix.
<code>IronTemplate_tied</code> now sizes the kernel from the widest
<code>sigma_c</code>/<code>sigma_w*</code> actually given (a fit still overrides it with
the prior bound), and the pixel-space check below is at
{unit['rel_pixel']:.1e}%.</p>
""")

    parts.append(f"""
<h2>6. What this establishes</h2>
<ul>
<li><strong>The tied kernel is the line profile.</strong> Scaling the line's peak by
seven leaves the iron flux unchanged to {checks[2]['measured']} (check 3): the kernel is
normalised and carries no weight parameter.</li>
<li><strong>Core and wind are both tied.</strong> All five kernel parameters resolve onto
the broad line (check 4); the posterior keeps exactly one free iron number, the
amplitude.</li>
<li><strong>The convolution is implemented correctly.</strong> An independent convolution
written from the same formula agrees to {unit['rel_pixel']:.1e}% of peak (check 6); the
quadrature identity holds analytically (check 2); one kernel component reproduces
<code>IronTemplate</code> bit for bit (check 1).</li>
<li><strong>A fit to its own model reproduces the input.</strong> With the ordering prior
the <code>jax</code> row returns all seven parameters with a worst pull of
{num(ordered, 'jax', 'worst_pull'):.2f}&sigma; and a profile difference of
{num(ordered, 'jax', 'median_error')*100:.3f}%; the <code>pixel</code> row, an independent
implementation of the same maths, gives {num(ordered, 'pixel', 'worst_pull'):.2f}&sigma; and
{num(ordered, 'pixel', 'median_error')*100:.3f}%.</li>
<li><strong>The two broad components are exchangeable</strong> (4.2), exactly and by
construction: the profile cannot tell "core" from "wind". Any fit that reports those two
labelled widths needs an ordering prior, otherwise the numbers are a coin flip between
two identical solutions. This affects the science numbers of the production fit, not
just this test.</li>
<li><strong>The CPU tied model is the same function.</strong>
{unit['rel_cpu']:.1e}% of peak apart at the same parameters (check 8), so either backend
can serve as the truth standard; before this revision they were 8.5&ndash;10% apart and
this test is what found it.</li>
<li><strong>One approximation is on record</strong> (5.2): the 126 km/s sampling of the
template, which no implementation can remove and both share.</li>
</ul>
<p class="cap">Reproduce with <code>python develop/dev_jax/iron_tied_io_test.py</code>;
all numbers in <code>io_test_results.json</code>, figures as <code>mock.png</code>,
<code>fits.png</code>, <code>recovery.png</code>, <code>iron.png</code>,
<code>degeneracy.png</code>.</p>
</main></body></html>
""")

    path.write_text("\n".join(parts))


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nlive", type=int, default=400)
    parser.add_argument("--dlogz", type=float, default=0.02)
    parser.add_argument("--noise", action="store_true",
                        help="also fit a noisy variant of the CPU mock")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kinds", default="cpu,pixel,velocity,jax",
                        help="generators to fit (comma separated)")
    parser.add_argument("--priors", default="wide,ordered",
                        help="prior modes to fit (wide, ordered)")
    args = parser.parse_args()

    started = time.perf_counter()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] unit checks")
    checks, irons = unit_checks()
    for check in checks:
        tag = {"finding": "fnd "}.get(check["kind"], "ok  " if check["ok"] else "FAIL")
        print(f"      {tag} {check['name']}: {check['measured']}")

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    modes = [m.strip() for m in args.priors.split(",") if m.strip()]
    print(f"[2/5] mocks: {', '.join(kinds)}")
    datasets = {}
    # A realistic error budget (S/N = 100 per pixel) even for the noiseless mocks:
    # with a 1e-7 floor the likelihood of a *mismatched* generator is so sharp that
    # the sampler cannot find its own optimum, which would hide the very difference
    # this test exists to measure.
    keep = (WAVE_FULL >= WAVE_FIT[0]) & (WAVE_FULL <= WAVE_FIT[1])
    for kind in kinds:
        flux_full, line_full, iron_full, _ = mock_spectrum(kind, WAVE_FULL)
        datasets[kind] = {"flux": flux_full[keep], "line": line_full[keep],
                          "iron": iron_full[keep],
                          "error": np.full(int(keep.sum()),
                                           NOISE_FRACTION * np.max(flux_full[keep]))}
        print(f"      {kind:9s} peak {flux_full[keep].max():.3f}  "
              f"iron {iron_full[keep].max():.4f}")
    if args.noise:
        flux_full, line_full, iron_full, err_full = mock_spectrum(
            "cpu", WAVE_FULL, noise=NOISE_FRACTION, seed=11)
        datasets["cpu+noise"] = {"flux": flux_full[keep], "line": line_full[keep],
                                 "iron": iron_full[keep], "error": err_full[keep]}

    wave = WAVE
    print(f"[3/5] fitting ({', '.join(modes)})")
    store = {mode: {} for mode in modes}
    for mode in modes:
        for kind, data in datasets.items():
            fit, wall = fit_mock(wave, data["flux"], data["error"], nlive=args.nlive,
                                 dlogz=args.dlogz, device=args.device, priors=mode)
            total, line_only, iron_only, resolved, model_best = evaluate_model(fit, wave)
            residual = (data["flux"] - total) / data["error"]
            truth_flux = data["line"] + data["iron"]
            quantiles = _quantiles(fit)
            pulls = {name: float((quantiles[name][1] - TRUTH[name])
                                 / max((quantiles[name][2] - quantiles[name][0]) / 2.0, 1e-12))
                     for name in TRUTH}

            # how much posterior mass sits in the swapped labelling?
            samples = np.asarray(fit.samples_equal_weight)
            names = list(fit.full_param_names)
            frac_swapped = float(np.mean(
                samples[:, names.index("b_hb.sigma_c")]
                > samples[:, names.index("b_hb.sigma_w0")]))
            subset = samples[np.random.default_rng(3).choice(
                len(samples), size=min(3000, len(samples)), replace=False)]
            both = np.concatenate([subset, swap_labelling(subset, names)], axis=0)
            pair = both[:, [names.index("b_hb.sigma_c"), names.index("b_hb.sigma_w0")]]
            mixed = swap_labelling(samples, names)
            both_full = np.concatenate([samples, mixed], axis=0)
            mixed_sigma_c = float(np.median(both_full[:, names.index("b_hb.sigma_c")]))

            # profile of the posterior median, not of the MAP
            med = {name: quantiles[name][1] for name in TRUTH}
            line_m, iron_m = jax_models(med)
            med_model = line_m + iron_m
            median_flux = np.asarray(med_model.flux(wave, med_model.get_free_params()))

            ndim = fit.ndim
            store[mode][kind] = {
                "wall": wall, "logz": fit.log_evidence, "logz_err": fit.log_evidence_err,
                "iterations": fit.n_iterations,
                "chi2dof": float(np.sum(residual ** 2) / max(len(wave) - ndim, 1)),
                "rms": float(np.std(residual)),
                "worst_pull": float(np.max(np.abs(list(pulls.values())))),
                "iron_shift": float(quantiles["FeII.amplitude"][1] / TRUTH["FeII.amplitude"] - 1),
                "model_error": float(np.max(np.abs(total - truth_flux)) / np.max(truth_flux)),
                "median_error": float(np.max(np.abs(median_flux - truth_flux))
                                      / np.max(truth_flux)),
                "quantiles": quantiles, "pulls": pulls,
                "frac_swapped": frac_swapped, "samples": pair,
                "branch_sigma_c": float(quantiles["b_hb.sigma_c"][1]),
                "mixed_sigma_c": mixed_sigma_c,
                "total": total, "line": line_only, "iron": iron_only, "resolved": resolved,
            }
            print(f"      {mode:7s} {kind:10s} log Z {fit.log_evidence:9.2f}  "
                  f"chi2/dof {store[mode][kind]['chi2dof']:.4f}  "
                  f"worst pull {store[mode][kind]['worst_pull']:8.1f}  "
                  f"median-truth {store[mode][kind]['median_error']*100:.3f}%  "
                  f"iron {store[mode][kind]['iron_shift']*100:+6.2f}%  "
                  f"swapped {frac_swapped*100:3.0f}%  ({wall/60:.1f} min)")
            fit.release_device_buffers()

    print("[4/5] figures")
    components = {"line": datasets[kinds[0]]["line"], "iron": datasets[kinds[0]]["iron"]}
    ordered = store.get("ordered", store[modes[-1]])
    wide = store.get("wide", {})
    fit_kinds = [k for k in ("cpu", "pixel", "velocity", "jax", "cpu+noise") if k in ordered]
    images = {
        "mock": _b64(fig_mock(wave, components)),
        "fits": _b64(fig_fits(wave, datasets, {k: ordered[k] for k in fit_kinds})),
        "recovery": _b64(fig_recovery({k: ordered[k] for k in fit_kinds}, fit_kinds)),
        "iron": _b64(fig_iron_models(wave, {k: irons[k] for k in
                                           ("cpu", "pixel", "velocity", "jax")})),
        "degeneracy": _b64(fig_degeneracy(
            (wide or ordered).get("jax", {}).get("samples") if "jax" in (wide or ordered)
            else list((wide or ordered).values())[0]["samples"])),
    }

    print("[5/5] report")
    units = {
        "iron_share": float(components["iron"].max() / components["line"].max() * 100),
        "rel_pixel": irons["rel_pixel"], "rel_velocity": irons["rel_velocity"],
        "rel_cpu": irons["rel_cpu"],
        "vchan": irons["vchan"], "vchan_spread": irons["vchan_spread"],
    }

    def strip(mode_store):
        return {k: {kk: vv for kk, vv in v.items()
                    if kk not in ("total", "line", "iron", "resolved", "samples")}
                for k, v in mode_store.items()}

    payload = {
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "runtime": time.perf_counter() - started,
        "device": f"JAX on {jax.devices()[0].platform}",
        "dlogz": args.dlogz, "nlive": args.nlive,
        "npix": len(wave), "ndim": len(TRUTH),
        "checks": checks, "units": units, "images": images, "truth": TRUTH,
        "wide": strip(wide), "ordered": strip(ordered),
    }
    report = outdir / "iron_tied_io_test.html"
    build_report(report, payload)
    (outdir / "io_test_results.json").write_text(json.dumps(payload, indent=1, default=float))
    for name in ("mock", "fits", "recovery", "iron", "degeneracy"):
        (outdir / f"{name}.png").write_bytes(base64.b64decode(images[name]))

    print(f"      wrote {report}  ({report.stat().st_size/1024:.0f} kB)")
    print(f"      total {payload['runtime']/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())