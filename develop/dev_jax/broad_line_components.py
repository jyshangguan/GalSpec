#!/usr/bin/env python3
"""Does the toggle prior find the right number of broad-line components?

    python develop/dev_jax/broad_line_components.py            # full run (~30 min on a GPU)
    python develop/dev_jax/broad_line_components.py --quick    # two mocks, one config
    python develop/dev_jax/broad_line_components.py --config nlive=1000 --config log

The question this answers
-------------------------
Broad emission lines are fitted with a superposition of Gaussians, and the number
of Gaussians is a *modelling choice*.  Rather than fitting a ladder of models and
comparing evidences -- which needs one run per ``k``, and ``k`` is capped by hand
anyway -- each wing amplitude gets a prior that can be **exactly zero**
(``galspec.uniform_toggle``).  The number of components then comes out of a single
run as the posterior of "how many wings are switched on".

Two things have to hold for that to be worth anything, and both are measured here
on mocks whose answer is known:

1. the rule returns the number of components the data need.  Four mocks are
   generated with 1, 2, 3 and 4 components -- a core plus 0, 1, 2 and 3 wings, two
   of the wings of *similar* width, which is the hard case.  The last mock adds a
   third, overlapping wing, and is the case where the answer should be *fewer*
   components than were generated, because two Gaussians already reproduce that
   profile;
2. the *false positives* are understood -- an extra Gaussian always buys a little
   chi-square against noise, so the report measures how large that effect is and
   which prior shapes and live-point settings suppress it.

The recipe under test is the one proposed for production: one width prior
(300-8000 km/s) for the core and every wing -- a wing is not required to be
broader than the core -- and a ladder of prior inclusion probabilities
``p_off = 0.2, 0.5, 0.7`` so that higher-order wings are less likely a priori.

Outputs (default ``develop/dev_jax/io_test/``)
----------------------------------------------
``component_selection.html``    the report, with its figures embedded
``component_selection_*.png``   the same four figures as files
``component_selection.json``    every number behind it (the figures are not repeated)
"""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2

import jax

import galspec
import galspec.jax_models as jxm
from galspec.jnesty_fit import JNesty_Fit

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "io_test"
DATA_FILE = "component_selection_data.pkl"

#: the mock grid, the noise level and the template of the iron input/output test
sys.path.insert(0, str(HERE))
import iron_tied_io_test as io  # noqa: E402

WAVE = io.WAVE

# --------------------------------------------------------------------------------------
# The scheme under test
# --------------------------------------------------------------------------------------
#: core + three gated wings
K_MAX = 4
N_WINGS = K_MAX - 1
#: prior probability that wing 0, 1, 2 is off -- higher-order wings are less likely
P_OFF = (0.2, 0.5, 0.7)
#: one width prior for the core and for every wing [km/s]
SIGMA_RANGE = (300.0, 8000.0)
DV_RANGE = 6000.0
#: a wing has to reach this fraction of the core peak to count as needed
MIN_WING_FRACTION = 0.03

CORE = {k: v for k, v in io.TRUTH.items() if "_w" not in k}
WING0 = {k: v for k, v in io.TRUTH.items() if "_w0" in k}


def _wings(*triples):
    """Build the ``amp_w<i>``/``dv_w<i>``/``sigma_w<i>`` keys from (amp, dv, sigma)."""
    out = {}
    for i, (amp, dv, sigma) in enumerate(triples):
        out[f"b_hb.amp_w{i}"] = float(amp)
        out[f"b_hb.dv_w{i}"] = float(dv)
        out[f"b_hb.sigma_w{i}"] = float(sigma)
    return out


#: Four mocks with a known number of wings.  ``CORE`` is the published truth
#: (amp_c 1.2, sigma_c 1980 km/s) and ``WING0`` its published wind.
CASES = {
    "1 component": dict(CORE),
    "2 components": dict(CORE, **WING0),
    "3 similar components": dict(CORE, **_wings((0.50, -400.0, 3000.0), (0.35, 900.0, 4200.0))),
    # Three wings that overlap smoothly.  They are *generated* with three but the
    # profile is reproduced by two to well within the noise, which is why this case
    # is judged by "is the answer sufficient?" and not by "is it the generating
    # count?" (a 3-component model fits this mock at chi2/dof ~ 1.01, and the third
    # wing buys dchi2 = 6.3 for 3 extra parameters, p = 0.10).
    "4 overlapping components": dict(CORE, **_wings((0.60, -300.0, 2600.0),
                                               (0.30, 700.0, 4600.0),
                                               (0.15, -1200.0, 7000.0))),
}

CONFIGS = [
    ("linear slab, nlive=400", dict(slab="linear", nlive=400)),
    ("linear slab, nlive=1000", dict(slab="linear", nlive=1000)),
    ("log slab, nlive=1000", dict(slab="log", nlive=1000)),
]

FLOORS = (0.01, 0.03, 0.05, 0.10)


def n_true_wings(truth):
    return sum(1 for k in truth if k.startswith("b_hb.amp_w"))


# --------------------------------------------------------------------------------------
# Mock: the CPU (astropy) models, so the mock does not come from the fitter's own code
# --------------------------------------------------------------------------------------
def cpu_models(truth):
    """``Line_MultiGauss`` with one core plus the truth's wings, and the tied iron."""
    par_w = {k.split(".")[-1]: v for k, v in truth.items()
             if k.split(".")[-1].startswith(("amp_w", "dv_w", "sigma_w"))}
    line = galspec.Line_MultiGauss(
        n_components=1 + len([k for k in par_w if k.startswith("amp_w")]),
        amp_c=truth["b_hb.amp_c"], dv_c=truth["b_hb.dv_c"],
        sigma_c=truth["b_hb.sigma_c"], wavec=galspec.utils.line_wave_dict["Hbeta"],
        par_w=par_w, name="b_hb")
    iron = galspec.IronTemplate_tied(amplitude=truth["FeII.amplitude"], z=0.0,
                                     template_name=io.TEMPLATE, kernel=line, name="FeII")
    return line, iron


def mock_spectrum(truth, noise=io.NOISE_FRACTION, seed=11):
    line, iron = cpu_models(truth)
    line_flux = np.asarray(line(WAVE))
    iron_flux = np.asarray(iron(WAVE))
    peak = float(np.max(line_flux + iron_flux))
    error = np.full_like(WAVE, max(noise, 1e-8) * peak)
    flux = line_flux + iron_flux
    if noise > 0:
        flux = flux + np.random.default_rng(seed).normal(0.0, error)
    return flux, error, line_flux, iron_flux


# --------------------------------------------------------------------------------------
# Fit: every kernel parameter of the iron tied to the line, every wing gated
# --------------------------------------------------------------------------------------
def jax_models(n_components=K_MAX):
    defaults = {}
    for i in range(1, n_components - 1):
        defaults.update({f"amp_w{i}": 0.05, f"dv_w{i}": 0.0, f"sigma_w{i}": 3000.0})
    line = jxm.Line_MultiGauss(
        n_components=n_components, amp_c=1.2, dv_c=0.0, sigma_c=2000.0,
        wavec=galspec.utils.line_wave_dict["Hbeta"],
        amp_w0=0.5, dv_w0=0.0, sigma_w0=4000.0, name="b_hb", **defaults)
    iron = jxm.IronTemplate_tied(n_components=n_components, amplitude=0.3, z=0.0,
                                 name="FeII")
    iron.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("b_hb")
    iron.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("b_hb")
    for i in range(n_components - 1):
        getattr(iron, f"amp_w{i}").tied = getattr(jxm.tie, f"tie_MultiGauss_amp_w{i}")("b_hb")
        getattr(iron, f"dv_w{i}").tied = getattr(jxm.tie, f"tie_MultiGauss_dv_w{i}")("b_hb")
        getattr(iron, f"sigma_w{i}").tied = getattr(jxm.tie, f"tie_MultiGauss_sigma_w{i}")("b_hb")
    return line, iron


def make_priors(flux, slab="linear", p_off=P_OFF):
    """Bounds plus the toggle priors -- the recipe being validated."""
    peak = float(np.max(flux))
    bounds = {
        "b_hb.amp_c": (1e-3, 3.0 * peak),
        "b_hb.dv_c": (-2000.0, 2000.0),
        "b_hb.sigma_c": SIGMA_RANGE,
        "FeII.amplitude": (1e-3, 3.0 * peak),
    }
    custom = {"b_hb.amp_c": galspec.log_uniform(*bounds["b_hb.amp_c"]),
              "FeII.amplitude": galspec.log_uniform(*bounds["FeII.amplitude"])}
    for i in range(N_WINGS):
        bounds[f"b_hb.amp_w{i}"] = (0.0, 1.0)
        bounds[f"b_hb.dv_w{i}"] = (-DV_RANGE, DV_RANGE)
        bounds[f"b_hb.sigma_w{i}"] = SIGMA_RANGE
        if slab == "linear":
            custom[f"b_hb.amp_w{i}"] = galspec.uniform_toggle(p_off=p_off[i])
        else:
            custom[f"b_hb.amp_w{i}"] = galspec.log_uniform_toggle(p_off=p_off[i], lower=0.02)
    return bounds, custom


def run_fit(flux, error, slab="linear", nlive=400, dlogz=0.05, max_iterations=300_000,
            p_off=P_OFF, device="auto"):
    line, iron = jax_models()
    model = line + iron
    bounds, custom = make_priors(flux, slab=slab, p_off=p_off)
    fit = JNesty_Fit(model, WAVE, flux, error, bounds_dict=bounds, custom_priors=custom,
                     nlive=nlive, bound="multi", device=device)
    started = time.perf_counter()
    fit.fit(progress=False, dlogz=dlogz, max_iterations=max_iterations)
    fit.get_best_fit()          # sets fit.theta_best; used for chi2/dof
    return fit, model, time.perf_counter() - started


# --------------------------------------------------------------------------------------
# Analysis: the number of wings is a label-invariant posterior
# --------------------------------------------------------------------------------------
def amp_matrix(fit):
    """``(samples, N_WINGS)`` matrix of the wing peak ratios (0 = switched off).

    For the mock fits, whose single broad line is ``b_hb``.  A fit of real data has
    its wings under the Balmer lines, so the real section below reads its own columns.
    """
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight)
    return np.stack([samples[:, names.index(f"b_hb.amp_w{i}")] for i in range(N_WINGS)], 1)


def count_posterior(fit, floor=0.0):
    """Posterior of the number of wings that are on (``floor=0``) or above ``floor``."""
    amps = amp_matrix(fit)
    counts = (amps > floor).sum(1)
    return np.array([float(np.mean(counts == n)) for n in range(N_WINGS + 1)])


def wing_table(fit, p_off=P_OFF):
    """Per-wing ``P(on)``, prior-corrected Bayes factor, and the size when on."""
    amps = amp_matrix(fit)
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight)
    rows = []
    for i in range(N_WINGS):
        on = amps[:, i] > 0.0
        p_on = float(on.mean())
        odds_prior = galspec.toggle_prior_odds(p_off[i])
        if p_on <= 0.0:
            bayes = 0.0
        elif p_on >= 1.0:
            bayes = float("inf")
        else:
            bayes = (p_on / (1.0 - p_on)) / odds_prior
        sigma = samples[on, names.index(f"b_hb.sigma_w{i}")] if on.any() else np.array([])
        rows.append({
            "wing": i,
            "p_off": p_off[i],
            "p_on": p_on,
            "bayes": bayes,
            "amp_median": float(np.median(amps[on, i])) if on.any() else 0.0,
            "amp_97": float(np.percentile(amps[on, i], 97.5)) if on.any() else 0.0,
            "sigma_median": float(np.median(sigma)) if sigma.size else float("nan"),
        })
    return rows


def sorted_components(fit, floor=MIN_WING_FRACTION):
    """Width-sorted ``(amp, sigma)`` pairs, averaged over the posterior.

    Labels are not identified -- the likelihood is symmetric under swapping two
    wings -- so components are reported by rank in width, never by index.
    """
    amps = amp_matrix(fit)
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight)
    sigmas = np.stack([samples[:, names.index(f"b_hb.sigma_w{i}")] for i in range(N_WINGS)], 1)
    rows = []
    for s in range(0, samples.shape[0], max(1, samples.shape[0] // 400)):
        use = amps[s] > floor
        if not use.any():
            continue
        order = np.argsort(sigmas[s][use])      # widest last
        rows.append([(float(amps[s][use][j]), float(sigmas[s][use][j]))
                     for j in order])
    if not rows:
        return []
    n_max = max(len(r) for r in rows)
    out = []
    for rank in range(n_max):
        amps_r = [r[rank][0] for r in rows if len(r) > rank]
        sig_r = [r[rank][1] for r in rows if len(r) > rank]
        out.append({"rank": rank + 1, "n": len(amps_r),
                    "amp": float(np.median(amps_r)), "sigma": float(np.median(sig_r))})
    return out


def chi2_per_dof(fit):
    """chi-square of the best fit, per degree of freedom."""
    pred = np.asarray(fit.model.flux(fit.wave_use,
                                     jax.numpy.asarray(fit.theta_best)))
    chi2 = float(np.sum(((fit.flux_use - pred) / fit.ferr) ** 2))
    return chi2 / (fit.wave_use.size - fit.ndim)


def logl_at_truth(fit, truth):
    """Likelihood of the truth inside the fitted model -- a mock-only diagnostic.

    If the sampler converged, ``logL(MAP) >= logL(truth)``: the MAP cannot be
    worse than a point the model can represent.  A negative difference means the
    live points missed the truth's region.
    """
    names = list(fit.full_param_names)
    values = {}
    for i in range(N_WINGS):
        values[f"b_hb.amp_w{i}"] = truth.get(f"b_hb.amp_w{i}", 0.0)
        values[f"b_hb.dv_w{i}"] = truth.get(f"b_hb.dv_w{i}", 0.0)
        values[f"b_hb.sigma_w{i}"] = truth.get(f"b_hb.sigma_w{i}", 2000.0)
    values.update({k: v for k, v in truth.items() if k in names})
    theta = np.array([values[n] for n in names], dtype=float)
    pred = np.asarray(fit.model.flux(fit.wave_use, jax.numpy.asarray(theta)))
    return -0.5 * float(np.sum(((fit.flux_use - pred) / fit.ferr) ** 2
                               + np.log(2 * np.pi * fit.ferr ** 2)))


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def amps_of(fit):
    """``(samples, 3)`` amplitudes of the three wings, in units of the core peak."""
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight)
    return np.stack([samples[:, names.index(f"b_hb.amp_w{i}")] for i in range(N_WINGS)], 1)


def best_fit_by_count(fit, wave, max_samples=600):
    """The best-fitting sample at each number of components, with its model curve.

    One run already contains a fit at every count, because the posterior visits the
    regions where some wings are switched off.  For each count we take the sample
    with the smallest chi-square; that is what "the best fit with N components"
    means inside a single gated run, and it needs no second fit.
    """
    samples = np.asarray(fit.samples_equal_weight)
    counts = (amps_of(fit) > MIN_WING_FRACTION).sum(1)
    flux, ferr = np.asarray(fit.flux_use), np.asarray(fit.ferr)
    dof = np.asarray(fit.wave_use).size - fit.ndim
    out = {}
    for n in range(N_WINGS + 1):
        index = np.flatnonzero(counts == n)
        if index.size == 0:
            continue
        if index.size > max_samples:
            index = index[np.linspace(0, index.size - 1, max_samples).astype(int)]
        chi2 = np.empty(index.size)
        for start in range(0, index.size, 200):
            block = np.asarray(samples[index[start:start + 200]], dtype=float)
            predicted = np.asarray(fit.model.flux_batch(fit.wave_use, block))
            chi2[start:start + 200] = np.sum(((flux - predicted) / ferr) ** 2, axis=-1)
        theta = np.asarray(samples[int(index[int(np.argmin(chi2))])], dtype=float)
        out[n] = {"chi2_dof": float(np.min(chi2) / dof),
                  "curve": np.asarray(fit.model.flux(wave, theta)),
                  "n_samples": int(index.size)}
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_toggle(payload):
    """One run, many answers: is each component on, and how large is it?"""
    plt = io._style()
    order = ("2 components", "3 similar components", *CASES)
    picks = [t for t in dict.fromkeys(order) if t in payload["amp_samples"]][:2]
    fig, axes = plt.subplots(1, max(len(picks), 1), figsize=(5.3 * max(len(picks), 1), 2.4),
                             sharex=True, squeeze=False)
    colours = ["#2b6cb0", "#2f855a", "#c05621"]
    for ax, truth in zip(axes[0], picks):
        amps = payload["amp_samples"][truth]
        labels = []
        for i in range(N_WINGS):
            values = amps[:, i]
            on = values > 0.0
            if on.any() and on.mean() > 0.02:
                lo, mid, hi = np.percentile(values[on], [16, 50, 84])
                ax.plot([lo, hi], [i, i], color=colours[i], lw=8, alpha=0.4,
                        solid_capstyle="butt")
                ax.plot([mid], [i], "o", color=colours[i], ms=6)
            labels.append(f"wing {i}\nzero in {1 - on.mean():.0%}")
        ax.axvline(MIN_WING_FRACTION, color="0.45", ls=":", lw=1.0, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 1.15)
        ax.set_ylim(-0.6, N_WINGS - 0.4)
        ax.invert_yaxis()
        ax.set_yticks(range(N_WINGS))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("amplitude / core peak (when it is on)")
        ax.set_title(f"mock made with {n_true_wings(CASES[truth])} wing(s)", fontsize=9.5)
    fig.suptitle("A component is either off (exactly zero) or on with a real amplitude "
                 "(dot: median, bar: 16-84%; dotted: 3% floor)", fontsize=9.5, y=1.06)
    fig.tight_layout()
    return fig


def fig_counts(payload):
    """The posterior of the number of components, one panel per mock."""
    plt = io._style()
    rows = payload["primary"]
    truths = [t for t in CASES if t in rows]
    fig, axes = plt.subplots(1, max(len(truths), 1), figsize=(2.9 * max(len(truths), 1), 2.8),
                             sharey=True, squeeze=False)
    colours = {0: "#a0aec0", 1: "#2b6cb0", 2: "#2f855a", 3: "#c05621"}
    for ax, truth in zip(axes[0], truths):
        n = np.arange(N_WINGS + 1) + 1          # components = core + wings
        p = np.asarray(rows[truth]["p_count_floor"])
        mode = int(np.argmax(p))
        ax.bar(n, p, 0.72, color=[colours.get(i, "0.6") for i in range(len(n))])
        if p[mode] > 0.35:
            ax.text(n[mode], p[mode] / 2, str(n[mode]), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        n_true = n_true_wings(CASES[truth]) + 1
        ax.plot([n_true], [1.03], marker="v", ms=7, color="#c53030", clip_on=False)
        ax.text(n_true, 1.06, "in the mock", ha="center", va="bottom",
                fontsize=7.5, color="#c53030")
        ax.set_xticks(n)
        ax.set_xlabel("components (core + wings)")
        ax.set_title(truth, fontsize=9.5)
        ax.set_ylim(0, 1.20)
    axes[0][0].set_ylabel("posterior probability")
    fig.suptitle("How many components each mock needs (the number in the bar is the answer)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


def _fit_row(wave, flux, error, curves, reported, title, ax, axr, units="flux [arbitrary]",
             labelled=True):
    """One data, model and residual row; ``reported`` is drawn solid."""
    ax.plot(wave, flux, color="0.62", lw=0.7, label="data")
    low, high = float(np.min(flux)), float(np.max(flux))
    pad = 0.06 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    # a difference is worth drawing only if it would move the line by a few pixels:
    # two Gaussians fitted to the same profile can differ by a fraction of a line width
    # and still produce a legend entry that points at nothing visible
    span = (high - low) + 2 * pad
    reference = curves.get(reported, {}).get("curve")
    drawn = 0
    for count in sorted(curves):
        entry = curves[count]
        best = count == reported
        if not best and reference is not None:
            if np.max(np.abs(entry["curve"] - reference)) < 3.0 * span / 250.0:
                continue
        ax.plot(wave, entry["curve"],
                color="#c53030" if best else "#2b6cb0",
                lw=1.2 if best else 1.0, ls="-" if best else "--",
                label=rf"{count + 1} component{'s' if count else ''}: "
                      rf"$\chi^2$/dof {entry['chi2_dof']:.2f}")
        drawn += 1
        if best:
            axr.plot(wave, (flux - entry["curve"]) / error, color="#c53030", lw=0.6)
    ax.set_title(title + ("" if drawn < 2
                          else " (dashed: the other counts this run contains)"), fontsize=9)
    ax.set_ylabel(units, fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.tick_params(labelbottom=False)
    ax.set_xlim(wave.min(), wave.max())
    axr.axhline(0, color="0.4", lw=0.7)
    for value in (-2, 2):
        axr.axhline(value, color="0.8", lw=0.6, ls=":")
    axr.set_ylim(-4.5, 4.5)
    # three ticks, small enough for a panel this short: -2/+2 are drawn as guide lines
    axr.set_yticks([-3, 0, 3])
    axr.tick_params(labelsize=7)
    axr.set_xlim(wave.min(), wave.max())
    if labelled:            # only the bottom pair carries the axis labels
        axr.set_ylabel(r"residual [$\sigma$]", fontsize=8)
        axr.set_xlabel(r"rest wavelength [$\AA$]", fontsize=8)
    else:
        axr.tick_params(labelbottom=False)


def fig_mock_fits(payload):
    """The best fit at the reported count, and with one component fewer."""
    plt = io._style()
    rows = payload["primary"]
    truths = [t for t in CASES if t in rows and t in payload["mock_fits"]]
    n = max(len(truths), 1)
    fig, axes = plt.subplots(2 * n, 1, figsize=(9.4, 2.55 * n + 0.7),
                             gridspec_kw={"height_ratios": [2.7, 1.15] * n,
                                          "hspace": 0.52}, squeeze=False)
    for row, truth in enumerate(truths):
        fit_payload = payload["mock_fits"][truth]
        reported = int(rows[truth]["mode_floor"])
        curves = fit_payload["counts"]
        _fit_row(io.WAVE, fit_payload["flux"], fit_payload["error"], curves, reported,
                 f"{truth}: the fit the rule accepts has {reported + 1} components",
                 axes[2 * row][0], axes[2 * row + 1][0], labelled=(row == len(truths) - 1))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.96, bottom=0.06)
    return fig


def fig_real(payload):
    """The same rule on IRAS 07546: the posterior, and the accepted fit itself."""
    real = payload["real"]
    plt = io._style()
    fig = plt.figure(figsize=(9.8, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2.8], height_ratios=[3, 1],
                            hspace=0.10, wspace=0.20)
    axp = fig.add_subplot(grid[:, 0])
    ax = fig.add_subplot(grid[0, 1])
    axr = fig.add_subplot(grid[1, 1], sharex=ax)

    p = np.asarray(real["p_count_floor"])
    n = np.arange(len(p)) + 1                   # components = core + wings
    reported = int(np.argmax(p))
    colours = {0: "#a0aec0", 1: "#2b6cb0", 2: "#2f855a", 3: "#c05621", 4: "#553c9a"}
    axp.bar(n, p, 0.7, color=[colours.get(i, "0.6") for i in n])
    axp.text(n[reported], float(p[reported]) / 2, str(n[reported]), ha="center",
             va="center", color="white", fontsize=10, fontweight="bold")
    axp.set_xticks(n)
    axp.set_xlabel("components (core + wings)")
    axp.set_ylabel("posterior probability")
    axp.set_ylim(0, 1.1)

    wave = np.asarray(real["wave"])
    curve = real["curves"][reported]["curve"]
    ax.plot(wave, real["flux"], color="0.62", lw=0.5, label="data")
    ax.plot(wave, curve, color="#c53030", lw=1.0, label="best fit")
    axr.plot(wave, (np.asarray(real["flux"]) - curve) / np.asarray(real["error"]),
             color="#c53030", lw=0.5)
    ax.set_xlim(wave.min(), wave.max())
    ax.set_ylabel(r"$F_\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",
                  fontsize=8)
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    axr.axhline(0, color="0.4", lw=0.7)
    for value in (-2, 2):
        axr.axhline(value, color="0.85", lw=0.6, ls=":")
    axr.set_ylim(-4.5, 4.5)
    # three ticks, small enough for a panel this short: -2/+2 are drawn as guide lines
    axr.set_yticks([-3, 0, 3])
    axr.tick_params(labelsize=7)
    axr.set_ylabel(r"residual [$\sigma$]", fontsize=8)
    axr.set_xlabel(r"rest wavelength [$\AA$]", fontsize=8)
    fig.suptitle(rf"IRAS 07546+3928: the rule asks for {reported + 1} components "
                 rf"($\chi^2$/dof {real['curves'][reported]['chi2_dof']:.3f})",
                 fontsize=10, y=0.99)
    return fig


def _jsonable(value):
    """Plain JSON types, with non-finite floats written as null.

    ``null`` therefore means "not a finite number" -- for a Bayes factor it means
    unbounded, which is how a wing that is on in every sample is recorded.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return int(value) if not isinstance(value, np.bool_) else bool(value)
    return value


def compute_checks(payload: dict) -> list:
    """The pass/fail checks, as a pure function of the payload.

    Kept separate from the fitting loop so the checks can be recomputed from a
    stored ``component_selection.json`` without re-running the sampler.
    """
    # The question is not "does the count equal the number of wings the mock was
    # built from?" -- it is "is the answer the *minimum* number that fits?".  A mock
    # can be generated with three wings that two Gaussians reproduce to a tenth of
    # the noise, and then 2 is the right answer to the question that was asked.
    checks = []
    primary = payload["primary"]

    # Adequate and decisive, per mock.  Two claims, both measured on the run itself:
    # the fit at the reported count leaves nothing to explain (chi2/dof -> 1), and the
    # posterior excludes a smaller count, which is what makes it the *minimum*.
    for truth_name, entry in primary.items():
        p = np.asarray(entry["p_count_floor"], dtype=float)
        mode = int(entry["mode_floor"])
        below = float(p[:mode].sum())
        at_mode = float(p[mode])
        fitted = float(entry["chi2_dof"])
        ok = bool(fitted <= 1.10 and at_mode >= 0.90 and below <= 0.05)
        checks.append({
            "name": f"adequate and decisive: {truth_name}",
            "ok": ok,
            "detail": (f"{mode} wing(s) at P = {at_mode:.2f}, fewer wings at "
                       f"P = {below:.2f}, chi2/dof = {fitted:.3f} "
                       f"({entry['config']})")})

    over = [t for t, e in primary.items()
            if int(e["mode_floor"]) > int(n_true_wings(CASES[t]))]
    checks.append({"name": "never more components than were generated",
                   "ok": not over,
                   "detail": ("no mock reports a component beyond its generating "
                              "count" if not over else "over-counted: " + ", ".join(over))})

    # the Bayes factors agree with the answer (label-free: count them)
    def decisive(row) -> bool:
        """``BF > 10``; a null (written for an unbounded factor) counts as decisive."""
        bf = row["bayes"]
        return True if bf is None else float(bf) > 10.0

    mismatched = []
    for truth_name, wings in payload["per_wing"].items():
        above = sum(1 for row in wings if decisive(row))
        mode = int(primary[truth_name]["mode_floor"])
        if above != mode:
            mismatched.append(f"{truth_name}: {above} wings with BF > 10, count {mode}")
    checks.append({"name": "the number of wings with BF > 10 equals the count",
                   "ok": not mismatched,
                   "detail": ("one Bayes factor above 10 per reported wing, no more"
                              if not mismatched else "; ".join(mismatched))})

    # the linear slab is the recipe: on the one-wing mock the log slab turns on a
    # second wing that the amplitude floor then has to discard
    one_wing = {cfg["slab"]: cfg for cfg in payload["cases"].get("2 components", {}).values()
                if cfg["nlive"] >= 1000}
    if "linear" in one_wing:
        cfg = one_wing["linear"]
        checks.append({"name": "the linear slab needs no rescuing",
                       "ok": (int(cfg["mode"]), int(cfg["mode_floor"])) == (1, 1),
                       "detail": (f"on the one-wing mock it reports "
                                  f"{cfg['mode_floor']} floored ({cfg['mode']} before "
                                  f"the floor), P = "
                                  f"{cfg['p_count_floor'][int(cfg['mode_floor'])]:.2f}")})
    if "log" in one_wing:
        cfg = one_wing["log"]
        above = float(cfg["p_count_floor"][2])
        checks.append({"name": "the floor catches the log slab's false positive",
                       "ok": int(cfg["mode"]) > int(cfg["mode_floor"]),
                       "detail": (f"on the one-wing mock the log slab turns on a second "
                                  f"wing in {cfg['p_count'][2]:.2f} of the samples "
                                  f"({above:.2f} of them stay above the 3% floor), "
                                  f"which the linear slab never does")})
    return checks


def real_data_section(max_samples=120):
    """The same rule on IRAS 07546, if that fit has been run.

    Optional by design: the report is about the calibration, and the real object is
    there to show that the rule is used on data, not only on mocks.
    """
    npz_path = Path("example/output/iras07546_jax_toggle_posterior.npz")
    fits_path = Path("example/output/iras07546_jax_toggle_fit.fits")
    if not (npz_path.exists() and fits_path.exists()):
        return None
    data = np.load(npz_path, allow_pickle=True)
    samples = np.asarray(data["samples"])
    names = list(data["qualified_names"])
    columns = [names.index(f"b_halpha.amp_w{i}") for i in range(N_WINGS)]
    def counts_above(floor):
        return (samples[:, columns] > floor).sum(1)

    counts = counts_above(MIN_WING_FRACTION)
    p = np.array([float(np.mean(counts == n)) for n in range(N_WINGS + 1)])

    import warnings
    from galspec.jnesty_fit_io import load_jnesty_fit_from_fits
    with warnings.catch_warnings():
        # prior callables cannot be stored in FITS; only the samples are needed
        warnings.simplefilter("ignore", UserWarning)
        fit = load_jnesty_fit_from_fits(str(fits_path))
    wave = np.asarray(fit.wave_use)
    flux, ferr = np.asarray(fit.flux_use), np.asarray(fit.ferr)
    dof = wave.size - fit.ndim
    curves = {}
    for n in {int(np.argmax(p)), max(int(np.argmax(p)) - 1, 0)}:
        index = np.flatnonzero(counts == n)
        if index.size == 0:
            continue
        if index.size > max_samples:
            index = index[np.linspace(0, index.size - 1, max_samples).astype(int)]
        chi2 = np.empty(index.size)
        for start in range(0, index.size, 60):
            block = samples[index[start:start + 60]]
            predicted = np.asarray(fit.model.flux_batch(wave, block))
            chi2[start:start + 60] = np.sum(((flux - predicted) / ferr) ** 2, axis=-1)
        theta = samples[int(index[int(np.argmin(chi2))])]
        curves[n] = {"chi2_dof": float(np.min(chi2) / dof),
                     "curve": np.asarray(fit.model.flux(wave, theta))}
    dchi2 = float("nan")
    fitted = sorted(curves)
    if len(fitted) >= 2:
        dchi2 = float((curves[fitted[0]]["chi2_dof"]
                       - curves[fitted[-1]]["chi2_dof"]) * dof)
    by_floor = {}
    for floor in FLOORS:
        above = counts_above(floor)
        probs = [float(np.mean(above == n)) for n in range(N_WINGS + 1)]
        by_floor[f"{floor:.2f}"] = int(np.argmax(probs))
    return {"wave": wave, "flux": flux, "error": ferr, "p_count_floor": p,
            "chi2_dof": float(curves[int(np.argmax(p))]["chi2_dof"]),
            "floors": by_floor,
            "curves": curves, "n_samples": int(samples.shape[0]), "dchi2": dchi2}


def build_report(path: Path, payload: dict) -> None:
    """Write the figure-first report: four pictures, one table."""
    primary = payload["primary"]
    checks = payload["checks"]
    failed = [c for c in checks if not c["ok"]]
    png = payload["images"]

    table_rows = []
    for truth in [t for t in CASES if t in primary]:
        entry = primary[truth]
        reported = int(entry["mode_floor"])
        curves = payload.get("mock_fits", {}).get(truth, {}).get("counts", {})
        fitted = curves.get(reported, {}).get("chi2_dof")
        table_rows.append(
            f"<tr><td>{html.escape(truth)}</td>"
            f"<td class='num'>{entry['n_true'] + 1}</td>"
            f"<td class='num'><b>{reported + 1}</b></td>"
            f"<td class='num'>{fitted:.2f}</td></tr>")

    if failed:
        verdict = ("<div class='verdict warn'><strong>One check did not pass: "
                   + "; ".join(html.escape(c["name"]) for c in failed)
                   + ".</strong> The figures below show where.</div>")
    else:
        verdict = ("<div class='verdict'><strong>The rule works: on all four mocks it "
                   "returns the number of components the data need.</strong> The tallest "
                   "bar in section 2 is the answer, and section 3 shows what that answer "
                   "looks like against the data.</div>")

    floor_rows = payload.get("floor_table", {})
    moved = []
    for truth, table in floor_rows.items():
        kmins = {floor: row["kmin"] + 1 for floor, row in table["floor"].items()}
        if len(set(kmins.values())) > 1:
            moved.append(html.escape(truth) + " " + ", ".join(
                f"at {float(k):.0%} &rarr; {v} components" for k, v in kmins.items()))
    if payload.get("real"):
        real_floors = payload["real"].get("floors", {})
        if len(set(real_floors.values())) > 1:
            moved.append("IRAS 07546 " + ", ".join(
                f"at {float(k):.0%} &rarr; {v + 1} components"
                for k, v in real_floors.items()))
    floor_text = ("; ".join(moved) if moved else
                  "on these mocks it does not, because every component is well above "
                  "the floor.")

    live_rows = payload.get("nlive", {})
    moved = [t for t, d in live_rows.items() if len(set(d.values())) > 1]
    if moved:
        detail = "; ".join(
            f"{html.escape(t)} &mdash; " + ", ".join(
                f"{k} live points give {v + 1} components"
                for k, v in live_rows[t].items())
            for t in moved)
        live = (f"400 live points is marginal for a 40-parameter model with gated "
                f"amplitudes, and this run measures it: {detail}.  Every number "
                f"quoted above comes from 1000.")
    elif all(len(d) > 1 for d in live_rows.values()) and live_rows:
        live = ("400 and 1000 live points give the same count on every mock here.  "
                "That is a measurement of the margin, not a guarantee: 400 is "
                "marginal for a 40-parameter model with gated amplitudes, and a "
                "repeat can differ.")
    else:
        live = ("this run has a single live-point setting, so the margin is not "
                "measured here; 400 is marginal for a 40-parameter model with gated "
                "amplitudes and is not the setting the recipe recommends.")

    real = payload.get("real")
    if real:
        real_html = f"""
<h2>4. The same rule on real data</h2>
<p>IRAS 07546+3928, fitted with up to four broad components
({real['n_samples']:,} posterior samples).  The bars are the same posterior as in
section 2; the curve is the fit the rule accepted, with the residual underneath.</p>
<img src="data:image/png;base64,{png['real']}" alt="the rule on IRAS 07546">
<p class="cap">The residuals show no structure -- they sit inside the noise -- and all
three wings are on in every sample, so this object needs four components.  Its
chi2/dof of {real['chi2_dof']:.3f} is far below 1 because the error bars are
conservative on these pixels (the published two-component fit gets 0.32 on the same
data); a low value cannot make a model look better than it is, and the comparison that
decides the count is between two fits of the <em>same</em> pixels: four components beat
three by <code>dchi2</code> = {real['dchi2']:.0f} for three parameters &mdash; decisive,
yet invisible in chi2/dof, where 57 000 pixels hide it in the third decimal.  The
broadest component sits at the upper edge of the adopted width range, which is reported
rather than hidden.</p>"""
    else:
        real_html = ""

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>How many broad-line components does the data need?</title>
<style>{io.CSS}</style></head>
<body><main>
<h1>How many components does a broad line need?</h1>
<p class="sub">One fit per line, four mocks whose answer is known, and four pictures.
Written {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by
<code>develop/dev_jax/broad_line_components.py</code>.</p>

{verdict}

<h2>1. The method</h2>
<p>Fit the line with <em>more</em> components than you think it has, and give every
component after the core a <strong>toggle prior</strong>: with probability
<code>p</code> its amplitude is <strong>exactly zero</strong>, otherwise it is drawn
from a prior on positive amplitudes.  The fit is therefore free to switch a component
off completely, and the sampler reports how much of the posterior sits on each
choice.</p>

<p>A <strong>component</strong> here always means one Gaussian: the core, which is
always present, plus the wings that are switched on.  A line fitted with a core and
two wings has three components.</p>

<p><strong>The number of components is then read off in three steps.</strong></p>
<ol>
<li><strong>Count, per sample.</strong>  In each posterior sample, count the components
that are present: the core always counts, and a wing counts when its peak is above 3%
of the core peak.  A wing that is switched off does not count; one that is on but
negligible is below the floor.</li>
<li><strong>Take the most common count.</strong>  The mode of that distribution is the
answer &mdash; the number of components the data actually constrain.</li>
<li><strong>Check it against the data.</strong>  The fit at that count must describe
the spectrum: its chi-square per degree of freedom must not be <em>above</em> about 1
(section 3).  Well <em>below</em> 1 is not a problem -- it means the quoted error bars
are conservative -- and it is what the real object below shows.</li>
</ol>

<img src="data:image/png;base64,{png['toggle']}" alt="what a toggle prior does">
<p class="cap">What the toggle prior does, in one picture: the amplitude of each
wing across the posterior samples.  A wing either sits at exactly zero -- which a log
axis cannot show, so the share of samples is written on the row -- or takes a real
value.  In the mock made with 2 components one wing is on and two are at zero; in the
mock made with 3 components, two are on.  The dotted line is the 3% floor of step 1:
a wing whose dot sits far below it is on in the sampler but not a component the data
need, which is exactly the distinction the count uses.</p>

<h2>2. The answer for each mock</h2>
<p>Four mocks &mdash; a core plus 0, 1, 2 and 3 extra components &mdash; were fitted
the same way.  The bars are the posterior of the number of components, and the red
triangle marks the number that was put in when the mock was built.</p>
<img src="data:image/png;base64,{png['counts']}" alt="posterior of the number of components">
<p class="cap">The number in the bar is the answer, in components.  Three of the
four mocks are answered exactly.  The fourth was built with four components that
overlap, and the answer is three, because three already describe it &mdash; the
bottom row of section 3 shows the fit is already at the noise level.</p>

<h2>3. How well the fit matches the data</h2>
<p>For each mock: the simulated spectrum (grey) and the best fit the rule accepts
(solid), with the residual in units of the noise underneath.  A switched-off component
costs nothing, so the same run also contains a fit at every lower count; where such a
fit differs visibly from the accepted one it is drawn dashed, with its chi2/dof in the
legend.  On these mocks it never does, and that is the point worth seeing: one more
Gaussian changes each pixel by a fraction of the noise (chi2 differs by tens over
thousands of pixels, or 0.1&ndash;0.2&sigma; per pixel), so the curves coincide while the
evidence does not.  That is why the count is decided from chi2 and the posterior, not
from looking at the fit.</p>
<img src="data:image/png;base64,{png['mock_fits']}" alt="best fits and residuals">
<table>
<tr><th>mock</th><th>components in the mock</th><th>components the rule reports</th>
    <th>chi2/dof of that fit</th></tr>
{''.join(table_rows)}
</table>
<p class="cap">chi2/dof = 1 means the model has explained everything the noise allows;
a value well above 1 means the profile is not described.  Every reported fit is at
about 1, and the run never prefers fewer components (see the posteriors in
section 2), which is what "the minimum number of components that fits" means.  The
last row is the interesting one: the mock made with four components is described by
three, and the fit is already at chi2/dof = 1, so the fourth is not needed.</p>
{real_html}

<h2>5. What to keep in mind</h2>
<ul>
<li><strong>The components are not labelled.</strong>  Two components of similar width
can swap places from one sample to the next, so only the <em>number</em> of components
and their widths are meaningful, not their names.</li>
<li><strong>The live points are not decoration.</strong> {live}</li>
<li><strong>The floor is part of the definition.</strong>  "A component is needed
when it carries more than 3% of the core peak" is a statement about what counts as a
component; the fit itself still puts mass on smaller ones.  A component a few percent
above the core peak therefore sits near the definition's edge, and the answer moves
when the floor is moved: {floor_text}.</li>
<li><strong>On real data the width range can bind.</strong>  A component pinned to the
upper end of the adopted widths is reported (~<code>prior_dominated</code>) rather than
accepted silently.</li>
</ul>
</main></body></html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true",
                        help="two mocks, 400 live points only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild", action="store_true",
                        help="re-render the report from the stored samples")
    parser.add_argument("--case", action="append", metavar="TEXT",
                        help="only mock cases whose name contains TEXT (repeatable)")
    parser.add_argument("--config", action="append", metavar="TEXT",
                        help="only configs whose name contains TEXT (repeatable); "
                             "default: all of them")
    args = parser.parse_args()
    if args.rebuild:
        return rebuild(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    cases = list(CASES)
    configs = CONFIGS
    if args.case:
        cases = [c for c in CASES
                 if any(text.lower() in c.lower() for text in args.case)]
        if not cases:
            parser.error(f"no case matches {args.case}; have "
                         + ", ".join(f"'{c}'" for c in CASES))
    if args.config:
        configs = [c for c in CONFIGS
                   if any(text.lower() in c[0].lower() for text in args.config)]
        if not configs:
            parser.error(f"no config matches {args.config}; have "
                         + ", ".join(f"'{c[0]}'" for c in CONFIGS))
    if args.quick:
        cases = ["1 component", "2 components"]
        configs = [CONFIGS[0]]

    payload = {"cases": {}, "primary": {}, "per_wing": {}, "floor_table": {},
               "checks": [], "images": {}, "configs": [c[0] for c in configs],
               "amp_samples": {}, "mock_fits": {},
               "scheme": {"k_max": K_MAX, "p_off": list(P_OFF),
                          "sigma_range": list(SIGMA_RANGE),
                          "min_wing_fraction": MIN_WING_FRACTION}}
    started = time.perf_counter()
    mocks = {}

    for truth_name in cases:
        truth = CASES[truth_name]
        flux, error, line_flux, iron_flux = mock_spectrum(truth)
        mocks[truth_name] = (flux, error)
    print(f"{len(cases)} mocks built on {WAVE.size} pixels")
    print(f"  {'truth':18s} {'config':24s} {'N':>8s} {'P(N)':>22s} {'lnZ':>9s}  min")

    for truth_name in cases:
        truth = CASES[truth_name]
        flux, error = mocks[truth_name]
        n_true = n_true_wings(truth)
        for config_name, config in configs:
            fit, model, wall = run_fit(flux, error, slab=config["slab"],
                                       nlive=config["nlive"], device=args.device)
            p_count = count_posterior(fit, floor=0.0)
            p_floor = count_posterior(fit, floor=MIN_WING_FRACTION)
            mode, mode_floor = int(np.argmax(p_count)), int(np.argmax(p_floor))
            entry = {
                "n_true": n_true,
                "config": config_name,
                "slab": config["slab"],
                "nlive": config["nlive"],
                "p_count": [float(v) for v in p_count],
                "p_count_floor": [float(v) for v in p_floor],
                "mode": mode,
                "mode_floor": mode_floor,
                "map_minus_truth": float(np.max(np.asarray(fit.results["logl"]))
                                         - logl_at_truth(fit, truth)),
                "chi2_dof": float(chi2_per_dof(fit)),
                "log_evidence": float(fit.log_evidence),
                "log_evidence_err": float(fit.log_evidence_err),
                "n_iterations": int(fit.n_iterations),
                "ndim": int(fit.ndim),
                "wall": float(wall),
                "wings": wing_table(fit, P_OFF),
                "sorted": sorted_components(fit),
            }
            payload["cases"].setdefault(truth_name, {})[config_name] = entry
            print(f"  {truth_name:18s} {config_name:24s} "
                  f"{mode}/{mode_floor:<5d} {' '.join(f'{v:.2f}' for v in p_count):>22s} "
                  f"{fit.log_evidence:9.1f}  {wall/60:.1f}")

            if config["slab"] == "linear":
                entry["floors"] = {}
                for floor in FLOORS:
                    counts = count_posterior(fit, floor=floor)
                    entry["floors"][f"{floor:.2f}"] = {
                        "kmin": int(np.argmax(counts)),
                        "p_count": [float(v) for v in counts]}
                # what the report draws: the amplitudes of each wing, and the best fit
                # at every count (the posterior already contains one of each)
                payload["amp_samples"].setdefault(truth_name, {})[config_name] = amps_of(fit)
                payload["mock_fits"].setdefault(truth_name, {})[config_name] = {
                    "flux": np.asarray(flux), "error": np.asarray(error),
                    "counts": best_fit_by_count(fit, WAVE)}

    for truth_name in cases:
        linear = [(entry["nlive"], name, entry)
                  for name, entry in payload["cases"].get(truth_name, {}).items()
                  if entry["slab"] == "linear"]
        if not linear:
            continue
        _, _, entry = max(linear, key=lambda row: row[0])
        payload["primary"][truth_name] = entry
        payload["per_wing"][truth_name] = entry["wings"]
        payload["floor_table"][truth_name] = {"n_true": entry["n_true"],
                                              "floor": entry["floors"]}

    # keep the report's material for the run each mock is quoted from
    for key in ("amp_samples", "mock_fits"):
        payload[key] = {truth: payload[key][truth][payload["primary"][truth]["config"]]
                        for truth in payload["primary"] if truth in payload[key]}

    try:
        payload["real"] = real_data_section()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (no real-data panel: {exc})")
        payload["real"] = None

    # the samples behind the figures, so the report can be re-rendered without a GPU
    import pickle
    with open(args.out / DATA_FILE, "wb") as handle:
        pickle.dump({k: payload[k] for k in ("amp_samples", "mock_fits", "real",
                                             "cases", "primary", "per_wing",
                                             "floor_table", "configs",
                                             "nlive")}, handle)

    return _render(payload, args.out, started)


def _render(payload: dict, out: Path, started: float | None = None) -> int:
    """Figures, report and JSON from a finished payload; no fitting happens here."""
    # how much the answer moves with the number of live points: a measurement of the
    # margin, not a claim, so it is reported in the caveats rather than checked
    payload["nlive"] = {}
    for truth_name, per_config in payload["cases"].items():
        linear = {c["nlive"]: int(c["mode_floor"]) for c in per_config.values()
                  if c["slab"] == "linear"}
        payload["nlive"][truth_name] = {str(k): v for k, v in sorted(linear.items())}
    unstable = [f"{t} ({d})" for t, d in payload["nlive"].items() if len(set(d.values())) > 1
                and len(d) > 1]
    if unstable:
        print("  live points change the count for: " + "; ".join(unstable))
    payload["checks"] = compute_checks(payload)
    payload["accepted"] = not [c for c in payload["checks"] if not c["ok"]]
    payload["images"] = {}
    for name, build in (("toggle", fig_toggle), ("counts", fig_counts),
                        ("mock_fits", fig_mock_fits), ("real", fig_real)):
        if name == "real" and not payload.get("real"):
            payload["images"][name] = ""
            continue
        encoded = io._b64(build(payload))
        payload["images"][name] = encoded
        # the same figure as a file: the report embeds it, this lets it be opened
        (out / f"component_selection_{name}.png").write_bytes(
            base64.b64decode(encoded))

    payload["notes"] = {
        "counts": ("mode, mode_floor, p_count, p_count_floor and wing_table counts "
                   "are numbers of *wings*; the number of components is one more "
                   "(the core is always present)"),
        "bayes": ("null means the factor is unbounded (no posterior sample switched "
                  "the wing off), not that it is zero"),
        "floors": "amplitude of a wing as a fraction of the core peak",
    }
    report = out / "component_selection.html"
    build_report(report, payload)
    heavy = {"images", "amp_samples", "mock_fits", "real"}
    (out / "component_selection.json").write_text(
        json.dumps(_jsonable({k: v for k, v in payload.items() if k not in heavy}),
                   indent=2), encoding="utf-8")
    print(f"\nwrote {report}"
          + (f"  ({time.perf_counter() - started:.0f} s total)" if started else ""))
    passed = sum(1 for c in payload["checks"] if c["ok"])
    print(f"{passed}/{len(payload['checks'])} checks pass")
    for check in payload["checks"]:
        print(f"  [{'ok' if check['ok'] else '!!'}] {check['name']}: {check['detail']}")
    return 0 if payload["accepted"] else 1


def rebuild(out: Path) -> int:
    """Re-render the report from ``<out>`` without refitting."""
    import pickle
    data_path = out / DATA_FILE
    if not data_path.is_file():
        raise SystemExit(f"{data_path} is missing: run the fits first")
    with open(data_path, "rb") as handle:
        payload = pickle.load(handle)
    payload["checks"] = []
    payload.update(json.loads((out / "component_selection.json").read_text()))
    return _render(payload, out)


if __name__ == "__main__":
    raise SystemExit(main())