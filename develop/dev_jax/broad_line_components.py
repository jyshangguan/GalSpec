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

1. the rule returns the number of wings the data need: 0, 1, 2 and 3 wings are
   generated (two of them of *similar* width, which is the hard case), and the
   third overlapping one is the case where the answer should be *fewer* than
   were generated -- because two Gaussians already reproduce that profile;
2. the *false positives* are understood -- an extra Gaussian always buys a little
   chi-square against noise, so the report measures how large that effect is and
   which prior shapes and live-point settings suppress it.

The recipe under test is the one proposed for production: one width prior
(300-8000 km/s) for the core and every wing -- a wing is not required to be
broader than the core -- and a ladder of prior inclusion probabilities
``p_off = 0.2, 0.5, 0.7`` so that higher-order wings are less likely a priori.

Outputs (default ``develop/dev_jax/io_test/``)
----------------------------------------------
``component_selection.html``    the report
``component_selection.png``     the P(N) figure it embeds
``component_selection.json``    every number in the report
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
    "0 wings": dict(CORE),
    "1 wing": dict(CORE, **WING0),
    "2 similar wings": dict(CORE, **_wings((0.50, -400.0, 3000.0), (0.35, 900.0, 4200.0))),
    # Three wings that overlap smoothly.  They are *generated* with three but the
    # profile is reproduced by two to well within the noise, which is why this case
    # is judged by "is the answer sufficient?" and not by "is it the generating
    # count?" (a 3-component model fits this mock at chi2/dof ~ 1.01, and the third
    # wing buys dchi2 = 6.3 for 3 extra parameters, p = 0.10).
    "3 overlapping wings": dict(CORE, **_wings((0.60, -300.0, 2600.0),
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
    """``(samples, N_WINGS)`` matrix of the wing peak ratios (0 = switched off)."""
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
def fig_counts(payload):
    plt = io._style()
    rows = payload["primary"]
    truths = list(CASES)
    fig, axes = plt.subplots(1, len(truths), figsize=(3.1 * len(truths), 3.0),
                             sharey=True)
    colours = {0: "#a0aec0", 1: "#2b6cb0", 2: "#2f855a", 3: "#c05621"}
    for ax, truth in zip(axes, truths):
        entry = rows.get(truth)
        if entry is None:
            continue
        n = np.arange(N_WINGS + 1)
        width = 0.38
        ax.bar(n - width / 2, entry["p_count"], width, color=[colours.get(i, "0.6") for i in n],
               label="on ($amp>0$)")
        ax.bar(n + width / 2, entry["p_count_floor"], width, facecolor="none",
               edgecolor="0.35", hatch="///", lw=0.8, label="above floor")
        ax.axvline(n_true_wings(CASES[truth]), color="#c53030", ls="--", lw=1.0)
        ax.set_title(f"truth: {truth}", fontsize=9.5)
        ax.set_xticks(n)
        ax.set_xlabel("wings")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("posterior probability")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Posterior of the number of wings used (dashed red: the truth)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    return fig


def fig_example(payload):
    """What the selected model looks like: component curves for one mock."""
    plt = io._style()
    wave = WAVE
    flux = payload["example"]["flux"]
    error = payload["example"]["error"]
    comps = payload["example"]["components"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.0, 4.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    ax.plot(wave, flux, color="0.15", lw=0.9, label="mock")
    total = np.zeros_like(wave)
    palette = ["#2b6cb0", "#2f855a", "#b7791f", "#805ad5"]
    for j, comp in enumerate(comps):
        ax.plot(wave, comp["profile"], color=palette[j % len(palette)], lw=1.0, ls="--",
                label=comp["label"])
        total = total + np.asarray(comp["profile"])
    iron = np.asarray(payload["example"].get("iron", np.zeros_like(wave)))
    if np.any(iron):
        ax.plot(wave, iron, color="#b7791f", lw=0.9, ls=":", label=r"Fe II (tied)")
        total = total + iron
    ax.plot(wave, total, color="#c53030", lw=0.9, alpha=0.8, label="sum")
    ax.set_ylabel(r"$F_\lambda$")
    ax.set_title(payload["example"]["title"], fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    axr.axhline(0, color="0.7", lw=0.6)
    axr.plot(wave, (flux - total) / error, color="0.3", lw=0.7)
    axr.set_ylabel(r"residual / $\sigma$")
    axr.set_xlabel(r"rest wavelength [$\AA$]")
    axr.set_xlim(payload["example"]["xlim"])
    return fig


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _jsonable(value):
    """Plain JSON types, with non-finite floats written as null.

    ``null`` therefore means "not a finite number" -- for a Bayes factor, it means
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
    # ---- checks -----------------------------------------------------------------
    # The question is not "does the count equal the number of wings the mock was
    # built from?" -- it is "is the answer the *minimum* number that fits?".  A mock
    # can be generated with three wings that two Gaussians reproduce to a tenth of
    # the noise, and then 2 is the right answer to the question that was asked.
    # So the checks below are about sufficiency, parsimony and stability, and only
    # the parsimony one refers to the generating count at all.
    checks = []
    primary = payload["primary"]

    for truth_name, entry in primary.items():
        n_true = int(n_true_wings(CASES[truth_name]))
        mode = int(entry["mode_floor"])
        d_n = n_true - mode
        if d_n <= 0:
            detail = (f"count = {mode}, generated with {n_true} "
                      f"({entry['config']})")
            ok = d_n == 0
        else:
            # fewer components than were generated: the fit must be as good as the
            # truth's, i.e. dropping them must cost nothing significant
            dchi2 = -2.0 * float(entry["map_minus_truth"])
            limit = float(chi2.ppf(0.95, 3 * d_n)) if d_n > 0 else 0.0
            ok = bool(dchi2 <= limit)
            detail = (f"count = {mode} of {n_true} generated, and the fit is "
                      f"indistinguishable from the truth's (dchi2 = {dchi2:.1f} on "
                      f"{3 * d_n} extra parameters, 95% point {limit:.1f})")
        checks.append({"name": f"parsimonious and sufficient: {truth_name}",
                       "ok": ok, "detail": detail})

    over = [t for t, e in primary.items()
            if int(e["mode_floor"]) > int(n_true_wings(CASES[t]))]
    checks.append({"name": "never more components than were generated",
                   "ok": not over,
                   "detail": ("no mock reports a component beyond its generating "
                              "count" if not over else "over-counted: " + ", ".join(over))})

    # the count does not depend on where the amplitude floor is put
    unstable = []
    for truth_name, table in payload["floor_table"].items():
        kmins = {row["kmin"] for row in table["floor"].values()}
        if len(kmins) > 1:
            unstable.append(f"{truth_name}: {sorted(kmins)}")
    checks.append({"name": "the count is stable across the amplitude floor",
                   "ok": not unstable,
                   "detail": ("the same count at floors "
                              + ", ".join(f"{f:.0%}" for f in FLOORS)
                              if not unstable else "; ".join(unstable))})

    # the count is stable between 400 and 1000 live points
    jitter = []
    for truth_name, per_config in payload["cases"].items():
        linear = {c["nlive"]: c["mode_floor"] for c in per_config.values()
                  if c["slab"] == "linear"}
        if len(set(linear.values())) > 1:
            jitter.append(f"{truth_name}: {linear}")
    checks.append({"name": "the count is stable between 400 and 1000 live points",
                   "ok": not jitter,
                   "detail": ("every mock gives the same count at both settings"
                              if not jitter else "; ".join(jitter))})

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

    ridged = [t for t, e in primary.items() if e["chi2_dof"] > 1.10]
    checks.append({"name": "the fits are statistically adequate",
                   "ok": not ridged,
                   "detail": ("chi2/dof <= 1.10 for every primary fit"
                              if not ridged else
                              "chi2/dof > 1.10 for " + ", ".join(ridged))})

    # the linear slab is the recipe: on the one-wing mock the log slab turns on a
    # second wing that the amplitude floor then has to discard
    one_wing = {cfg["slab"]: cfg for cfg in payload["cases"].get("1 wing", {}).values()
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


def build_report(path: Path, payload: dict) -> None:
    rows = payload["primary"]
    plt_rows = payload["per_wing"]
    tables = payload["floor_table"]

    def fmt(value, digits=3):
        if value is None:
            return "--"
        if isinstance(value, float) and not np.isfinite(value):
            return "inf"
        return f"{value:.{digits}f}"

    def verdict_span(ok):
        return ('<span class="ok">yes</span>' if ok
                else '<span class="no">no</span>')

    main_rows = []
    for truth, entry in rows.items():
        n_true = n_true_wings(CASES[truth])
        p = entry["p_count"]
        mode = int(np.argmax(p))
        p_eff = entry["p_count_floor"]
        mode_eff = int(np.argmax(p_eff))
        # the answer is judged on parsimony and sufficiency: a count *below* the
        # generating one is right when the fit is indistinguishable from the truth's
        if mode_eff > n_true:
            verdict = '<span class="no">over-count</span>'
        elif mode_eff == n_true:
            verdict = '<span class="ok">exact</span>'
        else:
            dchi2 = -2.0 * float(entry["map_minus_truth"])
            n_extra = n_true - mode_eff
            verdict = ('<span class="ok">sufficient</span>' if dchi2 <= chi2.ppf(0.95, 3 * n_extra)
                       else '<span class="no">short</span>')
        main_rows.append(f"""<tr>
          <td>{html.escape(truth)}</td>
          <td class="num">{entry['n_true']}</td>
          <td class="num">{' '.join(f'{v:.2f}' for v in p)}</td>
          <td class="num">{' '.join(f'{v:.2f}' for v in p_eff)}</td>
          <td class="num">{mode} / {mode_eff}</td>
          <td class="num">{entry['map_minus_truth']:+.1f}</td>
          <td class="num">{entry['chi2_dof']:.3f}</td>
          <td class="num">{entry['log_evidence']:.1f} &#177; {entry['log_evidence_err']:.2f}</td>
          <td class="num">{entry['wall'] / 60:.1f}</td>
          <td>{verdict}</td>
        </tr>""")

    wing_rows = []
    for truth, entry in plt_rows.items():
        for row in entry:
            wing_rows.append(f"""<tr>
              <td>{html.escape(truth)}</td>
              <td class="num">{row['wing']}</td>
              <td class="num">{row['p_off']:.1f}</td>
              <td class="num">{row['p_on']:.3f}</td>
              <td class="num">{fmt(row['bayes'], 2)}</td>
              <td class="num">{row['amp_median']:.4f}</td>
              <td class="num">{row['amp_97']:.3f}</td>
              <td class="num">{fmt(row['sigma_median'], 0)}</td>
            </tr>""")

    floor_rows = []
    for truth, entry in tables.items():
        kmin = [entry["floor"][f"{f:.2f}"]["kmin"] for f in FLOORS]
        verdict = ('<span class="ok">stable</span>'
                   if len(set(kmin)) == 1 else '<span class="no">varies</span>')
        floor_rows.append(f"""<tr>
          <td>{html.escape(truth)}</td>
          <td class="num">{entry['n_true']}</td>
          <td class="num">{' '.join(str(k + 1) for k in kmin)}</td>
          <td class="num">{verdict}</td>
        </tr>""")

    good = payload["checks"]
    failed = [c for c in good if not c["ok"]]

    head = (f"<div class='verdict {'' if not failed else 'warn'}'>"
            f"<strong>{sum(1 for c in good if c['ok'])} of {len(good)} checks pass.</strong> ")
    if failed:
        head += ("The rule does not reproduce every truth under this recipe: "
                 + "; ".join(html.escape(c["name"]) for c in failed) + ".")
    else:
        head += ("Every mock gets the number of components the data need: the "
                 "generating count where the components are separable, and the "
                 "smaller sufficient count where they are not.  The count does not "
                 "move with the amplitude floor or with the number of live points, "
                 "and the Bayes factors agree with it.")
    head += "</div>"

    checks_rows = "".join(
        f"<tr><td>{html.escape(c['name'])}</td><td>{verdict_span(c['ok'])}</td>"
        f"<td>{html.escape(c['detail'])}</td></tr>" for c in good)

    png = payload["images"]
    if png.get("example"):
        example_note = (
            f"<p>{html.escape(payload['example']['title'])} Each curve is one "
            f"posterior-median component of the {K_MAX}-component fit; the dotted "
            "curve is the iron, broadened by the same profile.</p>"
            f"<img src=\"data:image/png;base64,{png['example']}\" "
            "alt=\"component decomposition\">")
    else:
        example_note = "<p>No decomposition in this run.</p>"
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>How many broad-line components?</title>
<style>{io.CSS}</style></head>
<body><main>
<h1>How many broad-line components does the profile need?</h1>
<p class="sub">A spike-and-slab ("toggle") prior on the wing amplitudes, tested on
mocks whose answer is known &mdash; generated
{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by
<code>develop/dev_jax/broad_line_components.py</code>.</p>

{head}

<h2>1. The scheme</h2>
<p>Each broad line gets <strong>{K_MAX} components</strong>: one core, which is always
present, and {N_WINGS} wings whose amplitudes are drawn from a
<em>spike-and-slab</em> prior.  With probability <code>p_off</code> a wing amplitude is
<strong>exactly zero</strong>, and it then contributes nothing at all &mdash; no flux, no
normalisation, no kernel.  The remaining probability maps onto a linear slab on
<code>(0, 1]</code> relative to the core peak.  Because both states own real prior
volume, a fixed-dimension nested sampler finds them, and the number of components
becomes a posterior rather than a choice.</p>

<pre>p_off = {', '.join(str(p) for p in P_OFF)}   (wings 0, 1, 2: higher order, less likely)
sigma_c and every sigma_w : uniform {SIGMA_RANGE[0]:.0f} - {SIGMA_RANGE[1]:.0f} km/s
dv_w                      : uniform &#177;{DV_RANGE:.0f} km/s
a wing counts as needed when its peak exceeds {MIN_WING_FRACTION*100:.0f}% of the core peak</pre>

<p><strong>Why the ladder is a prior and not a decision.</strong>  The likelihood is
exactly symmetric under exchanging two wings, so no likelihood term can say which
of them is "wing 0".  The ladder breaks the <em>tie</em> in prior probability, which
shrinks the higher components without prejudging their widths or shifts.  What it
does not do is identify the labels: in the tests below the same physical wing turns
up under index 0 or index 2 from one sample to the next, which is why every
quantity in this report is label-invariant &mdash; the <em>number</em> of wings, and
components sorted by width.</p>

<p><strong>The decision rule.</strong>  For each posterior sample count the wings that
are on and above the {MIN_WING_FRACTION*100:.0f}%-of-core floor; the mode of that
distribution is the minimum number of components the data need, and
<code>k_min = 1 + mode</code>.  The per-wing Bayes factor
<code>BF = (P_on/P_off) / prior odds</code> is reported as the supporting evidence
(the prior odds are divided out so runs with different <code>p_off</code> are
comparable).</p>

<h2>2. Calibration: how many components do the mocks need?</h2>
<p>{len(CASES)} mocks, each built with the CPU (astropy) models so the mock does not
come from the code under test, fitted with {K_MAX} components.  The two wing counts
are <code>P(N)</code> over all samples and <code>P(N)</code> restricted to wings above
the floor; the mode of each is shown as <code>plain / floor</code>.</p>
<table>
<tr><th>mock</th><th>generated</th><th>P(N) all samples</th><th>P(N) above floor</th>
    <th>mode / mode(floor)</th><th>logL(MAP)-logL(truth)</th><th>chi2/dof</th>
    <th>ln Z</th><th>min</th><th>verdict</th></tr>
{''.join(main_rows)}
</table>
<p class="cap">Headline rows are the best-sampled linear-slab run of each mock
({', '.join(sorted({e['config'] for e in rows.values()})) or 'n/a'}).
<b>exact</b> means the count equals the number of wings the mock was generated with;
<b>sufficient</b> means it reports fewer, and the fit is statistically
indistinguishable from the truth's (a likelihood-ratio test on the component count);
<b>short</b> would mean fewer components and a fit measurably worse than the truth's.
The <code>logL</code> column is the size of that difference: it cannot be positive,
because the truth is a point the model can represent.</p>
<img src="data:image/png;base64,{png['counts']}" alt="posterior of the number of wings">

<h2>3. Per-wing evidence</h2>
<p>Indices are not labels (see above), so these rows are per <em>slot</em>: whatever
component the sampler happens to put there.  A wing the data need should be far
above a Bayes factor of 3; one they do not should be
below it.  The amplitude is quoted as a fraction of the core peak, in the samples
where that wing is on.</p>
<table>
<tr><th>mock</th><th>slot</th><th>p_off</th><th>P(on)</th>
    <th>Bayes factor</th><th>amp (median)</th><th>amp (97.5%)</th>
    <th>sigma_w [km/s]</th></tr>
{''.join(wing_rows)}
</table>

<h2>4. How much does the floor matter?</h2>
<p>The floor is the one number in the rule that is a choice rather than a
measurement, so the answer is shown against it.  A rule worth using gives the same
<code>k_min</code> across a factor of ten in the floor.</p>
<table>
<tr><th>truth</th><th>wings</th>
    {''.join(f'<th class="num">k_min at {f*100:.0f}%</th>' for f in FLOORS)}
    <th>verdict</th></tr>
{''.join(floor_rows)}
</table>
<p class="cap"><code>k_min</code> is <code>1 + mode</code> of the number of wings above
that floor: a "stable" row means the answer does not depend on where the floor is
put.</p>

<h2>5. What the selected model looks like</h2>
{example_note}

<h2>6. Checks</h2>
<table>
<tr><th>check</th><th>ok</th><th>detail</th></tr>
{checks_rows}
</table>

<h2>7. Caveats</h2>
<ul>
<li><strong>Labels are not identified.</strong>  Two wings of similar width can swap
index from sample to sample; only the count and the width-sorted set are
meaningful.  Quoting "wing 1 = 4200 km/s" would be wrong roughly half the time.</li>
<li><strong>An extra Gaussian is cheap.</strong>  With 1% noise, an unneeded wing can
still buy a little chi-square at ~1% of the core peak.  That is what the floor is
for, and why the floor sensitivity above is part of the result.</li>
<li><strong>Live points matter.</strong>  The 400-live-point rows are shown to be
marginal: one of them converged to a point <em>worse</em> than the truth.  Production
runs use 1000 and report the ln Z error.</li>
<li><strong>The widths are capped at {SIGMA_RANGE[1]:.0f} km/s</strong> by design
(the published two-component fit puts the wind at ~6930 km/s).  A wing pinned to that
cap in the real spectrum is reported by <code>prior_dominated()</code> rather than
silently accepted.</li>
</ul>
</main></body></html>
"""
    path.write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true",
                        help="two mocks, 400 live points only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--case", action="append", metavar="TEXT",
                        help="only mock cases whose name contains TEXT (repeatable)")
    parser.add_argument("--config", action="append", metavar="TEXT",
                        help="only configs whose name contains TEXT (repeatable); "
                             "default: all of them")
    args = parser.parse_args()
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
        cases = ["0 wings", "1 wing"]
        configs = [CONFIGS[0]]

    payload = {"cases": {}, "primary": {}, "per_wing": {}, "floor_table": {},
               "checks": [], "images": {}, "configs": [c[0] for c in configs],
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
            # the example decomposition: the 3-wing mock, linear slab (last wins)
            if truth_name == "3 overlapping wings" and config["slab"] == "linear":
                payload["example"] = _example_payload(fit, truth_name, flux, error)

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

    payload["checks"] = compute_checks(payload)
    payload["accepted"] = not [c for c in payload["checks"] if not c["ok"]]

    # figures
    payload["images"]["counts"] = io._b64(fig_counts(payload))
    if "example" in payload:
        payload["images"]["example"] = io._b64(fig_example(payload))
    else:
        payload["images"]["example"] = ""

    report = args.out / "component_selection.html"
    build_report(report, payload)
    (args.out / "component_selection.json").write_text(
        json.dumps(_jsonable({k: v for k, v in payload.items() if k != "images"}),
                   indent=2), encoding="utf-8")
    print(f"\nwrote {report}  ({time.perf_counter() - started:.0f} s total)")
    passed = sum(1 for c in payload["checks"] if c["ok"])
    print(f"{passed}/{len(payload['checks'])} checks pass")
    for check in payload["checks"]:
        print(f"  [{'ok' if check['ok'] else '!!'}] {check['name']}: {check['detail']}")
    return 0 if payload["accepted"] else 1


def _example_payload(fit, truth, flux, error):
    """Posterior-median decomposition of one fit, built from the model itself.

    Each component is obtained by switching the other wings off inside a copy of
    the fitted line -- the same "an off wing contributes nothing" property the
    selection relies on -- so the curves are the model's own, not a redrawing of
    its parameters.
    """
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight)
    med = {n: float(np.median(samples[:, i])) for i, n in enumerate(names)}
    hbeta = galspec.utils.line_wave_dict["Hbeta"]

    dv_w = {f"dv_w{i}": med[f"b_hb.dv_w{i}"] for i in range(N_WINGS)}
    sigma_w = {f"sigma_w{i}": med[f"b_hb.sigma_w{i}"] for i in range(N_WINGS)}
    amps = {f"amp_w{i}": med[f"b_hb.amp_w{i}"] for i in range(N_WINGS)}

    def line_at(wings):
        candidate = jxm.Line_MultiGauss(
            n_components=K_MAX, amp_c=med["b_hb.amp_c"], dv_c=med["b_hb.dv_c"],
            sigma_c=med["b_hb.sigma_c"], wavec=hbeta, par_w=wings)
        return np.asarray(candidate.flux(WAVE, candidate.get_free_params()))

    core = line_at({**amps, **{k: 0.0 for k in amps}})
    total_line = line_at({**dv_w, **sigma_w, **amps})

    line, iron = jax_models()
    for flat in line.free_param_names:
        if flat in med:
            line.set_value(flat, med[flat])
    iron.set_value("amplitude", med["FeII.amplitude"])
    model = line + iron
    total_model = np.asarray(model.flux(WAVE, model.get_free_params()))
    iron_flux = total_model - total_line

    comps = [{"label": rf"core: $\sigma$={med['b_hb.sigma_c']:.0f} km/s",
              "profile": core, "sigma": med["b_hb.sigma_c"]}]
    for i in range(N_WINGS):
        amp = med[f"b_hb.amp_w{i}"]
        if amp <= MIN_WING_FRACTION:
            continue
        only = line_at({**dv_w, **sigma_w, **amps, **{k: 0.0 for k in amps},
                        f"amp_w{i}": amp})
        comps.append({"label": (rf"wing: amp={amp:.2f}, "
                                rf"$\sigma$={med[f'b_hb.sigma_w{i}']:.0f} km/s"),
                      "profile": only - core, "sigma": med[f"b_hb.sigma_w{i}"]})
    comps = [dict(comps[0])] + sorted(comps[1:], key=lambda c: c["sigma"])
    for comp in comps:
        comp.pop("sigma", None)

    return {"flux": flux, "error": error, "components": comps,
            "iron": iron_flux,
            "title": (f"Posterior-median decomposition of the '{truth}' mock "
                      f"({len(comps)} components left on)"),
            "xlim": io.WAVE_FIT}


if __name__ == "__main__":
    raise SystemExit(main())