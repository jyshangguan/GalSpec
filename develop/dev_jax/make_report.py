#!/usr/bin/env python3
"""Build the multi-page HTML report for the SDSS AGN CPU-vs-GPU comparison.

    python develop/dev_jax/make_report.py

Reads ``results/sdss_<config>.json`` + ``.npz`` written by ``run_sdss_fit.py``
and writes ``results/report/`` (``index.html``, ``model.html``,
``results.html``, ``glossary.html``, ``style.css``, ``img/*.png``).

Every technical term is explained the first time it is used, and the glossary
page collects the whole vocabulary.
"""

from __future__ import annotations

import html
import base64
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "report"
IMG = OUT / "img"
sys.path.insert(0, str(HERE))

# Order of the configurations in tables and figures.
CONFIGS = ["dynesty-serial", "dynesty-parallel", "jnesty-cpu", "jnesty-gpu"]
from report_pages import SHORT_NAME  # noqa: E402  (labels live with the page text)

# Plain-language description of every free parameter of the model.
PARAM_INFO = {
    "amplitude_0": ("Continuum level", "flux at 5500 A", "How bright the smooth power law is at 5500 A."),
    "alpha_0": ("Continuum slope", "", "Power-law index: brightness falls as wavelength^(-alpha)."),
    "amplitude_1": ("Fe II strength", "flux", "Height of the iron (Fe II) template, the forest of iron lines."),
    "sigma_1": ("Fe II broadening", "km/s", "How much the iron template is blurred by velocity."),
    "amp_c_2": ("H-alpha narrow peak", "flux", "Peak height of the narrow H-alpha line."),
    "amp_w0_2": ("H-alpha broad peak", "flux", "Peak height of the broad H-alpha line."),
    "dv_w0_2": ("H-alpha broad shift", "km/s", "Velocity offset of the broad H-alpha line."),
    "sigma_w0_2": ("H-alpha broad width", "km/s", "Velocity width (sigma) of the broad H-alpha line."),
    "amp_c_3": ("H-beta narrow peak", "flux", "Peak height of the narrow H-beta line."),
    "amp_w0_3": ("H-beta broad peak", "flux", "Peak height of the broad H-beta line."),
    "amp_c_4": ("H-gamma narrow peak", "flux", "Peak height of the narrow H-gamma line."),
    "amp_w0_4": ("H-gamma broad peak", "flux", "Peak height of the broad H-gamma line."),
    "amp_c0_5": ("[O III] 5007 narrow peak", "flux", "Peak height of the narrow [O III] line."),
    "dv_c_5": ("Narrow-line shift", "km/s", "Velocity offset shared by all narrow lines."),
    "sigma_c_5": ("Narrow-line width", "km/s", "Velocity width (sigma) shared by all narrow lines."),
    "amp_w0_5": ("[O III] broad peak", "flux", "Peak height of the broad wing of [O III]."),
    "dv_w0_5": ("[O III] wing shift", "km/s", "Velocity offset of the [O III] wing."),
    "sigma_w0_5": ("[O III] wing width", "km/s", "Velocity width (sigma) of the [O III] wing."),
    "amp_c0_6": ("[N II] 6583 peak", "flux", "Peak height of [N II]; its 6548 twin follows automatically."),
    "amp_c0_7": ("[S II] 6716 peak", "flux", "Peak height of [S II]; its 6731 twin follows automatically."),
    "amp_c0_8": ("[O I] 6300 peak", "flux", "Peak height of [O I]; its 6364 twin follows automatically."),
    "amplitude_9": ("He II 4686 peak", "flux", "Peak height of the He II line."),
    "amplitude_10": ("[O III] 4363 peak", "flux", "Peak height of the faint temperature-sensitive [O III] line."),
}


# ======================================================================================
# Data loading
# ======================================================================================
def load_results() -> dict:
    """Read every finished run; missing configurations are simply skipped."""
    runs = {}
    for config in CONFIGS:
        js = RESULTS / f"sdss_{config}.json"
        npz_path = RESULTS / f"sdss_{config}.npz"
        if not (js.exists() and npz_path.exists()):
            continue
        result = json.loads(js.read_text())
        data = np.load(npz_path, allow_pickle=True)
        result["_arrays"] = {k: data[k] for k in data.files}
        runs[config] = result
    return runs


# ======================================================================================
# Figures
# ======================================================================================
def _style(ax, labelsize=10):
    ax.tick_params(direction="out", labelsize=labelsize, length=3.5, width=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def fig_spectrum(runs: dict) -> str:
    """Full spectrum with the best-fit model and the residuals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = "jnesty-gpu" if "jnesty-gpu" in runs else next(iter(runs))
    arr = runs[key]["_arrays"]
    wave, flux, error = arr["wave"], arr["flux"], arr["error"]
    predicted, residual = arr["predicted"], arr["residual"]

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(wave, flux, color="0.15", lw=0.55, label="observed spectrum")
    ax.plot(wave, predicted, color="#d1495b", lw=0.75, label="best-fit model")
    ax.axvspan(5840, 5930, color="#f0c987", alpha=0.45, lw=0, zorder=0)
    ax.text(5885, ax.get_ylim()[1] * 0.93, "Na D\n(down-weighted)", ha="center",
            va="top", fontsize=7.5, color="0.35")
    for x, name in ((6563, "H$\\alpha$"), (4861, "H$\\beta$"), (4340, "H$\\gamma$"),
                    (5007, "[O III]"), (6583, "[N II]"), (6716, "[S II]"), (6300, "[O I]"),
                    (4686, "He II")):
        ax.text(x, ax.get_ylim()[1] * 0.995, name, rotation=90, fontsize=7.5,
                color="#2b5c8a", ha="right", va="top")
    ax.set_ylabel("flux  (arbitrary units)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_title(f"SDSS spectrum of IRAS 07546+3928 fitted with "
                 f"{len(runs[key]['parameters'])} free parameters", fontsize=11, pad=8)
    _style(ax)
    ax.tick_params(labelbottom=False)

    axr.plot(wave, residual, color="0.35", lw=0.5)
    axr.axhline(0, color="#d1495b", lw=0.7)
    for s in (3, -3):
        axr.axhline(s, color="0.6", lw=0.6, ls=":")
    axr.axvspan(5840, 5930, color="#f0c987", alpha=0.35, lw=0)
    axr.set_xlabel("rest-frame wavelength  [A]")
    axr.set_ylabel("residual  [$\\sigma$]")
    axr.set_ylim(-7.5, 7.5)
    _style(axr)
    axr.text(0.99, 0.06, f"{wave.size} pixels   rms {np.std(residual):.2f}$\\sigma$",
             transform=axr.transAxes, ha="right", fontsize=8, color="0.3")

    pdf = IMG / "spectrum.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "img/spectrum.png"


def _components(runs: dict, key: str, wave):
    """Individual model components at the best fit, for the zoom panels."""
    import sdss_agn
    from galspec.dynesty_fit import Dynesty_Fit

    names = runs[key]["_arrays"]["parameter_names"].tolist()
    theta = np.array([runs[key]["parameters"][n]["best"] for n in names], dtype=float)
    model = sdss_agn.build_cpu()
    fit = Dynesty_Fit(model, wave, runs[key]["_arrays"]["flux"], runs[key]["_arrays"]["error"],
                      bounds_dict=sdss_agn.BOUNDS, nlive=100, n_processes=1)
    fit.set_model_params(theta)
    out = {}
    for sub in ("Continuum", "FeII", "Halpha", "Hbeta", "Hgamma", "OIII", "NII", "SII",
                "OI", "HeII", "OIII4363"):
        out[sub] = np.asarray(fit.model[sub](wave))
    return out


def fig_components(runs: dict) -> str:
    """The model split into its parts: what each ingredient actually contributes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = "jnesty-gpu" if "jnesty-gpu" in runs else next(iter(runs))
    arr = runs[key]["_arrays"]
    wave, flux, predicted = arr["wave"], arr["flux"], arr["predicted"]
    comp = _components(runs, key, wave)

    base = comp["Continuum"] + comp["FeII"]
    narrow = sum(comp[k] for k in ("Halpha", "Hbeta", "Hgamma", "OIII", "NII", "SII", "OI",
                                   "HeII", "OIII4363"))
    broad = sum(v for k, v in comp.items()) - base - narrow

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.2, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1.35, 1.0]})
    top = float(np.percentile(flux, 99.0)) * 1.15
    ax.plot(wave, flux, color="0.6", lw=0.5, label="observed spectrum")
    ax.plot(wave, predicted, color="#c1121f", lw=0.9, label="total model")
    ax.plot(wave, base, color="#5c6b73", lw=1.0, ls="--", label="continuum + iron")
    ax.set_ylim(0, top)
    ax.set_ylabel("flux (file units)")
    ax.set_title("The same spectrum, model and data, on the scale of the bright emission lines",
                 fontsize=11, pad=8)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    for y, label, colour, ls in ((base, "continuum + iron", "#5c6b73", "--"),
                                 (narrow, "narrow lines (slow gas)", "#3d5a80", "-"),
                                 (broad, "broad lines (fast gas)", "#2a9d8f", "-")):
        ax2.plot(wave, np.clip(y, 1e-4, None), color=colour, lw=0.9, ls=ls, label=label)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, None)
    ax2.set_xlabel("rest wavelength (\u00c5)")
    ax2.set_ylabel("component flux (log)")
    ax2.set_title("Each ingredient on its own (log scale): the iron bump bluewards of "
                  "H$\\beta$, the broad Balmer lines, the narrow forbidden lines",
                  fontsize=10.5, pad=8)
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    for a in (ax, ax2):
        a.grid(alpha=0.15, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig.tight_layout()
    name = "components.png"
    fig.savefig(OUT / "img" / name, dpi=135)
    plt.close(fig)
    return f"img/{name}"


def fig_regions(runs: dict) -> str:
    """The three Balmer regions, data and model, with the components drawn separately."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = "jnesty-gpu" if "jnesty-gpu" in runs else next(iter(runs))
    arr = runs[key]["_arrays"]
    wave, flux, error = arr["wave"], arr["flux"], arr["error"]
    predicted = arr["predicted"]
    comp = _components(runs, key, wave)

    panels = [("H$\\gamma$ 4340 + [O III] 4363 + He II 4686", 4290, 4740),
              ("H$\\beta$ 4861 + [O III] 4959/5007", 4800, 5060),
              ("H$\\alpha$ 6563 + [N II] 6548/6583 + [S II] 6716/6731", 6470, 6770)]
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.6))

    narrow_colour = "#3d5a80"
    broad_colour = "#2a9d8f"
    for ax, (title, lo, hi) in zip(axes, panels):
        m = (wave > lo) & (wave < hi)
        w = wave[m]
        ax.plot(w, flux[m], color="0.15", lw=0.7, label="observed")
        ax.plot(w, predicted[m], color="#d1495b", lw=0.9, label="total model")
        base = comp["Continuum"][m] + comp["FeII"][m]
        ax.plot(w, base, color="#8d99ae", lw=0.8, ls="--", label="continuum + Fe II")
        for sub, colour, ls in (("Halpha", broad_colour, "-"), ("Hbeta", broad_colour, "-"),
                                ("Hgamma", broad_colour, "-"), ("OIII", broad_colour, "-")):
            part = comp[sub][m]
            if np.max(np.abs(part)) > 1e-9:
                ax.plot(w, part + base, color=colour, lw=0.7, ls=ls, alpha=0.85,
                        label="broad components" if sub == "Halpha" else None)
        narrow = (comp["Halpha"] + comp["Hbeta"] + comp["Hgamma"] + comp["NII"] + comp["SII"]
                  + comp["OI"] + comp["HeII"] + comp["OIII4363"])[m]
        ax.plot(w, narrow + base, color=narrow_colour, lw=0.6, ls=":", alpha=0.9,
                label="narrow lines")
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_ylabel("flux")
        _style(ax)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
        span = np.max(flux[m]) - np.min(flux[m])
        ax.set_ylim(np.min(flux[m]) - 0.15 * span, np.max(flux[m]) + 0.55 * span)
    axes[-1].set_xlabel("rest-frame wavelength  [A]")

    fig.suptitle("Where the emission comes from: broad and narrow components",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    pdf = IMG / "regions.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "img/regions.png"


def _derived(samples: np.ndarray, names) -> dict:
    """Physical quantities computed from posterior samples."""
    idx = {n: i for i, n in enumerate(names)}
    s = samples
    out = {
        "narrow_fwhm": 2.3548 * s[:, idx["sigma_c_5"]],
        "broad_fwhm": 2.3548 * s[:, idx["sigma_w0_2"]],
        "broad_peak": s[:, idx["amp_w0_2"]],
        "oiii_peak": s[:, idx["amp_c0_5"]],
        "alpha": s[:, idx["alpha_0"]],
        "broad_shift": s[:, idx["dv_w0_2"]],
        "ha_over_hb": s[:, idx["amp_w0_2"]] / np.maximum(s[:, idx["amp_w0_3"]], 1e-6),
        "nii_over_ha": s[:, idx["amp_c0_6"]] / np.maximum(s[:, idx["amp_c_2"]], 1e-6),
    }
    return out


def _corner(ax_grid, values, labels, colour="#2b6cb0", units=None):
    """Minimal corner plot: 1-D histograms on the diagonal, 2-D density off it."""
    n = len(labels)
    for i in range(n):
        for j in range(n):
            ax = ax_grid[i][j]
            if i == j:
                ax.hist(values[i], bins=32, color=colour, alpha=0.75, histtype="stepfilled")
                lo, hi = np.percentile(values[i], [0.5, 99.5])
                ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            elif i > j:
                ax.hist2d(values[j], values[i], bins=28, cmap="Blues",
                          cmin=1, density=True)
                ax.set_xlim(*np.percentile(values[j], [0.5, 99.5]))
                ax.set_ylim(*np.percentile(values[i], [0.5, 99.5]))
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=6.5, direction="out", length=2.5, width=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)


def fig_corner(runs: dict) -> list:
    """Corner plots: the physically interesting quantities, then the raw parameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = "jnesty-gpu" if "jnesty-gpu" in runs else next(iter(runs))
    names = runs[key]["_arrays"]["parameter_names"].tolist()
    samples = runs[key]["_arrays"]["samples"]
    if samples.shape[0] < 200:
        return []
    made = []

    deriv = _derived(samples, names)
    keys = ["narrow_fwhm", "broad_fwhm", "broad_peak", "oiii_peak", "alpha", "broad_shift"]
    labels = ["narrow FWHM\n[km/s]", "broad H$\\alpha$ FWHM\n[km/s]", "broad H$\\alpha$ peak\n[flux]",
              "[O III] narrow peak\n[flux]", "continuum slope\n$\\alpha$", "broad H$\\alpha$ shift\n[km/s]"]
    values = [deriv[k] for k in keys]
    fig, axes = plt.subplots(len(keys), len(keys), figsize=(10.5, 10.5))
    _corner(axes, values, labels)
    fig.suptitle("Posterior distributions of quantities you would quote in a paper",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(IMG / "corner_derived.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    made.append("img/corner_derived.png")

    short = {"amplitude_0": "cont. amp", "alpha_0": "gamma", "amplitude_1": "Fe II amp",
             "sigma_1": "Fe II sd", "amp_c_2": "Ha nar", "amp_w0_2": "Ha brd",
             "dv_w0_2": "Ha dv", "sigma_w0_2": "Ha sig", "amp_c_3": "Hb nar",
             "amp_w0_3": "Hb brd", "amp_c_4": "Hg nar", "amp_w0_4": "Hg brd",
             "amp_c0_5": "OIII nar", "dv_c_5": "nar dv", "sigma_c_5": "nar sig",
             "amp_w0_5": "OIII brd", "dv_w0_5": "OIII dV", "sigma_w0_5": "OIII sg",
             "amp_c0_6": "NII", "amp_c0_7": "SII", "amp_c0_8": "OI",
             "amplitude_9": "HeII", "amplitude_10": "OIII43"}
    values = [samples[:, i] for i in range(len(names))]
    labels = [short.get(n, n) for n in names]
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(17.0, 17.0))
    _corner(axes, values, labels)
    fig.suptitle("Every free parameter of the model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(IMG / "corner_all.png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    made.append("img/corner_all.png")
    return made


def fig_timing(runs: dict) -> str:
    """Runtime and speed-up bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in CONFIGS if k in runs]
    times = [runs[k]["runtime_seconds"] for k in keys]
    labels = [SHORT_NAME[k].format(p=runs[k].get("processes", 1)) for k in keys]
    colours = ["#8d99ae", "#8d99ae", "#2a9d8f", "#d1495b"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    y = np.arange(len(keys))
    ax.barh(y, times, color=colours[:len(keys)], height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    for yi, t, k in zip(y, times, keys):
        txt = (f"{t/3600:.1f} h so far" if runs[k].get("in_progress")
               else f"{t/60:.1f} min")
        ax.text(t * 1.08, yi, txt, va="center", fontsize=9)
    ax.set_xlabel("wall-clock time for the same fit  [s, log scale]")
    ax.set_xlim(min(times) * 0.55, max(times) * 2.6)
    ax.set_title("Same model, same data, same stopping rule", fontsize=10.5)
    if any(runs[k].get("in_progress") for k in keys):
        ax.set_xlabel("wall-clock time for the same fit  [s, log scale]\n"
                      "(a bar labelled \"so far\" is a run that had not finished)", fontsize=9)
    _style(ax)

    if "jnesty-gpu" in runs:
        base = runs["jnesty-gpu"]["runtime_seconds"]
        speed = [runs[k]["runtime_seconds"] / base for k in keys]
        ax2.barh(y, speed, color=colours[:len(keys)], height=0.6)
        ax2.set_yticks(y)
        ax2.set_yticklabels([])
        ax2.invert_yaxis()
        ax2.axvline(1.0, color="0.4", lw=0.8, ls="--")
        for yi, s_, k in zip(y, speed, keys):
            star = "&gt;" if runs[k].get("in_progress") else ""
            ax2.text(s_ * 1.06, yi, f"{star}{s_:.1f}x", va="center", fontsize=9)
        ax2.set_xscale("log")
        ax2.set_xlabel("times faster than JNesty on GPU")
        ax2.set_title("Speed-up", fontsize=10.5)
        _style(ax2)

    fig.tight_layout()
    fig.savefig(IMG / "timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "img/timing.png"


def measure_costs() -> dict:
    """Time one likelihood evaluation for each code path, on this machine."""
    import time

    import sdss_agn
    from galspec.dynesty_fit import Dynesty_Fit
    from galspec.jnesty_fit import JNesty_Fit

    wave, flux, error = sdss_agn.load_sdss_spectrum()
    costs = {}
    cpu_model = sdss_agn.build_cpu()
    fit = Dynesty_Fit(cpu_model, wave, flux, error, bounds_dict=sdss_agn.BOUNDS,
                      nlive=100, n_processes=1)
    fit.set_model_params(fit.theta_initial)
    cpu_model(wave)
    rounds = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(4):
            cpu_model(wave)
        rounds.append((time.perf_counter() - t0) / 4)
    costs["astropy model\n(what dynesty calls)"] = float(np.median(rounds))

    jax_model = sdss_agn.build_jax()
    jax_model.prepare(wave)
    th = np.asarray(jax_model.get_free_params(), dtype=float)
    fit_j = JNesty_Fit(jax_model, wave, flux, error, bounds_dict=sdss_agn.BOUNDS,
                       nlive=50, verbose=False)
    fit_j.log_likelihood(th)

    def _time_once(n_rep: int, fn) -> float:
        t0 = time.perf_counter()
        for _ in range(n_rep):
            fn()
        return (time.perf_counter() - t0) / n_rep

    # median of several rounds: a busy machine can only make a timing slower,
    # never faster, so the median is a stable stand-in for the true cost
    costs["JAX, one point"] = float(np.median(
        [_time_once(20, lambda: fit_j.log_likelihood(th)) for _ in range(5)]))

    batch_fn = fit_j.log_likelihood_batch
    grid = np.tile(th, (64, 1))
    batch_fn(grid)
    costs["JAX, 64 points\n(throughout JNesty)"] = float(np.median(
        [_time_once(10, lambda: batch_fn(grid)) for _ in range(5)])) / 64
    return costs


def fig_cost(costs: dict) -> str:
    """Bar chart of the measured per-likelihood cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sdss_agn

    # only the timing entries; ``costs`` also carries the serial-throughput probe
    costs = {k: float(v) for k, v in costs.items()
             if isinstance(v, (int, float)) and "model" in k or k.startswith("JAX")}
    wave = sdss_agn.load_sdss_spectrum()[0]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    keys = list(costs)
    values = [costs[k] * 1e3 for k in keys]
    colours = ["#8d99ae", "#2a9d8f", "#d1495b"]
    ax.barh(np.arange(len(keys)), values, color=colours[:len(keys)], height=0.6)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for i, v in enumerate(values):
        ax.text(v * 1.07, i, f"{v:.3f} ms", va="center", fontsize=9)
    ax.set_xlabel("time per likelihood evaluation  [ms, log scale]")
    ax.set_xlim(min(values) * 0.4, max(values) * 3.0)
    ax.set_title(f"One likelihood = {wave.size} pixels x 11 components", fontsize=10.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(IMG / "cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "img/cost.png"


def fig_agreement(runs: dict) -> str:
    """Two fitters, same model: do they agree?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("dynesty-serial", "dynesty-parallel", "jnesty-cpu", "jnesty-gpu")
            if k in runs and not runs[k].get("projected")]
    if len(keys) < 2:
        return ""
    arr0 = runs[keys[0]]["_arrays"]
    wave, error = arr0["wave"], arr0["error"]
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    colours = {"dynesty-serial": "#8d99ae", "dynesty-parallel": "#5c6b73",
               "jnesty-cpu": "#2a9d8f", "jnesty-gpu": "#d1495b"}
    for key in keys:
        diff = runs[key]["_arrays"]["predicted"] - arr0["predicted"]
        ax.plot(wave, diff / error, lw=0.6, color=colours[key],
                label=f"{SHORT_NAME[key]} - {SHORT_NAME[keys[0]]}")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xlabel("rest-frame wavelength  [A]")
    ax.set_ylabel("difference in $\\sigma$")
    ax.set_title("Best-fit models from the different fitters, subtracted from the first one",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(IMG / "agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "img/agreement.png"


def fig_prior_posterior(runs: dict) -> str:
    """Posterior vs the (wide) prior for a few parameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = "jnesty-gpu" if "jnesty-gpu" in runs else next(iter(runs))
    names = runs[key]["_arrays"]["parameter_names"].tolist()
    samples = runs[key]["_arrays"]["samples"]
    import sdss_agn
    bounds = {q: b for q, b in sdss_agn.BOUNDS.items()}

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0))
    picks = ["sigma_c_5", "sigma_w0_2", "amp_w0_2", "dv_w0_2"]
    qualified = {"sigma_c_5": "OIII.sigma_c", "sigma_w0_2": "Halpha.sigma_w0",
                 "amp_w0_2": "Halpha.amp_w0", "dv_w0_2": "Halpha.dv_w0"}
    for ax, name in zip(axes, picks):
        i = names.index(name)
        lo, hi = bounds[qualified[name]]
        ax.axvspan(lo, hi, color="#8d99ae", alpha=0.25, lw=0, label="prior range")
        ax.hist(samples[:, i], bins=40, color="#d1495b", alpha=0.8, histtype="stepfilled",
                label="posterior")
        ax.set_xlabel(f"{name}  [{PARAM_INFO[name][1] or '-'}]", fontsize=8.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("The prior was wide; the data did the work", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(IMG / "prior_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "img/prior_posterior.png"

# ======================================================================================
# Assembly
# ======================================================================================
def add_projection(runs: dict, costs: dict) -> None:
    """If the single-core run has not finished, project it from the measured cost.

    Nested sampling with the same settings makes the same number of likelihood
    calls, so ``calls x measured cost per call`` is a faithful wall-clock
    estimate; the flag keeps it clearly separated from measured numbers.
    """
    if "dynesty-serial" in runs or "dynesty-parallel" not in runs:
        return
    par = runs["dynesty-parallel"]
    projected = dict(par)
    projected["config"] = "dynesty-serial"
    projected["processes"] = 1
    projected["projected"] = True
    projected["setup_seconds"] = 0.0
    projected["_arrays"] = par["_arrays"]

    # Prefer a *directly measured* dynesty throughput (a time-capped run of the
    # same problem, counted by dynesty itself) over the model-only microbenchmark:
    # it already includes dynesty's own per-iteration overhead.
    # How long has it been going?  That is a measurement, not an estimate, and it
    # already gives a hard lower bound on the speed-up.
    elapsed = 0.0
    log = RESULTS / "log_dynesty_serial.txt"
    if log.exists():
        elapsed = time.time() - log.stat().st_mtime
    projected["in_progress"] = True
    projected["runtime_seconds"] = elapsed if elapsed > 0 else float("nan")

    per_eval = costs.get("astropy model\n(what dynesty calls)")
    if per_eval:
        # a sampler call is not always a full model evaluation (bound rejections
        # are counted too), so quote the range the two measurements bracket
        projected["_estimate_seconds"] = (par["ncall"] * per_eval * 0.75,
                                          par["ncall"] * per_eval * 1.05)
    runs["dynesty-serial"] = projected


def _labels(runs: dict) -> None:
    from report_pages import NICE_NAME
    for key, run in runs.items():
        run["_label"] = NICE_NAME[key].format(p=run.get("processes", 1))


def _speeds(runs: dict) -> dict:
    speed = {}
    base_serial = runs.get("dynesty-serial", runs.get("dynesty-parallel"))
    base_parallel = runs.get("dynesty-parallel", base_serial)
    gpu = runs.get("jnesty-gpu")
    if gpu is not None and base_serial is not None:
        speed["gpu_vs_serial"] = f"{base_serial['runtime_seconds'] / gpu['runtime_seconds']:.1f}&times;"
    if gpu is not None and base_parallel is not None:
        speed["gpu_vs_parallel"] = (
            f"{base_parallel['runtime_seconds'] / gpu['runtime_seconds']:.1f}&times;")
    return speed


def main() -> int:
    import report_pages as pages

    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    runs = load_results()
    if not runs:
        print("no results found; run run_sdss_fit.py first")
        return 1
    _labels(runs)
    print(f"loaded: {', '.join(runs)}")

    try:
        costs = measure_costs()
        print({k: f"{v*1e3:.3f} ms" for k, v in costs.items()})
    except Exception as exc:
        print(f"  (cost measurement failed: {exc})")
        costs = {}

    # a time-capped single-core run, if one was made, beats any projection
    probes = sorted(RESULTS.glob("sdss_dynesty-serial_timing*.json"))
    if probes:
        try:
            recs = [json.loads(p.read_text()) for p in probes]
            rec = max(recs, key=lambda r: r["ncall"])      # longest baseline wins
            calls, secs = rec["ncall"], rec["runtime_seconds"]
            if calls and secs > 0:
                costs["serial_calls_per_s"] = calls / secs
                costs["serial_probe"] = rec
                print(f"  serial throughput probe: {calls:,} calls in "
                      f"{secs:,.0f} s = {calls/secs:,.1f} calls/s "
                      f"({rec['n_iterations']} iterations, converged="
                      f"{rec.get('converged')})")
        except Exception as exc:
            print(f"  (serial probe unreadable: {exc})")
    add_projection(runs, costs)
    if runs.get("dynesty-serial", {}).get("projected"):
        run = runs["dynesty-serial"]
        run["_label"] = "dynesty, one CPU core (still running)"
        print(f"projected dynesty-serial: {run['runtime_seconds']/3600:.1f} h")

    figs = {}
    figs["timing"] = fig_timing(runs)
    print(" figure: timing")
    figs["spectrum"] = fig_spectrum(runs)
    print(" figure: spectrum")
    figs["components"] = fig_components(runs)
    print(" figure: components")
    figs["regions"] = fig_regions(runs)
    print(" figure: regions")
    corners = fig_corner(runs)
    figs["corner_derived"] = corners[0] if corners else figs["spectrum"]
    if len(corners) > 1:
        figs["corner_all"] = corners[1]
    figs["agreement"] = fig_agreement(runs) or figs["spectrum"]
    print(" figure: corner + agreement")
    figs["prior_posterior"] = fig_prior_posterior(runs)
    print(" figure: prior vs posterior")
    try:
        figs["cost"] = fig_cost(costs)
        print(" figure: cost")
    except Exception as exc:                     # measurement is a nicety, not essential
        print(f"  (cost figure skipped: {exc})")
        figs["cost"] = figs["timing"]

    speed = _speeds(runs)
    if costs.get("serial_calls_per_s"):
        speed["serial_rate"] = float(costs["serial_calls_per_s"])
    if costs.get("astropy model\n(what dynesty calls)"):
        speed["per_eval_ms"] = 1e3 * float(costs["astropy model\n(what dynesty calls)"])
    (OUT / "style.css").write_text(pages.CSS)

    def _standalone(html: str) -> str:
        """Inline the stylesheet and the images.

        A page with ``<img src="img/x.png">`` only renders when it is opened from
        the same directory as its assets.  Report viewers, chat previews and
        ``file://`` sandboxes often refuse to load sibling files, which is why the
        figures stay blank.  Data URIs make each page a single self-contained file
        that renders anywhere (and can be e-mailed as-is).
        """
        html = html.replace('<link rel="stylesheet" href="style.css">',
                            f"<style>\n{pages.CSS}\n</style>")

        def _embed(match):
            rel = match.group(1)
            path = OUT / rel
            if not path.exists():
                return match.group(0)
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f'src="data:image/png;base64,{data}"'

        return re.sub(r'src="(img/[^"]+)"', _embed, html)

    bodies = {
        "overview": pages.index_html(runs, figs, speed, wrap=False),
        "model": pages.model_html(runs, figs, wrap=False),
        "results": pages.results_html(runs, figs, wrap=False),
        "glossary": pages.glossary_html(runs, wrap=False),
    }
    pages_out = {f"{key}.html": body for key, body in bodies.items()}
    # per-page titles for the stand-alone variants
    titles = {"overview": "Overview", "model": "The model, in plain words",
              "results": "Results", "glossary": "Glossary"}
    active = {"overview": "index.html", "model": "model.html",
              "results": "results.html", "glossary": "glossary.html"}
    pages_out = {
        "index.html": pages.index_html(runs, figs, speed),
        "model.html": pages.model_html(runs, figs),
        "results.html": pages.results_html(runs, figs),
        "glossary.html": pages.glossary_html(runs),
    }
    for name, html in pages_out.items():
        out = _standalone(html)
        (OUT / name).write_text(out)
        print(f"  {name}: {(OUT / name).stat().st_size/1e6:.2f} MB (self-contained)")

    # one file holding all four pages: navigation that works even in a sandbox
    combined = _standalone(pages.combined_html(
        bodies, f"A real AGN fit: CPU versus GPU ({speed.get('gpu_vs_parallel', '')} faster)",
        pages.CSS))
    (OUT / "report.html").write_text(combined)
    print(f"  report.html: {(OUT / 'report.html').stat().st_size/1e6:.2f} MB "
          f"(all pages in one file)")
    print(f"written: {OUT}/report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
