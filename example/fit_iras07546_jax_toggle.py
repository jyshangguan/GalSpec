#!/usr/bin/env python3
"""IRAS 07546+3928: four broad-line components, and the data decide how many.

    python example/fit_iras07546_jax_toggle.py             # the real run (hours)
    python example/fit_iras07546_jax_toggle.py --quick     # a few minutes

What this does
--------------
The published fit (``fit_iras07546_jax.py``) gives each broad Balmer line a core
plus **one** wind -- two components, because that is how the line was modelled.
Here the same lines get a core plus **three** wings, and each wing amplitude is
drawn from a *toggle* prior (``galspec.uniform_toggle``): with probability
``p_off`` it is **exactly zero**, and the wing then contributes nothing at all.
The number of components is no longer a choice made outside the fit; it is a
posterior, and one run answers "how many Gaussians do these profiles need?".

The recipe
----------
* three components x three lines (H-alpha, H-beta, H-gamma), one core plus three
  gated wings each; He I 5876 and He II 4686 keep a single broad Gaussian, and the
  Fe II kernel follows broad H-alpha component for component;
* ``sigma_c`` and every ``sigma_w`` share the prior 300-8000 km/s -- a wing is not
  required to be broader than the core, and the nested sampler is free to put a
  component anywhere in that range;
* ``p_off = 0.2, 0.5, 0.7`` for wings 0, 1, 2.  The likelihood cannot tell two
  wings apart (swapping them leaves the spectrum bit-identical), so the *prior*
  carries the only asymmetry there is: higher components are less likely.  That
  shrinks them without prejudging their widths or shifts;
* a wing counts as needed once its peak exceeds ``MIN_WING_FRACTION`` (3%) of the
  core peak.  Without that floor an extra Gaussian can hide at 1% of the core and
  buy a little chi-square out of the noise -- measured on mocks in
  ``develop/dev_jax/broad_line_components.py``, which also shows the rule
  recovering 0, 1, 2 and 3 wings from data with that many.

Reading the answer
------------------
The number of wings is a *label-invariant* count: for each posterior sample, count
the wings that are on and above the floor.  Components are then reported **sorted
by width**, never as "wing 0 = ...", because the sampler is free to call the same
physical wing 0 in one sample and 2 in the next.

Bayes factors are prior-corrected: ``BF = (P_on/P_off) / ((1-p_off)/p_off)``, so a
wing with ``p_off = 0.2`` is not credited for being likely a priori.

Outputs go to ``example/output/`` with the ``iras07546_jax_toggle_`` prefix:
``.npz`` (samples and data), ``.fits`` (the whole fit), ``_fit.png``,
``_corner.png``, ``_histograms.png``, ``_components.png``.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import galspec
from galspec import jax_models as jxm
from galspec.jnesty_fit_io import save_jnesty_fit_to_fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_iras07546_jax as base                                   # noqa: E402

# --------------------------------------------------------------------------------------
# The scheme
# --------------------------------------------------------------------------------------
#: wings after the core, so four components per broad Balmer line
N_WINGS = 3
#: prior probability that wing 0, 1, 2 is off: higher order, less likely a priori
P_OFF = (0.2, 0.5, 0.7)
#: one width prior for the core and for every wing [km/s]
SIGMA_RANGE = (300.0, 8000.0)
#: a wing counts as needed above this fraction of the core peak
MIN_WING_FRACTION = 0.03
#: the lines that get wings (the He lines keep one broad Gaussian)
WINGED_LINES = ("halpha", "hbeta", "hgamma")
#: the published answer, quoted from example/output/iras07546_jax_posterior.npz
PUBLISHED = {"free_parameters": 30, "log_evidence": 130493.85, "chi2_dof": 0.321}

NLIVE = 1000          # 40 free parameters: 400 was measured to be too few
DLOGZ = 0.01
MAX_ITERATIONS = 400_000


# --------------------------------------------------------------------------------------
# Priors: the published recipe, plus a toggle on every wing amplitude
# --------------------------------------------------------------------------------------
def make_toggle_priors(flux):
    """``(bounds, custom_priors, notes)`` with the wing amplitudes gated.

    Everything except the wing amplitudes is exactly what the published fit uses,
    so the two are comparable component by component.  Two things change:

    * every ``sigma_w`` gets the *core's* prior, 300-8000 km/s, because nothing
      says a wing must be broader than the core;
    * ``amp_w<i>`` gets ``galspec.uniform_toggle(p_off=P_OFF[i])`` -- half of the
      prior volume at exactly zero, the rest on a linear slab (0, 1].  The
      amplitude stays bounded by the core, which is what keeps the labels
      meaningful when only one component is present.
    """
    bounds, custom, notes = base.make_priors(flux, n_balmer_wings=N_WINGS)

    for i in range(N_WINGS):
        bounds[f"b_halpha.sigma_w{i}"] = SIGMA_RANGE
    gated = {}
    for line in WINGED_LINES:
        for i in range(N_WINGS):
            name = f"b_{line}.amp_w{i}"
            custom[name] = galspec.uniform_toggle(p_off=P_OFF[i])
            gated[name] = P_OFF[i]

    notes = [note for note in notes if note[0] != "broad wind"]
    notes += [
        ("broad components", f"1 core + {N_WINGS} wings for "
                             + ", ".join(f"b_{line}" for line in WINGED_LINES)),
        ("wing widths", f"{SIGMA_RANGE[0]:.0f}-{SIGMA_RANGE[1]:.0f} km/s, same as the "
                        "core: a wing need not be broader"),
        ("wing toggle", "p_off = " + ", ".join(f"{p:.1f}" for p in P_OFF)
                        + ": this fraction of the prior has amp_w = 0 exactly, so the "
                          "sampler can switch the wing off"),
        ("needed", f"a wing counts above {MIN_WING_FRACTION*100:.0f}% of the core peak"),
    ]
    return bounds, custom, notes


# --------------------------------------------------------------------------------------
# Reading the posterior
# --------------------------------------------------------------------------------------
def amp_matrix(fit):
    """``(samples, N_WINGS)`` wing amplitudes of broad H-alpha (0 = switched off)."""
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight)
    return samples, names, np.stack(
        [samples[:, names.index(f"b_halpha.amp_w{i}")] for i in range(N_WINGS)], 1)


def component_count(fit, floor=0.0):
    """Posterior of the number of wings that are on (``floor=0``) or above ``floor``.

    Label-invariant by construction: it counts, it does not name.
    """
    _, _, amps = amp_matrix(fit)
    counts = (amps > floor).sum(1)
    return np.array([float(np.mean(counts == n)) for n in range(N_WINGS + 1)])


def wing_evidence(fit):
    """Per-wing posterior probability, prior-corrected Bayes factor and size."""
    samples, names, amps = amp_matrix(fit)
    rows = []
    for i in range(N_WINGS):
        on = amps[:, i] > 0.0
        p_on = float(on.mean())
        if p_on <= 0.0:
            bayes = 0.0
        elif p_on >= 1.0:
            bayes = np.inf
        else:
            bayes = (p_on / (1.0 - p_on)) / galspec.toggle_prior_odds(P_OFF[i])
        sigma = samples[on, names.index(f"b_halpha.sigma_w{i}")] if on.any() else None
        dv = samples[on, names.index(f"b_halpha.dv_w{i}")] if on.any() else None
        rows.append({
            "wing": i,
            "p_off": P_OFF[i],
            "p_on": p_on,
            "bayes": bayes,
            "amp": float(np.median(amps[on, i])) if on.any() else 0.0,
            "amp_97": float(np.percentile(amps[on, i], 97.5)) if on.any() else 0.0,
            "sigma": float(np.median(sigma)) if sigma is not None else np.nan,
            "dv": float(np.median(dv)) if dv is not None else np.nan,
            "counted": bool(on.any() and np.median(amps[on, i]) > MIN_WING_FRACTION),
        })
    return rows


def sorted_components(fit, floor=MIN_WING_FRACTION):
    """Width-sorted components, with the flux fraction each one carries.

    Labels are not identified, so this is the honest way to present a
    decomposition: rank in width, plus how often the posterior has that many
    components at all.
    """
    samples, names, amps = amp_matrix(fit)
    sigmas = np.stack([samples[:, names.index(f"b_halpha.sigma_w{i}")]
                       for i in range(N_WINGS)], 1)
    dv = np.stack([samples[:, names.index(f"b_halpha.dv_w{i}")]
                   for i in range(N_WINGS)], 1)
    wavec = base.LINE_WAVE["Halpha"]
    h = 2.0 * np.sqrt(2.0 * np.log(2.0))

    # the core's own flux, for the fractions
    amp_c = samples[:, names.index("b_halpha.amp_c")]
    sigma_c = samples[:, names.index("b_halpha.sigma_c")]
    core_flux = amp_c * sigma_c * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3

    rows = []
    for rank in range(N_WINGS):
        amps_r, sig_r, dv_r, frac_r = [], [], [], []
        for s in range(samples.shape[0]):
            use = np.argsort(sigmas[s])          # narrowest first
            keep = [j for j in use if amps[s, j] > floor]
            if len(keep) <= rank:
                continue
            j = keep[rank]
            flux = (amp_c[s] * amps[s, j] * sigmas[s, j] * np.sqrt(2.0 * np.pi)
                    * wavec / 299792.458 * 1e3)
            total = core_flux[s] + sum(
                amp_c[s] * amps[s, k] * sigmas[s, k] * np.sqrt(2.0 * np.pi)
                * wavec / 299792.458 * 1e3 for k in keep)
            amps_r.append(amps[s, j])
            sig_r.append(sigmas[s, j])
            dv_r.append(dv[s, j])
            frac_r.append(flux / total)
        rows.append({
            "rank": rank,
            "n": len(amps_r),
            "amp": float(np.median(amps_r)) if amps_r else 0.0,
            "sigma": float(np.median(sig_r)) if sig_r else np.nan,
            "fwhm": float(np.median(sig_r)) * h if sig_r else np.nan,
            "dv": float(np.median(dv_r)) if dv_r else np.nan,
            "flux_fraction": float(np.median(frac_r)) if frac_r else 0.0,
        })
    return rows


def core_row(fit):
    """The core's own numbers (always present, so no toggle and no labels)."""
    samples, names, _ = amp_matrix(fit)
    amp_c = samples[:, names.index("b_halpha.amp_c")]
    sigma_c = samples[:, names.index("b_halpha.sigma_c")]
    dv_c = samples[:, names.index("b_halpha.dv_c")]
    wavec = base.LINE_WAVE["Halpha"]
    flux = amp_c * sigma_c * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3
    h = 2.0 * np.sqrt(2.0 * np.log(2.0))
    return {"amp": float(np.median(amp_c)),
            "sigma": float(np.median(sigma_c)),
            "fwhm": float(np.median(sigma_c)) * h,
            "dv": float(np.median(dv_c)),
            "flux": float(np.median(flux))}


def print_selection(fit):
    """The answer, in the order a reader needs it."""
    p_wings = component_count(fit, floor=0.0)
    p_needed = component_count(fit, floor=MIN_WING_FRACTION)
    cumulative = np.cumsum(p_needed)
    k_min = int(np.argmax(p_needed))

    print("\n  how many wings does the profile need?")
    print("  " + "-" * 74)
    above = f"P(N), above {MIN_WING_FRACTION * 100:.0f}%"
    print(f"  {'wings':>5s} {'components':>11s} {'P(N), all':>10s} "
          f"{above:>18s} {'cumulative':>11s}")
    for n in range(N_WINGS + 1):
        print(f"  {n:5d} {n + 1:11d} {p_wings[n]:10.3f} {p_needed[n]:18.3f} "
              f"{cumulative[n]:11.3f}")
    print(f"\n  k_min = 1 + {k_min} wings = {k_min + 1} components "
          f"(posterior probability {p_needed[k_min]:.3f})")

    print("\n  per-wing evidence (Bayes factor = posterior odds / prior odds)")
    print("  " + "-" * 74)
    print(f"  {'wing':>4s} {'p_off':>6s} {'P(on)':>7s} {'BF':>10s} {'amp':>7s} "
          f"{'amp 97.5%':>10s} {'FWHM [km/s]':>12s} {'dv [km/s]':>10s}")
    for row in wing_evidence(fit):
        h = 2.0 * np.sqrt(2.0 * np.log(2.0))
        bayes = "inf" if not np.isfinite(row["bayes"]) else f"{row['bayes']:.1f}"
        print(f"  {row['wing']:4d} {row['p_off']:6.1f} {row['p_on']:7.3f} "
              f"{bayes:>10s} {row['amp']:7.4f} {row['amp_97']:10.4f} "
              f"{row['sigma'] * h:12.0f} {row['dv']:10.0f}")

    core = core_row(fit)
    print("\n  components, sorted by width (labels are not identified)")
    print("  " + "-" * 74)
    print(f"  {'rank':>4s} {'kind':>8s} {'amp':>7s} {'amp/core':>9s} "
          f"{'FWHM [km/s]':>12s} {'dv [km/s]':>10s} {'flux fraction':>14s}")
    print(f"  {0:4d} {'core':>8s} {core['amp']:7.4f} {1.0:9.3f} "
          f"{core['fwhm']:12.0f} {core['dv']:10.0f} {'--':>14s}")
    for row in sorted_components(fit):
        if row["n"] < 1:
            continue
        print(f"  {row['rank'] + 1:4d} {'wing':>8s} {row['amp']:7.4f} "
              f"{row['amp']:9.3f} {row['fwhm']:12.0f} {row['dv']:10.0f} "
              f"{row['flux_fraction']:14.3f}")
    print(f"\n  (a wing counts above {MIN_WING_FRACTION*100:.0f}% of the core peak; "
          f"effective components = {k_min + 1})")
    return k_min + 1


def chi2_of_samples(fit, samples, chunk=400):
    """``chi2`` of each parameter vector, batched on the device.

    The posterior visits the regions where some wings are switched off, so the same
    run can report what the extra components buy -- something a fit at one fixed
    ``k`` cannot do without a second run.
    """
    model = fit.model
    wave = np.asarray(fit.wave_use)
    flux = np.asarray(fit.flux_use)
    ferr = np.asarray(fit.ferr)
    out = np.empty(samples.shape[0])
    for start in range(0, samples.shape[0], chunk):
        block = np.asarray(samples[start:start + chunk], dtype=float)
        predicted = np.asarray(model.flux_batch(wave, block))
        out[start:start + chunk] = np.sum(((flux - predicted) / ferr) ** 2, axis=-1)
    return out


def improvement_table(fit, floor=MIN_WING_FRACTION, chunk=400, max_samples=1500):
    """Best achievable ``chi2`` at each number of wings, from this one posterior.

    The minimum is taken over at most ``max_samples`` of the samples with each
    count, chosen at random: evaluating the model on all of them takes tens of
    minutes for a 57 000-pixel fit with a tabulated iron template, and the
    difference between counts is not sensitive to the exact tail of the sample.
    """
    _, _, amps = amp_matrix(fit)
    counts = (amps > floor).sum(1)
    samples = np.asarray(fit.samples_equal_weight)
    dof = float(np.asarray(fit.flux_use).size - samples.shape[1])
    rows = []
    for n in range(N_WINGS + 1):
        index = np.flatnonzero(counts == n)
        if index.size > max_samples:
            index = np.random.default_rng(0).choice(index, max_samples, replace=False)
        if index.size == 0:
            rows.append({"wings": n, "n_samples": 0, "chi2": np.nan,
                         "chi2_dof": np.nan, "dchi2": np.nan, "dof": 0})
            continue
        chi2 = float(np.min(chi2_of_samples(fit, samples[index], chunk)))
        rows.append({"wings": n, "n_samples": int(index.size), "chi2": chi2,
                     "chi2_dof": chi2 / dof, "dchi2": np.nan, "dof": 3 * n})
    # what each extra wing buys, relative to the best sample with one fewer
    usable = [row for row in rows if np.isfinite(row["chi2"])]
    for previous, row in zip(usable, usable[1:]):
        row["dchi2"] = previous["chi2"] - row["chi2"]
        row["dof"] = 3 * (row["wings"] - previous["wings"])
    return rows


def print_improvement(fit):
    """The size of what each extra component buys, and its cost in parameters."""
    rows = improvement_table(fit)
    if not any(np.isfinite(row["chi2"]) for row in rows):
        print("\n  (no sample had a floored count, cannot tabulate the improvement)")
        return rows
    print("\n  what each extra component buys (best sample at that count)")
    print("  " + "-" * 74)
    print(f"  {'wings':>5s} {'components':>11s} {'samples':>8s} {'chi2':>11s} "
          f"{'chi2/dof':>9s} {'dchi2':>8s} {'dof':>5s}")
    for row in rows:
        dchi2 = "--" if not np.isfinite(row["dchi2"]) else f"{row['dchi2']:8.1f}"
        chi2 = "--" if not np.isfinite(row["chi2"]) else f"{row['chi2']:11.1f}"
        chi2dof = "--" if not np.isfinite(row["chi2_dof"]) else f"{row['chi2_dof']:9.3f}"
        print(f"  {row['wings']:5d} {row['wings'] + 1:11d} {row['n_samples']:8d} "
              f"{chi2} {chi2dof} {dchi2} {row['dof']:5d}")
    best = min((row for row in rows if np.isfinite(row["chi2"])),
               key=lambda row: row["chi2"])
    print(f"\n  the best sample has {best['wings'] + 1} components "
          f"(chi2/dof = {best['chi2_dof']:.3f}); read the dchi2 column against the "
          f"dof spent")
    return rows


def report_prior_edges(fit):
    """Flag parameters pressed against a prior edge -- the width cap, mostly."""
    try:
        flagged = fit.prior_dominated(tolerance=0.01)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (prior_dominated unavailable: {exc})")
        return
    if not flagged:
        print("\n  no parameter sits within 1% of a prior edge")
        return
    qualified = dict(zip(fit.param_names, fit.full_param_names))
    print("\n  parameters pressed against a prior edge (prior_dominated):")
    for name, side in flagged.items():
        print(f"      {qualified.get(name, name):24s} at the {side} limit")
    pinned = [name for name in flagged if "sigma_w" in name or "sigma_c" in name]
    if pinned:
        print(f"  note: {len(pinned)} width(s) at an edge -- the cap, not the data, "
              f"is setting that profile; see SIGMA_RANGE in this file")


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_components(fit, wave, flux, model_cpu, error, outfile):
    """The decomposition itself: one curve per component, sorted by width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core = core_row(fit)
    rows = [row for row in sorted_components(fit) if row["n"] >= 1]
    plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.9,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.2, 4.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    ax.plot(wave, flux, color="0.15", lw=0.8, label="data")
    wavec = base.LINE_WAVE["Halpha"]
    colours = ["#2b6cb0", "#2f855a", "#b7791f", "#805ad5", "#c53030"]
    x = np.linspace(wavec - 400.0, wavec + 400.0, 1500)
    core_curve = core["amp"] * np.exp(
        -0.5 * ((x - wavec * (1.0 + core["dv"] / 299792.458))
                / (wavec * core["sigma"] / 299792.458)) ** 2)
    ax.plot(x, core_curve, color=colours[0], lw=1.1,
            label=rf"core: FWHM {core['fwhm']:.0f} km/s")
    for j, row in enumerate(rows):
        curve = row["amp"] * core["amp"] * np.exp(
            -0.5 * ((x - wavec * (1.0 + (core["dv"] + row["dv"]) / 299792.458))
                    / (wavec * row["sigma"] / 299792.458)) ** 2)
        ax.plot(x, curve, color=colours[(j + 1) % len(colours)], lw=1.1, ls="--",
                label=rf"wing: FWHM {row['fwhm']:.0f} km/s, "
                      rf"{row['flux_fraction'] * 100:.0f}% of the flux")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(r"$F_\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.set_title("Broad H-alpha: the components the sampler left switched on",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    axr.axhline(0, color="0.7", lw=0.6)
    sel = (wave > x[0]) & (wave < x[-1])
    axr.plot(wave[sel], (flux[sel] - np.asarray(model_cpu(wave))[sel]) / error[sel],
             color="0.3", lw=0.7)
    axr.set_ylabel(r"residual / $\sigma$")
    axr.set_xlabel(r"rest wavelength [$\AA$]")
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")
    return outfile


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nlive", type=int, default=NLIVE,
                        help=f"live points (default {NLIVE}: 40 free parameters)")
    parser.add_argument("--dlogz", type=float, default=DLOGZ)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--device", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--outdir", default=str(base.OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="rough answer quickly (nlive=200, dlogz=0.2)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.nlive, args.dlogz = 200, 0.2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("IRAS 07546+3928: how many Gaussian components do the broad lines need?")
    print("=" * 78)

    wave, flux, error = base.load_spectrum()
    print(f"\n[1/5] data: {wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A, "
          f"peak {flux.max():.2f}")

    model = base.apply_model_labels(base.build_model(jxm, n_balmer_wings=N_WINGS))
    print(f"\n[2/5] model: {len(model.free_param_names)} free parameters, "
          f"{1 + N_WINGS} components per broad Balmer line")

    bounds, custom, notes = make_toggle_priors(flux)
    print("\n[3/5] priors")
    for what, description in notes:
        print(f"      {what:18s} {description}")
    missing = [q for q in base.free_qualified_names(model) if q not in bounds]
    if missing:
        raise RuntimeError(f"no prior given for these free parameters: {missing}")
    toggled = sorted(n for n in custom if n.endswith(tuple(f"amp_w{i}" for i in range(N_WINGS))))
    print(f"      gated amplitudes  {len(toggled)} "
          f"({', '.join(toggled[:3])}, ...)")

    print("\n[4/5] running JNesty")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started = time.perf_counter()
        fit = base.run_fit(wave, flux, error, bounds, custom, nlive=args.nlive,
                           dlogz=args.dlogz, max_iterations=args.max_iterations,
                           device=args.device, progress=None,
                           n_balmer_wings=N_WINGS)
        wall = time.perf_counter() - started
    print(f"  wall clock      : {wall/60:.1f} min")

    print("\n[5/5] results")

    # Save first.  A three-hour fit must not be lost to a bug in the analysis or in
    # a plot, so everything that can fail is wrapped below.
    npz_path = outdir / "iras07546_jax_toggle_posterior.npz"
    np.savez_compressed(
        npz_path,
        samples=fit.samples_equal_weight,
        parameter_names=np.array(fit.param_names),
        qualified_names=np.array(fit.full_param_names),
        log_evidence=fit.log_evidence, log_evidence_error=fit.log_evidence_err,
        wave=wave, flux=flux, error=error,
        component_count_wings=component_count(fit, floor=MIN_WING_FRACTION),
        p_off=np.array(P_OFF), min_wing_fraction=MIN_WING_FRACTION,
        sigma_range=np.array(SIGMA_RANGE))
    print(f"  wrote {npz_path}")
    fits_path = outdir / "iras07546_jax_toggle_fit.fits"
    save_jnesty_fit_to_fits(fit, fits_path)
    print(f"  wrote {fits_path}")

    predicted = None
    try:
        predicted, _ = base.report_quality(fit, wave, flux, error)
        base.print_parameter_table(fit)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (quality report failed: {exc})")
    try:
        n_components = print_selection(fit)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (selection report failed: {exc})")
        n_components = 1
    try:
        report_prior_edges(fit)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (prior-edge report failed: {exc})")
    try:
        base.print_derived(fit)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (derived quantities failed: {exc})")
    try:
        print_improvement(fit)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (component-improvement table failed: {exc})")

    print(f"\n  the published two-component fit: "
          f"{PUBLISHED['free_parameters']} free parameters, "
          f"ln Z = {PUBLISHED['log_evidence']:.2f}, "
          f"chi2/dof = {PUBLISHED['chi2_dof']:.3f}")
    print(f"  this run: {fit.ndim} free parameters, "
          f"ln Z = {fit.log_evidence:.2f} +- {fit.log_evidence_err:.2f}")
    print("  (the two ln Z are not comparable: the toggled model is a prior mixture "
          "over\n   subsets.  Quote the component count from here and the evidence "
          "from a\n   plain-prior refit at that count.)")

    model_cpu = None
    try:
        model_cpu = base.check_parity(fit, wave, predicted)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (CPU/JAX parity check failed: {exc})")

    if not args.no_plots and model_cpu is not None:
        zooms = [("Hgamma_Hbeta", (4230, 5100), (0.0, 2.4)),
                 ("Halpha_NII_SII", (6420, 6800), (0.0, 5.0)),
                 ("HeI_NaD", (5750, 6000), (0.4, 1.4))]
        for label, function in (
                ("fit figure", lambda: base.plot_result(
                    wave, flux, model_cpu, outdir / "iras07546_jax_toggle_fit.png",
                    zooms, mask_list=list(base.MASK_WINDOWS), error=error)),
                ("components figure", lambda: plot_components(
                    fit, wave, flux, model_cpu, error,
                    outdir / "iras07546_jax_toggle_components.png")),
                ("corner plot", lambda: base.plot_corner(
                    fit, outdir / "iras07546_jax_toggle_corner.png", max_params=18)),
                ("histograms", lambda: base.plot_histograms(
                    fit, outdir / "iras07546_jax_toggle_histograms.png"))):
            try:
                function()
            except Exception as exc:                           # noqa: BLE001
                print(f"  ({label} failed: {exc})")
    elif not args.no_plots:
        print("  (the plots need the astropy twin, which failed to build: skipped)")

    fit.release_device_buffers()
    print("\n  released the sampler's device buffers")
    print(f"\n  k_min = {n_components} components "
          f"(1 core + {n_components - 1} wings): the account above is the evidence.")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())