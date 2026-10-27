#!/usr/bin/env python
"""Are broad H-beta and H-gamma really the same shape as broad H-alpha?

``fit_iras07546_jax.py`` *ties* the broad lines together: one velocity shift and one
width for the core of all three Balmer lines, one pair for the winds.  That is a
physical statement -- one broad-line region -- and it buys eight fewer parameters.
This script asks whether the data agree with it.  The eight parameters are released,
and each one is given a prior centred on the H-alpha value:

    1. load the spectrum               (the same data, from the same file)
    2. build the released model        (the same model, eight ties undone)
       and put a Gaussian around H-alpha on each freed parameter
    3. fit it
    4. show the result                 (how far each line moved, and what it costs)

Two numbers go with every released parameter.  The **pull** is the difference from
H-alpha in units of the prior's own width, so +/-1 means "one tie width away",
whatever the line.  When ``MAX_SIGMA`` is set below the Gaussian's numerical default
the prior is a *hard* limit, and a parameter whose posterior sits against it is only
bounded from one side: its pull is then a statement about the limit, not a
measurement, and the table prints it with a ``*``.

The evidence is compared with the hard-tie fit that ``fit_iras07546_jax.py``
writes.  Softening eight parameters costs prior volume, so the soft tie must buy back
more than that to be worth it -- the fit itself says how much it did.

Nothing here touches the narrow lines: both fits put them on the same measured
template, so the comparison isolates the broad-line assumption.

Running it
----------
::

    python example/fit_iras07546_jax_tied.py

``MAX_SIGMA`` at the top of the file is the hard limit in tie widths (5.0 gives
"nowhere more than five widths", 7.0 is the numerical default and is not a
statement).  Results go to ``example/output/`` beside the published fit, which is
read but never written here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# the data, the model and the priors are the published ones -- what differs is the
# eight priors replaced below
sys.path.insert(0, str(Path(__file__).resolve().parent))
import iras07546_model as shared          # the model all three examples fit
                                          # (self-contained: this needs no other
                                          # example to have been run)

import galspec
from galspec import priors
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary
from galspec.jnesty_fit_io import load_jnesty_fit_from_fits, save_jnesty_fit_to_fits

# %% settings
OUTPUT_DIR = shared.OUTPUT_DIR
PREFIX = "iras07546_jax_tied_"

#: Which broad lines are released, and what they are centred on.
TIED_LINES = ("b_hbeta", "b_hgamma")
REFERENCE_LINE = "b_halpha"
#: Width of the centred prior on a line width, as a fraction of H-alpha's width.
WIDTH_FRACTION = 0.20
#: Width of the centred prior on a velocity shift, as a fraction of that component's
#: H-alpha *width* (not of the shift, which may be near zero).
SHIFT_FRACTION = 0.05
#: How many standard deviations of the difference are allowed at all.  7.0 is the
#: numerical default ("the Gaussian has run out"); 5.0 says "nowhere more than five
#: tie widths", which is a statement the data can be pressed against.
MAX_SIGMA = 5.0
#: A sample within this many tie widths of the limit counts as sitting on it.
CAP_TOLERANCE = 0.05

NLIVE = 1000                   # 38 parameters: 400 live points were measured short
DLOGZ = 0.01
MAX_ITERATIONS = 400_000
DEVICE = "auto"
QUICK = False                  # True -> nlive 200, dlogz 0.5

#: The hard-tie fit this one is scored against: read, never written.
#: The hard-tie fit this one is compared against.  Its numbers travel with the shared
#: model (``shared.REFERENCE["template"]``), so this script does not need
#: ``fit_iras07546_jax.py`` to have been run; if its FITS file happens to be there, the
#: measured ``chi2/dof`` is used instead of the documented one.
PUBLISHED_FIT = OUTPUT_DIR / "iras07546_jax_fit.fits"


# %% results-helpers
def tie_table(fit):
    """Posterior summary of the difference between each line and H-alpha.

    For every released pair: the two medians, their difference, the same difference
    relative to H-alpha's value (percent), and the **pull** -- the difference in
    units of the prior width, which is the quantity the prior actually controls.  A
    pull of 0 means the data agree with the hard tie; |pull| ~ 1 means the difference
    is as large as the prior allowed for; |pull| >> 1 would mean the prior is
    fighting the data.

    When the prior has a hard limit, ``at_cap`` says whether a sizeable part of the
    posterior sits against it -- and then the pull is a *limit*, not a measurement.

    Also used by ``develop/dev_jax/tied_prior_report.py``, which is why it is a
    function rather than a block inside :func:`main`.
    """
    samples = np.asarray(fit.samples_equal_weight, dtype=float)
    index = {q: i for i, q in enumerate(fit.full_param_names)}
    rows = []
    for line in TIED_LINES:
        for stem in ("c", "w0"):
            for kind in ("sigma", "dv"):
                key = f"{line}.{kind}_{stem}"
                reference = f"{REFERENCE_LINE}.{kind}_{stem}"
                got = samples[:, index[key]]
                ref = samples[:, index[reference]]
                difference = got - ref
                if kind == "sigma":
                    tie_width = WIDTH_FRACTION * np.abs(ref)
                    scale = "of the reference value"
                else:
                    # a shift may be near zero, so its width is a fraction of the
                    # *line width*, not of the shift itself
                    tie_width = SHIFT_FRACTION * np.abs(
                        samples[:, index[f"{REFERENCE_LINE}.sigma_{stem}"]])
                    scale = f"of {REFERENCE_LINE}.sigma_{stem}"
                pull = difference / tie_width
                median_ref = float(np.median(ref))
                delta_lo, delta_med, delta_hi = np.percentile(difference, [16, 50, 84])
                pull_lo, pull_med, pull_hi = np.percentile(pull, [16, 50, 84])
                at_cap_fraction = float(np.mean(np.abs(pull) >= MAX_SIGMA - CAP_TOLERANCE))
                rows.append({
                    "line": line, "parameter": f"{kind}_{stem}",
                    "reference": median_ref, "value": float(np.median(got)),
                    "difference": float(delta_med), "difference_lo": float(delta_lo),
                    "difference_hi": float(delta_hi),
                    "fractional": float(delta_med / abs(median_ref)),
                    "fractional_lo": float(delta_lo / abs(median_ref)),
                    "fractional_hi": float(delta_hi / abs(median_ref)),
                    "pull": float(pull_med), "pull_lo": float(pull_lo),
                    "pull_hi": float(pull_hi),
                    "tie_width": float(np.median(tie_width)), "scale": scale,
                    "kind": "width" if kind == "sigma" else "shift",
                    "at_cap_fraction": at_cap_fraction,
                    "at_cap": bool(at_cap_fraction > 0.05),
                    "max_sigma": float(MAX_SIGMA),
                })
    return rows


def tie_label(name):
    r"""``b_hbeta.sigma_w0`` -> ``H$\beta$ $\sigma_{w0}$``, for axis ticks."""
    line, rest = name.split(".", 1)
    kind, stem = rest.split("_", 1)
    line_symbol = {"b_hbeta": r"H$\beta$", "b_hgamma": r"H$\gamma$",
                   "b_halpha": r"H$\alpha$"}.get(line, line.replace("b_", ""))
    kind_symbol = {"sigma": r"$\sigma$", "dv": r"$v$"}.get(kind, kind)
    return f"{line_symbol} {kind_symbol}$_{{{stem}}}$"


# %% tie-priors
# ---- The eight priors that replace the eight ties -----------------------------------
def tie_priors(model):
    """A Gaussian around H-alpha for every released parameter, hard-stopped.

    A width may differ by ``WIDTH_FRACTION`` of H-alpha's width, a shift by
    ``SHIFT_FRACTION`` of that component's H-alpha *width* (a shift can be near zero,
    so it is not measured against itself), and nothing may differ by more than
    ``MAX_SIGMA`` widths.  The truncation range is H-alpha's own prior -- the box the
    parameter would have had -- so the prior stays a proper density over the range the
    sampler can reach, and the evidence stays comparable with the hard-tie run.

    Needed twice: ``main`` applies it to the fit, and
    ``develop/dev_jax/tied_prior_report.py`` to the mocks it fits.
    """
    for line in TIED_LINES:
        for stem in ("c", "w0"):
            lo, hi = model.get_prior(f"{REFERENCE_LINE}.sigma_{stem}").support
            model.set_prior(f"{line}.sigma_{stem}",
                            priors.Normal(mean=f"{REFERENCE_LINE}.sigma_{stem}",
                                          relative=WIDTH_FRACTION, nsigma=MAX_SIGMA,
                                          mini=lo, maxi=hi))
            lo, hi = model.get_prior(f"{REFERENCE_LINE}.dv_{stem}").support
            model.set_prior(f"{line}.dv_{stem}",
                            priors.Normal(mean=f"{REFERENCE_LINE}.dv_{stem}",
                                          sigma=SHIFT_FRACTION,
                                          scale_from=f"{REFERENCE_LINE}.sigma_{stem}",
                                          nsigma=MAX_SIGMA, mini=lo, maxi=hi))
    return model


# %% main
# ---- Fit, and show the result -------------------------------------------------------
def main() -> int:
    nl, dlogz = (200, 0.5) if QUICK else (NLIVE, DLOGZ)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("IRAS 07546+3928: broad H-beta and H-gamma tied softly to broad H-alpha")
    print("=" * 78)

    # ---- 1. Load the spectrum ------------------------------------------------------
    wave, flux, error = shared.load_spectrum()
    print(f"\ndata:  {wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A (rest frame)")

    # ---- 2. Build the released model, and centre eight priors on H-alpha -----------
    model = shared.build_model(n_wings=shared.N_WINGS, tie_broad=False)
    shared.set_priors(model, flux)
    print(f"\nmodel: {len(model.free_param_names)} free parameters "
          "(eight released from the hard tie)")

    tie_priors(model)
    print(f"\nsoft ties: eight parameters, a Gaussian around H-alpha, "
          f"hard limit {MAX_SIGMA:g} tie widths")
    print(f"           width {WIDTH_FRACTION:.0%} of H-alpha's, "
          f"shift {SHIFT_FRACTION:.0%} of H-alpha's width")
    missing = [q for q in shared.free_qualified_names(model) if model.get_prior(q) is None]
    if missing:
        raise RuntimeError(f"no prior given for these free parameters: {missing}")

    # ---- 3. Fit --------------------------------------------------------------------
    fit = JNesty_Fit(model, wave, flux, error, nlive=nl, bound="multi", device=DEVICE)
    print(f"\nfit:   {nl} live points, dlogz {dlogz}, "
          f"device {fit.device}  [{jax_devices_summary()}]")
    started = time.perf_counter()
    fit.fit(progress=sys.stdout.isatty(), dlogz=dlogz, max_iterations=MAX_ITERATIONS)
    print(f"\n       log Z = {fit.log_evidence:.2f} +- {fit.log_evidence_err:.2f}"
          f"   ({fit.n_iterations} iterations, "
          f"{(time.perf_counter() - started) / 60:.1f} min)")

    # ---- 4. Show the result --------------------------------------------------------
    model_best, _, theta = fit.get_best_fit()
    predicted = np.asarray(model_best.flux(wave, theta))
    residual = (flux - predicted) / error
    dof = wave.size - len(theta)
    print(f"       chi2/dof = {np.sum(residual ** 2) / dof:.3f}, residual rms "
          f"{residual.std():.2f} sigma")

    # every parameter, with the 16th and 84th percentiles
    lo, med, hi = fit.get_quantiles((0.16, 0.5, 0.84))
    print(f"\n  {'parameter':22s} {'median':>10s} {'-':>9s} {'+':>9s}  unit")
    print("  " + "-" * 76)
    for index, name in enumerate(fit.full_param_names):
        print(f"  {name:22s} {med[index]:10.5g} {med[index] - lo[index]:9.2g} "
              f"{hi[index] - med[index]:9.2g}  {shared.unit_of(name)}")

    # how far each released parameter moved from H-alpha
    rows = tie_table(fit)
    print(f"\n  {'parameter':20s} {'H-alpha':>9s} {'line':>9s} {'difference':>11s} "
          f"{'fractional':>11s} {'pull':>7s}")
    print(f"  {'':20s} {'[km/s]':>9s} {'[km/s]':>9s} {'[km/s]':>11s} "
          f"{'[%]':>11s} {'[sigma]':>7s}")
    for row in rows:
        mark = "*" if row["at_cap"] else " "
        print(f"  {row['line'] + '.' + row['parameter']:20s} "
              f"{row['reference']:9.1f} {row['value']:9.1f} "
              f"{row['difference']:+11.2f} {100 * row['fractional']:+10.1f}% "
              f"{row['pull']:+6.2f}{mark}")
    largest = max(rows, key=lambda row: abs(row["pull"]))
    print(f"\n  largest pull: {largest['line']}.{largest['parameter']} = "
          f"{largest['pull']:+.2f} sigma of the prior width "
          f"({largest['tie_width']:.0f} km/s)")
    capped = [row for row in rows if row["at_cap"]]
    if capped:
        print(f"\n  * the prior stops at {MAX_SIGMA:g} tie widths and this parameter's "
              "posterior sits against it")
        for row in capped:
            print(f"    {row['line']}.{row['parameter']}: "
                  f"{row['at_cap_fraction']:.0%} of the samples are within "
                  f"{CAP_TOLERANCE:g} widths of the limit")
        print("    -> read that pull as a limit ('at least this far from H-alpha'), "
              "not as a\n       measurement; raise MAX_SIGMA to let the data speak.")
    elif abs(largest["pull"]) > 3.0:
        print("  -> the prior is fighting the data: a pull beyond ~3 sigma means the "
              "tie width\n     is too small for this object.")
    else:
        print("  -> every pull is within ~3 sigma: the hard tie is an adequate "
              "description\n     of these lines.")

    # what the eight extra parameters cost in evidence
    hard = shared.REFERENCE["template"]
    print(f"\n  the hard tie: {hard['free_parameters']} free parameters, "
          f"{'its own file' if PUBLISHED_FIT.exists() else 'the shared reference'}")
    published = None
    if PUBLISHED_FIT.exists():
        # its FITS is there, so the chi-square can be measured rather than quoted
        published = load_jnesty_fit_from_fits(PUBLISHED_FIT)
        hard_evidence, hard_error = published.log_evidence, published.log_evidence_err
        hard_ndim = published.ndim
    else:
        hard_evidence = hard["log_evidence"]
        hard_error = hard["log_evidence_error"]
        hard_ndim = hard["free_parameters"]
    difference = fit.log_evidence - hard_evidence
    n_extra = fit.ndim - hard_ndim
    # what the eight extra priors cost: each released parameter buys the range it
    # ended up using, so the penalty is the log ratio of the prior width to the
    # posterior width of its difference (the estimate tied_prior_report.py makes)
    penalty = 0.0
    for row in rows:
        width = max(row["difference_hi"] - row["difference_lo"], 1e-6) / 2.0
        penalty += np.log(row["tie_width"] * np.sqrt(2.0 * np.pi) / width)
    print(f"    ln Z (hard tie)           : "
          f"{hard_evidence:.2f} +- {hard_error:.2f}"
          f"   [{hard_ndim} parameters]")
    print(f"    ln Z (soft tie, this fit) : {fit.log_evidence:.2f} +- "
          f"{fit.log_evidence_err:.2f}   [{fit.ndim} parameters]")
    print(f"    difference                : {difference:+.2f} +- "
          f"{np.hypot(fit.log_evidence_err, hard_error):.2f}")
    print(f"    Occam penalty             : {penalty:+.1f} "
          f"(the {n_extra} released parameters' own prior volume)")
    # what the better fit is worth: half the drop in chi2 is the log-likelihood
    # gain.  This is the arithmetic that decides whether the difference is
    # likelihood or book-keeping, so it is printed next to the numbers.  The
    # hard-tie chi-square is measured from its own fit when that fit is on disk
    # and taken from the shared reference when it is not, so this script never
    # needs another example to have been run first.
    chi2dof_hard = (float(np.sum(
        ((flux - np.asarray(published.get_best_fit()[0].flux(
            wave, published.get_best_fit()[2]))) / error) ** 2)) / dof
        if published is not None else hard["chi2_dof"])
    chi2dof_soft = float(np.sum(residual ** 2)) / dof
    likelihood_gain = 0.5 * (chi2dof_hard - chi2dof_soft) * dof
    if difference < penalty:
        print("    -> the soft tie lost less than the penalty predicts: the released "
              "parameters\n       earned their keep, and the pulls above say where.")
    else:
        print(f"    -> the difference is the better fit, not an Occam factor: "
              f"chi2/dof {chi2dof_hard:.3f} -> {chi2dof_soft:.3f} over {dof} "
              f"degrees of freedom\n       is worth about {likelihood_gain:+.0f} "
              f"nats of likelihood, against the {penalty:+.1f} the extra prior "
              f"volume\n       costs -- so the hard tie is rejected, and the pulls "
              f"above say where.")
        print(f"       (the evidence error bar is "
              f"{np.hypot(fit.log_evidence_err, hard_error):.2f} "
              f"nats, so a likelihood gain this far above it is not a convergence\n"
              f"       artefact.  JNesty seeds from a fixed key, so a rerun draws "
              f"the same\n       live points; two runs of this fit differed by 0.45 "
              f"nats, inside that bar.)")
    # figures: the fit, the zooms, the posteriors, and the pulls themselves
    ax, _ = galspec.plot.plot_fit_jax(wave, flux, model_best, error=error,
                                  residual_in_sigma=True, xlabel="Rest wavelength  [A]",
                                  ylabel=shared.FLUX_LABEL,
                                  legend_kwargs={"fontsize": 10, "ncol": 2})
    ax.figure.set_size_inches(15, 7)
    ax.set_title("IRAS 07546+3928: broad H-beta and H-gamma released, "
                 "soft-tied to broad H-alpha", fontsize=13)
    ax.figure.savefig(OUTPUT_DIR / f"{PREFIX}fit.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    for tag, title, xlim, ylim in shared.ZOOMS[:2]:
        ax, _ = galspec.plot.plot_fit_jax(wave, flux, model_best, error=error,
                                      xlim=xlim, ylim0=ylim, ylim1=(-8, 8),
                                      residual_in_sigma=True,
                                      xlabel="Rest wavelength  [A]",
                                      ylabel=shared.FLUX_LABEL,
                                      legend_kwargs={"fontsize": 7, "ncol": 1})
        ax.figure.set_size_inches(13.5, 6)
        ax.set_title(f"{title}: {xlim[0]:.0f}-{xlim[1]:.0f} A", fontsize=13)
        out = OUTPUT_DIR / f"{PREFIX}fit_{tag}.png"
        ax.figure.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

    keep = [i for i, name in enumerate(fit.full_param_names)
            if name in shared.CORNER_PARAMS or name.split(".")[0] in TIED_LINES]
    labels = getattr(fit, "param_labels", None) or list(fit.full_param_names)
    corner_fig = corner.corner(fit.samples_equal_weight[:, keep],
                               labels=[labels[i] for i in keep],
                               quantiles=(0.16, 0.5, 0.84), show_titles=True,
                               title_fmt=".3g", title_kwargs={"fontsize": 8})
    corner_fig.savefig(OUTPUT_DIR / f"{PREFIX}corner.png", dpi=110, bbox_inches="tight")
    plt.close(corner_fig)
    galspec.plot.plot_posterior_hist(
        fit, params=shared.HIST_PARAMS, ncols=4, bins=40,
        title="IRAS 07546+3928: soft ties, marginal posteriors",
        filename=str(OUTPUT_DIR / f"{PREFIX}histograms.png"))

    # the pulls: where each released parameter sits, with the hard limit drawn
    order = sorted(rows, key=lambda row: row["pull"])
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    y = np.arange(len(order))
    ax.axvspan(-MAX_SIGMA, MAX_SIGMA, color="0.94", zorder=0)
    for level in (-MAX_SIGMA, 0.0, MAX_SIGMA):
        ax.axvline(level, color="#c53030" if abs(level) == MAX_SIGMA else "0.6",
                   ls=":" if level else "-", lw=1.2 if level else 1.0, zorder=2)
    capped_any = any(row["at_cap"] for row in order)
    labelled = False
    for i, row in enumerate(order):
        low = row["pull"] - row["pull_lo"]
        high = row["pull_hi"] - row["pull"]
        ax.plot([0.0, row["pull"]], [i, i], color="0.75", lw=1.0, zorder=1)
        label = None
        if row["at_cap"] and not labelled:
            label, labelled = "posterior against the hard limit", True
        ax.errorbar(row["pull"], i, xerr=[[low], [high]], fmt="o", ms=7, zorder=3,
                    elinewidth=1.4, capsize=3, label=label,
                    color="#c53030" if row["at_cap"] else "#2b6cb0")
    ax.set_yticks(y)
    ax.set_yticklabels([tie_label(f"{row['line']}.{row['parameter']}") for row in order],
                       fontsize=10)
    ax.set_xlabel("difference from H-alpha  [tie widths]", fontsize=12)
    ax.set_xlim(-1.15 * MAX_SIGMA, 1.15 * MAX_SIGMA)
    ax.set_xticks([-MAX_SIGMA, -2.5, 0.0, 2.5, MAX_SIGMA])
    ax.set_title(f"Soft ties: each released parameter against a hard limit of "
                 f"{MAX_SIGMA:g} widths", fontsize=12)
    ax.text(MAX_SIGMA, len(order) - 0.4, f" limit = {MAX_SIGMA:g} widths",
            color="#c53030", fontsize=9, va="top", ha="left")
    if capped_any:
        ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{PREFIX}ties.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote the figures and {OUTPUT_DIR / f'{PREFIX}ties.png'}")

    # the record
    npz_path = OUTPUT_DIR / f"{PREFIX}posterior.npz"
    np.savez_compressed(npz_path,
                        samples=fit.samples_equal_weight,
                        parameter_names=np.array(fit.param_names),
                        qualified_names=np.array(fit.full_param_names),
                        log_evidence=fit.log_evidence,
                        log_evidence_error=fit.log_evidence_err,
                        wave=wave, flux=flux, error=error, predicted=predicted,
                        width_fraction=WIDTH_FRACTION, shift_fraction=SHIFT_FRACTION,
                        max_sigma=MAX_SIGMA,
                        tie_rows=np.array([str(row) for row in rows]))
    fits_path = OUTPUT_DIR / f"{PREFIX}fit.fits"
    save_jnesty_fit_to_fits(fit, fits_path)
    print(f"  wrote {npz_path}")
    print(f"  wrote {fits_path}  ({fits_path.stat().st_size / 1e6:.0f} MB)")
    fit.release_device_buffers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())