"""IRAS 07546+3928: broad H-beta and H-gamma *similar* to, not equal to, H-alpha.

The published fit (``fit_iras07546_jax.py``) makes one hard assumption about the
broad Balmer lines: they share the kinematics of broad H-alpha.  ``dv_c``,
``sigma_c`` and the wind's ``dv_w0``/``sigma_w0`` of H-beta and H-gamma are *tied*
to H-alpha's values, so the three profiles differ only in amplitude.  That is a
strong statement -- it says the broad-line region emitting H-gamma is moving
exactly like the one emitting H-alpha.

This script replaces each tie with a **prior centred on H-alpha's value**:
the parameter is free, but a priori it is expected near H-alpha's, within a width
set by that component's own scale.

    widths  (sigma_c, sigma_w0, ...)      sigma = 20% of H-alpha's width
    shifts  (dv_c,    dv_w0,    ...)      sigma = 5%  of H-alpha's width

Per component: the core is tied to the core, wing *i* to wing *i*.  The 20% says
how much broader or narrower H-beta may be; the 5% says how far its centre may sit
from H-alpha's, measured against the line width rather than against the (small and
possibly zero) shift itself.  Both numbers are choices, not measurements, and the
script exposes them as ``--width-fraction`` and ``--shift-fraction``.

``--max-sigma`` (default 7) puts a **hard limit** on the offset: ``--max-sigma 5``
says "nowhere more than five tie widths from H-alpha".  That is a different
statement from a wider Gaussian -- the prior is exactly zero beyond it -- so a
posterior that piles up against the limit means the data wanted to go further,
and the pull saturates at the cap instead of measuring a difference.  The table
marks such rows.

Eight parameters leave the tie (four per line), so the model has 38 free
parameters instead of 30.  What that buys is a *measurement*: the posterior of
``theta_line - theta_Halpha`` per component, reported here in km/s, as a fraction
of H-alpha's value, and as a **pull** in units of the prior width.  A pull near 0
says the data are happy with H-alpha's kinematics; a pull of several sigma says
they are not, and that the hard tie was hiding it.

The evidence is reported too, against the published hard-tie fit, but read it with
care: a soft tie has eight more parameters and each costs an Occam penalty of
about ``ln(sigma_tie * sqrt(2 pi) / posterior_width)``, so ``ln Z`` is expected to
drop even when the extra freedom is useless.  The pulls are the part that answers
"are the profiles the same?", and they do not depend on the tie width.

Usage::

    python example/fit_iras07546_jax_tied.py              # the real run (~1 h on one GPU)
    python example/fit_iras07546_jax_tied.py --quick      # a few minutes
    python example/fit_iras07546_jax_tied.py --width-fraction 0.5   # a looser tie
    python example/fit_iras07546_jax_tied.py --max-sigma 5          # a hard limit

Outputs go to ``example/output/`` with the ``iras07546_jax_tied_`` prefix:
``_fit.fits`` (reloadable with :func:`galspec.jnesty_fit_io.load_jnesty_fit_from_fits`),
``_posterior.npz``, ``_fit.png`` (the whole spectrum), ``_fit_Hgamma_Hbeta.png`` and
``_fit_Halpha_NII_SII.png`` (the two zooms), ``_histograms.png``, ``_corner.png`` and
``_ties.png`` (the difference from H-alpha, per component).  Nothing published by the
other scripts is overwritten.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fit_iras07546_jax as base                                     # noqa: E402
import galspec                                                       # noqa: E402
from galspec import jax_models as jxm                                # noqa: E402
from galspec import priors as priors_                                # noqa: E402

OUTPUT_DIR = base.OUTPUT_DIR
PREFIX = "iras07546_jax_tied_"

#: Which broad lines get the soft tie, and which one they are centred on.
TIED_LINES = ("b_hbeta", "b_hgamma")
REFERENCE_LINE = "b_halpha"

#: Width of the centred prior on a line width, as a fraction of H-alpha's width.
WIDTH_FRACTION = 0.20
#: Width of the centred prior on a velocity shift, as a fraction of that
#: component's H-alpha *width* (not of the shift, which may be near zero).
SHIFT_FRACTION = 0.05
#: How many standard deviations of the difference are allowed at all.  Seven is
#: the numerical default ("the Gaussian has run out"); a smaller number is a
#: deliberate statement, e.g. 5 for "nowhere more than five tie widths".
MAX_SIGMA = 7.0
#: A sample within this many tie widths of the limit counts as "at the cap".
CAP_TOLERANCE = 0.05

#: 38 free parameters: 400 live points were measured to be too few at 40.
NLIVE = 1000
DLOGZ = 0.01
MAX_ITERATIONS = 400_000

#: The published hard-tie fit, read from example/output (never written here).
PUBLISHED_FIT = OUTPUT_DIR / "iras07546_jax_fit.fits"


# --------------------------------------------------------------------------------------
# 1. The free parameters and the priors that constrain them
# --------------------------------------------------------------------------------------
def component_stems(n_balmer_wings=1):
    """Names of the Gaussian components: the core, then each wind.

    Component *i* of H-beta is centred on component *i* of H-alpha, never on a
    different one: the core is compared with the core and wing 1 with wing 1.
    """
    return ["c"] + [f"w{i}" for i in range(n_balmer_wings)]


def tied_names(n_balmer_wings=1):
    """``(line_parameter, reference_parameter)`` for every soft tie."""
    pairs = []
    for line in TIED_LINES:
        for stem in component_stems(n_balmer_wings):
            for kind in ("sigma", "dv"):
                pairs.append((f"{line}.{kind}_{stem}",
                              f"{REFERENCE_LINE}.{kind}_{stem}"))
    return pairs


def reference_support(model, name):
    """``(lower, upper)`` of a parameter that a centred prior will be truncated to.

    The released parameter keeps the *same box* as the H-alpha parameter it is
    centred on, and that box doubles as the truncation range of the Gaussian: the
    prior stays a proper density over the parameter's own range, which is what
    keeps its evidence comparable with the hard-tie run.
    """
    prior = model.get_prior(name)
    support = getattr(prior, "support", None)
    if support is None:
        raise RuntimeError(
            f"{name} has no explicit prior range, so the parameter centred on it "
            "would fall back to a default box. Give it a priors.Uniform(...) or "
            "priors.LogUniform(...) first (base.apply_priors does that).")
    return support


def apply_ties(model, n_balmer_wings=1, width_fraction=WIDTH_FRACTION,
               shift_fraction=SHIFT_FRACTION, max_sigma=MAX_SIGMA):
    """Put the soft ties on the model: eight parameters, eight lines of prior.

    Each one says where the parameter is expected to be, in terms of another
    parameter's *value*: the width of H-beta's core is a Gaussian around the width
    of H-alpha's core, and likewise per wind component and for the shifts.  The
    fitter works out that H-alpha's value has to be drawn first.

    Returns
    -------
    rows : list of dict
        What was set, for the printed summary.
    """
    rows = []
    for stem in component_stems(n_balmer_wings):
        for line in TIED_LINES:
            # -- the width: a Gaussian around H-alpha's width, 20% wide
            name = f"{line}.sigma_{stem}"
            reference = f"{REFERENCE_LINE}.sigma_{stem}"
            lower, upper = reference_support(model, reference)
            model.set_prior(name, priors_.Normal(
                mean=reference, relative=width_fraction, nsigma=max_sigma,
                mini=lower, maxi=upper))
            rows.append({"parameter": name, "reference": reference,
                         "kind": "width",
                         "scale": f"{width_fraction:.0%} of the reference"})
            # -- the shift: a Gaussian around H-alpha's shift, 5% of the width wide
            name = f"{line}.dv_{stem}"
            reference = f"{REFERENCE_LINE}.dv_{stem}"
            width_from = f"{REFERENCE_LINE}.sigma_{stem}"
            # the box is the *shift's* own range (it may be negative); the width
            # that says how far it may go comes from the line width
            lower, upper = reference_support(model, reference)
            model.set_prior(name, priors_.Normal(
                mean=reference, sigma=shift_fraction, scale_from=width_from,
                nsigma=max_sigma, mini=lower, maxi=upper))
            rows.append({"parameter": name, "reference": reference,
                         "kind": "shift",
                         "scale": f"{shift_fraction:.0%} of {width_from}"})
    return rows


# --------------------------------------------------------------------------------------
# 2. The fit
# --------------------------------------------------------------------------------------
def run_tied_fit(wave, flux, error, model, nlive=NLIVE, dlogz=DLOGZ,
                 max_iterations=MAX_ITERATIONS, device="auto", progress=None):
    """Sample the posterior of the soft-tied model with JNesty.

    ``model`` must already carry its priors (``base.apply_priors`` then
    ``apply_ties``), which is all the sampler needs: it reads them off the model.
    """
    return base.run_fit(wave, flux, error, nlive=nlive, dlogz=dlogz,
                        max_iterations=max_iterations, device=device,
                        progress=progress, model=model, priors=False)


# --------------------------------------------------------------------------------------
# 3. What the tie measured
# --------------------------------------------------------------------------------------
def tie_table(fit, width_fraction=WIDTH_FRACTION, shift_fraction=SHIFT_FRACTION,
              n_balmer_wings=None, max_sigma=MAX_SIGMA):
    """Posterior summary of the difference between each line and H-alpha.

    For every tied pair: the two medians, their difference, the same difference
    relative to H-alpha's value (percent), and the **pull** -- the difference in
    units of the prior width, which is the quantity the prior actually controls.

    A pull of 0 means the data are consistent with the hard tie; |pull| ~ 1 means
    the difference is as large as the prior allowed for; |pull| >> 1 would mean the
    prior is fighting the data.

    When the prior has a hard limit (``max_sigma`` smaller than the numerical
    default), ``at_cap`` says whether a sizeable part of the posterior sits
    against it -- and then the pull is a *limit*, not a measurement.
    """
    if n_balmer_wings is None:
        n_balmer_wings = base.broad_wing_count(fit)
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight, dtype=float)
    median = fit.get_quantiles((0.5,))[0]
    rows = []
    for line, reference in tied_names(n_balmer_wings):
        line_index, reference_index = names.index(line), names.index(reference)
        difference = samples[:, line_index] - samples[:, reference_index]
        kind = "width" if ".sigma_" in line else "shift"
        stem = line.split(".", 1)[1].split("_", 1)[1]   # dv_c -> c, dv_w0 -> w0
        if kind == "width":
            tie_width = width_fraction * np.abs(samples[:, reference_index])
            scale = "of the reference value"
        else:
            width_index = names.index(f"{REFERENCE_LINE}.sigma_{stem}")
            tie_width = shift_fraction * np.abs(samples[:, width_index])
            scale = f"of {REFERENCE_LINE}.sigma_{stem}"
        pull = difference / tie_width
        delta_lo, delta_med, delta_hi = np.percentile(difference, [16, 50, 84])
        at_cap_fraction = float(np.mean(np.abs(pull) >= max_sigma - CAP_TOLERANCE))
        rows.append({
            "line": line.split(".")[0], "parameter": line.split(".", 1)[1],
            "reference": median[reference_index], "value": median[line_index],
            "difference": float(delta_med),
            "difference_lo": float(delta_lo), "difference_hi": float(delta_hi),
            "fractional": float(delta_med / abs(median[reference_index])),
            "fractional_lo": float(delta_lo / abs(median[reference_index])),
            "fractional_hi": float(delta_hi / abs(median[reference_index])),
            "pull": float(np.median(pull)),
            "pull_lo": float(np.percentile(pull, 16)),
            "pull_hi": float(np.percentile(pull, 84)),
            "tie_width": float(np.median(tie_width)), "scale": scale,
            "kind": kind,
            "at_cap_fraction": at_cap_fraction,
            "at_cap": bool(at_cap_fraction > 0.05),
            "max_sigma": float(max_sigma),
        })
    return rows


def tie_label(name):
    """``'b_hbeta.sigma_w0'`` -> ``'H$\\beta$ $\\sigma_{w0}$'`` for an axis tick."""
    line, rest = name.split(".", 1)
    kind, stem = rest.split("_", 1)
    line_symbol = {"b_hbeta": r"H$\beta$", "b_hgamma": r"H$\gamma$",
                   "b_halpha": r"H$\alpha$"}.get(line, line.replace("b_", ""))
    kind_symbol = {"sigma": r"$\sigma$", "dv": r"$v$"}.get(kind, kind)
    return f"{line_symbol} {kind_symbol}$_{{{stem}}}$"


def print_tie_table(rows):
    """Print the tie measurement as a table."""
    def nice(value):
        return f"{value:+9.2f}"

    print(f"\n  {'parameter':22s} {'H-alpha':>9s} {'line':>9s} "
          f"{'difference':>10s} {'fractional':>11s} {'pull':>7s}")
    print(f"  {'':22s} {'[km/s]':>9s} {'[km/s]':>9s} {'[km/s]':>10s} "
          f"{'[%]':>11s} {'[sigma]':>7s}")
    for row in rows:
        mark = "*" if row.get("at_cap") else " "
        print(f"  {row['line'] + '.' + row['parameter']:22s} "
              f"{row['reference']:9.1f} {row['value']:9.1f} "
              f"{nice(row['difference'])} "
              f"{100 * row['fractional']:+10.2f}% "
              f"{nice(row['pull'])}{mark}")
    print("  (the fractional column divides by H-alpha's own value; for a shift that\n"
          "   value can be small or zero, so the pull -- measured against the tie "
          "width -- is the\n   scale that matters)")
    worst = max(rows, key=lambda row: abs(row["pull"]))
    print(f"\n  largest pull: {worst['line']}.{worst['parameter']} "
          f"= {worst['pull']:+.2f} sigma of the prior width "
          f"({worst['tie_width']:.0f} km/s)")
    capped = [row for row in rows if row.get("at_cap")]
    if capped:
        limit = capped[0].get("max_sigma")
        print(f"\n  * the prior stops at {limit:g} tie widths and this parameter's "
              f"posterior sits against it")
        for row in capped:
            print(f"    {row['line']}.{row['parameter']}: "
                  f"{100 * row['at_cap_fraction']:.0f}% of the samples are within "
                  f"{CAP_TOLERANCE:g} widths of the limit")
        print("    -> read that pull as a limit ('at least this far from H-alpha'), "
              "not as a measurement; raise --max-sigma to let the data speak.")
    if abs(worst["pull"]) < 1.0:
        print("  -> every profile is within the tie width of H-alpha's: the hard tie "
              "was a good approximation.")
    elif abs(worst["pull"]) < 3.0:
        print("  -> at least one profile sits about a tie width away from H-alpha's; "
              "the data prefer a difference the hard tie could not express.")
    else:
        print("  -> the prior is fighting the data: a pull beyond ~3 sigma means the "
              "tie width is too small for this object.")


def read_published_evidence(path=PUBLISHED_FIT):
    """``(log Z, error)`` of the published hard-tie fit, or ``None``."""
    try:
        from astropy.io import fits
        header = fits.open(path)[0].header
    except Exception:                                   # pragma: no cover - optional
        return None
    evidence = header.get("LOGZ", header.get("LOGEVIDENCE"))
    error = header.get("LOGZERR", header.get("LOGEVIDENCE_ERR"))
    if evidence is None:
        return None
    return float(evidence), (float(error) if error is not None else float("nan"))


def print_evidence_comparison(fit, rows):
    """Bayes factor against the hard tie, with the Occam penalty named."""
    published = read_published_evidence()
    if published is None:
        print("\n  no published hard-tie fit at example/output/iras07546_jax_fit.fits; "
              "skipping the evidence comparison")
        return None
    hard_logz, hard_error = published
    difference = fit.log_evidence - hard_logz
    combined = np.hypot(fit.log_evidence_err, hard_error)
    # what the extra freedom costs even if it is useless
    penalty = 0.0
    for row in rows:
        posterior_width = max(abs(row["difference_hi"] - row["difference_lo"]) / 2.0,
                              1e-6)
        penalty += np.log(row["tie_width"] * np.sqrt(2.0 * np.pi) / posterior_width)
    print(f"\n  ln Z (hard tie, published): {hard_logz:.2f}"
          f" +- {hard_error:.2f}   [30 parameters]")
    print(f"  ln Z (soft tie, this fit) : {fit.log_evidence:.2f}"
          f" +- {fit.log_evidence_err:.2f}   [{fit.ndim} parameters]")
    print(f"  difference                : {difference:+.2f} +- {combined:.2f}")
    print(f"  Occam penalty expected    : about {penalty:+.1f} "
          "(8 parameters, each with its own prior volume)")
    print("  -> the difference is what softening the tie costs; the pulls above say "
          "whether the freedom was needed.")
    return {"hard_logz": hard_logz, "hard_error": hard_error,
            "difference": float(difference), "difference_error": float(combined),
            "occam_penalty": float(penalty)}


# --------------------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------------------
def plot_ties(rows, outfile, title="IRAS 07546+3928: how far H-$\\beta$ and "
                                   "H-$\\gamma$ are from H-$\\alpha$",
              max_sigma=None):
    """Draw the tie measurement: one row per parameter, in units of the prior width.

    Built by hand rather than with :func:`galspec.plot.plot_posterior_hist` because
    what has to be shown is a *difference between two parameters* against the width
    of its prior, which that function does not do.

    The horizontal bar on each row is the 16-84% interval of the difference, so it is
    usually shorter than the marker: the data measure these differences to a few
    percent of the tie width, which is the point.

    When the prior has a hard limit, a dashed vertical line at ``+-max_sigma`` marks
    it: a marker sitting on that line is a bound, not a measurement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    kinds = [("width", r"velocity width $\sigma$", r"$\sigma$ difference"),
             ("shift", r"velocity shift $v$", r"$v$ difference")]
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 1.15 * len(rows) / 2 + 2.9),
                             sharey=True)
    for ax, (kind, panel_title, xlabel) in zip(axes, kinds):
        subset = [row for row in rows if row["kind"] == kind]
        for position, row in enumerate(subset):
            capped = row.get("at_cap", False)
            ax.errorbar([row["pull"]], [position],
                        xerr=[[max(row["pull"] - row["pull_lo"], 0.0)],
                              [max(row["pull_hi"] - row["pull"], 0.0)]],
                        fmt="o", ms=5.5,
                        color="#c53030" if capped else "#2b6cb0",
                        ecolor="#c53030" if capped else "#2b6cb0",
                        elinewidth=1.8, capsize=3.0, capthick=1.4,
                        label=None, zorder=4)
            ax.annotate(f"{row['difference']:+.0f} km/s"
                        + (" (at the limit)" if capped else ""),
                        (row["pull"], position), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8.0,
                        color="#c53030" if capped else "0.30", zorder=5)
        ax.axvline(0.0, color="#c53030", lw=1.3, zorder=3)
        for sigma, alpha in ((2.0, 0.09), (1.0, 0.16)):
            ax.axvspan(-sigma, sigma, color="#b7791f", alpha=alpha, lw=0, zorder=1)
        if max_sigma is not None:
            for value in (-max_sigma, max_sigma):
                ax.axvline(value, color="0.25", lw=1.1, ls=(0, (4, 3)), zorder=3)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels([tie_label(f"{row['line']}.{row['parameter']}")
                            for row in subset], fontsize=9)
        ax.set_xlabel(xlabel + " from H-$\\alpha$, in units of the tie width")
        ax.set_title(panel_title, fontsize=10)
        ax.grid(axis="x", color="0.92", lw=0.8)
        ax.set_axisbelow(True)
        # room for the annotations above the top row and below the bottom one
        ax.set_ylim(-0.75, len(subset) - 1 + 0.95)
        highest = max([abs(row["pull_hi"]) for row in subset] +
                      [abs(row["pull_lo"]) for row in subset] +
                      [2.0] + ([max_sigma] if max_sigma is not None else []))
        ax.set_xlim(-highest - 0.6, highest + 0.6)

    handles = [Line2D([], [], color="#c53030", lw=1.3,
                      label="identical to H-$\\alpha$ (the hard tie)"),
               Patch(facecolor="#b7791f", alpha=0.16, lw=0,
                     label="the prior: 1 tie width"),
               Patch(facecolor="#b7791f", alpha=0.09, lw=0,
                     label="2 tie widths"),
               Line2D([], [], color="#2b6cb0", marker="o", ms=5.5, lw=1.8,
                      label="difference and its 16-84% interval")]
    if max_sigma is not None:
        handles.append(Line2D([], [], color="0.25", lw=1.1, ls=(0, (4, 3)),
                              label=f"hard limit of the prior: {max_sigma:g} tie widths"))
    if any(row.get("at_cap") for row in rows):
        handles.append(Line2D([], [], color="#c53030", marker="o", ms=5.5, lw=0,
                              label="posterior against the limit (a bound, "
                                    "not a measurement)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")


# --------------------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nlive", type=int, default=NLIVE,
                        help=f"live points (default {NLIVE}: {38} free parameters)")
    parser.add_argument("--dlogz", type=float, default=DLOGZ)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--device", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--outdir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true",
                        default=None, help="force the progress bar on")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="never draw the progress bar")
    parser.add_argument("--broad-wings", type=int, default=1,
                        help="wind components after the broad core (default 1)")
    parser.add_argument("--width-fraction", type=float, default=WIDTH_FRACTION,
                        help=f"prior width on a line width, as a fraction of "
                             f"H-alpha's (default {WIDTH_FRACTION})")
    parser.add_argument("--shift-fraction", type=float, default=SHIFT_FRACTION,
                        help=f"prior width on a shift, as a fraction of that "
                             f"component's H-alpha width (default {SHIFT_FRACTION})")
    parser.add_argument("--max-sigma", type=float, default=MAX_SIGMA,
                        help="hard limit on the offset, in units of the prior width "
                             f"(default {MAX_SIGMA:g}; try 5 for 'no wider offsets')")
    parser.add_argument("--quick", action="store_true",
                        help="rough answer quickly (nlive=200, dlogz=0.2)")
    args = parser.parse_args(argv)
    if args.quick:
        args.nlive, args.dlogz = 200, 0.2
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("IRAS 07546+3928: broad H-beta and H-gamma tied softly to broad H-alpha")
    print("=" * 78)

    wave, flux, error = base.load_spectrum()
    print(f"\n[1/5] data: {wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A (rest frame)")

    # -- the model, with the Balmer ties released -------------------------------
    model = base.apply_model_labels(base.build_model(jxm, n_balmer_wings=args.broad_wings,
                                                     tie_broad_balmer=False))
    print(f"\n[2/5] model: {len(model.free_param_names)} free parameters "
          f"({len(base.build_model(jxm, n_balmer_wings=args.broad_wings).free_param_names)} "
          f"with the hard tie)")
    for line in TIED_LINES:
        print(f"      {line}: own dv_c, sigma_c, dv_w0, sigma_w0 (was: tied to H-alpha)")

    # -- the priors ------------------------------------------------------------
    notes = base.apply_priors(model, flux, n_balmer_wings=args.broad_wings)
    tie_rows = apply_ties(model, n_balmer_wings=args.broad_wings,
                          width_fraction=args.width_fraction,
                          shift_fraction=args.shift_fraction,
                          max_sigma=args.max_sigma)
    print(f"\n[3/5] soft ties ({len(tie_rows)} parameters), "
          f"hard limit {args.max_sigma:g} tie widths:")
    for row in tie_rows:
        print(f"      {row['parameter']:22s} = Gaussian around "
              f"{row['reference']:22s} ({row['scale']})")
    missing = [name for name in base.free_qualified_names(model)
               if model.get_prior(name) is None]
    if missing:
        raise RuntimeError(f"no prior given for these free parameters: {sorted(missing)}")

    # -- the sampler -----------------------------------------------------------
    print(f"\n[4/5] running JNesty ({args.nlive} live points)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = run_tied_fit(wave, flux, error, model,
                           nlive=args.nlive, dlogz=args.dlogz,
                           max_iterations=args.max_iterations, device=args.device,
                           progress=args.progress)

    # -- results ---------------------------------------------------------------
    print("\n[5/5] results")
    predicted, _ = base.report_quality(fit, wave, flux, error)
    base.print_parameter_table(fit)
    rows = tie_table(fit, width_fraction=args.width_fraction,
                     shift_fraction=args.shift_fraction,
                     max_sigma=args.max_sigma)
    print_tie_table(rows)
    comparison = print_evidence_comparison(fit, rows)
    model_cpu = base.check_parity(fit, wave, predicted, tie_broad_balmer=False)

    npz_path = outdir / f"{PREFIX}posterior.npz"
    np.savez_compressed(npz_path,
                        samples=fit.samples_equal_weight,
                        parameter_names=np.array(fit.param_names),
                        qualified_names=np.array(fit.full_param_names),
                        log_evidence=fit.log_evidence,
                        log_evidence_error=fit.log_evidence_err,
                        wave=wave, flux=flux, error=error, predicted=predicted,
                        width_fraction=args.width_fraction,
                        shift_fraction=args.shift_fraction,
                        max_sigma=args.max_sigma,
                        tie_rows=np.array([str(row) for row in rows]))
    print(f"\n  wrote {npz_path}")
    fits_path = outdir / f"{PREFIX}fit.fits"
    base.save_jnesty_fit_to_fits(fit, fits_path)
    print(f"  wrote {fits_path}")
    print(f"  reload it with "
          f"galspec.jnesty_fit_io.load_jnesty_fit_from_fits({str(fits_path)!r})")

    if not args.no_plots:
        zooms = [("Hgamma_Hbeta", (4230, 5100), (0.0, 2.4)),
                 ("Halpha_NII_SII", (6420, 6800), (0.0, 5.0))]
        base.plot_result(wave, flux, model_cpu, outdir / f"{PREFIX}fit.png",
                         zooms, mask_list=list(base.MASK_WINDOWS), error=error)
        base.plot_corner(fit, outdir / f"{PREFIX}corner.png")
        base.plot_histograms(fit, outdir / f"{PREFIX}histograms.png")
        plot_ties(rows, outdir / f"{PREFIX}ties.png",
                  max_sigma=args.max_sigma if args.max_sigma < 7.0 else None)

    fit.release_device_buffers()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())