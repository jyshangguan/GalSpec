# %% imports
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

import matplotlib
if "get_ipython" not in globals():       # a batch script: there is no window to show
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

# GalSpec is used straight from the checkout (it is not pip-installed), so put the
# repository root on the path.  This file lives in <repo>/example/, which is also
# where the notebook runs, so the two share this cell.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE) not in sys.path:            # so ``import iras07546_model`` works however
    sys.path.insert(0, str(HERE))        # this file is loaded (script, notebook, test)

import galspec
from galspec import jax_models as jxm
from galspec import priors
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary
from galspec.jnesty_fit_io import save_jnesty_fit_to_fits
from galspec.utils import line_wave_dict as LINE_WAVE


# ---- everything the three worked examples share ---------------------------------
# The spectrum loader, the narrow-line template, the seventeen components and their
# ties, and the priors all live in one module, so ``fit_iras07546_jax_tied.py`` and
# ``fit_iras07546_jax_toggle.py`` import the same model instead of each other's
# output files.  Any one of the three runs on its own, from this folder:
#
#     python fit_iras07546_jax.py
from iras07546_model import (             # noqa: E402
    CORNER_PARAMS, DATA_FILE, E_BV, ERROR_FRACTION, FLUX_LABEL, HIST_PARAMS,
    LINE_WAVE, MODEL_LABELS, NARROW_REFERENCE, N_WINGS, OUTPUT_DIR,
    PHOTON_FRACTION, REDSHIFT, REFERENCE, TEMPLATE_FILE, WAVE_MAX, WAVE_MIN, ZOOMS,
    build_model, free_qualified_names, load_spectrum, load_template, set_priors,
    template_fwhm, unit_of,
)

# %% settings
# ---- how hard to work on it ---------------------------------------------------------
NLIVE = 400                    # live points kept by the sampler
DLOGZ = 0.01                   # stop when < 1% of the evidence may remain unseen
MAX_ITERATIONS = 200_000
DEVICE = "auto"                # "gpu", "cpu" or "auto"
QUICK = False                  # True -> nlive 120, dlogz 0.5 (a rough answer)


# ---- main: the four steps of a fit, top to bottom ----------------------------------
def main() -> int:
    print("=" * 78)
    print("IRAS 07546+3928: 4200-6800 A, fitted with JNesty on the GPU")
    print("=" * 78)

    # %% step-1-data
    # ---- 1. Load the spectrum ------------------------------------------------------
    wave, flux, error = load_spectrum()
    print(f"\ndata: {wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A (rest frame)")
    print(f"      peak flux {flux.max():.2f}, line-free level "
          f"{np.percentile(flux, 15):.2f}  [1e-14 erg/s/cm2/A]")

    # %% step-2-model
    # ---- 2. Load the narrow-line profile, build the model, give every parameter a prior
    # The profile is the one thing this model does not fit.  It comes from stage 1,
    # which fitted the [O III] doublet alone: two components of known ratio, a
    # polynomial continuum and an explicit Fe II component, with the evidence choosing
    # between two and three kinematic components.  Everything about the narrow lines
    # below is that profile plus an amplitude.
    template = load_template()
    tvel, tflux = template
    print(f"\ntemplate: {TEMPLATE_FILE.name}, {tvel.size} points, "
          f"{tvel[0]:+.0f} to {tvel[-1]:+.0f} km/s, "
          f"FWHM {template_fwhm(tvel, tflux):.0f} km/s")

    model = build_model(template, N_WINGS)
    set_priors(model, flux)
    print(f"\nmodel: {len(model.submodel_names)} components, no LSF convolution, "
          f"{len(model.free_param_names)} free parameters")
    print(f"       narrow lines are the template: "
          f"{sum(isinstance(m, (jxm.Line_template, jxm.Line_template_doublet)) for m in model)} of the "
          f"{len(model.submodel_names)} components, one shift, no widths")
    print("       " + ", ".join(model.submodel_names))
    missing = [q for q in free_qualified_names(model) if model.get_prior(q) is None]
    if missing:
        raise RuntimeError(f"no prior given for these free parameters: {missing}")

    # %% step-3-fitter
    # ---- 3. Fit: nested sampling on the GPU ----------------------------------------
    # QUICK is the one shortcut: fewer live points, a looser stopping criterion
    nlive, dlogz = (120, 0.5) if QUICK else (NLIVE, DLOGZ)
    fit = JNesty_Fit(model, wave, flux, error, nlive=nlive, bound="multi", device=DEVICE)
    print(f"\nfit:   {nlive} live points, dlogz {dlogz}, "
          f"device {fit.device}  [{jax_devices_summary()}]")

    # %% step-4-fit
    started = time.perf_counter()
    fit.fit(progress=sys.stdout.isatty(), dlogz=dlogz, max_iterations=MAX_ITERATIONS)
    print(f"\n       log Z = {fit.log_evidence:.2f} +- {fit.log_evidence_err:.2f}"
          f"   ({fit.n_iterations} iterations, "
          f"{(time.perf_counter() - started) / 60:.1f} min)")

    # %% step-5-results
    # ---- 4. Present the results: the parameter table --------------------------------
    model_best, _, theta = fit.get_best_fit()
    predicted = np.asarray(model_best.flux(wave, theta))
    residual = (flux - predicted) / error
    dof = wave.size - len(theta)
    print(f"       chi2/dof = {np.sum(residual ** 2) / dof:.3f}, residual rms "
          f"{residual.std():.2f} sigma, worst pixel {np.abs(residual).max():.1f} sigma")

    # the numbers, with the 16th and 84th percentiles of the posterior
    lo, med, hi = fit.get_quantiles((0.16, 0.5, 0.84))
    print(f"\n  {'parameter':22s} {'median':>10s} {'-':>9s} {'+':>9s}  unit")
    print("  " + "-" * 76)
    for index, name in enumerate(fit.full_param_names):
        print(f"  {name:22s} {med[index]:10.5g} {med[index] - lo[index]:9.2g} "
              f"{hi[index] - med[index]:9.2g}  {unit_of(name)}")

    # the scoreboard.  The published fit is the same spectrum, the same continuum,
    # the same broad lines and the same priors, with a Gaussian per narrow line and
    # one width shared between them: the only difference is the narrow-line profile.
    # Nested sampling compares models by their evidence, so this is the number that
    # says whether freezing the shape was worth three fewer parameters.
    template_row = REFERENCE["template"]
    print(f"\n  {'model':26s} {'free':>5s} {'ln Z':>12s} {'chi2/dof':>9s}")
    print("  " + "-" * 56)
    for label, row in REFERENCE.items():
        mark = "*" if label == "template" else " "
        print(f" {mark}{label:26s} {row['free_parameters']:5d} "
              f"{row['log_evidence']:12.2f} {row['chi2_dof']:9.3f}")
    delta = template_row["log_evidence"] - REFERENCE["published"]["log_evidence"]
    print(f"  {'':27s} {'':5s} Delta = {delta:+.2f}")
    if delta < 0:
        print("\n  The template costs a little evidence: the narrow Balmer and [N II]")
        print("  lines are ~16% broader than [O III], and a frozen [O III] shape cannot")
        print("  follow them.  What it buys is three fewer parameters, a better fit to")
        print("  [O III] itself, and a profile that is measured rather than assumed --")
        print("  stage 1 measures that ratio and prints it; see also")
        print("  develop/dev_template/README.md.")

    # %% step-6-figure
    # the fit itself: every component, and the residuals in units of the error.
    # The legend sits outside the axes: seventeen components do not fit inside a
    # panel without covering the data.
    fig = plt.figure(figsize=(17.0, 8.0))
    ax = fig.add_axes([0.055, 0.20, 0.80, 0.72])
    axr = fig.add_axes([0.055, 0.07, 0.80, 0.13])
    galspec.plot.plot_fit(wave, flux, model_best, ax=ax, axr=axr, error=error,
                          residual_in_sigma=True,
                          xlabel="Rest wavelength  [A]", ylabel=FLUX_LABEL,
                          legend_kwargs={"fontsize": 10, "ncol": 2,
                                         "loc": "center left",
                                         "bbox_to_anchor": (1.005, 0.5),
                                         "frameon": False})
    ax.set_title("IRAS 07546+3928: narrow lines on the measured [O III] profile",
                 fontsize=13)
    fig_path = OUTPUT_DIR / "iras07546_jax_fit.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {fig_path}")

    # %% step-7-zooms
    # and three zooms, where the lines actually are
    for tag, title, xlim, ylim in ZOOMS:
        ax, _ = galspec.plot.plot_fit(wave, flux, model_best, error=error,
                                      xlim=xlim, ylim0=ylim, ylim1=(-8, 8),
                                      residual_in_sigma=True,
                                      xlabel="Rest wavelength  [A]", ylabel=FLUX_LABEL,
                                      legend_kwargs={"fontsize": 8, "ncol": 2,
                                                     "loc": "upper left",
                                                     "frameon": True})
        ax.figure.set_size_inches(13.5, 6)
        ax.set_title(f"{title}: {xlim[0]:.0f}-{xlim[1]:.0f} A", fontsize=13)
        out = OUTPUT_DIR / f"iras07546_jax_fit_{tag}.png"
        ax.figure.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  wrote {out}")

    # %% step-8-posteriors
    # and the joint and the marginal posteriors: the pairs, then one histogram each
    keep = [i for i, name in enumerate(fit.full_param_names) if name in CORNER_PARAMS]
    labels = getattr(fit, "param_labels", None) or list(fit.full_param_names)
    try:
        corner_fig = corner.corner(fit.samples_equal_weight[:, keep],
                                   labels=[labels[i] for i in keep],
                                   quantiles=(0.16, 0.5, 0.84), show_titles=True,
                                   title_fmt=".3g", title_kwargs={"fontsize": 8})
    except ValueError:
        # corner refuses a parameter that never moved, which happens when the sampler
        # is stopped before it starts moving (QUICK with a tiny MAX_ITERATIONS).
        print("  corner plot skipped: a selected parameter has no posterior range")
    else:
        corner_fig.savefig(OUTPUT_DIR / "iras07546_jax_corner.png", dpi=110,
                           bbox_inches="tight")
    galspec.plot.plot_posterior_hist(
        fit, params=HIST_PARAMS, ncols=4, bins=40,
        title="IRAS 07546+3928: marginal posteriors",
        filename=str(OUTPUT_DIR / "iras07546_jax_histograms.png"))
    print(f"  wrote {OUTPUT_DIR / 'iras07546_jax_corner.png'} and "
          f"{OUTPUT_DIR / 'iras07546_jax_histograms.png'}")

    # %% step-9-save
    # the record: the samples as plain arrays, and the whole fit as one FITS file
    npz_path = OUTPUT_DIR / "iras07546_jax_posterior.npz"
    np.savez_compressed(npz_path,
                        samples=fit.samples_equal_weight,
                        parameter_names=np.array(fit.param_names),
                        qualified_names=np.array(fit.full_param_names),
                        log_evidence=fit.log_evidence,
                        log_evidence_error=fit.log_evidence_err,
                        wave=wave, flux=flux, error=error, predicted=predicted)
    fits_path = OUTPUT_DIR / "iras07546_jax_fit.fits"
    save_jnesty_fit_to_fits(fit, fits_path)
    print(f"  wrote {npz_path}")
    print(f"  wrote {fits_path}  ({fits_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  reload it with galspec.jnesty_fit_io.load_jnesty_fit_from_fits("
          f"{str(fits_path)!r})")
    fit.release_device_buffers()

    # %% entry
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
