#!/usr/bin/env python
"""How many Gaussian components do the broad Balmer lines need?

The published fit gives each broad Balmer line a core and one wind: two components,
a fixed number.  This script lets the data decide instead, by putting a prior on each
wing amplitude that has a finite chance of being *exactly* zero -- a spike and slab.
That fraction of the prior volume switches the wing off completely, so a single run
with a fixed number of parameters explores "how many components does this profile
need?" and returns a posterior probability for each answer.

    1. load the spectrum          (the same data, from the same file)
    2. build the model            (four components per broad Balmer line)
       and gate every wing amplitude at zero
    3. fit it
    4. show the result            (P(N components), per-wing Bayes factors, and the
                                   components ranked by width)

Why the labels are not used
---------------------------
Four Gaussians can describe the same profile in more than one way, and the posterior
visits all of them.  So the table ranks the components *by width* and reports how much
flux each carries; it never claims that "wing 1" is a particular physical structure.
The count is the answer, and it is label-invariant by construction.

Why the evidence is not comparable
----------------------------------
The model is a mixture of sub-models, so its normalisation is not the same integral as
a plain prior over the same parameters.  The ln Z printed here says how much better one
*number of components* fits, not how it compares with the published fit.  Quote the
component count from here, and the evidence from a plain-prior refit at that count.

Running it
----------
::

    python example/fit_iras07546_jax_toggle.py

``P_OFF`` and ``MIN_WING_FRACTION`` at the top of the file say how likely each extra
wing is a priori and when a wing counts as present; the widths are the published
recipe's, 300-8000 km/s for the core and for every wing.  The fit takes hours with
the settings below and minutes with ``QUICK = True``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_iras07546_jax as plain

import galspec
from galspec import priors
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary
from galspec.jnesty_fit_io import load_jnesty_fit_from_fits, save_jnesty_fit_to_fits

# %% settings
OUTPUT_DIR = plain.OUTPUT_DIR
PREFIX = "iras07546_jax_toggle_"

#: Wings after the core, so four components per broad Balmer line.
N_WINGS = 3
#: Prior probability that wing 0, 1, 2 is off: the higher order, the less likely.
P_OFF = (0.2, 0.5, 0.7)
#: A wing counts as present above this fraction of the core peak.
MIN_WING_FRACTION = 0.03
#: The lines that get wings (the He lines keep one broad Gaussian).
WINGED_LINES = ("halpha", "hbeta", "hgamma")
#: The published fit, for the closing comparison: its own file when it is there, and
#: the numbers `example/output/` gives when it is not.
PUBLISHED_FIT = OUTPUT_DIR / "iras07546_jax_fit.fits"
PUBLISHED = {"free_parameters": 30, "log_evidence": 130894.66, "chi2_dof": 0.307}

NLIVE = 1000                   # 40 free parameters: 400 was measured to be too few
DLOGZ = 0.01
MAX_ITERATIONS = 400_000
DEVICE = "auto"
QUICK = False                  # True -> nlive 200, dlogz 0.2

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))        # sigma -> FWHM


# %% results-helpers
def amp_matrix(fit):
    """``(samples, names, amps)``: the wing amplitudes of broad H-alpha, one per wing."""
    names = list(fit.full_param_names)
    samples = np.asarray(fit.samples_equal_weight)
    amps = np.stack([samples[:, names.index(f"b_halpha.amp_w{i}")]
                     for i in range(N_WINGS)], 1)
    return samples, names, amps


def component_count(fit, floor=0.0):
    """Posterior probability of 0, 1, ... ``N_WINGS`` wings being on.

    ``floor=0`` counts a wing as on if it is not exactly zero (i.e. the spike was not
    visited); ``floor=MIN_WING_FRACTION`` counts it only if it is big enough to
    matter.  Label-invariant by construction: it counts, it does not name.
    """
    _, _, amps = amp_matrix(fit)
    counts = (amps > floor).sum(1)
    return np.array([float(np.mean(counts == n)) for n in range(N_WINGS + 1)])


def wing_evidence(fit):
    """Per wing: posterior probability, prior-corrected Bayes factor, and its size.

    The Bayes factor divides the posterior odds by the prior odds, so a value near 1
    means the data did not move the wing away from what the prior said, and a large
    value means they did.
    """
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
            "wing": i, "p_off": P_OFF[i], "p_on": p_on, "bayes": bayes,
            "amp": float(np.median(amps[on, i])) if on.any() else 0.0,
            "amp_97": float(np.percentile(amps[on, i], 97.5)) if on.any() else 0.0,
            "sigma": float(np.median(sigma)) if sigma is not None else np.nan,
            "dv": float(np.median(dv)) if dv is not None else np.nan,
            "counted": bool(on.any() and np.median(amps[on, i]) > MIN_WING_FRACTION),
        })
    return rows


def sorted_components(fit, floor=MIN_WING_FRACTION, max_samples=2000):
    """The components ranked by width, with the fraction of the broad flux each carries.

    This is the honest way to present a decomposition whose labels are not identified:
    rank in width, and say how often the posterior has that many components at all.
    """
    samples, names, amps = amp_matrix(fit)
    sigmas = np.stack([samples[:, names.index(f"b_halpha.sigma_w{i}")]
                       for i in range(N_WINGS)], 1)
    dv = np.stack([samples[:, names.index(f"b_halpha.dv_w{i}")]
                   for i in range(N_WINGS)], 1)
    scale = base_scale()                     # a flux is amplitude*sigma*scale

    amp_c = samples[:, names.index("b_halpha.amp_c")]
    sigma_c = samples[:, names.index("b_halpha.sigma_c")]
    core_flux = amp_c * sigma_c * np.sqrt(2.0 * np.pi) * scale

    step = max(1, samples.shape[0] // max_samples)
    rows = []
    for rank in range(N_WINGS):
        used_amp, used_sigma, used_dv, used_fraction = [], [], [], []
        for s in range(0, samples.shape[0], step):
            order = np.argsort(sigmas[s])                 # narrowest first
            keep = [j for j in order if amps[s, j] > floor]
            if len(keep) <= rank:
                continue
            j = keep[rank]
            flux = amp_c[s] * amps[s, j] * sigmas[s, j] * np.sqrt(2.0 * np.pi) * scale
            total = core_flux[s] + sum(amp_c[s] * amps[s, k] * sigmas[s, k]
                                       * np.sqrt(2.0 * np.pi) * scale for k in keep)
            used_amp.append(amps[s, j])
            used_sigma.append(sigmas[s, j])
            used_dv.append(dv[s, j])
            used_fraction.append(flux / total)
        rows.append({
            "rank": rank,
            "n": len(used_amp),
            "amp": float(np.median(used_amp)) if used_amp else 0.0,
            "sigma": float(np.median(used_sigma)) if used_sigma else np.nan,
            "fwhm": float(np.median(used_sigma)) * FWHM if used_sigma else np.nan,
            "dv": float(np.median(used_dv)) if used_dv else np.nan,
            "flux_fraction": float(np.median(used_fraction)) if used_fraction else 0.0,
        })
    return rows


def base_scale():
    """``lambda / c`` with lambda = H-alpha: turns ``amplitude x sigma`` into a flux.

    Amplitudes are in 1e-14 erg/s/cm2/A at 6563 A, so the result is in
    1e-14 erg/s/cm2 x (km/s) / (km/s); the tables divide by 1e3 to quote 1e-17.
    """
    return plain.LINE_WAVE["Halpha"] / 299792.458


# %% toggle-priors
# ---- The published priors, with the wing amplitudes gated ---------------------------
def toggle_priors(model, flux):
    """The published recipe plus a spike and slab on every wing amplitude.

    Everything except the wing amplitudes is what the published fit uses, so the two
    are comparable component by component -- including the widths: the published
    recipe already gives a wing the core's 300-8000 km/s box, because nothing says a
    wing must be broader than its core.  One thing changes:

    * ``amp_w<i>`` gets a :class:`galspec.priors.SpikeAndSlab` with ``p_off=P_OFF[i]``:
      that fraction of the prior volume is at exactly zero, the rest on a linear slab
      (0, 1].  The amplitude stays bounded by the core, which is what keeps the labels
      meaningful when only one component is present.

    Needed twice: ``main`` applies it to the fit, and
    ``develop/dev_jax/prior_api_report.py`` compares its prior transform with the
    pre-prior-API spelling.
    """
    plain.set_priors(model, flux)          # core and wing already share 300-8000 km/s
    for line in WINGED_LINES:
        for i in range(N_WINGS):
            model.set_prior(f"b_{line}.amp_w{i}",
                            priors.SpikeAndSlab(p_off=P_OFF[i], lower=0.0, upper=1.0))
    return model


# %% main
# ---- Fit, and show the result -------------------------------------------------------
def main() -> int:
    nl, dlogz = (200, 0.2) if QUICK else (NLIVE, DLOGZ)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("IRAS 07546+3928: how many Gaussian components do the broad lines need?")
    print("=" * 78)

    # ---- 1. Load the spectrum ------------------------------------------------------
    wave, flux, error = plain.load_spectrum()
    print(f"\ndata:  {wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A (rest frame), "
          f"peak {flux.max():.2f}")

    # ---- 2. Build the model, and gate every wing amplitude at zero -----------------
    model = toggle_priors(plain.build_model(n_wings=N_WINGS), flux)
    print(f"\nmodel: {len(model.free_param_names)} free parameters, "
          f"{1 + N_WINGS} components per broad Balmer line")
    print(f"       wings off a priori: p_off = "
          + ", ".join(f"{p:.1f}" for p in P_OFF)
          + f" (of {len(WINGED_LINES) * N_WINGS} gated amplitudes)")
    print(f"       a wing counts above {MIN_WING_FRACTION * 100:.0f}% of the core peak")
    missing = [q for q in plain.free_qualified_names(model) if model.get_prior(q) is None]
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
    # Save first: a three-hour fit must not be lost to a bug in the analysis below.
    npz_path = OUTPUT_DIR / f"{PREFIX}posterior.npz"
    np.savez_compressed(npz_path,
                        samples=fit.samples_equal_weight,
                        parameter_names=np.array(fit.param_names),
                        qualified_names=np.array(fit.full_param_names),
                        log_evidence=fit.log_evidence,
                        log_evidence_error=fit.log_evidence_err,
                        wave=wave, flux=flux, error=error,
                        component_count_wings=component_count(fit,
                                                              floor=MIN_WING_FRACTION),
                        p_off=np.array(P_OFF), min_wing_fraction=MIN_WING_FRACTION,
                        # recorded from the model, so the file says what was used
                        sigma_range=np.array(
                            model.get_prior("b_halpha.sigma_w0").support))
    fits_path = OUTPUT_DIR / f"{PREFIX}fit.fits"
    save_jnesty_fit_to_fits(fit, fits_path)
    print(f"\n  wrote {npz_path}")
    print(f"  wrote {fits_path}  ({fits_path.stat().st_size / 1e6:.0f} MB)")

    model_best, _, theta = fit.get_best_fit()
    predicted = np.asarray(model_best.flux(wave, theta))
    residual = (flux - predicted) / error
    dof = wave.size - len(theta)
    print(f"  chi2/dof = {np.sum(residual ** 2) / dof:.3f}, residual rms "
          f"{residual.std():.2f} sigma")

    # the answer: how many components?
    p_wings = component_count(fit, floor=0.0)
    p_needed = component_count(fit, floor=MIN_WING_FRACTION)
    cumulative = np.cumsum(p_needed)
    k_min = int(np.argmax(p_needed))
    print("\n  how many wings does the profile need?")
    print("  " + "-" * 74)
    print(f"  {'wings':>5s} {'components':>11s} {'P(N), all':>10s} "
          f"{'P(N), above ' + f'{MIN_WING_FRACTION * 100:.0f}%':>18s} {'cumulative':>11s}")
    for n in range(N_WINGS + 1):
        print(f"  {n:5d} {n + 1:11d} {p_wings[n]:10.3f} {p_needed[n]:18.3f} "
              f"{cumulative[n]:11.3f}")
    print(f"\n  k_min = 1 + {k_min} wings = {k_min + 1} components "
          f"(posterior probability {p_needed[k_min]:.3f})")

    # per wing: did the data want it?
    print("\n  per-wing evidence (Bayes factor = posterior odds / prior odds)")
    print("  " + "-" * 74)
    print(f"  {'wing':>4s} {'p_off':>6s} {'P(on)':>7s} {'BF':>10s} {'amp':>7s} "
          f"{'amp 97.5%':>10s} {'FWHM [km/s]':>12s} {'dv [km/s]':>10s}")
    for row in wing_evidence(fit):
        bayes = "inf" if not np.isfinite(row["bayes"]) else f"{row['bayes']:.1f}"
        print(f"  {row['wing']:4d} {row['p_off']:6.1f} {row['p_on']:7.3f} "
              f"{bayes:>10s} {row['amp']:7.4f} {row['amp_97']:10.4f} "
              f"{row['sigma'] * FWHM:12.0f} {row['dv']:10.0f}")

    # the components themselves, ranked by width
    by_name = {q: i for i, q in enumerate(fit.full_param_names)}
    sigma_c = fit.samples_equal_weight[:, by_name["b_halpha.sigma_c"]]
    dv_c = fit.samples_equal_weight[:, by_name["b_halpha.dv_c"]]
    print("\n  components, sorted by width (labels are not identified)")
    print("  " + "-" * 74)
    print(f"  {'rank':>4s} {'kind':>8s} {'amp/core':>9s} "
          f"{'FWHM [km/s]':>12s} {'dv [km/s]':>10s} {'flux fraction':>14s}")
    print(f"  {0:4d} {'core':>8s} {1.0:9.3f} "
          f"{np.median(sigma_c) * FWHM:12.0f} {np.median(dv_c):10.0f} {'--':>14s}")
    for row in sorted_components(fit):
        if row["n"] < 1:
            continue
        print(f"  {row['rank'] + 1:4d} {'wing':>8s} {row['amp']:9.3f} "
              f"{row['fwhm']:12.0f} {row['dv']:10.0f} {row['flux_fraction']:14.3f}")
    print(f"\n  (a wing counts above {MIN_WING_FRACTION * 100:.0f}% of the core peak; "
          f"effective components = {k_min + 1})")

    # the caveat, and the published numbers to compare against
    evidence, parameters, source = (PUBLISHED["log_evidence"],
                                    PUBLISHED["free_parameters"], "documented")
    if PUBLISHED_FIT.exists():
        reference = load_jnesty_fit_from_fits(PUBLISHED_FIT)
        evidence, parameters, source = (reference.log_evidence, reference.ndim,
                                        PUBLISHED_FIT.name)
        del reference
    print(f"\n  the published two-component fit: {parameters} free parameters, "
          f"ln Z = {evidence:.2f}, chi2/dof = {PUBLISHED['chi2_dof']:.3f}  [{source}]")
    print(f"  this run: {fit.ndim} free parameters, "
          f"ln Z = {fit.log_evidence:.2f} +- {fit.log_evidence_err:.2f}")
    print("  (the two ln Z are not comparable: the toggled model is a prior mixture "
          "over\n   subsets.  Quote the component count from here and the evidence "
          "from a\n   plain-prior refit at that count.)")

    # figures
    ax, _ = galspec.plot.plot_fit(wave, flux, model_best, error=error,
                                  residual_in_sigma=True, xlabel="Rest wavelength  [A]",
                                  ylabel=plain.FLUX_LABEL,
                                  legend_kwargs={"fontsize": 10, "ncol": 2})
    ax.figure.set_size_inches(15, 7)
    ax.set_title(f"IRAS 07546+3928: {1 + N_WINGS} components per broad Balmer line, "
                 f"{k_min + 1} of them needed", fontsize=13)
    ax.figure.savefig(OUTPUT_DIR / f"{PREFIX}fit.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    keep = [i for i, name in enumerate(fit.full_param_names)
            if name in plain.CORNER_PARAMS
            or name.split(".")[-1].startswith("amp_w")]
    labels = getattr(fit, "param_labels", None) or list(fit.full_param_names)
    import corner
    corner_fig = corner.corner(fit.samples_equal_weight[:, keep],
                               labels=[labels[i] for i in keep],
                               quantiles=(0.16, 0.5, 0.84), show_titles=True,
                               title_fmt=".3g", title_kwargs={"fontsize": 8})
    corner_fig.savefig(OUTPUT_DIR / f"{PREFIX}corner.png", dpi=110, bbox_inches="tight")
    plt.close(corner_fig)
    galspec.plot.plot_posterior_hist(
        fit, params=plain.HIST_PARAMS, ncols=4, bins=40,
        title="IRAS 07546+3928: with the wings gated, marginal posteriors",
        filename=str(OUTPUT_DIR / f"{PREFIX}histograms.png"))

    # the answer as a picture: the posterior of the component count, and the wings
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4),
                                      gridspec_kw={"width_ratios": [1.1, 1]})
    counts = np.arange(N_WINGS + 2)
    left.bar(counts, np.append(p_needed, 0.0), color="#2b6cb0", width=0.65)
    left.bar(counts, np.append(p_wings - p_needed, 0.0), bottom=np.append(p_needed, 0.0),
             color="#bee3f8", width=0.65, label="not bigger than the floor")
    left.set_xticks(counts)
    left.set_xlabel("number of components per broad Balmer line")
    left.set_ylabel("posterior probability")
    left.set_title(f"P(N): {k_min + 1} components are enough", fontsize=10)
    left.legend(fontsize=8, frameon=False)

    rows = wing_evidence(fit)
    right.bar([row["wing"] for row in rows], [row["p_on"] for row in rows],
              color="#2f855a", width=0.6)
    right.axhline(1.0 - np.mean(P_OFF), color="0.5", ls=":", lw=1.0)
    right.set_xticks([row["wing"] for row in rows])
    right.set_xlabel("wing")
    right.set_ylabel("P(on)")
    right.set_ylim(0, 1.05)
    right.set_title("per-wing posterior probability of being on", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{PREFIX}components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote the figures and {OUTPUT_DIR / f'{PREFIX}components.png'}")

    fit.release_device_buffers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())