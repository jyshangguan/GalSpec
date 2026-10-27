"""Measure the narrow-line profile of IRAS 07546+3928 on the [O III] doublet.

The step before ``fit_iras07546_jax.py``.  Every narrow line in that fit shares one
*measured* profile instead of carrying its own Gaussian, and this script measures it:
a polynomial local continuum, the Fe II pseudo-continuum that sits under [O III]
5007, and a three-component ``Line_MultiGauss_doublet`` -- a core plus two winds --
fitted to the 4920-5080 A window.  ``gen_template`` then turns the fitted doublet into
the single-line profile that ``Line_template`` gives to all ten narrow lines, which is
what makes the global fit cost 27 parameters instead of 30.

Three decisions are worth naming, because the template inherits them:

* **Fe II stays in the model.**  [O III] 5007 sits where the iron pseudo-continuum is
  still strong.  Without it the window fit has nowhere to put that flux but a very
  broad wing on [O III] itself, and that wing would then be added under every narrow
  line in the spectrum.
* **The continuum is a polynomial on a mapped abscissa.**  [O III] sits on the damped
  red wing of broad H-beta, so the local continuum is curved; evaluated on
  ``(x - 4930) / 130`` mapped onto [-1, 1], one prior shape covers every order.
* **The wing width is capped at 1500 km/s.**  Broader than that is indistinguishable
  from the continuum across this window, and a component that broad is a continuum
  term wearing a line's name.

The three-component count is not assumed: it is what the evidence chose between two
and three in ``develop/dev_template/stage1_o3_template.py``, which also scans the
polynomial order and the window cut.  This script keeps the run that produces the
template and skips the scans.

Run it with::

    python example/fit_iras07546_o3_template.py

It writes ``example/data/iras07546_narrow_template.txt`` (the file the global fit
reads), two figures into ``example/output/``, and prints the numbers the walkthrough
quotes.  ``QUICK = True`` trades the answer for a minute.
"""

# %% imports
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
if "get_ipython" not in globals():       # a batch script: there is no window to show
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import galspec
from galspec import jax_models as jxm
from galspec import priors
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary
from galspec.utils import line_wave_dict as LINE_WAVE

from iras07546_model import (            # noqa: E402
    FLUX_LABEL, OUTPUT_DIR, free_qualified_names, load_spectrum,
)

# %% settings
# ---- what to fit --------------------------------------------------------------------
#: The window around the doublet [A, rest frame].  160 A is wide enough for both
#: peaks and for the local continuum to be visible, narrow enough that one polynomial
#: describes it.
WINDOW = (4920.0, 5080.0)
#: Order of the local continuum, on the abscissa mapped onto [-1, 1].
POLY_DEGREE = 3
POLY_WINDOW = (-1.0, 1.0)
#: Components of the doublet: a core plus two winds -- the count the evidence chose.
N_COMPONENTS = 3
#: [O III] 5007/4959 from atomic physics; the ratio is tied, not fitted.
O3_RATIO = 2.98
#: Allowed width of a wing component [km/s]: an outflow of the narrow-line region.
WING_SIGMA_MIN, WING_SIGMA_MAX = 200.0, 1500.0
#: Allowed broadening of the Fe II template [km/s].
IRON_SIGMA_MAX = 3000.0

#: Template velocity grid: +-4000 km/s at 1 km/s.  At 0.046 A/pixel near 5000 A one
#: pixel is 2.8 km/s, so the grid is finer than the data by a factor of three.
TEMPLATE_VMAX = 4000.0
TEMPLATE_STEP = 1.0
#: The profile must have decayed to this fraction of its peak at both grid edges, or
#: the grid is too narrow and every narrow line would get a truncated wing.
TEMPLATE_EDGE_FLOOR = 1e-3

NLIVE = 400
DLOGZ = 0.01
MAX_ITERATIONS = 200_000
DEVICE = "auto"
QUICK = False                  # True -> nlive 200, dlogz 0.1, for a minute-long check

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))        # sigma -> FWHM
#: Where the template goes: the file the global fit reads.
TEMPLATE_FILE = Path(os.environ.get("GALSPEC_TEMPLATE_OUT",
                                    HERE / "data" / "iras07546_narrow_template.txt"))


# %% load
def read_window(wave, flux, error, window=WINDOW):
    """The pixels inside ``window`` [A], and nothing else.

    The fit is *local*: 2848 pixels of the 57 000, so nothing outside the window can
    pull on the profile.  That is the point of a local fit -- a global one would let
    the broad H-beta wing shape move the narrow profile.
    """
    use = (wave >= window[0]) & (wave <= window[1])
    return wave[use], flux[use], error[use]


# %% model
def build_window_model(n_components=N_COMPONENTS, degree=POLY_DEGREE, window=WINDOW):
    """Polynomial continuum + Fe II + an [O III] doublet of ``n_components`` Gaussians.

    Initial values are the published global fit's, so the sampler starts where the
    spectrum is known to be; they are values, not an answer -- the window is fitted
    from scratch and its priors are below.
    """
    continuum = jxm.Polynomial1D(degree=degree, domain=tuple(window),
                                 window=POLY_WINDOW, name="Cont")
    continuum.label = f"Continuum (degree {degree})"
    o3 = jxm.Line_MultiGauss_doublet(
        n_components=n_components,
        amp_c0=1.9, amp_c1=1.9 / O3_RATIO, dv_c=50.0, sigma_c=280.0,
        wavec0=LINE_WAVE["OIII_5007"], wavec1=LINE_WAVE["OIII_4959"],
        name="OIII",
        par_w={f"{stem}_w{i}": value
               for i in range(n_components - 1)
               for stem, value in (("amp", 0.09), ("dv", -200.0), ("sigma", 900.0))})
    # the 5007/4959 ratio is atomic: tied, not fitted
    o3.amp_c1.tied = jxm.tie.tie_MultiGauss_doublet_ratio("OIII", O3_RATIO)

    # amplitude and width are fitted; a velocity shift is not, because across 160 A
    # it is degenerate with the shape of the continuum
    iron = jxm.IronTemplate(amplitude=0.25, sigma=1200.0, z=0.0,
                            template_name="park2022", sigma_max=IRON_SIGMA_MAX,
                            name="FeII", fixed={"z": True, "dv": True})
    iron.label = "Fe II"
    o3.label = "[O III]"
    return continuum + iron + o3


def set_window_priors(model, flux):
    """One prior per free parameter, scaled from the line-free level of the window.

    The continuum coefficients are ``O(1)`` because the abscissa is mapped onto
    [-1, 1]; on raw wavelength a cubic coefficient is of order 1e-11 and no single
    prior shape fits all four orders.
    """
    floor = float(np.percentile(flux, 15.0))
    scale = max(abs(floor), 1e-3)

    for index in range(model["Cont"].degree + 1):
        model["Cont"].set_prior(f"c{index}", priors.Uniform(-2.0 * floor, 2.0 * floor))
    model["FeII"].set_prior("amplitude", priors.LogUniform(1e-3 * scale, 5.0 * scale))
    model["FeII"].set_prior("sigma", priors.Uniform(400.0, IRON_SIGMA_MAX))

    o3 = model["OIII"]
    o3.set_prior("amp_c0", priors.LogUniform(1e-2 * scale, 20.0 * scale))
    o3.set_prior("dv_c", priors.Uniform(-600.0, 600.0))
    o3.set_prior("sigma_c", priors.Uniform(50.0, 800.0))
    for i in range(o3.n_components - 1):
        o3.set_prior(f"amp_w{i}", priors.Uniform(0.0, 1.0))
        o3.set_prior(f"dv_w{i}", priors.Uniform(-1500.0, 1500.0))
        o3.set_prior(f"sigma_w{i}", priors.Uniform(WING_SIGMA_MIN, WING_SIGMA_MAX))

    missing = [name for name in free_qualified_names(model)
               if model.get_prior(name) is None]
    if missing:
        raise RuntimeError(f"no prior on {missing}: the fit would sample the default box")
    return model


# %% fit
def fit_window(n_components=N_COMPONENTS, wave=None, flux=None, error=None,
               progress=False):
    """Nested-sampling fit of the window; returns ``(fit, wave, flux, error)``."""
    if wave is None:
        wave, flux, error = load_spectrum()
    w, f, e = read_window(wave, flux, error)

    model = set_window_priors(build_window_model(n_components), f)
    nlive, dlogz = (200, 0.1) if QUICK else (NLIVE, DLOGZ)
    fit = JNesty_Fit(model, w, f, e, nlive=nlive, bound="multi", device=DEVICE)
    start = time.perf_counter()
    fit.fit(progress=progress and sys.stdout.isatty(), dlogz=dlogz,
            max_iterations=MAX_ITERATIONS)
    fit.seconds = time.perf_counter() - start
    fit.window_data = (w, f, e)
    return fit, w, f, e


# %% template
def half_maximum_width(velc, flux):
    """FWHM of a normalised profile, by linear interpolation on each flank."""
    peak = int(np.argmax(flux))
    right_f, right_v = flux[peak:], velc[peak:]
    left_f, left_v = flux[peak::-1], velc[peak::-1]
    if right_f.min() > 0.5 or left_f.min() > 0.5:
        raise ValueError("the profile never falls below half maximum inside the grid")
    return float(np.interp(0.5, right_f[::-1], right_v[::-1])
                 - np.interp(0.5, left_f[::-1], left_v[::-1]))


def make_template(o3_model, vmax=TEMPLATE_VMAX, step=TEMPLATE_STEP):
    """The single-line profile of the fitted doublet, on a velocity grid.

    ``gen_template`` returns the *shape*: the core at zero velocity, the wings at
    their own ``dv_w``, normalised to unit peak.  The line's own ``dv_c`` is not part
    of it -- that belongs to each line that uses the template, which is why one
    profile can serve H-alpha and [O III] at once.

    ``o3_model`` must be holding its *fitted* values: the method reads the current
    state, not the construction defaults.
    """
    velc = np.arange(-vmax, vmax + step / 2.0, step)
    flux = np.asarray(o3_model.gen_template(velc, normalized=True), dtype=float)

    edges = (float(flux[0]), float(flux[-1]))
    if max(edges) > TEMPLATE_EDGE_FLOOR:
        raise ValueError(
            f"the template has not decayed at the grid edge ({max(edges):.2e} of peak); "
            f"widen TEMPLATE_VMAX (now {vmax:g} km/s)")

    qc = {"peak_velocity": float(velc[int(np.argmax(flux))]),
          "fwhm": half_maximum_width(velc, flux),
          "edge_flux_start": edges[0], "edge_flux_end": edges[1],
          "grid_points": int(velc.size), "grid_step": float(step)}
    qc["sigma_equivalent"] = qc["fwhm"] / FWHM
    return velc, flux, qc


def profile_envelope(fit, velc, n_samples=400):
    """The 16-84% envelope of the profile over the posterior.

    The decomposition into a core and two wings is degenerate: the components trade
    amplitude against width at almost constant likelihood, so the maximum-likelihood
    profile is one draw from a family.  The *shape* they share is much better
    determined, and this measures by how much.  The model is left at the best fit.
    """
    theta = np.asarray(fit.get_best_fit()[2], dtype=float)
    samples = np.asarray(fit.samples_equal_weight)
    keep = np.linspace(0, samples.shape[0] - 1,
                       min(n_samples, samples.shape[0])).astype(int)

    stack = np.empty((keep.size, velc.size))
    try:
        for row, index in enumerate(keep):
            fit.model.set_free_params(samples[index])
            stack[row] = np.asarray(fit.model["OIII"].gen_template(velc))
    finally:
        fit.model.set_free_params(theta)

    low, median, high = np.percentile(stack, [16.0, 50.0, 84.0], axis=0)
    band = float(np.max(0.5 * (high - low)[np.abs(velc) < 800.0]))
    return {"low": low, "median": median, "high": high, "band": band,
            "fwhm_median": half_maximum_width(velc, median)}


# %% verify
def verify_frozen_profile(velc, flux_template, degree=POLY_DEGREE):
    """Refit the window on the frozen profile: does it still describe the data?

    The free fit had three Gaussians per line; this one has the profile and two
    amplitudes with one shared shift.  ``chi2/dof`` moving by a few per cent is
    expected (the profile is a smoother description); moving by orders of magnitude
    would mean the template is wrong.
    """
    wave, flux, error = load_spectrum()
    w, f, e = read_window(wave, flux, error)

    continuum = jxm.Polynomial1D(degree=degree, domain=tuple(WINDOW),
                                 window=POLY_WINDOW, name="Cont")
    o3 = jxm.Line_template_doublet(
        template_velc=velc, template_flux=flux_template,
        amp_c0=1.9, amp_c1=1.9 / O3_RATIO, dv_c=50.0,
        wavec0=LINE_WAVE["OIII_5007"], wavec1=LINE_WAVE["OIII_4959"], name="OIII")
    o3.amp_c1.tied = jxm.tie.tie_MultiGauss_doublet_ratio("OIII", O3_RATIO)
    model = continuum + o3

    floor = float(np.percentile(f, 15.0))
    for index in range(degree + 1):
        model["Cont"].set_prior(f"c{index}", priors.Uniform(-2.0 * floor, 2.0 * floor))
    model["OIII"].set_prior("amp_c0",
                            priors.LogUniform(1e-2 * abs(floor), 20.0 * abs(floor)))
    model["OIII"].set_prior("dv_c", priors.Uniform(-600.0, 600.0))

    nlive, dlogz = (200, 0.1) if QUICK else (NLIVE, DLOGZ)
    fit = JNesty_Fit(model, w, f, e, nlive=nlive, bound="multi", device=DEVICE)
    fit.fit(progress=False, dlogz=dlogz, max_iterations=MAX_ITERATIONS)

    best, _, _ = fit.get_best_fit()
    residual = (f - np.asarray(best(w))) / e
    dof = max(w.size - len(best.free_param_names), 1)
    return {"chi2_dof": float(np.sum(residual ** 2) / dof), "dof": int(dof)}


# %% figures
def plot_window(fit, n_components, path):
    """The window fit: data, total, polynomial, iron, and the doublet's Gaussians."""
    wave, flux, error = fit.window_data
    model, _, _ = fit.get_best_fit()
    ax, _ = galspec.plot_fit_jax(wave, flux, model, error=error,
                                 residual_in_sigma=True, ylim1=(-1.5, 1.5),
                                 xlabel="Rest wavelength  [A]",
                                 ylabel=FLUX_LABEL,
                                 legend_kwargs={"fontsize": 9, "ncol": 2,
                                                "loc": "upper left", "frameon": True})
    fig = ax.figure
    fig.set_size_inches(11.0, 6.0)
    fig.suptitle(f"[O III] window: degree-{POLY_DEGREE} polynomial + Fe II "
                 f"+ {n_components}-component doublet", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_template(velc, flux, qc, path, envelope=None):
    """The extracted profile in velocity space, with the posterior envelope."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    if envelope is not None:
        ax.fill_between(velc, envelope["low"], envelope["high"],
                        color="#2b6cb0", alpha=0.22, lw=0,
                        label="16-84% of the posterior")
        ax.plot(velc, envelope["median"], color="#1a365d", lw=0.8, ls="--",
                label="posterior median")
    ax.plot(velc, flux, color="#2b6cb0", lw=1.4, label="best fit")
    ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
    ax.axvline(qc["peak_velocity"], color="#b5364a", lw=0.8, ls="--")
    ax.annotate(f"FWHM = {qc['fwhm']:.0f} km/s", (0.97, 0.92),
                xycoords="axes fraction", ha="right", fontsize=10)
    ax.set_xlabel("velocity  [km/s]")
    ax.set_ylabel("normalised flux")
    ax.set_title("[O III] narrow-line template")
    ax.set_xlim(velc[0], velc[-1])
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# %% save
def save_template(velc, flux, qc, stage, path=TEMPLATE_FILE):
    """Write the template as two plain columns, with its provenance on top.

    The file is what ``Line_template`` reads, and a template should outlive the code
    that produced it: the header says which window, which continuum order and how
    many components produced it, and how well the frozen profile did on that window.
    Two columns and nothing else, so any tool can read it.
    """
    path = Path(path)
    header = ["narrow-line template of IRAS 07546+3928",
              "measured on the [O III] 4959/5007 doublet, normalised to unit peak",
              "generated by example/fit_iras07546_o3_template.py",
              f"window {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A, degree-{POLY_DEGREE} "
              f"local continuum, {stage['n_components']}-component doublet, "
              f"ln Z = {stage['log_evidence']:.2f}",
              f"FWHM {qc['fwhm']:.1f} km/s  (sigma_equivalent "
              f"{qc['sigma_equivalent']:.1f} km/s), peak at "
              f"{qc['peak_velocity']:+.1f} km/s",
              f"the frozen profile fits the window at chi2/dof = {stage['chi2_dof']:.3f}",
              "col 1: velocity [km/s]   col 2: normalised flux"]
    np.savetxt(path, np.column_stack([velc, flux]), fmt="%.6f",
               header="\n".join(header), comments="# ")
    print(f"  wrote {path}")
    return path


# %% entry
def main() -> int:
    print("=" * 78)
    print("Stage 1: measure the narrow-line profile on the [O III] doublet")
    print("=" * 78)
    print(f"device: {jax_devices_summary()}")
    if QUICK:
        print("QUICK = True: this is a rough answer, not the published template")

    wave, flux, error = load_spectrum()
    w, f, _ = read_window(wave, flux, error)
    print(f"window {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A: {w.size} pixels, "
          f"line-free level {np.percentile(f, 15.0):.3f}")

    print("\n-- fit the window --")
    fit, _, _, _ = fit_window(wave=wave, flux=flux, error=error)
    log_evidence, log_evidence_err = fit.get_evidence()
    print(f"  ln Z = {log_evidence:.2f} +- {log_evidence_err:.2f}   "
          f"({fit.n_iterations} iterations, {fit.seconds / 60.0:.1f} min)")

    print("\n-- the profile --")
    best, _, _ = fit.get_best_fit()
    velc, template_flux, qc = make_template(best["OIII"])
    envelope = profile_envelope(fit, velc)
    qc["posterior_fwhm_median"] = envelope["fwhm_median"]
    qc["posterior_band"] = envelope["band"]
    print(f"  FWHM = {qc['fwhm']:.1f} km/s (sigma_equiv {qc['sigma_equivalent']:.1f} km/s), "
          f"peak at {qc['peak_velocity']:+.1f} km/s")
    print(f"  posterior: median FWHM {envelope['fwhm_median']:.1f} km/s, "
          f"envelope +-{envelope['band']:.3f} of peak inside +-800 km/s")

    print("\n-- does the frozen profile still describe the window? --")
    verification = verify_frozen_profile(velc, template_flux)
    print(f"  chi2/dof = {verification['chi2_dof']:.3f} "
          f"over {verification['dof']} degrees of freedom")

    print("\n-- figures --")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_window(fit, N_COMPONENTS, OUTPUT_DIR / "iras07546_o3_window_fit.png")
    plot_template(velc, template_flux, qc,
                  OUTPUT_DIR / "iras07546_narrow_template_profile.png", envelope=envelope)

    print("\n-- the file the global fit reads --")
    stage = {"n_components": N_COMPONENTS, "log_evidence": log_evidence,
             "chi2_dof": verification["chi2_dof"], "window": list(WINDOW),
             "poly_degree": POLY_DEGREE}
    save_template(velc, template_flux, qc, stage)

    print("\nDone.  Next: python example/fit_iras07546_jax.py, which reads that file")
    print("and gives the profile to every narrow line.")
    fit.release_device_buffers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
