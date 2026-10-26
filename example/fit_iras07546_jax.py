#!/usr/bin/env python
"""Fit the spectrum of IRAS 07546+3928 with JNesty on the GPU.

The four steps are the file: load the spectrum, build the model, fit it, show the
result.  Nothing is optimised before nested sampling starts, so the priors cannot
inherit a preliminary answer, and nothing is masked.

What is fitted
--------------
4200-6800 A (rest frame) of the quasar IRAS 07546+3928 -- 56 956 pixels, from
H-gamma out to [S II] -- in one go, so the continuum and the lines are constrained
by the same posterior::

    power-law continuum        flux density at 5500 A + slope
  + Fe II template             broadened by broad H-alpha itself
  + H-alpha, H-beta, H-gamma   a narrow Gaussian + a broad MultiGauss
  + He I 5876, He II 4686      a narrow Gaussian + one broad Gaussian
  + [O III] 4959/5007          a narrow core + one broad wing
  + [N II], [O I]              narrow doublets, ratios fixed by atomic physics
  + [S II] 6716/6731           both peaks free: the ratio is a density diagnostic
  + [O III] 4363               a single narrow Gaussian

Two models per Balmer line
--------------------------
``n_halpha`` is the narrow component.  ``b_halpha`` is the broad one: component 0
(``amp_c``, ``dv_c``, ``sigma_c``) is the broad core -- "core" names a component,
it is not a narrow line -- and ``amp_w0``/``dv_w0``/``sigma_w0`` add the broader
wind.  The wind amplitude is a *ratio* to the core, bounded to [0, 1]: the two
components of a ``Line_MultiGauss`` are exchangeable (swapping them leaves the
spectrum bit-identical), and the swap sends ``amp_w0 -> 1/amp_w0``, which is
outside that interval.  So the labels mean what the names say, and the widths keep
their full data-driven range.  Widths and shifts are shared across the three
Balmer lines -- one velocity structure for the broad-line region -- while each line
keeps its own peak, which is how the Balmer decrement falls out of the fit.

The error on every pixel
------------------------
The FITS file has flux only, no uncertainty, so the error is built from the data:
the pixel-to-pixel scatter of a running median (read noise and sky), plus 5% of the
flux itself in quadrature for the photon noise of the source.  On the bright
continuum that second term dominates, which is the honest budget for a template
fit: the residuals then come out at ~0.6 sigma and chi2/dof at ~0.3, i.e. the error
bars are conservative rather than optimistic.

No instrumental smoothing is applied.  At 0.046 A per pixel this spectrum grossly
oversamples R ~ 2000 (the instrument contributes ~150 km/s while the narrow lines
are ~700 km/s), so the fitted narrow width is the observed one -- gas plus
instrument, exactly as in ``wrk_sagan_iras07546.ipynb``.

Running it
----------
::

    python example/fit_iras07546_jax.py

Change ``NLIVE``, ``DLOGZ`` or ``QUICK`` at the top of the file; ``QUICK = True``
is a rough answer in a few minutes.  Results go to ``example/output/`` (override
with the ``GALSPEC_OUTPUT_DIR`` environment variable): the figures, an ``.npz``
with the posterior samples, and the FITS file that holds the whole fit -- samples,
evidence, priors, the model and the data it was made against.  The FITS file is the
one to keep: :func:`galspec.jnesty_fit_io.load_jnesty_fit_from_fits` turns it back
into a working ``JNesty_Fit``, from which the same plots and numbers can be rebuilt
without running the sampler again.

Units
-----
The FITS file stores flux densities of order 1e-13 erg/s/cm2/A and they are
rescaled by 1e14 here, so every number is in 1e-14 erg/s/cm2/A.  Integrated line
fluxes are quoted in 1e-17 erg/s/cm2 and velocities in km/s.
"""

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

import galspec
from galspec import jax_models as jxm
from galspec import priors
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary
from galspec.jnesty_fit_io import save_jnesty_fit_to_fits
from galspec.utils import line_wave_dict as LINE_WAVE

# %% settings
# ---- what to fit, and how hard -----------------------------------------------------
DATA_FILE = HERE / "data" / "ir07546sed.fits"
#: Where the figures and the saved fit go.  The environment variable exists so that
#: a test run cannot overwrite the published products in ``example/output/``.
OUTPUT_DIR = Path(os.environ.get("GALSPEC_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDSHIFT = 0.0953              # NASA/IPAC value for IRAS 07546+3928
E_BV = 0.176                   # Galactic extinction along this line of sight
WAVE_MIN, WAVE_MAX = 4200.0, 6800.0    # rest frame: H-gamma out to [S II] 6731
ERROR_FRACTION = 0.005         # floor on the pixel-to-pixel noise
PHOTON_FRACTION = 0.05         # 5% of the flux: photon noise of the source itself

NLIVE = 400                    # live points kept by the sampler
DLOGZ = 0.01                   # stop when < 1% of the evidence may remain unseen
MAX_ITERATIONS = 200_000
DEVICE = "auto"                # "gpu", "cpu" or "auto"
QUICK = False                  # True -> nlive 120, dlogz 0.5 (a rough answer)
N_WINGS = 1                    # wind components after the broad core

#: The three panels to draw besides the full spectrum: (file tag, title, x limits, y
#: limits).  The y limits are set by hand because the automatic ones follow the highest
#: pixel in the window, and that leaves [O III] 5007 cut off at the top of the frame.
ZOOMS = [("Hgamma_Hbeta", r"H$\gamma$ + H$\beta$", (4230, 5100), (0.0, 3.3)),
         ("Halpha_NII_SII", r"H$\alpha$ + [N II] + [S II]", (6420, 6800), (0.0, 5.0)),
         ("HeI_NaD", "He I 5876 + Na D", (5750, 6000), (0.45, 0.63))]
#: The parameters the corner plot shows: thirty of them are unreadable as a square.
CORNER_PARAMS = ("OIII.dv_c", "OIII.sigma_c", "b_halpha.dv_c", "b_halpha.sigma_c",
                 "b_halpha.dv_w0", "b_halpha.sigma_w0", "Continuum.amplitude",
                 "Continuum.alpha", "FeII.amplitude", "b_halpha.amp_c",
                 "b_halpha.amp_w0", "n_halpha.amplitude", "b_hei.amp_c",
                 "n_hei.amplitude", "b_heii.amp_c")
#: The parameters worth a one-dimensional posterior: the numbers the fit exists for.
HIST_PARAMS = ("b_halpha.amp_c", "b_halpha.amp_w0", "b_halpha.dv_c",
               "b_halpha.sigma_c", "b_halpha.dv_w0", "b_halpha.sigma_w0",
               "b_hbeta.amp_c", "b_hbeta.amp_w0", "n_halpha.amplitude",
               "OIII.sigma_c", "OIII.dv_c", "b_hei.amp_c", "n_hei.amplitude",
               "b_heii.amp_c", "SII.amp_c0", "SII.amp_c1", "Continuum.alpha")
#: Display names, so the legend says "BH$\alpha$ (broad)" and not "b_halpha".
MODEL_LABELS = {
    "Continuum": "Continuum", "FeII": "Fe II",
    "n_halpha": r"BH$\alpha$ (narrow)", "b_halpha": r"BH$\alpha$ (broad)",
    "n_hbeta": r"BH$\beta$ (narrow)", "b_hbeta": r"BH$\beta$ (broad)",
    "n_hgamma": r"BH$\gamma$ (narrow)", "b_hgamma": r"BH$\gamma$ (broad)",
    "n_hei": r"He I (narrow)", "b_hei": r"He I (broad)", "b_heii": r"He II (broad)",
    "OIII": r"[O III]", "NII": r"[N II]", "SII": r"[S II]", "OI": r"[O I]",
    "OIII4363": r"[O III] 4363", "HeII": r"He II", "HeI": r"He I",
}
FLUX_LABEL = r"$F_\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]"


# %% load
# ---- 1. Load the spectrum ----------------------------------------------------------
def load_spectrum():
    """Dereddened, rest-frame spectrum between ``WAVE_MIN`` and ``WAVE_MAX``.

    Returns ``(wave, flux, error)``: the error is measured from the data, since the
    file carries none.  Subtract a running median (about 30 A), take the median of
    the absolute remainder in the same window -- which ignores the few pixels that
    belong to emission lines -- divide by 0.6745 to turn it into a Gaussian sigma,
    and smooth over a few hundred A.  Then add ``PHOTON_FRACTION`` of the flux in
    quadrature: that is the photon noise of the source, plus what the model does not
    describe exactly (flux calibration, the shape of the Fe II template, departures
    from a pure power law).
    """
    from astropy.io import fits
    from extinction import ccm89, remove

    with fits.open(DATA_FILE) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=float) * 1e14
    wave = header["CRVAL1"] + header["CD1_1"] * np.arange(flux.size)

    flux = remove(ccm89(wave, E_BV, r_v=3.1, unit="aa"), flux)   # Milky Way extinction
    wave = wave / (1.0 + REDSHIFT)                                # observed -> rest frame

    use = (wave > WAVE_MIN) & (wave < WAVE_MAX)
    wave, flux = wave[use], flux[use]

    window = 751
    baseline = median_filter(flux, size=window, mode="nearest")
    scatter = median_filter(np.abs(flux - baseline), size=window, mode="nearest") / 0.6745
    scatter = median_filter(scatter, size=4 * window + 1, mode="nearest")
    statistical = np.maximum(scatter, ERROR_FRACTION * np.abs(flux))
    error = np.hypot(statistical, PHOTON_FRACTION * np.abs(flux))
    return wave, flux, error


# %% helpers
def free_qualified_names(model):
    """``Submodel.parameter`` spelling of every free parameter."""
    pairs = zip(model.param_names, model.qualified_param_names)
    free = set(model.free_param_names)
    return [q for flat, q in pairs if flat in free]


class _TieFrom:
    """Copy one parameter of another submodel (the plain-Gaussian case)."""

    def __init__(self, submodel: str, parameter: str):
        self.submodel = submodel
        self.parameter = parameter

    def __call__(self, model):
        return getattr(model[self.submodel], self.parameter)


# %% model
# ---- 2. Build the model ------------------------------------------------------------
def build_model(n_wings=N_WINGS, tie_broad=True):
    """Seventeen components, summed; then the ties that make them physical.

    ``n_wings`` is how many wind components follow the broad core, so the broad
    Balmer lines get ``1 + n_wings`` Gaussians each (the published fit uses one).
    ``tie_broad`` decides whether broad H-beta and H-gamma *share* the profile shape
    of broad H-alpha; ``fit_iras07546_jax_tied.py`` releases it and constrains the
    freed parameters with a Gaussian instead, which is the difference between "the
    same" and "similar".
    """
    def wing_defaults(amp_w0):
        """Initial wind components: component 0 is the wind, i>0 are extra wings."""
        out = {}
        for i in range(n_wings):
            out[f"amp_w{i}"] = amp_w0 if i == 0 else 0.05
            out[f"dv_w{i}"] = 0.0
            out[f"sigma_w{i}"] = 6000.0 if i == 0 else 9000.0
        return out

    # -- continuum: a power law anchored at 5500 A, plus the iron pseudo-continuum --
    continuum = jxm.PowerLaw1D(amplitude=0.55, x_0=5500.0, alpha=1.0,
                               fixed={"x_0": True}, name="Continuum")
    # The iron template is broadened by the profile of broad H-alpha itself, tied
    # component by component below.  A Line_MultiGauss already is a weighted sum and
    # the convolution renormalises it, so the core amplitude cancels and only the
    # wind ratio survives as a weight: the iron has one free parameter, its amplitude.
    iron = jxm.IronTemplate_tied(n_components=1 + n_wings, amplitude=0.2, z=0.0,
                                 dv_c=0.0, sigma_c=1500.0, name="FeII",
                                 **wing_defaults(0.10))
    iron.z.fixed = True

    # -- the Balmer lines: a narrow Gaussian and a broad MultiGauss, separately -----
    n_halpha = jxm.Line_Gaussian(amplitude=2.2, dv=0.0, sigma=700.0,
                                 wavec=LINE_WAVE["Halpha"], name="n_halpha")
    n_hbeta = jxm.Line_Gaussian(amplitude=0.8, dv=0.0, sigma=700.0,
                                wavec=LINE_WAVE["Hbeta"], name="n_hbeta")
    n_hgamma = jxm.Line_Gaussian(amplitude=0.4, dv=0.0, sigma=700.0,
                                 wavec=LINE_WAVE["Hgamma"], name="n_hgamma")

    b_halpha = jxm.Line_MultiGauss(n_components=1 + n_wings, amp_c=0.65, dv_c=0.0,
                                   sigma_c=1500.0, wavec=LINE_WAVE["Halpha"],
                                   name="b_halpha", **wing_defaults(0.10))
    b_hbeta = jxm.Line_MultiGauss(n_components=1 + n_wings, amp_c=0.22, dv_c=0.0,
                                  sigma_c=1500.0, wavec=LINE_WAVE["Hbeta"],
                                  name="b_hbeta", **wing_defaults(0.20))
    b_hgamma = jxm.Line_MultiGauss(n_components=1 + n_wings, amp_c=0.10, dv_c=0.0,
                                   sigma_c=1500.0, wavec=LINE_WAVE["Hgamma"],
                                   name="b_hgamma", **wing_defaults(0.20))

    # -- He I 5876 and He II 4686: one broad Gaussian each --------------------------
    # He I sits where Na D absorption used to be masked; with the mask gone it has to
    # be modelled.  Broad He II is strong enough to measure in a narrow-line Seyfert 1
    # and sits where the iron and the Balmer wing overlap, so it needs a broad foot.
    n_hei = jxm.Line_Gaussian(amplitude=0.12, dv=0.0, sigma=700.0,
                              wavec=LINE_WAVE["HeI_5876"], name="n_hei")
    b_hei = jxm.Line_MultiGauss(n_components=1, amp_c=0.06, dv_c=0.0, sigma_c=1500.0,
                                wavec=LINE_WAVE["HeI_5876"], name="b_hei")
    b_heii = jxm.Line_MultiGauss(n_components=1, amp_c=0.05, dv_c=0.0, sigma_c=1500.0,
                                 wavec=LINE_WAVE["HeII_4686"], name="b_heii")
    he2 = jxm.Line_Gaussian(amplitude=0.08, dv=0.0, sigma=700.0,
                            wavec=LINE_WAVE["HeII_4686"], name="HeII")

    # -- the forbidden lines -------------------------------------------------------
    # [N II] and [O I] ratios are atomic physics; [S II] is not -- its ratio is a
    # density diagnostic, so both peaks stay free and the ratio comes out of the fit.
    o3 = jxm.Line_MultiGauss_doublet(n_components=2, amp_c0=1.8, amp_c1=1.8 / 2.98,
                                     dv_c=0.0, sigma_c=700.0,
                                     wavec0=LINE_WAVE["OIII_5007"],
                                     wavec1=LINE_WAVE["OIII_4959"], name="OIII",
                                     amp_w0=0.3, dv_w0=0.0, sigma_w0=1800.0)
    n2 = jxm.Line_MultiGauss_doublet(n_components=1, amp_c0=1.0, amp_c1=1.0 / 2.96,
                                     dv_c=0.0, sigma_c=700.0,
                                     wavec0=LINE_WAVE["NII_6583"],
                                     wavec1=LINE_WAVE["NII_6548"], name="NII")
    s2 = jxm.Line_MultiGauss_doublet(n_components=1, amp_c0=0.10, amp_c1=0.10,
                                     dv_c=0.0, sigma_c=700.0,
                                     wavec0=LINE_WAVE["SII_6716"],
                                     wavec1=LINE_WAVE["SII_6731"], name="SII")
    o1 = jxm.Line_MultiGauss_doublet(n_components=1, amp_c0=0.05, amp_c1=0.05 / 3.0,
                                     dv_c=0.0, sigma_c=700.0,
                                     wavec0=LINE_WAVE["OI_6300"],
                                     wavec1=LINE_WAVE["OI_6364"], name="OI")
    o3_4363 = jxm.Line_Gaussian(amplitude=0.05, dv=0.0, sigma=700.0,
                                wavec=LINE_WAVE["OIII_4363"], name="OIII4363")

    model = (continuum + iron + n_halpha + b_halpha + n_hbeta + b_hbeta
             + n_hgamma + b_hgamma + o3 + n2 + s2 + o1 + he2 + b_heii
             + o3_4363 + n_hei + b_hei)

    # -- ties: one narrow kinematics, measured from [O III] ------------------------
    for line in (n_halpha, n_hbeta, n_hgamma, n_hei, he2, o3_4363):
        line.dv.tied = _TieFrom("OIII", "dv_c")
        line.sigma.tied = _TieFrom("OIII", "sigma_c")
    for line in (n2, s2, o1):
        line.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("OIII")
        line.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("OIII")
    o3.dv_w0.tied = _TieFrom("OIII", "dv_c")    # the wing sits at the core velocity

    # -- ties: broad lines share one velocity structure, not one peak --------------
    for line in (b_hbeta, b_hgamma) if tie_broad else ():
        line.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("b_halpha")
        line.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("b_halpha")
        for i in range(n_wings):
            getattr(line, f"dv_w{i}").tied = getattr(
                jxm.tie, f"tie_MultiGauss_dv_w{i}")("b_halpha")
            getattr(line, f"sigma_w{i}").tied = getattr(
                jxm.tie, f"tie_MultiGauss_sigma_w{i}")("b_halpha")
    for line in (b_hei, b_heii):
        line.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("b_halpha")
        line.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("b_halpha")

    # -- ties: the iron kernel IS the broad H-alpha profile ------------------------
    iron.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c("b_halpha")
    iron.sigma_c.tied = jxm.tie.tie_MultiGauss_sigma_c("b_halpha")
    for i in range(n_wings):
        for stem in ("amp_w", "dv_w", "sigma_w"):
            getattr(iron, f"{stem}{i}").tied = getattr(
                jxm.tie, f"tie_MultiGauss_{stem}{i}")("b_halpha")

    # -- ties: doublet ratios fixed by atomic physics ------------------------------
    o3.amp_c1.tied = jxm.tie.tie_MultiGauss_doublet_ratio("OIII", 2.98)
    n2.amp_c1.tied = jxm.tie.tie_MultiGauss_doublet_ratio("NII", 2.96)
    o1.amp_c1.tied = jxm.tie.tie_MultiGauss_doublet_ratio("OI", 3.00)

    # display names, so the figure legend says which line it is
    for name in model.submodel_names:
        if name in MODEL_LABELS:
            model[name].label = MODEL_LABELS[name]
    return model


# %% priors
# ---- 3. Attach a prior to every free parameter --------------------------------------
def set_priors(model, flux):
    """Wide priors, derived from the spectrum rather than from a previous fit.

    Two shapes do the work.  ``LogUniform`` gives equal probability per factor of
    ten, which is what a peak that may be four decades away from the brightest pixel
    needs -- a linear box would put 99% of the prior at the bright end and never
    explore the faint lines.  ``Uniform`` is right for a quantity whose range is
    already the physical one (a ratio bounded by 1, a velocity shift).

    Everything here scales with the data, so the recipe transfers to another object
    without hand-tuning.  A parameter left without a prior would be uniform over its
    default bounds, which is how a stray parameter ends up sampling the whole box;
    :func:`main` refuses to start in that case.
    """
    n_wings = model["b_halpha"].n_components - 1
    peak = float(np.max(flux))
    continuum = float(np.percentile(flux, 15.0))
    amplitude = (1e-3, 3.0 * peak)      # a line peak: invisible to three times the peak

    # continuum and iron
    model["Continuum"].amplitude.prior = priors.LogUniform(0.2 * continuum, 4.0 * continuum)
    model["Continuum"].alpha.prior = priors.Uniform(-5.0, 5.0)
    model["FeII"].amplitude.prior = priors.LogUniform(1e-3, 3.0 * continuum)

    # every line peak, one per model
    for line in ("halpha", "hbeta", "hgamma", "hei"):
        model[f"n_{line}"].amplitude.prior = priors.LogUniform(*amplitude)
    for line in ("HeII", "OIII4363"):
        model[line].amplitude.prior = priors.LogUniform(*amplitude)
    for line in ("OIII", "NII"):
        model[line].amp_c0.prior = priors.LogUniform(*amplitude)
    for peak_name in ("amp_c0", "amp_c1"):          # [S II]: both peaks, free ratio
        getattr(model["SII"], peak_name).prior = priors.LogUniform(*amplitude)
    model["OI"].amp_c0.prior = priors.LogUniform(*amplitude)
    for line in ("b_hei", "b_heii"):
        model[line].amp_c.prior = priors.LogUniform(*amplitude)

    # narrow kinematics: [O III] sets the shift and the width of every narrow line
    model["OIII"].dv_c.prior = priors.Uniform(-1000.0, 1000.0)
    model["OIII"].sigma_c.prior = priors.Uniform(30.0, 800.0)
    model["OIII"].amp_w0.prior = priors.Uniform(0.0, 1.0)
    model["OIII"].sigma_w0.prior = priors.Uniform(300.0, 4000.0)

    # broad kinematics: the only place where the broad-line region is free to move
    model["b_halpha"].dv_c.prior = priors.Uniform(-2500.0, 2500.0)
    model["b_halpha"].sigma_c.prior = priors.Uniform(300.0, 8000.0)
    for i in range(n_wings):
        getattr(model["b_halpha"], f"dv_w{i}").prior = priors.Uniform(-6000.0, 6000.0)
        getattr(model["b_halpha"], f"sigma_w{i}").prior = priors.Uniform(800.0, 25000.0)

    # each broad line keeps its own peak and its own wind ratio (the Balmer decrement)
    for line in ("halpha", "hbeta", "hgamma"):
        broad = model[f"b_{line}"]
        broad.amp_c.prior = priors.LogUniform(*amplitude)
        for i in range(n_wings):
            getattr(broad, f"amp_w{i}").prior = priors.Uniform(0.0, 1.0)
    return model


def unit_of(name):
    """Units of a parameter, read off its suffix."""
    base = name.split(".", 1)[-1]
    if base.startswith("amp_w"):
        return "ratio to core"
    if base.startswith(("amp_c", "amplitude")):
        return "1e-14 erg/s/cm2/A"
    if base.startswith(("dv", "sigma")):
        return "km/s"
    return ""


# %% fit-and-show
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
    # ---- 2. Build the model, and give every free parameter a prior ------------------
    model = build_model(N_WINGS)
    set_priors(model, flux)
    print(f"\nmodel: {len(model.submodel_names)} components, no LSF convolution, "
          f"{len(model.free_param_names)} free parameters")
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
    ax.set_title("IRAS 07546+3928: each line split into a narrow and a broad model",
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
    corner_fig = corner.corner(fit.samples_equal_weight[:, keep],
                               labels=[labels[i] for i in keep],
                               quantiles=(0.16, 0.5, 0.84), show_titles=True,
                               title_fmt=".3g", title_kwargs={"fontsize": 8})
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