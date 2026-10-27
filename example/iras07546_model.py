"""The model that ``fit_iras07546_jax*.py`` all fit, in one place.

IRAS 07546+3928 at 4200-6800 A: a power-law continuum, the iron pseudo-continuum
tied to the profile of broad H-alpha, ten narrow lines on one *measured* profile,
and the broad Balmer lines with a core and a wind.

Three scripts fit it and each is runnable on its own -- ``fit_iras07546_jax.py``
(the hard ties), ``fit_iras07546_jax_tied.py`` (the broad kinematics released and
centred on H-alpha) and ``fit_iras07546_jax_toggle.py`` (a spike-and-slab prior on
every wind amplitude, so the number of broad components is a posterior). They import
this module rather than each other, and it needs nothing but the two data files that
travel with the repository::

    example/data/ir07546sed.fits                 the spectrum
    example/data/iras07546_narrow_template.txt   the narrow-line profile

so any one of them runs first, on its own::

    python example/fit_iras07546_jax_toggle.py

The narrow-line profile is measured by ``develop/dev_template/stage1_o3_template.py``
and is not re-measured here; ``GALSPEC_TEMPLATE_FILE`` points at a different
extraction if you want to test whether the answer depends on that choice.
"""

# %% imports
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

# GalSpec is used straight from the checkout (it is not pip-installed), so put the
# repository root on the path.  This file lives in <repo>/example/, and the three
# scripts import it from there.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from galspec import jax_models as jxm          # noqa: E402
from galspec import priors                     # noqa: E402
from galspec.utils import line_wave_dict as LINE_WAVE   # noqa: E402

# %% settings
# ---- what to fit --------------------------------------------------------------------
DATA_FILE = HERE / "data" / "ir07546sed.fits"
#: The narrow-line profile: velocity [km/s] and normalised flux, from stage 1
#: (``develop/dev_template/stage1_o3_template.py``).  It travels with the repo, so
#: the example runs without re-measuring it; point ``GALSPEC_TEMPLATE_FILE`` at a
#: fresh extraction to see what changes.
TEMPLATE_FILE = Path(os.environ.get("GALSPEC_TEMPLATE_FILE",
                                    HERE / "data" / "iras07546_narrow_template.txt"))
#: Where the figures and the saved fit go.  The environment variable exists so that
#: a test run cannot overwrite the published products in ``example/output/``.
OUTPUT_DIR = Path(os.environ.get("GALSPEC_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDSHIFT = 0.0953              # NASA/IPAC value for IRAS 07546+3928
E_BV = 0.176                   # Galactic extinction along this line of sight
WAVE_MIN, WAVE_MAX = 4200.0, 6800.0    # rest frame: H-gamma out to [S II] 6731
ERROR_FRACTION = 0.005         # floor on the pixel-to-pixel noise
PHOTON_FRACTION = 0.05         # 5% of the flux: photon noise of the source itself

#: What the two models answer on this spectrum.  ``published`` is the fit this file
#: used to perform -- a Gaussian per narrow line, one width shared between them --
#: and it is the number the template has to be judged against, because it is the
#: same data, the same continuum, the same broad lines and the same priors.  Both
#: used 400 live points and dlogz 0.01.  The template model has been implemented
#: twice -- here and in develop/dev_template/stage2_full_template.py -- and the two
#: agree to 0.01 nats, which is the point of the fixed key: two implementations, one
#: number.  What the fixed key does *not* buy is bitwise identity: three runs of this
#: example gave 130 656.34, 130 656.22 and 130 658.06, from the last bits of a GPU
#: sum feeding the sampler's ordering.  So quote the evidence to about a nat -- the
#: difference this model is judged on is two hundred of them.
REFERENCE = {
    "published": {"free_parameters": 30, "log_evidence": 130889.60,
                  "log_evidence_error": 0.62, "chi2_dof": 0.307,
                  "minutes": 10.2},
    "template": {"free_parameters": 27, "log_evidence": 130658.06,
                 "log_evidence_error": 0.58, "chi2_dof": 0.316,
                 "minutes": 7.4},
}

N_WINGS = 1                    # wind components after the broad core
#: Submodel whose ``dv_c`` every narrow line is tied to -- the shift measured on
#: [O III] in stage 1 and re-fitted here.
NARROW_REFERENCE = "OIII"

#: The three panels to draw besides the full spectrum: (file tag, title, x limits, y
#: limits).  The y limits are set by hand because the automatic ones follow the highest
#: pixel in the window, and that leaves [O III] 5007 cut off at the top of the frame.
ZOOMS = [("Hgamma_Hbeta", r"H$\gamma$ + H$\beta$", (4230, 5100), (0.0, 3.3)),
         ("Halpha_NII_SII", r"H$\alpha$ + [N II] + [S II]", (6420, 6800), (0.0, 5.0)),
         ("HeI_NaD", "He I 5876 + Na D", (5750, 6000), (0.45, 0.63))]
#: The parameters the corner plot shows: thirty of them are unreadable as a square.
CORNER_PARAMS = ("OIII.dv_c", "b_halpha.dv_c", "b_halpha.sigma_c",
                 "b_halpha.dv_w0", "b_halpha.sigma_w0", "Continuum.amplitude",
                 "Continuum.alpha", "FeII.amplitude", "b_halpha.amp_c",
                 "b_halpha.amp_w0", "n_halpha.amplitude", "b_hei.amp_c",
                 "n_hei.amplitude", "b_heii.amp_c")
#: The parameters worth a one-dimensional posterior: the numbers the fit exists for.
HIST_PARAMS = ("b_halpha.amp_c", "b_halpha.amp_w0", "b_halpha.dv_c",
               "b_halpha.sigma_c", "b_halpha.dv_w0", "b_halpha.sigma_w0",
               "b_hbeta.amp_c", "b_hbeta.amp_w0", "n_halpha.amplitude",
               "n_hbeta.amplitude", "OIII.dv_c", "b_hei.amp_c", "n_hei.amplitude",
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
def load_template(path=TEMPLATE_FILE):
    """Read the narrow-line profile: ``(velocity, normalised flux)``.

    Two columns of plain text, because a template should outlive the code that
    produced it: the numbers can be plotted or fitted without importing GalSpec.
    The grid has to be uniform -- ``Line_template`` turns a uniform grid into an
    affine gather, which is what makes ten templated lines cost the same as ten
    Gaussians (``develop/dev_template/template_model_test.py`` measures it) -- so
    that is checked rather than assumed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate it with "
            f"python develop/dev_template/stage1_o3_template.py")
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path}: expected two columns, got shape {data.shape}")
    velc, tflux = data[:, 0], data[:, 1]
    steps = np.diff(velc)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-6):
        raise ValueError(f"{path}: the velocity column must increase uniformly")
    return velc, tflux


def template_fwhm(velc, tflux):
    """FWHM of the template in km/s, by linear interpolation at half maximum."""
    peak = int(np.argmax(tflux))
    right = tflux[peak:]
    left = tflux[peak::-1]
    if right.min() > 0.5 or left.min() > 0.5:
        return float("nan")
    return float(np.interp(0.5, right[::-1], velc[peak:][::-1])
                 - np.interp(0.5, left[::-1], velc[peak::-1][::-1]))


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
def build_model(template=None, n_wings=N_WINGS, tie_broad=True):
    """Seventeen components, summed; then the ties that make them physical.

    ``template`` is ``(velocity, flux)`` -- the narrow-line profile.  When it is
    omitted the committed file is read, so the model can be built and its prior
    count checked without touching the data.

    ``n_wings`` is how many wind components follow the broad core, so the broad
    Balmer lines get ``1 + n_wings`` Gaussians each (the published fit uses one).
    ``tie_broad`` decides whether broad H-beta and H-gamma *share* the profile shape
    of broad H-alpha; ``fit_iras07546_jax_tied.py`` releases it and constrains the
    freed parameters with a Gaussian instead, which is the difference between "the
    same" and "similar".

    What the template changes is only the narrow lines.  Where the published model
    had a ``Line_Gaussian`` per narrow line, and a ``Line_MultiGauss_doublet`` with
    a core and a wing for [O III], there is now one measured profile: an amplitude
    per line, the atomic ratio for the second line of a doublet, and a single shift
    shared by all ten.  No width is fitted anywhere, so ``NARROW_REFERENCE`` has no
    ``sigma_c`` to offer and nothing ties to it.
    """
    if template is None:
        template = load_template()
    velc, tflux = template

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

    # -- the narrow lines: one frozen profile, one amplitude each -------------------
    n_halpha = jxm.Line_template(template_velc=velc, template_flux=tflux,
                                 amplitude=2.2, dv=0.0,
                                 wavec=LINE_WAVE["Halpha"], name="n_halpha")
    n_hbeta = jxm.Line_template(template_velc=velc, template_flux=tflux,
                                amplitude=0.8, dv=0.0,
                                wavec=LINE_WAVE["Hbeta"], name="n_hbeta")
    n_hgamma = jxm.Line_template(template_velc=velc, template_flux=tflux,
                                 amplitude=0.4, dv=0.0,
                                 wavec=LINE_WAVE["Hgamma"], name="n_hgamma")
    n_hei = jxm.Line_template(template_velc=velc, template_flux=tflux,
                              amplitude=0.12, dv=0.0,
                              wavec=LINE_WAVE["HeI_5876"], name="n_hei")
    he2 = jxm.Line_template(template_velc=velc, template_flux=tflux,
                            amplitude=0.08, dv=0.0,
                            wavec=LINE_WAVE["HeII_4686"], name="HeII")
    o3_4363 = jxm.Line_template(template_velc=velc, template_flux=tflux,
                                amplitude=0.05, dv=0.0,
                                wavec=LINE_WAVE["OIII_4363"], name="OIII4363")

    # -- the broad lines: unchanged from the published fit -------------------------
    b_halpha = jxm.Line_MultiGauss(n_components=1 + n_wings, amp_c=0.65, dv_c=0.0,
                                   sigma_c=1500.0, wavec=LINE_WAVE["Halpha"],
                                   name="b_halpha", **wing_defaults(0.10))
    b_hbeta = jxm.Line_MultiGauss(n_components=1 + n_wings, amp_c=0.22, dv_c=0.0,
                                  sigma_c=1500.0, wavec=LINE_WAVE["Hbeta"],
                                  name="b_hbeta", **wing_defaults(0.20))
    b_hgamma = jxm.Line_MultiGauss(n_components=1 + n_wings, amp_c=0.10, dv_c=0.0,
                                   sigma_c=1500.0, wavec=LINE_WAVE["Hgamma"],
                                   name="b_hgamma", **wing_defaults(0.20))
    b_hei = jxm.Line_MultiGauss(n_components=1, amp_c=0.06, dv_c=0.0, sigma_c=1500.0,
                                wavec=LINE_WAVE["HeI_5876"], name="b_hei")
    b_heii = jxm.Line_MultiGauss(n_components=1, amp_c=0.05, dv_c=0.0, sigma_c=1500.0,
                                 wavec=LINE_WAVE["HeII_4686"], name="b_heii")

    # -- the forbidden lines, as template doublets ---------------------------------
    # [N II] and [O I] ratios are atomic physics; [S II] is not -- its ratio is a
    # density diagnostic, so both peaks stay free and the ratio comes out of the fit.
    o3 = jxm.Line_template_doublet(template_velc=velc, template_flux=tflux,
                                   amp_c0=1.8, amp_c1=1.8 / 2.98, dv_c=0.0,
                                   wavec0=LINE_WAVE["OIII_5007"],
                                   wavec1=LINE_WAVE["OIII_4959"], name="OIII")
    n2 = jxm.Line_template_doublet(template_velc=velc, template_flux=tflux,
                                   amp_c0=1.0, amp_c1=1.0 / 2.96, dv_c=0.0,
                                   wavec0=LINE_WAVE["NII_6583"],
                                   wavec1=LINE_WAVE["NII_6548"], name="NII")
    s2 = jxm.Line_template_doublet(template_velc=velc, template_flux=tflux,
                                   amp_c0=0.10, amp_c1=0.10, dv_c=0.0,
                                   wavec0=LINE_WAVE["SII_6716"],
                                   wavec1=LINE_WAVE["SII_6731"], name="SII")
    o1 = jxm.Line_template_doublet(template_velc=velc, template_flux=tflux,
                                   amp_c0=0.05, amp_c1=0.05 / 3.0, dv_c=0.0,
                                   wavec0=LINE_WAVE["OI_6300"],
                                   wavec1=LINE_WAVE["OI_6364"], name="OI")

    model = (continuum + iron + n_halpha + b_halpha + n_hbeta + b_hbeta
             + n_hgamma + b_hgamma + o3 + n2 + s2 + o1 + he2 + b_heii
             + o3_4363 + n_hei + b_hei)

    # -- ties: every narrow line sits at the velocity of [O III] -------------------
    # The widths are not tied because there is nothing left to tie: the profile is
    # the template, and stage 1 already fixed its width on [O III].
    for line in (n_halpha, n_hbeta, n_hgamma, n_hei, he2, o3_4363):
        line.dv.tied = jxm.tie.tie_MultiGauss_dv_c(NARROW_REFERENCE)
    for line in (n2, s2, o1):
        line.dv_c.tied = jxm.tie.tie_MultiGauss_dv_c(NARROW_REFERENCE)

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

    The template shortens this list considerably.  There is no narrow ``sigma`` to
    give a range to -- the width is in the file -- and no [O III] wind ratio or wing
    width, because the profile that used to be a Gaussian plus a wing is now one
    shape.  What is left for the narrow lines is one shift, which is what ties them
    together: nine amplitudes and one velocity.
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

    # narrow kinematics: [O III] sets the shift of every narrow line.  The width is
    # not a parameter here -- it is in the template, measured in stage 1 -- and the
    # box below spans the whole range a narrow-line region shift is known to take.
    model[NARROW_REFERENCE].dv_c.prior = priors.Uniform(-1000.0, 1000.0)

    # broad kinematics: the only place where the broad-line region is free to move.
    # Core and wind share one width box -- a wing is not known in advance to be broader
    # than the core, and letting it run to 25 000 km/s only puts prior where the data
    # are not.
    model["b_halpha"].dv_c.prior = priors.Uniform(-2500.0, 2500.0)
    model["b_halpha"].sigma_c.prior = priors.Uniform(300.0, 8000.0)
    for i in range(n_wings):
        getattr(model["b_halpha"], f"dv_w{i}").prior = priors.Uniform(-6000.0, 6000.0)
        getattr(model["b_halpha"], f"sigma_w{i}").prior = priors.Uniform(300.0, 8000.0)

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
