"""A realistic SDSS AGN fit: H-gamma, H-beta and H-alpha regions at once.

The spectrum is the SDSS spectrum of IRAS 07546+3928 that ships with GalSpec
(``example/data/ir07546sed.fits``).  It is dereddened, shifted to rest frame and
cut to 4200-6900 A, so one model covers H-gamma (4340 A), H-beta (4861 A) and
H-alpha (6563 A) together with every strong forbidden line in between.

The model is the kind of fit an AGN paper would actually run:

    power law continuum
  + Fe II template (iron emission, the pseudo-continuum blue of H-beta)
  + H-alpha, H-beta, H-gamma: broad + narrow Gaussian each
  + [O III] 4959/5007 (narrow core + broad wing)
  + [N II] 6548/6584, [S II] 6716/6731, [O I] 6300/6363
  + He II 4686, [O III] 4363 (narrow, kinematics tied)

Twenty-five free parameters.  The physical ties that make that possible are the
standard ones: the narrow lines all share one velocity shift and one width (set
by [O III]), the broad Balmer lines share one velocity shift and one width (set
by H-alpha), and the doublet ratios are fixed by atomic physics.

Both fitters see exactly this model: :func:`build_cpu` returns the astropy
version for ``galspec.Dynesty_Fit`` and :func:`build_jax` the JAX version for
``galspec.JNesty_Fit``.

    python develop/dev_jax/sdss_agn.py            # parity + parameter summary
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import galspec
from galspec import jax_models as jxm
from galspec.utils import line_wave_dict as LINE_WAVE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_FILE = ROOT / "example" / "data" / "ir07546sed.fits"

REDSHIFT = 0.0953
E_BV = 0.176
WAVE_MIN, WAVE_MAX = 4200.0, 6900.0
RESOLVING_POWER = 2000.0     # SDSS spectrograph R at these wavelengths
ERROR_FRACTION = 0.010       # per-pixel noise, measured from the data itself
SYSTEMATIC_FRACTION = 0.030  # flux calibration + Fe II template + higher-order continuum


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_sdss_spectrum():
    """Dereddened, rest-frame SDSS spectrum of IRAS 07546+3928, 4200-6900 A.

    The FITS file contains flux only (no error extension), so the per-pixel
    uncertainty is taken as a constant fraction of the flux; 1% matches the
    pixel-to-pixel scatter measured in the line-free continuum windows.
    """
    from astropy.io import fits
    from extinction import ccm89, remove

    with fits.open(DATA_FILE) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=float) * 1e14
    wave = header["CRVAL1"] + header["CD1_1"] * np.arange(flux.size)
    flux = remove(ccm89(wave, E_BV, r_v=3.1, unit="aa"), flux)
    wave = wave / (1.0 + REDSHIFT)
    use = (wave > WAVE_MIN) & (wave < WAVE_MAX)
    wave, flux = wave[use], flux[use]
    noise = ERROR_FRACTION * np.abs(flux)
    systematic = SYSTEMATIC_FRACTION * np.abs(flux)
    error = np.maximum(np.hypot(noise, systematic), 1e-4)
    # The host galaxy's Na D absorption (5840-5930 A, ~3.5% deep) is not part of
    # the model, so those pixels are down-weighted rather than removed: the
    # wavelength grid has to stay uniform for the instrumental-resolution kernel.
    error[(wave > 5840.0) & (wave < 5930.0)] *= 3.0
    return wave, flux, error


# --------------------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------------------
# Qualified names ("Submodel.parameter") are accepted by both fitters, which keeps
# this readable for a model with eleven components.
def make_bounds(flux):
    """Priors wide enough to bracket the answer without any least-squares fit.

    The bounds come from the data alone, not from a previous fit, so the same
    recipe works on a new spectrum with no user intervention:

    * every emission-line amplitude is allowed between 1e-3 and twice the peak
      flux of the spectrum.  Amplitudes get a log-uniform prior (both fitters do
      that for ``amplitude``/``amp*`` parameters automatically), so "between
      1e-3 and twice the peak" means "anywhere from invisible to the strongest
      feature in the spectrum", which is 3-4 dex of freedom per line;
    * the narrow-line width spans 30-1000 km/s (FWHM ~70-2400 km/s): from a
      barely resolved line up to the widest narrow-line region known;
    * the broad Balmer width spans 500-20000 km/s: from a line as wide as the
      narrow lines up to the broadest line-wing/broad absorption line region.
    """
    peak = float(np.max(flux))
    continuum = float(np.percentile(flux, 15.0))      # line-free level
    amp = (1e-3, 2.0 * peak)
    return {
        # power-law continuum: amplitude at 5500 A, slope
        "Continuum.amplitude": (0.1 * continuum, 4.0 * continuum),
        "Continuum.alpha": (-5.0, 5.0),
        # Fe II pseudo-continuum: cannot be brighter than the real continuum
        "FeII.amplitude": (1e-3, 2.0 * continuum),
        # broad Balmer lines: width and shift free over a wide range
        "Halpha.amp_c": amp,
        "Halpha.amp_w0": amp,
        "Halpha.dv_w0": (-5000.0, 5000.0),
        "Halpha.sigma_w0": (500.0, 20000.0),
        "Hbeta.amp_c": amp,
        "Hbeta.amp_w0": amp,
        "Hgamma.amp_c": amp,
        "Hgamma.amp_w0": amp,
        # [O III] 4959/5007: narrow core plus a broad wing
        "OIII.amp_c0": amp,
        "OIII.dv_c": (-1500.0, 1500.0),
        "OIII.sigma_c": (30.0, 1000.0),
        "OIII.amp_w0": amp,
        "OIII.sigma_w0": (500.0, 5000.0),
        # forbidden lines whose kinematics follow [O III]
        "NII.amp_c0": amp,
        "SII.amp_c0": amp,
        "OI.amp_c0": amp,
        "HeII.amplitude": amp,
        "OIII4363.amplitude": amp,
    }


BOUNDS = make_bounds(load_sdss_spectrum()[1])

#: Parameters held fixed, with the reason (reported on the model page).
FIXED_PARAMS = {
    "line wavelengths": "laboratory values, known to far better than 1 km/s",
    "FeII.z": "the iron lines are at the galaxy redshift",
    "FeII.stddev": "3000 km/s, a typical broad-line-region value; degenerate with the continuum",
}


# --------------------------------------------------------------------------------------
# Ties: narrow-line kinematics follow [O III]
# --------------------------------------------------------------------------------------
class _TieNarrowDv:
    """Velocity shift of a single Gaussian, copied from the [O III] core."""

    def __call__(self, model):
        return model["OIII"].dv_c


class _TieNarrowSigma:
    """Width of a single Gaussian, copied from the [O III] core."""

    def __call__(self, model):
        return model["OIII"].sigma_c


# --------------------------------------------------------------------------------------
# The model, in the two flavours
# --------------------------------------------------------------------------------------
def _build(module, tie):
    """Assemble the model from either the astropy or the JAX classes.

    ``module`` only has to supply ``PowerLaw1D``, ``IronTemplate``,
    ``Line_MultiGauss``, ``Line_MultiGauss_doublet`` and ``Line_Gaussian``;
    ``tie`` supplies the tie helpers.  Building both from one function keeps the
    two models identical by construction.
    """
    continuum = module.PowerLaw1D(amplitude=0.52, x_0=5500.0, alpha=1.0,
                                  fixed={"x_0": True}, name="Continuum")

    iron = module.IronTemplate(amplitude=0.10, stddev=3000.0, z=0.0, name="FeII")
    iron.z.fixed = True
    # The template's broadening is degenerate with the continuum: freed, it runs to
    # whatever upper limit it is given while chi2/dof stays the same, so we fix it at
    # a representative broad-line-region value instead of fitting noise.
    iron.stddev.fixed = True

    halpha = module.Line_MultiGauss(
        n_components=2, amp_c=1.60, dv_c=20.0, sigma_c=250.0,
        wavec=LINE_WAVE["Halpha"], name="Halpha",
        amp_w0=1.30, dv_w0=0.0, sigma_w0=2200.0)
    hbeta = module.Line_MultiGauss(
        n_components=2, amp_c=0.45, dv_c=20.0, sigma_c=250.0,
        wavec=LINE_WAVE["Hbeta"], name="Hbeta",
        amp_w0=0.55, dv_w0=0.0, sigma_w0=2200.0)
    hgamma = module.Line_MultiGauss(
        n_components=2, amp_c=0.16, dv_c=20.0, sigma_c=250.0,
        wavec=LINE_WAVE["Hgamma"], name="Hgamma",
        amp_w0=0.22, dv_w0=0.0, sigma_w0=2200.0)

    o3 = module.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.90, amp_c1=1.90 / 2.98, dv_c=20.0, sigma_c=250.0,
        wavec0=LINE_WAVE["OIII_5007"], wavec1=LINE_WAVE["OIII_4959"],
        name="OIII", amp_w0=0.45, dv_w0=0.0, sigma_w0=700.0)
    n2 = module.Line_MultiGauss_doublet(
        n_components=1, amp_c0=1.10, amp_c1=1.10 / 2.96, dv_c=20.0, sigma_c=250.0,
        wavec0=LINE_WAVE["NII_6583"], wavec1=LINE_WAVE["NII_6548"], name="NII")
    s2 = module.Line_MultiGauss_doublet(
        n_components=1, amp_c0=0.07, amp_c1=0.07, dv_c=20.0, sigma_c=250.0,
        wavec0=LINE_WAVE["SII_6716"], wavec1=LINE_WAVE["SII_6731"], name="SII")
    o1 = module.Line_MultiGauss_doublet(
        n_components=1, amp_c0=0.05, amp_c1=0.05 / 3.0, dv_c=20.0, sigma_c=250.0,
        wavec0=LINE_WAVE["OI_6300"], wavec1=LINE_WAVE["OI_6364"], name="OI")

    he2 = module.Line_Gaussian(amplitude=0.09, dv=20.0, sigma=250.0,
                               wavec=LINE_WAVE["HeII_4686"], name="HeII")
    o3_4363 = module.Line_Gaussian(amplitude=0.03, dv=20.0, sigma=250.0,
                                   wavec=LINE_WAVE["OIII_4363"], name="OIII4363")

    model = (continuum + iron + halpha + hbeta + hgamma + o3 + n2 + s2 + o1
             + he2 + o3_4363)

    # --- narrow lines: one velocity shift and one width, set by [O III] core ---
    for line in (halpha, hbeta, hgamma):
        line.dv_c.tied = tie.tie_MultiGauss_dv_c("OIII")
        line.sigma_c.tied = tie.tie_MultiGauss_sigma_c("OIII")
    for line in (n2, s2, o1):
        line.dv_c.tied = tie.tie_MultiGauss_dv_c("OIII")
        line.sigma_c.tied = tie.tie_MultiGauss_sigma_c("OIII")
    for line in (he2, o3_4363):
        line.dv.tied = _TieNarrowDv()
        line.sigma.tied = _TieNarrowSigma()

    # --- broad Balmer lines: one velocity shift and one width, set by H-alpha --
    o3.dv_w0.tied = _TieNarrowDv()      # the [O III] wing sits at the core velocity

    for line in (hbeta, hgamma):
        line.dv_w0.tied = tie.tie_MultiGauss_dv_w0("Halpha")
        line.sigma_w0.tied = tie.tie_MultiGauss_sigma_w0("Halpha")

    # --- doublet ratios fixed by atomic physics -------------------------------
    o3.amp_c1.tied = tie.tie_MultiGauss_doublet_ratio("OIII", 2.98)
    n2.amp_c1.tied = tie.tie_MultiGauss_doublet_ratio("NII", 2.96)
    s2.amp_c1.tied = tie.tie_MultiGauss_doublet_ratio("SII", 1.00)
    o1.amp_c1.tied = tie.tie_MultiGauss_doublet_ratio("OI", 3.00)
    return model


def build_cpu(resolving_power=RESOLVING_POWER, wavec=5500.0):
    """The astropy model for :class:`galspec.Dynesty_Fit`.

    The spectrum is convolved with the spectrograph's line-spread function
    (a Gaussian of ``FWHM = wavec / resolving_power``).  Without it a 0.05 A
    sampled SDSS spectrum cannot be fitted at all: the model would try to
    reproduce the instrumental profile with the physical line width.
    """
    from astropy.modeling import models

    class _Module:
        PowerLaw1D = staticmethod(models.PowerLaw1D)
        Line_Gaussian = staticmethod(galspec.Line_Gaussian)

    _Module.IronTemplate = staticmethod(galspec.IronTemplate)
    _Module.Line_MultiGauss = staticmethod(galspec.Line_MultiGauss)
    _Module.Line_MultiGauss_doublet = staticmethod(galspec.Line_MultiGauss_doublet)
    model = _build(_Module, galspec)
    if resolving_power is None:
        return model
    return galspec.convolve_lsf(model, wavec=wavec, resolving_power=resolving_power)


def build_jax(resolving_power=RESOLVING_POWER, wavec=5500.0):
    """The JAX model for :class:`galspec.JNesty_Fit` (same model, same LSF)."""
    model = _build(jxm, jxm.tie)
    if resolving_power is None:
        return model
    return jxm.convolve_lsf(model, wavec=wavec, resolving_power=resolving_power)


def free_parameter_names(model):
    return list(model.free_param_names)


# --------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------
def main() -> int:
    wave, flux, error = load_sdss_spectrum()
    cpu_model, jax_model = build_cpu(), build_jax()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from galspec.dynesty_fit import Dynesty_Fit
        fit = Dynesty_Fit(cpu_model, wave, flux, error, bounds_dict=BOUNDS,
                          nlive=50, n_processes=1)

    names_cpu = list(fit.param_names)
    names_jax = list(jax_model.free_param_names)
    print(f"data      : {wave.size} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A rest frame")
    print(f"free params ({len(names_jax)}):")
    for qualified, name in zip(fit.full_param_names, names_jax):
        lower, upper = BOUNDS.get(qualified, ("default", "default"))
        print(f"   {name:14s} {qualified:26s} prior ({lower}, {upper})")
    assert names_cpu == names_jax, (names_cpu, names_jax)

    theta = np.asarray(fit.theta_initial, dtype=float)
    fit.set_model_params(theta)                 # also applies the ties
    predicted_cpu = np.asarray(fit.model(wave))   # fit.model is a working copy
    jax_model.prepare(wave)
    predicted_jax = np.asarray(jax_model.flux(wave, theta))

    scale = np.max(np.abs(predicted_cpu))
    print(f"\npeak model flux: {scale:.3f}")
    print(f"CPU vs JAX    : max rel diff {np.max(np.abs(predicted_cpu - predicted_jax)) / scale:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())