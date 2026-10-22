#!/usr/bin/env python
"""Fit the SDSS spectrum of IRAS 07546+3928 with JNesty on the GPU.

What this script does
---------------------
It takes the spectrum of the quasar IRAS 07546+3928 that ships with GalSpec
(``example/data/ir07546sed.fits``), dereddens it, moves it to the rest frame and
fits the **whole optical spectrum** -- 3680 to 7500 A -- in one go, so that the
continuum and the three Balmer regions are constrained by the same posterior:

    power-law continuum  (flux density at 5500 A + slope)
  + Fe II template       (iron emission: the pseudo-continuum blue of H-beta)
  + H-alpha, H-beta, H-gamma, H-delta:
        one narrow Gaussian + TWO broad Gaussians each
  + [O III] 4959/5007 (narrow core + broad wing), [N II] 6548/6584,
    [S II] 6716/6731, [O I] 6300/6364
  + the higher Balmer lines H-epsilon 3970, H8 3889, H9 3835, H11 3770
  + He II 4686, [O III] 4363, [Ne III] 3869, [O II] 3727,
    [Ar III] 7136, [O II] 7319/7330

The narrow lines are tied together: they all share one velocity shift and one
width, measured from the [O III] core.  The broad Gaussians are tied across the
Balmer lines the same way (broad component 1 of H-alpha sets the width of broad
component 1 in H-beta, H-gamma and H-delta), and the doublet ratios follow
atomic physics -- except for [S II] 6716/6731, which is left free because it is
an electron-density diagnostic.

Two broad Gaussians per line let the model describe a classical broad-line
region plus a broader, more extreme-velocity component: the profile shape that a
single Gaussian always misses in a real AGN spectrum.

No instrumental smoothing
-------------------------
This spectrum is sampled finely enough (0.046 A per pixel, about 30 pixels per
resolution element) that the instrumental line-spread function is small next to
the widths being measured: the narrow lines are ~700 km/s FWHM while the
spectrograph contributes ~150 km/s.  The model is therefore *not* convolved,
exactly as in ``example/wrk_sagan_iras07546.ipynb``.  The price is that the
fitted "narrow" width is the observed width -- gas plus instrument -- so it comes
out a little wider than with the LSF folded in.

How the fit is done
-------------------
No least-squares step anywhere: nothing is optimised before nested sampling
starts, so the priors cannot be biased by a preliminary answer.  JNesty samples
the posterior directly, in batches on the GPU, using the JAX version of the same
model (:mod:`galspec.jax_models`).  The priors are derived from the data alone
and are deliberately wide.

The result is drawn with :func:`galspec.plot.plot_fit`, the package's own
plotting function, so every model component, the total model and the residuals
appear in one figure.

Run it
------
    python example/fit_iras07546_jax.py                 # ~30-45 min on one GPU
    python example/fit_iras07546_jax.py --quick         # ~4 min, rough answer

Results are written to ``example/output/``: figures, an ``.npz`` with the
posterior samples, and a FITS file that holds the whole fit -- samples, evidence,
priors, the model itself and the data it was made against.  The FITS file is the
one to keep: ``galspec.jnesty_fit_io.load_jnesty_fit_from_fits`` turns it back
into a working ``JNesty_Fit``, from which the same plots and numbers can be
rebuilt without running the sampler again.

Units
-----
The FITS file stores flux densities of order 1e-13 erg/s/cm2/A.  The script
rescales them by 1e14, so every number below is in units of
1e-14 erg/s/cm2/A; integrated line fluxes are converted to the customary
1e-17 erg/s/cm2.  Velocities are in km/s.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

# GalSpec is used straight from the checkout (it is not pip-installed), so put the
# repository root on the path: this file lives in <repo>/example/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import galspec
from galspec import jax_models as jxm
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary
from galspec.jnesty_fit_io import save_jnesty_fit_to_fits
from galspec.utils import line_wave_dict as LINE_WAVE

# --------------------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "ir07546sed.fits"
OUTPUT_DIR = HERE / "output"

REDSHIFT = 0.0953          # NASA/IPAC value for IRAS 07546+3928
E_BV = 0.176               # Galactic extinction along this line of sight (Schlafly & Finkbeiner)
WAVE_MIN, WAVE_MAX = 3680.0, 7500.0   # rest frame: the whole optical spectrum

ERROR_FRACTION = 0.005     # floor on the pixel-to-pixel noise
SYSTEMATIC_FRACTION = 0.030  # flux calibration, Fe II template, hot-star mismatch
#: Regions the model does not describe.  Their pixels are down-weighted rather
#: than cut, because the wavelength grid has to stay evenly spaced.
MASK_WINDOWS = [(3905.0, 3995.0),    # Ca II H+K absorption from the host galaxy
                (5840.0, 5930.0),    # Na D absorption (He I 5876 sits inside it)
                (6860.0, 7100.0)]    # residual O2 A-band absorption at 7600 A observed

NLIVE = 400                # live points kept by the sampler
DLOGZ = 0.01               # stop when less than 0.01 of the evidence may remain unseen

#: Lines that are not in ``line_wave_dict``; vacuum rest wavelengths in A.
EXTRA_LINES = {
    "NeIII_3869": 3868.76,
    "ArIII_7136": 7135.79,
    "OII_7325": 7325.0,   # blend of the 7318.9/7330.2 pair, unresolved here
    "Hepsilon_3970": 3970.07,   # the higher Balmer lines sit in the blue end
    "H8_3889": 3889.05,
    "H9_3835": 3835.39,     # n = 9 -> 2
    "H11_3770": 3770.63,   # n = 11 -> 2
}

#: The Balmer lines and the other line peaks (used by the priors and the tables).
BALMER_NAMES = ("Halpha", "Hbeta", "Hgamma", "Hdelta")
SINGLE_LINES = ("HeII", "OIII4363", "NeIII", "OII3727", "ArIII", "OII7325",
                "Hepsilon", "H8", "H9", "H11")


# --------------------------------------------------------------------------------------
# 2. The data
# --------------------------------------------------------------------------------------
def load_spectrum():
    """Dereddened, rest-frame spectrum between ``WAVE_MIN`` and ``WAVE_MAX``.

    The FITS file has flux only -- no error extension -- so the uncertainty is
    built from the data itself (see :func:`estimate_error`).
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
    return wave, flux, mask_regions(wave, estimate_error(flux))


def estimate_error(flux, window=751):
    """Per-pixel uncertainty measured from the spectrum itself.

    The pixel-to-pixel scatter is read off the data: subtract a running median
    (``window`` pixels, about 30 A), take the median of the absolute remainder in
    the same window -- which ignores the few pixels belonging to emission lines --
    divide by 0.6745 to turn it into a Gaussian sigma, and smooth the result over
    a few hundred A.  That gives a wavelength-dependent noise level which grows
    from ~0.3% of the flux at 4700 A to ~2-4% at the red end.

    A 3% systematic term is then added in quadrature.  It stands for everything
    the model does not describe exactly -- flux calibration, the shape of the
    Fe II template, departures from a pure power law -- and without it the fit
    would chase pixel noise in the line-free continuum.
    """
    baseline = median_filter(flux, size=window, mode="nearest")
    scatter = median_filter(np.abs(flux - baseline), size=window, mode="nearest") / 0.6745
    scatter = median_filter(scatter, size=4 * window + 1, mode="nearest")
    return np.hypot(np.maximum(scatter, ERROR_FRACTION * np.abs(flux)),
                    SYSTEMATIC_FRACTION * np.abs(flux))


def mask_regions(wave, error):
    """Down-weight the windows the model does not describe."""
    error = error.copy()
    for lo, hi in MASK_WINDOWS:
        error[(wave > lo) & (wave < hi)] *= 3.0
    return error


# --------------------------------------------------------------------------------------
# 3. The model
# --------------------------------------------------------------------------------------
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


def _tie_fn(backend, name):
    """``tie`` helpers of either backend (``jxm.tie.*`` or ``galspec.*``)."""
    return getattr(getattr(backend, "tie", backend), name)


def _power_law(backend):
    """The power-law class of either backend.

    The JAX models ship their own; on the astropy side it comes straight from
    :mod:`astropy.modeling.models`, which is what the other GalSpec examples use.
    """
    if backend is jxm:
        return jxm.PowerLaw1D
    from astropy.modeling import models as astro_models
    return astro_models.PowerLaw1D


def build_model(backend=jxm):
    """The full model, with three Gaussians per Balmer line.

    ``backend`` is either :mod:`galspec.jax_models` -- the models the fit uses --
    or :mod:`galspec` itself, the astropy equivalents.  Both expose the same
    constructors, tie helpers and parameter names, so one function builds the
    model twice: once to fit, once to hand to :func:`galspec.plot.plot_fit`.

    Each Balmer line is a ``Line_MultiGauss`` with ``n_components=3``: a narrow
    core (``amp_c``, ``dv_c``, ``sigma_c``), a broad component (``amp_w0``,
    ``dv_w0``, ``sigma_w0``) and a second, broader one (``amp_w1``, ``dv_w1``,
    ``sigma_w1``).  The wind amplitudes are *ratios* to the core amplitude, which
    is how the GalSpec models are written.

    No convolution is applied: see the module docstring.
    """
    # -- continuum: a power law anchored at 5500 A, plus the iron pseudo-continuum --
    continuum = _power_law(backend)(amplitude=0.55, x_0=5500.0, alpha=1.0,
                                    fixed={"x_0": True}, name="Continuum")
    iron = backend.IronTemplate(amplitude=0.2, stddev=3000.0, z=0.0, name="FeII")
    iron.z.fixed = True
    # The template width is degenerate with the continuum slope: left free it runs
    # to whatever limit it is given while chi2/dof stays put.  Fix it at a typical
    # broad-line value instead of fitting noise.
    iron.stddev.fixed = True

    # -- Balmer lines: narrow core + two broad Gaussians each -------------------
    def balmer(name, wavec, amp_c, amp_w0):
        return backend.Line_MultiGauss(
            n_components=3, amp_c=amp_c, dv_c=0.0, sigma_c=700.0,
            wavec=wavec, name=name, amp_w0=amp_w0, dv_w0=0.0, sigma_w0=2400.0,
            amp_w1=0.05, dv_w1=0.0, sigma_w1=6000.0)

    halpha = balmer("Halpha", LINE_WAVE["Halpha"], 2.2, 0.28)
    hbeta = balmer("Hbeta", LINE_WAVE["Hbeta"], 0.8, 0.30)
    hgamma = balmer("Hgamma", LINE_WAVE["Hgamma"], 0.4, 0.30)
    hdelta = balmer("Hdelta", LINE_WAVE["Hdelta"], 0.2, 0.30)

    # -- forbidden lines -------------------------------------------------------
    o3 = backend.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.8, amp_c1=1.8 / 2.98, dv_c=0.0, sigma_c=700.0,
        wavec0=LINE_WAVE["OIII_5007"], wavec1=LINE_WAVE["OIII_4959"],
        name="OIII", amp_w0=0.3, dv_w0=0.0, sigma_w0=1800.0)
    n2 = backend.Line_MultiGauss_doublet(
        n_components=1, amp_c0=1.0, amp_c1=1.0 / 2.96, dv_c=0.0, sigma_c=700.0,
        wavec0=LINE_WAVE["NII_6583"], wavec1=LINE_WAVE["NII_6548"], name="NII")
    # [S II]: the two peaks are left independent -- their ratio is a density
    # diagnostic, so it has to come out of the fit rather than go in as a guess.
    s2 = backend.Line_MultiGauss_doublet(
        n_components=1, amp_c0=0.10, amp_c1=0.10, dv_c=0.0, sigma_c=700.0,
        wavec0=LINE_WAVE["SII_6716"], wavec1=LINE_WAVE["SII_6731"], name="SII")
    o1 = backend.Line_MultiGauss_doublet(
        n_components=1, amp_c0=0.05, amp_c1=0.05 / 3.0, dv_c=0.0, sigma_c=700.0,
        wavec0=LINE_WAVE["OI_6300"], wavec1=LINE_WAVE["OI_6364"], name="OI")

    # -- single narrow Gaussians ----------------------------------------------
    def narrow(name, wavec, amplitude):
        return backend.Line_Gaussian(amplitude=amplitude, dv=0.0, sigma=700.0,
                                     wavec=wavec, name=name)

    he2 = narrow("HeII", LINE_WAVE["HeII_4686"], 0.08)
    o3_4363 = narrow("OIII4363", LINE_WAVE["OIII_4363"], 0.05)
    ne3 = narrow("NeIII", EXTRA_LINES["NeIII_3869"], 0.12)
    o2_3727 = narrow("OII3727", 0.5 * (LINE_WAVE["OII_3726"] + LINE_WAVE["OII_3728"]), 0.5)
    ar3 = narrow("ArIII", EXTRA_LINES["ArIII_7136"], 0.01)
    o2_7325 = narrow("OII7325", EXTRA_LINES["OII_7325"], 0.01)
    # The higher Balmer lines are strong enough to matter at the blue end and are
    # modelled with a narrow core only: their broad wings are weak, and giving
    # four more lines two broad Gaussians each would not be constrained by data
    # this far down the Balmer series.
    heps = narrow("Hepsilon", EXTRA_LINES["Hepsilon_3970"], 0.08)
    h8 = narrow("H8", EXTRA_LINES["H8_3889"], 0.05)
    h9 = narrow("H9", EXTRA_LINES["H9_3835"], 0.04)
    h11 = narrow("H11", EXTRA_LINES["H11_3770"], 0.04)

    model = (continuum + iron + halpha + hbeta + hgamma + hdelta
             + o3 + n2 + s2 + o1 + he2 + o3_4363 + ne3 + o2_3727 + ar3 + o2_7325
             + heps + h8 + h9 + h11)

    # -- ties: one narrow kinematics for every narrow line, [O III] as reference --
    for line in (halpha, hbeta, hgamma, hdelta, n2, s2, o1):
        line.dv_c.tied = _tie_fn(backend, "tie_MultiGauss_dv_c")("OIII")
        line.sigma_c.tied = _tie_fn(backend, "tie_MultiGauss_sigma_c")("OIII")
    for line in (he2, o3_4363, ne3, o2_3727, ar3, o2_7325, heps, h8, h9, h11):
        line.dv.tied = _TieFrom("OIII", "dv_c")
        line.sigma.tied = _TieFrom("OIII", "sigma_c")

    # -- ties: broad component 0 and broad component 1, each shared by the Balmers --
    # They are separate structures, so they get separate kinematics: a classical
    # broad-line region and a broader one can rotate or be shifted differently.
    for line in (hbeta, hgamma, hdelta):
        line.dv_w0.tied = _tie_fn(backend, "tie_MultiGauss_dv_w0")("Halpha")
        line.sigma_w0.tied = _tie_fn(backend, "tie_MultiGauss_sigma_w0")("Halpha")
        line.dv_w1.tied = _tie_fn(backend, "tie_MultiGauss_dv_w1")("Halpha")
        line.sigma_w1.tied = _tie_fn(backend, "tie_MultiGauss_sigma_w1")("Halpha")
    o3.dv_w0.tied = _TieFrom("OIII", "dv_c")     # the wing sits at the core velocity

    # -- doublet ratios fixed by atomic physics --------------------------------
    # (not [S II]: its ratio is density-sensitive and stays a free parameter)
    o3.amp_c1.tied = _tie_fn(backend, "tie_MultiGauss_doublet_ratio")("OIII", 2.98)
    n2.amp_c1.tied = _tie_fn(backend, "tie_MultiGauss_doublet_ratio")("NII", 2.96)
    o1.amp_c1.tied = _tie_fn(backend, "tie_MultiGauss_doublet_ratio")("OI", 3.00)

    return model


# --------------------------------------------------------------------------------------
# 4. Priors: wide, and taken from the data rather than from a previous fit
# --------------------------------------------------------------------------------------
def log_uniform(lower: float, upper: float):
    """Prior that is uniform per decade between ``lower`` and ``upper``.

    Bounds passed in ``bounds_dict`` are *linear*-uniform by design (that is what
    "amplitude between 1 and 2" means literally), but a quantity that may span
    four decades wants equal probability per factor of ten: otherwise 99% of the
    prior sits at the bright end and faint lines are never explored.  A callable
    handed to ``custom_priors`` says exactly what is meant.
    """
    lo, hi = np.log10(lower), np.log10(upper)

    def transform(u):
        return 10.0 ** (lo + (hi - lo) * u)

    return transform


def make_priors(flux):
    """Return ``(bounds, custom_priors, notes)`` for the whole model.

    Everything is derived from the spectrum itself, so the recipe transfers to
    another object without hand-tuning:

    * a line peak may be anything from 1e-3 (invisible) to 3x the brightest pixel
      of the spectrum, log-uniform -- four decades of freedom per line;
    * a broad component may be up to 8x its own narrow core, log-uniform;
    * narrow lines: 30-800 km/s of velocity dispersion (FWHM 70-1900 km/s).  No
      LSF is folded in, so this is the *observed* width;
    * broad component 0: 300-8000 km/s (FWHM up to 19000 km/s), shifted by up to
      +-2500 km/s;
    * broad component 1: 800-25000 km/s (FWHM up to 59000 km/s), shifted by up to
      +-6000 km/s -- wide enough for a line wing, and deliberately overlapping
      component 0 so the data decide where one ends;
    * continuum: 0.2-4x the level of the line-free windows, slope -5 to 5.
    """
    peak = float(np.max(flux))
    continuum = float(np.percentile(flux, 15.0))
    amp = (1e-3, 3.0 * peak)      # line peaks
    ratio = (1e-3, 8.0)           # broad / narrow amplitude

    bounds = {
        "Continuum.amplitude": (0.2 * continuum, 4.0 * continuum),
        "Continuum.alpha": (-5.0, 5.0),
        "FeII.amplitude": (1e-3, 3.0 * continuum),
        # narrow kinematics: [O III] sets them for every narrow line
        "OIII.dv_c": (-1000.0, 1000.0),
        "OIII.sigma_c": (30.0, 800.0),
        "OIII.amp_c0": amp,
        # [O III] broad wing
        "OIII.amp_w0": ratio,
        "OIII.sigma_w0": (300.0, 4000.0),
        # other line peaks
        "NII.amp_c0": amp,
        "SII.amp_c0": amp, "SII.amp_c1": amp,
        "OI.amp_c0": amp,
        # broad component 0, shared by the Balmer lines
        "Halpha.dv_w0": (-2500.0, 2500.0),
        "Halpha.sigma_w0": (300.0, 8000.0),
        # broad component 1, shared by the Balmer lines
        "Halpha.dv_w1": (-6000.0, 6000.0),
        "Halpha.sigma_w1": (800.0, 25000.0),
    }
    for name in BALMER_NAMES:
        bounds[f"{name}.amp_c"] = amp
        bounds[f"{name}.amp_w0"] = ratio
        bounds[f"{name}.amp_w1"] = ratio
    for name in SINGLE_LINES:
        bounds[f"{name}.amplitude"] = amp

    # everything that is a peak height or a broad/narrow ratio gets a log-uniform prior
    log_params = [name for name in bounds
                  if name.endswith(("amp_c", "amp_c0", "amp_c1", "amp_w0", "amp_w1",
                                    "amplitude"))]
    custom = {name: log_uniform(*bounds[name]) for name in log_params}

    notes = [
        ("continuum", "0.2-4x the level of the line-free windows, slope -5 to 5"),
        ("line peaks", "1e-3 to 3x the brightest pixel, log-uniform"),
        ("broad/narrow ratio", "1e-3 to 8, log-uniform"),
        ("narrow lines", "30-800 km/s, one width for all, set by [O III]"),
        ("broad component 0", "300-8000 km/s, one width and shift for all Balmers"),
        ("broad component 1", "800-25000 km/s, its own width and shift"),
        ("[S II] 6716/6731", "both peaks free: the ratio is a density diagnostic"),
    ]
    return bounds, custom, notes


# --------------------------------------------------------------------------------------
# 5. Run the fit
# --------------------------------------------------------------------------------------
def run_fit(wave, flux, error, bounds, custom_priors, nlive=NLIVE, dlogz=DLOGZ,
            max_iterations=200_000, device="auto", progress=None):
    """Sample the posterior with JNesty.  No optimiser is involved anywhere."""
    if progress is None:
        # Only draw the bar when someone is watching: in a batch job it would be
        # thousands of carriage-return updates in the log.
        progress = sys.stdout.isatty()
    model = build_model(jxm)
    fit = JNesty_Fit(model, wave, flux, error,
                     bounds_dict=bounds, custom_priors=custom_priors,
                     nlive=nlive, bound="multi", device=device)
    print(f"  free parameters : {fit.ndim}")
    print(f"  device          : {fit.device}  [{jax_devices_summary()}]")
    print(f"  live points     : {fit.nlive}")
    print(f"  stopping        : dlogz = {dlogz}\n")
    started = time.perf_counter()
    fit.fit(progress=progress, dlogz=dlogz, max_iterations=max_iterations)
    wall = time.perf_counter() - started
    print(f"\n  log Z           : {fit.log_evidence:.2f} +- {fit.log_evidence_err:.2f}")
    print(f"  iterations      : {fit.n_iterations}")
    print(f"  wall clock      : {wall/60:.1f} min")
    return fit


# --------------------------------------------------------------------------------------
# 6. Reading the answer
# --------------------------------------------------------------------------------------
#: Units per parameter suffix, for the summary table.
PARAM_UNITS = {
    ".amp_c": "1e-14 erg/s/cm2/A", ".amp_c0": "1e-14 erg/s/cm2/A",
    ".amp_c1": "1e-14 erg/s/cm2/A", ".amp_w0": "(ratio to core)",
    ".amp_w1": "(ratio to core)", ".amplitude": "1e-14 erg/s/cm2/A",
    ".alpha": "", ".dv_c": "km/s", ".sigma_c": "km/s", ".dv_w0": "km/s",
    ".sigma_w0": "km/s", ".dv_w1": "km/s", ".sigma_w1": "km/s",
}

PARAM_NOTES = {
    "Continuum.amplitude": "continuum flux density at 5500 A",
    "Continuum.alpha": "continuum slope (F_nu ~ nu^-alpha)",
    "FeII.amplitude": "iron pseudo-continuum strength",
    "OIII.amp_c0": "[O III] 5007 narrow peak", "OIII.amp_w0": "[O III] wing / core",
    "OIII.dv_c": "velocity shift of all narrow lines",
    "OIII.sigma_c": "velocity width of all narrow lines",
    "OIII.sigma_w0": "width of the [O III] wing",
    "NII.amp_c0": "[N II] 6583 peak",
    "SII.amp_c0": "[S II] 6716 peak", "SII.amp_c1": "[S II] 6731 peak",
    "OI.amp_c0": "[O I] 6300 peak",
    "HeII.amplitude": "He II 4686 peak", "OIII4363.amplitude": "[O III] 4363 peak",
    "NeIII.amplitude": "[Ne III] 3869 peak", "OII3727.amplitude": "[O II] 3727 peak",
    "ArIII.amplitude": "[Ar III] 7136 peak", "OII7325.amplitude": "[O II] 7320 peak",
    "Hepsilon.amplitude": "H-epsilon 3970 peak", "H8.amplitude": "H8 3889 peak",
    "H9.amplitude": "H9 3835 peak", "H11.amplitude": "H11 3770 peak",
    "Halpha.dv_w0": "shift of broad component 0 (all Balmers)",
    "Halpha.sigma_w0": "width of broad component 0 (all Balmers)",
    "Halpha.dv_w1": "shift of broad component 1 (all Balmers)",
    "Halpha.sigma_w1": "width of broad component 1 (all Balmers)",
}
for _name in BALMER_NAMES:
    PARAM_NOTES.setdefault(f"{_name}.amp_c", f"narrow {_name} peak")
    PARAM_NOTES.setdefault(f"{_name}.amp_w0", f"broad {_name} / narrow")
    PARAM_NOTES.setdefault(f"{_name}.amp_w1", f"second broad {_name} / narrow")


def parameter_table(fit):
    """Median and 16th/84th percentiles for every free parameter."""
    lo, med, hi = fit.get_quantiles((0.16, 0.5, 0.84))
    qualified = fit.full_param_names
    rows = []
    for index, name in enumerate(fit.param_names):
        q = qualified[index]
        unit = ""
        for suffix, label in PARAM_UNITS.items():
            if q.endswith(suffix):
                unit = label
                break
        rows.append((q, name, med[index], med[index] - lo[index], hi[index] - med[index],
                     unit, PARAM_NOTES.get(q, "")))
    return rows


def print_parameter_table(fit):
    print(f"\n  {'parameter':22s} {'value':>10s} {'-':>8s} {'+':>8s}  unit")
    print("  " + "-" * 78)
    for q, _, med, err_lo, err_hi, unit, _ in parameter_table(fit):
        print(f"  {q:22s} {med:10.3f} {err_lo:8.3f} {err_hi:8.3f}  {unit}")


def derived_quantities(fit):
    """Line widths and integrated fluxes, with credible intervals.

    These are the numbers a paper quotes, and they are computed per posterior
    sample rather than from the medians, so their error bars carry the
    correlations in the fit.
    """
    samples = fit.samples_equal_weight
    index = {q: i for i, q in enumerate(fit.full_param_names)}
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))       # sigma -> FWHM, 2.3548

    def get(sample, key):
        return sample[..., index[key]]

    out = {}
    for line in BALMER_NAMES:
        wavec = LINE_WAVE[line]
        block = {}
        for label, comp in (("narrow", "c"), ("broad 0", "w0"), ("broad 1", "w1")):
            if comp == "c":
                amplitude = get(samples, f"{line}.amp_c")
                sigma = get(samples, "OIII.sigma_c")
            else:
                amplitude = get(samples, f"{line}.amp_c") * get(samples, f"{line}.amp_{comp}")
                sigma = get(samples, f"Halpha.sigma_{comp}")
            # F = A sigma sqrt(2 pi) (lambda / c); A is in 1e-14 units and the
            # answer is quoted in 1e-17 erg/s/cm2, hence the 1e3.
            flux = amplitude * sigma * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3
            block[label] = {
                "fwhm": (np.median(fwhm * sigma), *np.percentile(fwhm * sigma, [16, 84])),
                "flux": (np.median(flux), *np.percentile(flux, [16, 84])),
            }
        broad = block["broad 0"]["flux"][0] + block["broad 1"]["flux"][0]
        block["broad/narrow"] = (broad / block["narrow"]["flux"][0], np.nan, np.nan)
        out[line] = block

    # The [S II] ratio: I(6716)/I(6731) falls from ~1.4 in the low-density limit
    # to ~0.4 at high density -- one of the few density diagnostics in the optical
    # window.  amp_c0 is 6716 and amp_c1 is 6731.
    ratio = get(samples, "SII.amp_c0") / get(samples, "SII.amp_c1")
    out["[S II] 6716/6731"] = {"ratio": (np.median(ratio), *np.percentile(ratio, [16, 84]))}
    return out


def print_derived(fit):
    print("\n  line     component      FWHM [km/s]              flux [1e-17 erg/s/cm2]")
    print("  " + "-" * 76)
    for line, block in derived_quantities(fit).items():
        if "ratio" in block:
            r = block["ratio"]
            print(f"  {line}  = {r[0]:.3f} (+{r[2]-r[0]:.3f}/-{r[0]-r[1]:.3f})"
                  "   [density diagnostic]")
            continue
        for label in ("narrow", "broad 0", "broad 1"):
            fw, fl = block[label]["fwhm"], block[label]["flux"]
            print(f"  {line:8s} {label:10s} {fw[0]:8.0f} (+{fw[2]-fw[0]:.0f}/-{fw[0]-fw[1]:.0f})"
                  f"        {fl[0]:9.1f} (+{fl[2]-fl[0]:.1f}/-{fl[0]-fl[1]:.1f})")
        print(f"  {'':8s} {'broad/narrow':10s} {block['broad/narrow'][0]:8.2f}")


def report_quality(fit, wave, flux, error):
    """chi2 of the best-fitting sample, and the residuals in units of the error."""
    model, _, theta = fit.get_best_fit()
    predicted = np.asarray(model.flux(wave, theta))
    residual = (flux - predicted) / error
    chi2 = float(np.sum(residual ** 2))
    dof = wave.size - len(theta)
    print(f"\n  pixels          : {wave.size:,}")
    print(f"  reduced chi2    : {chi2/dof:.3f}  (chi2 {chi2:.0f} / dof {dof})")
    print(f"  residual rms    : {np.std(residual):.2f} sigma")
    print(f"  worst pixel     : {np.max(np.abs(residual)):.1f} sigma")
    return predicted, residual


# --------------------------------------------------------------------------------------
# 7. Plots
# --------------------------------------------------------------------------------------
def cpu_model_with_values(fit):
    """The astropy twin of the fitted model, at the best-fitting sample.

    :func:`galspec.plot.plot_fit` walks an astropy compound model and draws every
    component, so the fitted JAX model is mirrored into its astropy equivalent
    here.

    The values copied across are the *tie-resolved* ones (``param_dict()``): an
    astropy compound model does not re-derive a tied parameter when it is
    evaluated -- ties are a fitting-time constraint there -- so the twin has to be
    given the widths and shifts that the ties produced, otherwise every
    ``dv_c``/``sigma_c`` would still hold its construction value.

    :func:`check_parity` then verifies the twin against the JAX model, which also
    re-proves CPU/JAX parity on this very fit.
    """
    model_jax, _, _ = fit.get_best_fit()
    resolved = model_jax.param_dict()
    model_cpu = build_model(galspec)
    for flat, qualified in zip(model_jax.param_names, model_jax.qualified_param_names):
        value = resolved.get(flat)
        if value is None:
            continue
        submodel, parameter = qualified.split(".", 1)
        getattr(model_cpu[submodel], parameter).value = float(value)
    return model_cpu


def check_parity(fit, wave, predicted):
    """Assert that the astropy twin reproduces the JAX model."""
    model_cpu = cpu_model_with_values(fit)
    cpu = np.asarray(model_cpu(wave))
    difference = float(np.max(np.abs(cpu - predicted)) / np.max(np.abs(predicted)))
    print(f"  CPU/JAX parity  : max difference {difference:.2e} of the peak flux")
    if difference > 1e-6:
        raise RuntimeError("the astropy twin does not reproduce the JAX model")
    return model_cpu


def plot_result(wave, flux, model_cpu, outfile, zoom_windows, mask_list):
    """Plot the fit with :func:`galspec.plot.plot_fit`: all components + residuals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flux_label = (r"$F_\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$"
                  r" $\AA^{-1}$]")

    # -- the whole spectrum ----------------------------------------------------
    fig = plt.figure(figsize=(16.0, 8.0))
    ax = fig.add_axes([0.055, 0.20, 0.885, 0.72])
    axr = fig.add_axes([0.055, 0.07, 0.885, 0.13])
    galspec.plot.plot_fit(wave, flux, model_cpu, ax=ax, axr=axr, mask_list=mask_list,
                          xlabel="Rest wavelength  [A]", ylabel=flux_label,
                          legend_kwargs={"fontsize": 10.5, "ncol": 3})
    ax.set_title("IRAS 07546+3928: whole optical spectrum, "
                 "two broad Gaussians per Balmer line", fontsize=13)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")

    # -- zooms around the interesting regions ---------------------------------
    for tag, xlim, ylim in zoom_windows:
        fig = plt.figure(figsize=(13.0, 7.0))
        ax = fig.add_axes([0.07, 0.20, 0.87, 0.72])
        axr = fig.add_axes([0.07, 0.07, 0.87, 0.13])
        galspec.plot.plot_fit(wave, flux, model_cpu, ax=ax, axr=axr, xlim=xlim,
                              ylim0=ylim, mask_list=mask_list,
                              xlabel="Rest wavelength  [A]", ylabel=flux_label,
                              legend_kwargs={"fontsize": 9.5, "ncol": 2})
        ax.set_title(f"{tag}: {xlim[0]:.0f}-{xlim[1]:.0f} A", fontsize=12)
        out = Path(outfile).with_name(Path(outfile).stem + f"_{tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def plot_corner(fit, outfile, max_params=14):
    """Corner plot of the parameters people actually quote."""
    try:
        import corner
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (corner is not installed: skipping the corner plot)")
        return None

    wanted = ("OIII.dv_c", "OIII.sigma_c", "Halpha.dv_w0", "Halpha.sigma_w0",
              "Halpha.dv_w1", "Halpha.sigma_w1", "Continuum.amplitude",
              "Continuum.alpha", "FeII.amplitude", "Halpha.amp_c", "Halpha.amp_w0",
              "Halpha.amp_w1", "Hbeta.amp_c", "Hbeta.amp_w0")
    keep = [i for i, name in enumerate(fit.full_param_names) if name in wanted][:max_params]
    fig = corner.corner(fit.samples_equal_weight[:, keep],
                        labels=[fit.full_param_names[i] for i in keep],
                        quantiles=(0.16, 0.5, 0.84), show_titles=True, title_fmt=".2f",
                        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 9})
    fig.savefig(outfile, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")
    return outfile


# --------------------------------------------------------------------------------------
# 8. Main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nlive", type=int, default=NLIVE,
                        help=f"live points (default {NLIVE})")
    parser.add_argument("--dlogz", type=float, default=DLOGZ,
                        help=f"stop when the remaining evidence is below this (default {DLOGZ})")
    parser.add_argument("--max-iterations", type=int, default=200_000)
    parser.add_argument("--device", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--outdir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true",
                        default=None, help="force the progress bar on")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="never draw the progress bar")
    parser.add_argument("--quick", action="store_true",
                        help="rough answer in a few minutes (nlive=120, dlogz=0.5)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.nlive, args.dlogz = 120, 0.5

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("IRAS 07546+3928: whole optical spectrum, fitted with JNesty on the GPU")
    print("=" * 78)

    wave, flux, error = load_spectrum()
    print(f"\n[1/5] data: {wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A (rest frame)")
    print(f"      peak flux {flux.max():.2f}, line-free level "
          f"{np.percentile(flux, 15):.2f}  [1e-14 erg/s/cm2/A]")
    for lo, hi in ((3800, 4200), (4200, 5000), (5000, 6000), (6000, 7000), (7000, 7500)):
        sel = (wave > lo) & (wave < hi)
        print(f"      {lo}-{hi} A: median flux {np.median(flux[sel]):5.2f}, "
              f"median error {np.median(error[sel]):5.3f} "
              f"({100*np.median(error[sel]/flux[sel]):4.1f}%)")

    model = build_model(jxm)
    submodels = sorted({q.split(".")[0] for q in model.qualified_param_names})
    print(f"\n[2/5] model: {len(submodels)} components, no LSF convolution, "
          f"{len(model.free_param_names)} free parameters")
    print("      " + ", ".join(submodels))

    bounds, custom_priors, notes = make_priors(flux)
    print("\n[3/5] priors (no least-squares fit is used to set them):")
    for what, description in notes:
        print(f"      {what:20s} {description}")

    missing = [q for q in free_qualified_names(model) if q not in bounds]
    if missing:
        raise RuntimeError(f"no prior given for these free parameters: {missing}")

    print("\n[4/5] running JNesty")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = run_fit(wave, flux, error, bounds, custom_priors, nlive=args.nlive,
                      dlogz=args.dlogz, max_iterations=args.max_iterations,
                      device=args.device, progress=args.progress)

    print("\n[5/5] results")
    predicted, _ = report_quality(fit, wave, flux, error)
    print_parameter_table(fit)
    print_derived(fit)
    model_cpu = check_parity(fit, wave, predicted)

    # The FITS file holds everything the .npz does, plus the model, the priors,
    # the sampler's own trajectory data and the data the fit was made against.
    npz_path = outdir / "iras07546_jax_posterior.npz"
    np.savez_compressed(npz_path,
                        samples=fit.samples_equal_weight,
                        parameter_names=np.array(fit.param_names),
                        qualified_names=np.array(fit.full_param_names),
                        log_evidence=fit.log_evidence,
                        log_evidence_error=fit.log_evidence_err,
                        wave=wave, flux=flux, error=error, predicted=predicted)
    print(f"\n  wrote {npz_path}")
    fits_path = outdir / "iras07546_jax_fit.fits"
    save_jnesty_fit_to_fits(fit, fits_path)
    print(f"  reload it with "
          f"galspec.jnesty_fit_io.load_jnesty_fit_from_fits({str(fits_path)!r})")

    if not args.no_plots:
        zooms = [("Hgamma_Hbeta", (4230, 5100), (0.0, 2.4)),
                 ("Halpha_NII_SII", (6420, 6800), (0.0, 5.0)),
                 ("blue_end", (3700, 4250), (0.6, 1.7))]
        plot_result(wave, flux, model_cpu, outdir / "iras07546_jax_fit.png",
                    zooms, mask_list=list(MASK_WINDOWS))
        plot_corner(fit, outdir / "iras07546_jax_corner.png")

    # The sampler's GPU buffers (sized by max_iterations) are no longer needed:
    # the results and the model live in host memory and the FITS file keeps a
    # full copy, so the card can be handed back to other jobs.
    fit.release_device_buffers()
    print("\n  released the sampler's device buffers")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())