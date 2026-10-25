#!/usr/bin/env python
"""Fit the SDSS spectrum of IRAS 07546+3928 with JNesty on the GPU.

What this script does
---------------------
It takes the spectrum of the quasar IRAS 07546+3928 that ships with GalSpec
(``example/data/ir07546sed.fits``), dereddens it, moves it to the rest frame and
fits **4200 to 6800 A** -- from H-gamma to the [S II] doublet -- in one go, so
that the continuum and the three Balmer regions are constrained by the same
posterior:

    power-law continuum  (flux density at 5500 A + slope)
  + Fe II template       (iron emission: the pseudo-continuum blue of H-beta)
  + H-alpha, H-beta, H-gamma:  two models each --
        a narrow Gaussian (n_*) and a broad MultiGauss (b_*)
  + He I 5876 with a narrow Gaussian and a broad component
  + He II 4686 with a narrow Gaussian and one broad Gaussian
  + [O III] 4959/5007 (narrow core + broad wing), [N II] 6548/6584,
    [S II] 6716/6731, [O I] 6300/6364
  + [O III] 4363

Everything blueward of H-gamma is left out: H-delta and the higher Balmer lines,
[Ne III] 3869, [O II] 3727.  The data start at 4200 A, so H-gamma keeps its
narrow core and the inner part of its broad wings but is cut on the blue side.
The red end stops at 6800 A, before the atmospheric O2 A band, so [Ar III] 7136
and [O II] 7319/7330 are out of range too -- nothing at either end has to be
masked.

Two models per line, as in ``example/wrk_sagan_iras07546.ipynb``
---------------------------------------------------------------
Each Balmer line (and He I 5876) is described twice, and the two models know
nothing about each other:

``n_halpha``, ``n_hbeta``, ``n_hgamma``, ``n_hei``
    ``Line_Gaussian``: the narrow component, one per line.  All of them are tied
    to the [O III] core -- one velocity shift and one width for every narrow
    line in the spectrum.

``b_halpha``, ``b_hbeta``, ``b_hgamma``, ``b_hei``, ``b_heii``
    ``Line_MultiGauss``: the broad component only.  Component 0 (``amp_c``,
    ``dv_c``, ``sigma_c``) is the broad core -- "core" here is a name for
    component 0, *not* a narrow line -- and ``amp_w0``/``dv_w0``/``sigma_w0``
    add the broader wind.  The widths and shifts are tied across the lines, so
    the broad-line region has one velocity structure; each line keeps its own
    peak and its own wind ratio, which is how the Balmer decrement and the
    broad/narrow ratio fall out of the fit.

The Fe II template is the *tied* model: instead of one Gaussian width it is
broadened by the profile of broad H-alpha, tied component by component --
``dv_c``/``sigma_c`` to the broad core, ``amp_w0``/``dv_w0``/``sigma_w0`` to the
wind.  Its parameters carry the same names as the ``Line_MultiGauss`` they come
from, ``n_components=2`` means a core plus one wind, and it has no weights of its
own: a ``Line_MultiGauss`` already is a weighted sum and the convolution
renormalises it, so the core amplitude cancels and the wind ratio (H-alpha's own
``amp_w0``) is the only weight left.  The iron therefore has exactly one free
number, its amplitude, and shares the velocity structure of the broad-line region
instead of inventing a dispersion for itself.

Two broad components per line let the model describe a classical broad-line
region plus a broader, more extreme-velocity component: the profile shape that a
single Gaussian always misses in a real AGN spectrum.

One prior that is not just "wide"
---------------------------------
The two components of a ``Line_MultiGauss`` are exchangeable: swapping them (with
the amplitudes and shifts that go with them) leaves the spectrum bit-identical, so
with a prior that allows ``amp_w0 > 1`` the posterior is bimodal and "core" and
"wind" mean nothing -- a fit reports whichever mode it happened to fill.

One number settles it.  A wind is by definition no brighter than its core, so
``amp_w0`` gets ``[0, 1]``, uniform.  The swap sends ``amp_w0 -> 1/amp_w0``, i.e.
outside that interval, so the mirrored solution is excluded and the posterior has
a single mode -- while the *widths* keep the full data-driven range (broad core
300-8000 km/s, wind 800-25000 km/s) instead of having to be split apart to keep
the labels meaningful.  ``develop/dev_jax/io_test/iron_tied_io_test.html``
measures both cases.

The error on every pixel
------------------------
The FITS file has flux only, no uncertainty, so the error is built from the data
*and from the source*: the pixel-to-pixel scatter measured from a running median
(which captures the read noise and the sky), plus **5% of the flux itself** in
quadrature, which stands for the photon noise of the source.  The counts scale with
the flux, so the uncertainty must too.

The consequence is visible in the goodness of fit: with a 5% floor the residuals
come out at 0.57 sigma and chi2/dof lands at 0.32, because the true pixel noise
here is 2-3% and the floor is larger than that.  That is the safe direction -- the
error bars are conservative rather than optimistic, which matters for a template
fit that also carries flux-calibration and template-shape uncertainty.  Set
``PHOTON_FRACTION`` to ~0.02 to let pixel noise dominate instead.

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
    python example/fit_iras07546_jax.py                 # ~10-20 min on one GPU
    python example/fit_iras07546_jax.py --quick         # ~3 min, rough answer

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
from astropy.modeling import Fittable1DModel

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
WAVE_MIN, WAVE_MAX = 4200.0, 6800.0   # rest frame: H-gamma out to [S II] 6731

ERROR_FRACTION = 0.005        # floor on the pixel-to-pixel noise
PHOTON_FRACTION = 0.05        # 5% of the flux: photon noise of the source itself
#: Regions the model does not describe.  Nothing is left to exclude: the blue end
#: starts at H-gamma, so the Ca II H+K absorption of the old 3680 A cut is gone;
#: the Na D window used to be masked because Na D is unmodelled, but He I 5876 sits
#: inside it and is now fitted, so the window is back in the fit; and the red end
#: stops at 6800 A, short of the atmospheric O2 A band.  The list is kept because
#: :func:`galspec.plot.plot_fit` takes one.
MASK_WINDOWS = []

NLIVE = 400                # live points kept by the sampler
DLOGZ = 0.01               # stop when less than 0.01 of the evidence may remain unseen

#: The Balmer lines fitted (narrow core + two broad components each).  H-delta and
#: the higher Balmer lines sit below the 4200 A cut and are not modelled.
BALMER_NAMES = ("Halpha", "Hbeta", "Hgamma")
#: Lines modelled as a narrow core plus ONE broad component.
TWO_COMPONENT_LINES = ("HeI",)
#: Single narrow Gaussians.
SINGLE_LINES = ("HeII", "OIII4363")


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

    A term of ``PHOTON_FRACTION`` (5%) times the flux is then added in quadrature.
    That is the photon noise of the source itself, which a spectrum with no error
    extension would otherwise hide: the counts scale with the flux, so the
    uncertainty does too, and on the bright continuum it dominates everything
    else.  It also absorbs what the model does not describe exactly -- flux
    calibration, the shape of the Fe II template, departures from a pure power
    law -- which is why a 5% floor is not merely conservative but the honest
    error budget for a template fit of this kind.
    """
    baseline = median_filter(flux, size=window, mode="nearest")
    scatter = median_filter(np.abs(flux - baseline), size=window, mode="nearest") / 0.6745
    scatter = median_filter(scatter, size=4 * window + 1, mode="nearest")
    statistical = np.maximum(scatter, ERROR_FRACTION * np.abs(flux))
    return np.hypot(statistical, PHOTON_FRACTION * np.abs(flux))


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


def broad_wing_count(fit):
    """Wings of the broad-Balmer ``Line_MultiGauss``: 1 for the published fit.

    The number lives in the model (``n_components``), not in the priors, so a
    script that builds a four-component model gets a four-component astropy twin
    and a four-row derived-quantities table without passing anything else around.
    """
    model = getattr(fit, "model", fit)
    try:
        return int(model["b_halpha"].n_components) - 1
    except Exception:                                     # noqa: BLE001
        names = set(getattr(fit, "full_param_names", ()))
        count = 0
        while f"b_halpha.sigma_w{count}" in names:
            count += 1
        return count


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


#: Display names for the models, with the maths a paper would print.  Plot labels
#: are built from these plus the parameter symbol, e.g. ``BH$\alpha$:$\sigma_{w0}$``
#: for the width of the broad component of H-alpha.
MODEL_LABELS = {
    "Continuum": "Continuum",
    "FeII": "Fe II",
    # one label per *model*: the narrow Gaussian and the broad MultiGauss of the
    # same line are separate objects, and the figure draws them separately
    "n_halpha": r"BH$\alpha$ (narrow)", "b_halpha": r"BH$\alpha$ (broad)",
    "n_hbeta": r"BH$\beta$ (narrow)", "b_hbeta": r"BH$\beta$ (broad)",
    "n_hgamma": r"BH$\gamma$ (narrow)", "b_hgamma": r"BH$\gamma$ (broad)",
    "n_hei": r"He I (narrow)", "b_hei": r"He I (broad)",
    "b_heii": r"He II (broad)",
    "OIII": r"[O III]", "NII": r"[N II]", "SII": r"[S II]", "OI": r"[O I]",
    "OIII4363": r"[O III] 4363", "HeII": r"He II", "HeI": r"He I",
}


def apply_model_labels(model, labels=None):
    """Give every submodel a display name for plots.

    ``galspec.param_labels`` turns a parameter into ``model-label:parameter-symbol``,
    so setting ``label`` is what puts the line's name in front of the symbol.
    """
    mapping = MODEL_LABELS if labels is None else labels
    for name in model.submodel_names:
        display = mapping.get(name)
        if display is not None:
            model[name].label = display
    return model


def build_model(backend=jxm, include_iron=None, n_balmer_wings=1,
                tie_broad_balmer=True):
    """The full model: two models per line, narrow and broad.

    ``include_iron`` defaults to "yes for the JAX model, no for the astropy twin":
    the twin is used for plotting, and there the iron enters as a curve evaluated
    from the JAX model itself.  That is not because the two constructions disagree
    -- they agree to 1e-15 (``io_test/cpu_vs_jax.html``) -- but because the astropy
    ``IronTemplate_tied`` takes the line profile as a *kernel object* and exposes
    only ``amplitude``/``stddev``, where the JAX one takes ``n_components`` and the
    five kernel parameters, so the parameter-by-parameter copy in
    :func:`cpu_model_with_values` has nothing to set.  See there.

    ``backend`` is either :mod:`galspec.jax_models` -- the models the fit uses --
    or :mod:`galspec` itself, the astropy equivalents.  Both expose the same
    constructors, tie helpers and parameter names, so one function builds the
    model twice: once to fit, once to hand to :func:`galspec.plot.plot_fit`.

    Every Balmer line (and He I 5876) is **two** models, as in
    ``example/wrk_sagan_iras07546.ipynb``:

    ``n_<line>``
        a ``Line_Gaussian`` for the narrow component, with its shift and width
        tied to the [O III] core -- so all the narrow lines share one kinematics;
    ``b_<line>``
        a ``Line_MultiGauss`` for the broad component only.  Its component 0
        (``amp_c``, ``dv_c``, ``sigma_c``) is the broad core, *not* a narrow line,
        and ``amp_w0``/``dv_w0``/``sigma_w0`` add the broader wind.

    The two are summed into the spectrum; neither knows about the other.  The wind
    amplitude is a *ratio* to the core amplitude, which is how the GalSpec models
    are written, so the Balmer decrement and the broad/narrow ratio both fall out
    of the fit.

    ``n_balmer_wings`` sets how many wind components follow the core (so
    ``n_components = 1 + n_balmer_wings``).  The published fit uses one; the
    component-count study in ``example/fit_iras07546_jax_toggle.py`` asks the
    sampler for up to three and switches them on and off.  He I 5876 and He II
    4686 keep a single broad Gaussian either way.

    ``tie_broad_balmer`` decides whether broad H-beta and H-gamma *share* the
    profile shape of broad H-alpha (the default: the shifts and widths are tied,
    component by component) or keep their own.  The soft-tie fit in
    ``example/fit_iras07546_jax_tied.py`` sets it to ``False`` and constrains the
    released parameters with a Gaussian centred on H-alpha instead, which is the
    difference between "the same" and "similar".

    No convolution is applied: see the module docstring.
    """
    if include_iron is None:
        # the astropy twin is drawn with the iron curve taken from the JAX model
        include_iron = backend is jxm

    # -- continuum: a power law anchored at 5500 A, plus the iron pseudo-continuum --
    continuum = _power_law(backend)(amplitude=0.55, x_0=5500.0, alpha=1.0,
                                    fixed={"x_0": True}, name="Continuum")
    # The iron used to be fitted with a single velocity dispersion and a velocity
    # shift, i.e. ``backend.IronTemplate_new(amplitude=..., sigma=..., z=..., dv=...)``.
    # Those lines are kept here because they are the reference against which the
    # tied version was checked (the two agree exactly when the kernel has one
    # component of width ``sigma`` and zero shift):
    #
    #     iron = backend.IronTemplate_new(amplitude=0.2, sigma=2500.0, z=0.0,
    #                                     dv=0.0, name="FeII")
    #     iron.z.fixed = True
    #
    # What the fit uses instead: the same template, broadened by *the profile of
    # broad H-alpha*.  The kernel parameters carry the same names as the
    # ``Line_MultiGauss`` they are tied to -- ``n_components=2`` is a core plus one
    # wind, exactly as for the line -- and there are no weights of their own: a
    # ``Line_MultiGauss`` already is a weighted sum, and the convolution renormalises
    # it, so the core amplitude cancels and only the wind ratio (the line's own
    # ``amp_w0``) survives as a weight.  The iron then has one free parameter, its
    # amplitude: the redshift is known, and the shifts, widths and weights are
    # broad H-alpha's.
    iron = None
    if include_iron:
        par_w = {}
        if n_balmer_wings >= 1:
            par_w.update(amp_w0=0.10, dv_w0=0.0, sigma_w0=6000.0)
        for i in range(1, n_balmer_wings):
            par_w.update({f"amp_w{i}": 0.05, f"dv_w{i}": 0.0, f"sigma_w{i}": 9000.0})
        iron = backend.IronTemplate_tied(n_components=1 + n_balmer_wings,
                                         amplitude=0.2, z=0.0, dv_c=0.0,
                                         sigma_c=1500.0, name="FeII", **par_w)
        iron.z.fixed = True

    # -- narrow components: one Gaussian per line, one kinematics for all --------
    def narrow(name, wavec, amplitude, model_name):
        return backend.Line_Gaussian(amplitude=amplitude, dv=0.0, sigma=700.0,
                                     wavec=wavec, name=model_name)

    # -- broad components: a Line_MultiGauss, component 0 = broad core -----------
    def broad(model_name, wavec, amp_c, amp_w0=0.2):
        par_w = {}
        if n_balmer_wings >= 1:
            par_w.update(amp_w0=amp_w0, dv_w0=0.0, sigma_w0=6000.0)
        for i in range(1, n_balmer_wings):
            par_w.update({f"amp_w{i}": 0.05, f"dv_w{i}": 0.0, f"sigma_w{i}": 9000.0})
        return backend.Line_MultiGauss(n_components=1 + n_balmer_wings, amp_c=amp_c,
                                       dv_c=0.0, sigma_c=1500.0, wavec=wavec,
                                       name=model_name, **par_w)

    n_halpha = narrow("Halpha", LINE_WAVE["Halpha"], 2.2, "n_halpha")
    n_hbeta = narrow("Hbeta", LINE_WAVE["Hbeta"], 0.8, "n_hbeta")
    n_hgamma = narrow("Hgamma", LINE_WAVE["Hgamma"], 0.4, "n_hgamma")

    b_halpha = broad("b_halpha", LINE_WAVE["Halpha"], 0.65, 0.10)
    b_hbeta = broad("b_hbeta", LINE_WAVE["Hbeta"], 0.22, 0.20)
    b_hgamma = broad("b_hgamma", LINE_WAVE["Hgamma"], 0.10, 0.20)

    # He I 5876 sits on top of what used to be the Na D mask.  With the mask gone it
    # has to be modelled: a narrow line from the narrow-line region plus a broad foot
    # from the broad-line region.  One broad component is enough there.
    n_hei = narrow("HeI", LINE_WAVE["HeI_5876"], 0.12, "n_hei")
    b_hei = backend.Line_MultiGauss(n_components=1, amp_c=0.06, dv_c=0.0,
                                    sigma_c=1500.0, wavec=LINE_WAVE["HeI_5876"],
                                    name="b_hei")

    # He II 4686 gets a broad component as well: in a narrow-line Seyfert 1 the
    # broad He II is strong enough to be measured, and one Gaussian is enough -- the
    # line sits on the same broad-line kinematics as He I 5876.  It has to be there
    # on the blue side of H-beta, where the iron and the Balmer wing overlap.
    b_heii = backend.Line_MultiGauss(n_components=1, amp_c=0.05, dv_c=0.0,
                                     sigma_c=1500.0, wavec=LINE_WAVE["HeII_4686"],
                                     name="b_heii")

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

    # -- other single narrow Gaussians ----------------------------------------
    he2 = narrow("HeII", LINE_WAVE["HeII_4686"], 0.08, "HeII")
    o3_4363 = narrow("OIII4363", LINE_WAVE["OIII_4363"], 0.05, "OIII4363")

    # Nothing blueward of H-gamma or redward of [S II] is modelled: the data start
    # at 4200 A and stop at 6800 A.  Dropped with the old 3680-7500 A range: H-delta,
    # H-epsilon, H8, H9, H11, [Ne III] 3869, [O II] 3727, [Ar III] 7136, [O II] 7325.
    components = [continuum]
    if iron is not None:
        components.append(iron)
    components += [n_halpha, b_halpha, n_hbeta, b_hbeta, n_hgamma, b_hgamma,
                   o3, n2, s2, o1, he2, b_heii, o3_4363, n_hei, b_hei]
    model = components[0]
    for _component in components[1:]:
        model = model + _component

    # -- ties: one narrow kinematics for every narrow line, [O III] as reference --
    for line in (n_halpha, n_hbeta, n_hgamma, n_hei):
        line.dv.tied = _TieFrom("OIII", "dv_c")
        line.sigma.tied = _TieFrom("OIII", "sigma_c")
    for line in (n2, s2, o1):
        line.dv_c.tied = _tie_fn(backend, "tie_MultiGauss_dv_c")("OIII")
        line.sigma_c.tied = _tie_fn(backend, "tie_MultiGauss_sigma_c")("OIII")
    for line in (he2, o3_4363):
        line.dv.tied = _TieFrom("OIII", "dv_c")
        line.sigma.tied = _TieFrom("OIII", "sigma_c")

    # -- ties: one broad kinematics, set by the broad H-alpha ---------------------
    # The broad core and the wind are separate structures, so they get separate
    # kinematics: a classical broad-line region and a broader one can rotate or be
    # shifted differently.  What is shared is the *shape*; each line keeps its own
    # peak and its own ratio, which is how the Balmer decrement comes out.
    for line in (b_hbeta, b_hgamma) if tie_broad_balmer else ():
        line.dv_c.tied = _tie_fn(backend, "tie_MultiGauss_dv_c")("b_halpha")
        line.sigma_c.tied = _tie_fn(backend, "tie_MultiGauss_sigma_c")("b_halpha")
        for i in range(n_balmer_wings):
            getattr(line, f"dv_w{i}").tied = _tie_fn(
                backend, f"tie_MultiGauss_dv_w{i}")("b_halpha")
            getattr(line, f"sigma_w{i}").tied = _tie_fn(
                backend, f"tie_MultiGauss_sigma_w{i}")("b_halpha")
    for line in (b_hei, b_heii):
        line.dv_c.tied = _tie_fn(backend, "tie_MultiGauss_dv_c")("b_halpha")
        line.sigma_c.tied = _tie_fn(backend, "tie_MultiGauss_sigma_c")("b_halpha")
    o3.dv_w0.tied = _TieFrom("OIII", "dv_c")     # the wing sits at the core velocity

    # -- ties: the iron kernel IS the broad H-alpha profile ----------------------
    # Kernels component for component: the core takes the core's shift and width,
    # the wind takes the wind's, ratios included.  Nothing is left to the iron but
    # its amplitude.  The compound above already exists, which is what lets the tie
    # names resolve.
    if iron is not None:
        iron.dv_c.tied = _tie_fn(backend, "tie_MultiGauss_dv_c")("b_halpha")
        iron.sigma_c.tied = _tie_fn(backend, "tie_MultiGauss_sigma_c")("b_halpha")
        for i in range(n_balmer_wings):
            for stem in ("amp_w", "dv_w", "sigma_w"):
                getattr(iron, f"{stem}{i}").tied = _tie_fn(
                    backend, f"tie_MultiGauss_{stem}{i}")("b_halpha")

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


def make_priors(flux, n_balmer_wings=1):
    """Return ``(bounds, custom_priors, notes)`` for the whole model.

    Everything is derived from the spectrum itself, so the recipe transfers to
    another object without hand-tuning:

    * a line peak may be anything from 1e-3 (invisible) to 3x the brightest pixel
      of the spectrum, log-uniform -- four decades of freedom per line;
    * a broad component may be up to 8x its own narrow core, log-uniform;
    * a *wind* component is 0-1 of its core, uniform.  That is the whole range a
      ratio can take in one direction, and it is what breaks the degeneracy:
      component 0 and component 1 of a ``Line_MultiGauss`` are exchangeable --
      swapping them leaves the spectrum bit-identical -- but the swap sends
      ``amp_w0 -> 1/amp_w0``, which for ``amp_w0 < 1`` lies outside the prior.  So
      the posterior has a single mode and the labels mean what the names say,
      without any constraint being put on the widths;
    * narrow lines: 30-800 km/s of velocity dispersion (FWHM 70-1900 km/s).  No
      LSF is folded in, so this is the *observed* width;
    * broad core (``b_halpha.sigma_c``): 300-8000 km/s, shifted by up to
      +-2500 km/s;
    * broad wind (``b_halpha.sigma_w0``): 800-25000 km/s (FWHM up to 59000 km/s),
      shifted by up to +-6000 km/s;
    * any further wings (``n_balmer_wings > 1``) reuse exactly those two priors, so
      a wing is free to be narrower than the core and the ladder of components does
      not smuggle in a width ordering.  Switching a wing *off* is a different prior,
      applied by the study script through ``custom_priors``.

    ``develop/dev_jax/iron_tied_io_test.py`` measures the degeneracy with and
    without the amplitude limit.
    """
    peak = float(np.max(flux))
    continuum = float(np.percentile(flux, 15.0))
    amp = (1e-3, 3.0 * peak)      # line peaks
    ratio = (1e-3, 8.0)           # broad / narrow amplitude
    wind = (0.0, 1.0)             # wind / core: bounded, and that breaks the
                                  # core-wind exchange degeneracy (swap -> 1/amp_w0)

    bounds = {
        "Continuum.amplitude": (0.2 * continuum, 4.0 * continuum),
        "Continuum.alpha": (-5.0, 5.0),
        # the iron has ONE free parameter now: its amplitude (the kernel is H-alpha)
        "FeII.amplitude": (1e-3, 3.0 * continuum),
        # narrow kinematics: [O III] sets them for every narrow line
        "OIII.dv_c": (-1000.0, 1000.0),
        "OIII.sigma_c": (30.0, 800.0),
        "OIII.amp_c0": amp,
        # [O III] broad wing
        "OIII.amp_w0": wind,
        "OIII.sigma_w0": (300.0, 4000.0),
        # other line peaks
        "NII.amp_c0": amp,
        "SII.amp_c0": amp, "SII.amp_c1": amp,
        "OI.amp_c0": amp,
        # broad H-alpha: the only place where the broad kinematics are free
        "b_halpha.dv_c": (-2500.0, 2500.0),
        "b_halpha.sigma_c": (300.0, 8000.0),
        "b_halpha.amp_w0": wind,
        "b_halpha.dv_w0": (-6000.0, 6000.0),
        "b_halpha.sigma_w0": (800.0, 25000.0),
    }
    for i in range(1, n_balmer_wings):        # extra wings: same priors as the first
        bounds[f"b_halpha.amp_w{i}"] = wind
        bounds[f"b_halpha.dv_w{i}"] = (-6000.0, 6000.0)
        bounds[f"b_halpha.sigma_w{i}"] = (800.0, 25000.0)
    # narrow peaks, one per line
    for name in BALMER_NAMES:
        bounds[f"n_{name.lower()}.amplitude"] = amp
    bounds["n_hei.amplitude"] = amp
    # broad peaks and ratios: each line keeps its own (the Balmer decrement)
    for name in BALMER_NAMES:
        bounds[f"b_{name.lower()}.amp_c"] = amp
        bounds[f"b_{name.lower()}.amp_w0"] = wind
        for i in range(1, n_balmer_wings):
            bounds[f"b_{name.lower()}.amp_w{i}"] = wind
    bounds["b_hei.amp_c"] = amp
    bounds["b_heii.amp_c"] = amp
    for name in SINGLE_LINES:
        bounds[f"{name}.amplitude"] = amp

    # Peak heights get a log-uniform prior.  The wind ratios do not: they are
    # bounded above by 1, and a uniform prior over [0, 1] is the unbiased one there.
    log_params = [name for name in bounds
                  if name.endswith(("amp_c", "amp_c0", "amp_c1", "amplitude"))]
    custom = {name: log_uniform(*bounds[name]) for name in log_params}

    notes = [
        ("continuum", "0.2-4x the level of the line-free windows, slope -5 to 5"),
        ("line peaks", "1e-3 to 3x the brightest pixel, log-uniform"),
        ("broad/narrow ratio", "1e-3 to 8, log-uniform"),
        ("wind/core ratio", "0-1, uniform: no wind is brighter than its core, and "
                            "the bound breaks the core-wind exchange"),
        ("narrow lines", "30-800 km/s, one width for all, set by [O III]"),
        ("broad core", "300-8000 km/s, one width and shift for all broad lines"),
        ("broad wind", "800-25000 km/s, its own width and shift"
                       + (f"; {n_balmer_wings} wings, same priors"
                          if n_balmer_wings > 1 else "")),
        ("iron template", "amplitude only: the broadening is broad H-alpha"),
        ("He I 5876", "narrow Gaussian + one broad component, same ratio prior"),
        ("[S II] 6716/6731", "both peaks free: the ratio is a density diagnostic"),
    ]
    return bounds, custom, notes


# --------------------------------------------------------------------------------------
# 5. Run the fit
# --------------------------------------------------------------------------------------
def run_fit(wave, flux, error, bounds, custom_priors, nlive=NLIVE, dlogz=DLOGZ,
            max_iterations=200_000, device="auto", progress=None, n_balmer_wings=1,
            tie_broad_balmer=True):
    """Sample the posterior with JNesty.  No optimiser is involved anywhere.

    ``n_balmer_wings`` must match the ``make_priors`` call that produced
    ``bounds``: it decides how many components the model has.
    """
    if progress is None:
        # Only draw the bar when someone is watching: in a batch job it would be
        # thousands of carriage-return updates in the log.
        progress = sys.stdout.isatty()
    model = apply_model_labels(build_model(jxm,     # so the fit carries plot labels
                                          n_balmer_wings=n_balmer_wings,
                                          tie_broad_balmer=tie_broad_balmer))
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
    ".sigma": "km/s", ".dv": "km/s",
    ".amp_w2": "(ratio to core)", ".amp_w3": "(ratio to core)",
    ".dv_w2": "km/s", ".sigma_w2": "km/s",
    ".dv_w3": "km/s", ".sigma_w3": "km/s",
}

PARAM_NOTES = {
    "Continuum.amplitude": "continuum flux density at 5500 A",
    "Continuum.alpha": "continuum slope (F_nu ~ nu^-alpha)",
    "FeII.amplitude": "iron strength (broadened by broad H-alpha)",
    "OIII.amp_c0": "[O III] 5007 narrow peak", "OIII.amp_w0": "[O III] wing / core",
    "OIII.dv_c": "velocity shift of all narrow lines",
    "OIII.sigma_c": "velocity width of all narrow lines",
    "OIII.sigma_w0": "width of the [O III] wing",
    "NII.amp_c0": "[N II] 6583 peak",
    "SII.amp_c0": "[S II] 6716 peak", "SII.amp_c1": "[S II] 6731 peak",
    "OI.amp_c0": "[O I] 6300 peak",
    "HeII.amplitude": "He II 4686 peak", "OIII4363.amplitude": "[O III] 4363 peak",
    "b_heii.amp_c": "broad He II 4686 peak (one Gaussian)",
    "b_hei.amp_c": "broad He I 5876 peak (one Gaussian)",
    "b_halpha.dv_c": "shift of the broad core (all broad lines)",
    "b_halpha.sigma_c": "width of the broad core (all broad lines)",
    "b_halpha.amp_w0": "wind / broad core of H-alpha",
    "b_halpha.dv_w0": "shift of the wind, relative to the core",
    "b_halpha.sigma_w0": "width of the wind (all broad lines)",
}
for _i in range(1, 4):
    PARAM_NOTES.setdefault(f"b_halpha.dv_w{_i}",
                           f"shift of wing {_i}, relative to the core")
    PARAM_NOTES.setdefault(f"b_halpha.sigma_w{_i}",
                           f"width of wing {_i} (all broad lines)")
for _name in BALMER_NAMES:
    _low = _name.lower()
    PARAM_NOTES.setdefault(f"n_{_low}.amplitude", f"narrow {_name} peak")
    PARAM_NOTES.setdefault(f"b_{_low}.amp_c", f"broad core of {_name}")
    PARAM_NOTES.setdefault(f"b_{_low}.amp_w0", f"wind / broad core of {_name}")
    for _i in range(1, 4):
        PARAM_NOTES.setdefault(f"b_{_low}.amp_w{_i}",
                               f"wing {_i} / broad core of {_name}")
PARAM_NOTES.setdefault("n_hei.amplitude", "narrow He I 5876 peak")
PARAM_NOTES.setdefault("b_hei.amp_c", "broad core of He I 5876")
PARAM_NOTES.setdefault("b_heii.amp_c", "broad He II 4686 (single Gaussian)")


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
    n_wings = broad_wing_count(fit)

    def get(sample, key):
        return sample[..., index[key]]

    out = {}
    for line in BALMER_NAMES:
        low = line.lower()
        wavec = LINE_WAVE[line]
        block = {}
        # the narrow Gaussian and the broad MultiGauss are separate models, so each
        # component is read from the model it actually lives in.  One wing keeps the
        # name it has always had in the tables; more wings are numbered, and each
        # line's own ratio pairs with the *shared* width of that wing.
        components = [("narrow", get(samples, f"n_{low}.amplitude"),
                       get(samples, "OIII.sigma_c")),
                      ("broad core", get(samples, f"b_{low}.amp_c"),
                       get(samples, "b_halpha.sigma_c"))]
        for i in range(n_wings):
            label = "broad wind" if n_wings == 1 else f"broad wing {i}"
            components.append((label,
                               get(samples, f"b_{low}.amp_c") * get(samples, f"b_{low}.amp_w{i}"),
                               get(samples, f"b_halpha.sigma_w{i}")))
        for label, amplitude, sigma in components:
            # F = A sigma sqrt(2 pi) (lambda / c); A is in 1e-14 units and the
            # answer is quoted in 1e-17 erg/s/cm2, hence the 1e3.
            flux = amplitude * sigma * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3
            block[label] = {
                "fwhm": (np.median(fwhm * sigma), *np.percentile(fwhm * sigma, [16, 84])),
                "flux": (np.median(flux), *np.percentile(flux, [16, 84])),
            }
        broad = block["broad core"]["flux"][0] + sum(
            block["broad wind" if n_wings == 1 else f"broad wing {i}"]["flux"][0]
            for i in range(n_wings))
        block["broad/narrow"] = (broad / block["narrow"]["flux"][0], np.nan, np.nan)
        out[line] = block

    # He I 5876 has one broad component rather than two
    wavec = LINE_WAVE["HeI_5876"]
    for label, amplitude, sigma in (
            ("narrow", get(samples, "n_hei.amplitude"), get(samples, "OIII.sigma_c")),
            ("broad core", get(samples, "b_hei.amp_c"),
             get(samples, "b_halpha.sigma_c"))):
        flux = amplitude * sigma * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3
        out.setdefault("He I 5876", {})[label] = {
            "fwhm": (np.median(fwhm * sigma), *np.percentile(fwhm * sigma, [16, 84])),
            "flux": (np.median(flux), *np.percentile(flux, [16, 84])),
        }
    he1 = out["He I 5876"]
    he1["broad/narrow"] = (he1["broad core"]["flux"][0] / he1["narrow"]["flux"][0],
                           np.nan, np.nan)

    # He II 4686 is the same construction with a single broad Gaussian
    wavec = LINE_WAVE["HeII_4686"]
    for label, amplitude, sigma in (
            ("narrow", get(samples, "HeII.amplitude"), get(samples, "OIII.sigma_c")),
            ("broad core", get(samples, "b_heii.amp_c"),
             get(samples, "b_halpha.sigma_c"))):
        flux = amplitude * sigma * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3
        out.setdefault("He II 4686", {})[label] = {
            "fwhm": (np.median(fwhm * sigma), *np.percentile(fwhm * sigma, [16, 84])),
            "flux": (np.median(flux), *np.percentile(flux, [16, 84])),
        }
    he2 = out["He II 4686"]
    he2["broad/narrow"] = (he2["broad core"]["flux"][0] / he2["narrow"]["flux"][0],
                           np.nan, np.nan)

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
        for label in ("narrow", "broad core", "broad wind"):
            if label not in block:
                continue
            fw, fl = block[label]["fwhm"], block[label]["flux"]
            print(f"  {line:8s} {label:11s} {fw[0]:8.0f} (+{fw[2]-fw[0]:.0f}/-{fw[0]-fw[1]:.0f})"
                  f"        {fl[0]:9.1f} (+{fl[2]-fl[0]:.1f}/-{fl[0]-fl[1]:.1f})")
        print(f"  {'':8s} {'broad/narrow':11s} {block['broad/narrow'][0]:8.2f}")


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
class _TabulatedCurve(Fittable1DModel):
    """A curve already evaluated on a grid, presented as an astropy component.

    The iron is the one component without a parameter-for-parameter astropy twin:
    ``galspec.IronTemplate_tied`` takes the broad line as a *kernel object* and
    exposes only ``amplitude``/``stddev``, while ``jxm.IronTemplate_tied`` takes
    ``n_components`` plus the five kernel parameters.  The two broaden the template
    the same way and agree to 1e-15 (``io_test/cpu_vs_jax.html``), so this is a
    plumbing difference, not a modelling one; even so, the wrapper carries the flux
    the fitted JAX model actually produces, so the figure and the fit agree bit for
    bit rather than to 1e-15.
    """

    n_inputs = 1
    n_outputs = 1

    def __init__(self, wave, flux, name=None, **kwargs):
        super().__init__(**kwargs)
        self._wave = np.asarray(wave, dtype=float)
        self._flux = np.asarray(flux, dtype=float)
        if name is not None:
            self.name = name

    def evaluate(self, x):
        return np.interp(np.asarray(x, dtype=float), self._wave, self._flux)


def cpu_model_with_values(fit, wave, tie_broad_balmer=True):
    """The astropy twin of the fitted model, at the best-fitting sample.

    :func:`galspec.plot.plot_fit` walks an astropy compound model and draws every
    component, so the fitted JAX model is mirrored into its astropy equivalent
    here.

    ``tie_broad_balmer`` must match the model that was fitted: a soft-tied fit
    released those ties, and copying its resolved values onto a twin that still
    has them tied would re-impose H-alpha's widths on H-beta.

    The values copied across are the *tie-resolved* ones (``param_dict()``): an
    astropy compound model does not re-derive a tied parameter when it is
    evaluated -- ties are a fitting-time constraint there -- so the twin has to be
    given the widths and shifts that the ties produced, otherwise every
    ``dv_c``/``sigma_c`` would still hold its construction value.

    The iron is handled differently from the other components: its astropy twin is
    a curve, not a model, because the tied iron has no astropy counterpart with the
    same parameters (see :class:`_TabulatedCurve`).  Its flux is what the JAX model
    produces, obtained by subtracting the model with the iron amplitude set to zero.

    :func:`check_parity` then verifies the twin against the JAX model, which also
    re-proves CPU/JAX parity on this very fit.
    """
    model_jax, _, theta = fit.get_best_fit()
    resolved = model_jax.param_dict()

    # ``flux`` takes the *free* parameters, in free_param_names order: map the
    # qualified name onto that index, not onto the full parameter list
    free = set(model_jax.free_param_names)
    free_qualified = [q for q, flat
                      in zip(model_jax.qualified_param_names, model_jax.param_names)
                      if flat in free]
    theta_no_iron = np.array(theta, dtype=float).copy()
    theta_no_iron[free_qualified.index("FeII.amplitude")] = 0.0
    iron_flux = (np.asarray(model_jax.flux(wave, theta))
                 - np.asarray(model_jax.flux(wave, theta_no_iron)))

    iron = _TabulatedCurve(wave, iron_flux, name="FeII")
    parts = list(build_model(galspec,                # astropy leaves, no iron model
                             n_balmer_wings=broad_wing_count(fit),
                             tie_broad_balmer=tie_broad_balmer))
    parts.insert(1, iron)                       # right after the continuum
    model_cpu = parts[0]
    for part in parts[1:]:
        model_cpu = model_cpu + part
    apply_model_labels(model_cpu)

    for flat, qualified in zip(model_jax.param_names, model_jax.qualified_param_names):
        value = resolved.get(flat)
        if value is None or qualified.startswith("FeII."):
            continue
        submodel, parameter = qualified.split(".", 1)
        getattr(model_cpu[submodel], parameter).value = float(value)
    return model_cpu


def check_parity(fit, wave, predicted, tie_broad_balmer=True):
    """Assert that the astropy twin reproduces the JAX model."""
    model_cpu = cpu_model_with_values(fit, wave, tie_broad_balmer=tie_broad_balmer)
    cpu = np.asarray(model_cpu(wave))
    difference = float(np.max(np.abs(cpu - predicted)) / np.max(np.abs(predicted)))
    print(f"  CPU/JAX parity  : max difference {difference:.2e} of the peak flux")
    if difference > 1e-6:
        raise RuntimeError("the astropy twin does not reproduce the JAX model")
    return model_cpu


def plot_result(wave, flux, model_cpu, outfile, zoom_windows, mask_list, error=None):
    """Plot the fit with :func:`galspec.plot.plot_fit`: all components + residuals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flux_label = (r"$F_\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$"
                  r" $\AA^{-1}$]")

    # -- the whole spectrum ----------------------------------------------------
    fig = plt.figure(figsize=(17.0, 8.0))
    ax = fig.add_axes([0.055, 0.20, 0.82, 0.72])
    axr = fig.add_axes([0.055, 0.07, 0.82, 0.13])
    galspec.plot.plot_fit(wave, flux, model_cpu, ax=ax, axr=axr, mask_list=mask_list,
                          xlabel="Rest wavelength  [A]", ylabel=flux_label,
                          error=error, residual_in_sigma=True,
                          legend_kwargs={"fontsize": 10.5, "ncol": 2,
                                         "loc": "center left",
                                         "bbox_to_anchor": (1.005, 0.5),
                                         "frameon": False})
    ax.set_title("IRAS 07546+3928: 4200-6800 A, each line split into a narrow and a "
                 "broad model, iron tied to broad H-alpha", fontsize=13)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")

    # -- zooms around the interesting regions ---------------------------------
    # The legend lives outside the axes: inside, an opaque frame sat exactly on top
    # of the [O II] 3727 peak in the blue-end panel.
    for tag, xlim, ylim in zoom_windows:
        fig = plt.figure(figsize=(15.5, 7.0))
        ax = fig.add_axes([0.06, 0.20, 0.70, 0.72])
        axr = fig.add_axes([0.06, 0.07, 0.70, 0.13])
        galspec.plot.plot_fit(wave, flux, model_cpu, ax=ax, axr=axr, xlim=xlim,
                              ylim0=ylim, ylim1=_residual_limits(wave, flux, error,
                                                                 model_cpu, xlim),
                              mask_list=mask_list, error=error,
                              residual_in_sigma=True,
                              xlabel="Rest wavelength  [A]", ylabel=flux_label,
                              legend_kwargs={"fontsize": 9.0, "ncol": 1,
                                             "loc": "center left",
                                             "bbox_to_anchor": (1.005, 0.5),
                                             "frameon": False})
        ax.set_title(f"{tag}: {xlim[0]:.0f}-{xlim[1]:.0f} A", fontsize=12)
        out = Path(outfile).with_name(Path(outfile).stem + f"_{tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def _residual_limits(wave, flux, error, model_cpu, xlim, margin=0.6):
    """Symmetric residual limits for one panel, in sigma.

    Without this every panel shares one scale set by the single worst pixel in the
    spectrum, so a zoom on a well-fitted region shows a flat line.
    """
    inside = (wave >= xlim[0]) & (wave <= xlim[1])
    residual = (flux - np.asarray(model_cpu(wave)))[inside]
    if error is not None:
        residual = residual / np.asarray(error, dtype=float)[inside]
    if residual.size == 0:
        return None
    scale = float(np.nanpercentile(np.abs(residual), 99.5))
    return (-1.25 * scale - margin * 0.1, 1.25 * scale + margin * 0.1)


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

    wanted = ("OIII.dv_c", "OIII.sigma_c", "b_halpha.dv_c", "b_halpha.sigma_c",
              "b_halpha.dv_w0", "b_halpha.sigma_w0", "Continuum.amplitude",
              "Continuum.alpha", "FeII.amplitude", "b_halpha.amp_c",
              "b_halpha.amp_w0", "b_halpha.amp_w1", "b_halpha.amp_w2",
              "b_halpha.sigma_w1", "b_halpha.sigma_w2",
              "n_halpha.amplitude", "b_hei.amp_c",
              "n_hei.amplitude", "b_heii.amp_c")
    keep = [i for i, name in enumerate(fit.full_param_names) if name in wanted][:max_params]
    # maths labels: "BH$\alpha$:$\sigma_{w0}$" instead of "b_halpha.sigma_w0"
    labels = getattr(fit, "param_labels", None) or fit.full_param_names
    fig = corner.corner(fit.samples_equal_weight[:, keep],
                        labels=[labels[i] for i in keep],
                        quantiles=(0.16, 0.5, 0.84), show_titles=True, title_fmt=".3g",
                        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 9})
    fig.savefig(outfile, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")
    return outfile


#: Parameters worth a histogram: the numbers the fit exists to measure.
HIST_PARAMS = ("b_halpha.amp_c", "b_halpha.amp_w0", "b_halpha.dv_c",
               "b_halpha.sigma_c", "b_halpha.dv_w0", "b_halpha.sigma_w0",
               "b_hbeta.amp_c", "b_hbeta.amp_w0", "n_halpha.amplitude",
               "OIII.sigma_c", "OIII.dv_c", "b_hei.amp_c", "n_hei.amplitude",
               "b_heii.amp_c", "SII.amp_c0", "SII.amp_c1", "Continuum.alpha",
               # present only when the model has more than one wing
               "b_halpha.amp_w1", "b_halpha.sigma_w1", "b_halpha.dv_w1",
               "b_halpha.amp_w2", "b_halpha.sigma_w2", "b_halpha.dv_w2")


def plot_histograms(fit, outfile, params=HIST_PARAMS):
    """One-dimensional posteriors, with the median and the 16-84 band.

    A corner plot of thirty parameters is unreadable; these are the marginals worth
    showing, and they carry the numbers ``print_parameter_table`` quotes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(fit.full_param_names)
    wanted = [name for name in params if name in names]
    fig, axes = galspec.plot.plot_posterior_hist(
        fit, params=wanted, ncols=4, bins=40,
        title="IRAS 07546+3928: marginal posteriors", filename=str(outfile))
    plt.close(fig)
    print(f"  wrote {outfile} ({len(wanted)} panels)")
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
    parser.add_argument("--broad-wings", type=int, default=1,
                        help="wind components after the broad core (default 1)")
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
    print("IRAS 07546+3928: 4200-6800 A, fitted with JNesty on the GPU")
    print("=" * 78)

    wave, flux, error = load_spectrum()
    print(f"\n[1/5] data: {wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A (rest frame)")
    print(f"      peak flux {flux.max():.2f}, line-free level "
          f"{np.percentile(flux, 15):.2f}  [1e-14 erg/s/cm2/A]")
    for lo, hi in ((4200, 4600), (4600, 5400), (5400, 6000), (6000, 6800)):
        sel = (wave > lo) & (wave < hi)
        print(f"      {lo}-{hi} A: median flux {np.median(flux[sel]):5.2f}, "
              f"median error {np.median(error[sel]):5.3f} "
              f"({100*np.median(error[sel]/flux[sel]):4.1f}%)")

    model = apply_model_labels(build_model(jxm, n_balmer_wings=args.broad_wings))
    submodels = sorted({q.split(".")[0] for q in model.qualified_param_names})
    print(f"\n[2/5] model: {len(submodels)} components, no LSF convolution, "
          f"{len(model.free_param_names)} free parameters")
    print("      " + ", ".join(submodels))

    bounds, custom_priors, notes = make_priors(flux,
                                               n_balmer_wings=args.broad_wings)
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
                      device=args.device, progress=args.progress,
                      n_balmer_wings=args.broad_wings)

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
                 ("HeI_NaD", (5750, 6000), (0.4, 1.4))]
        plot_result(wave, flux, model_cpu, outdir / "iras07546_jax_fit.png",
                    zooms, mask_list=list(MASK_WINDOWS), error=error)
        plot_corner(fit, outdir / "iras07546_jax_corner.png")
        plot_histograms(fit, outdir / "iras07546_jax_histograms.png")

    # The sampler's GPU buffers (sized by max_iterations) are no longer needed:
    # the results and the model live in host memory and the FITS file keeps a
    # full copy, so the card can be handed back to other jobs.
    fit.release_device_buffers()
    print("\n  released the sampler's device buffers")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())