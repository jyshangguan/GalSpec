"""Generate ``example/fit_iras07546_jax.ipynb``.

The notebook is the script ``fit_iras07546_jax.py`` broken into cells, with prose
between them.  Keeping the generator in the repository means the two stay in
step and the notebook can be regenerated after an edit::

    python example/_build_notebook.py

The notebook is *not* produced from the script text -- the cells below carry the
explanations that make them worth reading -- so edit them here.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "fit_iras07546_jax.ipynb"


imports = '''\
import sys
import time
import warnings
from pathlib import Path

# The repository is used straight from a checkout (it is not installed), so put
# its root on the path: this notebook lives in <repo>/example/.
REPO = Path.cwd()
if not (REPO / "galspec").is_dir():
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

import jax
import jax.numpy as jnp

import galspec
from galspec import jax_models as jxm
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary
from galspec.jnesty_fit_io import save_jnesty_fit_to_fits, load_jnesty_fit_from_fits
from galspec.utils import line_wave_dict as LINE_WAVE

warnings.simplefilter("ignore")
%matplotlib inline
'''

settings = '''\
# ---- how the fit is set up -------------------------------------------------
EXAMPLE_DIR = Path.cwd()
if not (EXAMPLE_DIR / "data" / "ir07546sed.fits").exists():
    EXAMPLE_DIR = REPO / "example"          # run from somewhere else
DATA_FILE = EXAMPLE_DIR / "data" / "ir07546sed.fits"

REDSHIFT = 0.0953          # NASA/IPAC value for IRAS 07546+3928
E_BV = 0.176               # Galactic extinction towards it (Schlafly & Finkbeiner)
WAVE_MIN, WAVE_MAX = 4200.0, 6800.0   # rest frame: H-gamma out to [S II] 6731

ERROR_FRACTION = 0.005     # floor on the pixel-to-pixel noise
PHOTON_FRACTION = 0.05     # 5% of the flux: photon noise of the source itself
# Nothing is masked any more: the blue end starts at H-gamma (so no Ca II H+K from
# the host), He I 5876 is fitted instead of masking Na D, and the red end stops
# before the atmospheric O2 A band.  The list stays because plot_fit takes one.
MASK_WINDOWS = []

NLIVE = 400                # live points the sampler keeps
DLOGZ = 0.01               # stop when < 1% of the evidence may remain unseen

BALMER_NAMES = ("Halpha", "Hbeta", "Hgamma")
TWO_COMPONENT_LINES = ("HeI",)      # narrow core + one broad component
SINGLE_LINES = ("HeII", "OIII4363")
'''

load = '''\
from astropy.io import fits
from extinction import ccm89, remove


def estimate_error(flux, window=751):
    """Per-pixel uncertainty measured from the spectrum itself.

    Subtract a running median (about 30 A), take the median of the absolute
    remainder in the same window -- which ignores the few pixels that belong to
    emission lines -- divide by 0.6745 to turn it into a Gaussian sigma, and
    smooth over a few hundred A.  A term of 5% of the flux is then added in
    quadrature: that is the photon noise of the source, which scales with the
    flux and dominates on the bright continuum.  It also stands for flux
    calibration and the shape of the Fe II template.
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


def load_spectrum():
    """Dereddened, rest-frame spectrum between WAVE_MIN and WAVE_MAX."""
    with fits.open(DATA_FILE) as hdul:
        header = hdul[0].header
        flux_array = np.asarray(hdul[0].data, dtype=float) * 1e14
    wave_array = header["CRVAL1"] + header["CD1_1"] * np.arange(flux_array.size)

    flux_array = remove(ccm89(wave_array, E_BV, r_v=3.1, unit="aa"), flux_array)  # Milky Way
    wave_array = wave_array / (1.0 + REDSHIFT)                      # observed -> rest frame

    use = (wave_array > WAVE_MIN) & (wave_array < WAVE_MAX)
    wave_array, flux_array = wave_array[use], flux_array[use]
    return wave_array, flux_array, mask_regions(wave_array, estimate_error(flux_array))


wave, flux, error = load_spectrum()
print(f"{wave.size:,} pixels, {wave[0]:.0f}-{wave[-1]:.0f} A rest frame, "
      f"{np.median(np.diff(wave)):.4f} A per pixel")
print(f"peak flux {flux.max():.2f}, line-free level {np.percentile(flux, 15):.2f}  "
      "[1e-14 erg/s/cm2/A]")
for lo, hi in ((4200, 4600), (4600, 5400), (5400, 6000), (6000, 6800)):
    sel = (wave > lo) & (wave < hi)
    print(f"  {lo}-{hi} A: median flux {np.median(flux[sel]):5.2f}, "
          f"median error {np.median(error[sel]):5.3f} "
          f"({100*np.median(error[sel]/flux[sel]):4.1f}% of the flux)")
'''

plot_data = '''\
fig, axs = plt.subplots(2, 1, figsize=(13.0, 5.4), sharex=True,
                        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06})
step = slice(None, None, 5)
axs[0].plot(wave[step], flux[step], color="0.35", lw=0.6, label="spectrum")
axs[0].plot(wave[step], error[step], color="#b5364a", lw=0.8, label="1-sigma error")
for (lo, hi) in MASK_WINDOWS:
    axs[0].axvspan(lo, hi, color="0.85", zorder=0)
    axs[1].axvspan(lo, hi, color="0.85", zorder=0)
top = np.percentile(flux, 99.5)
for centre, text in ((LINE_WAVE["Hgamma"], "H$\\\\gamma$"),
                     (LINE_WAVE["Hbeta"], "H$\\\\beta$"),
                     (LINE_WAVE["HeI_5876"], "He I"),
                     (LINE_WAVE["Halpha"], "H$\\\\alpha$")):
    axs[0].axvline(centre, color="#2b6cb0", lw=0.7, ls=":")
    axs[0].text(centre, top, " " + text, fontsize=9, color="#2b6cb0")
axs[0].set_ylabel("flux density  [1e-14 erg/s/cm$^2$/A]")
axs[0].set_title("IRAS 07546+3928: 4200-6800 A, fitted in one go")
axs[0].legend(fontsize=9, frameon=False, loc="upper right")
axs[1].plot(wave[step], 100*error[step]/np.abs(flux[step]), color="#b5364a", lw=0.8)
axs[1].set_ylabel("error [%]")
axs[1].set_xlabel("rest wavelength  [A]")
axs[1].set_ylim(0, 9)
plt.tight_layout()
'''

helpers = '''\
class _TieFrom:
    """Copy one parameter of another submodel (the plain-Gaussian case)."""

    def __init__(self, submodel, parameter):
        self.submodel = submodel
        self.parameter = parameter

    def __call__(self, model):
        return getattr(model[self.submodel], self.parameter)


def _tie_fn(backend, name):
    """``tie`` helpers of either backend (``jxm.tie.*`` or ``galspec.*``)."""
    return getattr(getattr(backend, "tie", backend), name)


def _power_law(backend):
    """The power-law class of either backend."""
    if backend is jxm:
        return jxm.PowerLaw1D
    from astropy.modeling import models as astro_models
    return astro_models.PowerLaw1D


def free_qualified_names(model):
    """``Submodel.parameter`` spelling of every free parameter."""
    pairs = zip(model.param_names, model.qualified_param_names)
    free = set(model.free_param_names)
    return [q for flat, q in pairs if flat in free]
'''

build = '''\
def build_model(backend=jxm, include_iron=None):
    """The full model: narrow core + two broad Gaussians per Balmer line.

    ``backend`` is either ``galspec.jax_models`` (what the fit uses) or
    ``galspec`` itself (the astropy equivalents).  Both expose the same
    constructors, tie helpers and parameter names, so one function builds the
    model twice: once to fit, once to hand to ``galspec.plot.plot_fit``.

    Each Balmer line is a ``Line_MultiGauss`` with ``n_components=3``: a narrow
    core (amp_c, dv_c, sigma_c), a broad component (amp_w0, dv_w0, sigma_w0) and
    a second, broader one (amp_w1, dv_w1, sigma_w1).  The wind amplitudes are
    ratios to the core amplitude, which is how the GalSpec models are written.
    No convolution is applied: see the introduction.
    """
    if include_iron is None:
        # the astropy twin used for plotting gets the iron as a curve instead
        include_iron = backend is jxm

    # -- continuum: a power law anchored at 5500 A, plus the iron pseudo-continuum --
    continuum = _power_law(backend)(amplitude=0.55, x_0=5500.0, alpha=1.0,
                                    fixed={"x_0": True}, name="Continuum")
    # The iron: an older version of this notebook fitted it with one velocity
    # dispersion and one shift,
    #
    #     iron = backend.IronTemplate_new(amplitude=0.2, sigma=2500.0, z=0.0,
    #                                     dv=0.0, name="FeII")
    #
    # and those two lines are kept here as the reference the tied model was checked
    # against.  What the fit uses is the *tied* template: it is broadened by a
    # multi-Gaussian kernel whose shifts, widths and weights ARE the broad
    # components of H-alpha (tied below), so the iron has a single free parameter,
    # its amplitude.
    iron = None
    if include_iron:
        iron = backend.IronTemplate_tied(n_components=2, amplitude=0.2, z=0.0,
                                         sigma_0=2400.0, dv_0=0.0, w_0=0.28,
                                         sigma_1=6000.0, dv_1=0.0, w_1=0.05,
                                         name="FeII")
        iron.z.fixed = True

    # -- Balmer lines: narrow core + two broad Gaussians each -------------------
    def balmer(name, wavec, amp_c, amp_w0):
        return backend.Line_MultiGauss(
            n_components=3, amp_c=amp_c, dv_c=0.0, sigma_c=700.0,
            wavec=wavec, name=name, amp_w0=amp_w0, dv_w0=0.0, sigma_w0=2400.0,
            amp_w1=0.05, dv_w1=0.0, sigma_w1=6000.0)

    halpha = balmer("Halpha", LINE_WAVE["Halpha"], 2.2, 0.28)
    hbeta = balmer("Hbeta", LINE_WAVE["Hbeta"], 0.8, 0.30)
    hgamma = balmer("Hgamma", LINE_WAVE["Hgamma"], 0.4, 0.30)

    # -- forbidden lines -------------------------------------------------------
    o3 = backend.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.8, amp_c1=1.8 / 2.98, dv_c=0.0, sigma_c=700.0,
        wavec0=LINE_WAVE["OIII_5007"], wavec1=LINE_WAVE["OIII_4959"],
        name="OIII", amp_w0=0.3, dv_w0=0.0, sigma_w0=1800.0)
    n2 = backend.Line_MultiGauss_doublet(
        n_components=1, amp_c0=1.0, amp_c1=1.0 / 2.96, dv_c=0.0, sigma_c=700.0,
        wavec0=LINE_WAVE["NII_6583"], wavec1=LINE_WAVE["NII_6548"], name="NII")
    # [S II]: the two peaks stay independent -- their ratio is a density
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

    # He I 5876: narrow core plus a broad foot, the same two-part treatment as the
    # Balmers but with one broad component.  It sits where the Na D mask used to be:
    # with the mask gone, the line has to be modelled rather than hidden.
    he1 = backend.Line_MultiGauss(
        n_components=2, amp_c=0.12, dv_c=0.0, sigma_c=700.0,
        wavec=LINE_WAVE["HeI_5876"], name="HeI",
        amp_w0=0.5, dv_w0=0.0, sigma_w0=2400.0)

    # Nothing blueward of H-gamma or redward of [S II] is modelled: the data are
    # 4200-6800 A.  Gone with the old range: H-delta, H-epsilon, H8-H11,
    # [Ne III] 3869, [O II] 3727, [Ar III] 7136, [O II] 7325.
    components = [continuum]
    if iron is not None:
        components.append(iron)
    components += [halpha, hbeta, hgamma, o3, n2, s2, o1, he2, o3_4363, he1]
    model = components[0]
    for _component in components[1:]:
        model = model + _component

    # -- ties: one narrow kinematics for every narrow line, [O III] as reference --
    for line in (halpha, hbeta, hgamma, n2, s2, o1, he1):
        line.dv_c.tied = _tie_fn(backend, "tie_MultiGauss_dv_c")("OIII")
        line.sigma_c.tied = _tie_fn(backend, "tie_MultiGauss_sigma_c")("OIII")
    for line in (he2, o3_4363):
        line.dv.tied = _TieFrom("OIII", "dv_c")
        line.sigma.tied = _TieFrom("OIII", "sigma_c")

    # -- broad components: two of them, each shared by the Balmer lines --------
    for line in (hbeta, hgamma, he1):
        line.dv_w0.tied = _tie_fn(backend, "tie_MultiGauss_dv_w0")("Halpha")
        line.sigma_w0.tied = _tie_fn(backend, "tie_MultiGauss_sigma_w0")("Halpha")
    for line in (hbeta, hgamma):
        line.dv_w1.tied = _tie_fn(backend, "tie_MultiGauss_dv_w1")("Halpha")
        line.sigma_w1.tied = _tie_fn(backend, "tie_MultiGauss_sigma_w1")("Halpha")
    o3.dv_w0.tied = _TieFrom("OIII", "dv_c")   # the wing sits at the core velocity

    # -- ties: the iron kernel IS broad H-alpha -------------------------------
    # Component k of the kernel takes the shift, width and weight of broad
    # component k of H-alpha.  The iron therefore cannot broaden independently of
    # the line it is emitted with.  The compound above already exists, which is what
    # lets these tie references resolve.
    if iron is not None:
        iron.dv_0.tied = _tie_fn(backend, "tie_MultiGauss_dv_w0")("Halpha")
        iron.sigma_0.tied = _tie_fn(backend, "tie_MultiGauss_sigma_w0")("Halpha")
        iron.w_0.tied = _tie_fn(backend, "tie_MultiGauss_amp_w0")("Halpha")
        iron.dv_1.tied = _tie_fn(backend, "tie_MultiGauss_dv_w1")("Halpha")
        iron.sigma_1.tied = _tie_fn(backend, "tie_MultiGauss_sigma_w1")("Halpha")
        iron.w_1.tied = _tie_fn(backend, "tie_MultiGauss_amp_w1")("Halpha")

    # -- doublet ratios fixed by atomic physics --------------------------------
    # (not [S II]: its ratio is density-sensitive and stays a free parameter)
    o3.amp_c1.tied = _tie_fn(backend, "tie_MultiGauss_doublet_ratio")("OIII", 2.98)
    n2.amp_c1.tied = _tie_fn(backend, "tie_MultiGauss_doublet_ratio")("NII", 2.96)
    o1.amp_c1.tied = _tie_fn(backend, "tie_MultiGauss_doublet_ratio")("OI", 3.00)

    return model


model = build_model(jxm)

# Display names for plots: a line's `name` stays "Halpha" (it is the key you index
# with), but `label` may contain maths, and param_labels combines it with the
# parameter symbol -- "BH$\\alpha$:$\\sigma_{w0}$" for the broad-width parameter.
MODEL_LABELS = {
    "Continuum": "Continuum", "FeII": "Fe II",
    "Halpha": r"BH$\\alpha$", "Hbeta": r"BH$\\beta$", "Hgamma": r"BH$\\gamma$",
    "OIII": r"[O III]", "NII": r"[N II]", "SII": r"[S II]", "OI": r"[O I]",
    "OIII4363": r"[O III] 4363", "HeII": r"He II", "HeI": r"He I",
}
for _name in model.submodel_names:
    if _name in MODEL_LABELS:
        model[_name].label = MODEL_LABELS[_name]
'''

show_params = '''\
print(f"{len(model.free_param_names)} free parameters "
      f"({len(model.qualified_param_names)} in total, the rest are fixed or tied):\\n")
for q in free_qualified_names(model):
    print("   ", q)
'''

priors = '''\
def log_uniform(lower, upper):
    """Prior that is uniform per decade between ``lower`` and ``upper``."""
    lo, hi = np.log10(lower), np.log10(upper)

    def transform(u):
        return 10.0 ** (lo + (hi - lo) * u)

    return transform


def make_priors(flux):
    """Bounds and log-uniform priors for the whole model."""
    peak = float(np.max(flux))
    continuum = float(np.percentile(flux, 15.0))
    amp = (1e-3, 3.0 * peak)      # line peaks
    ratio = (1e-3, 8.0)           # broad / narrow amplitude

    bounds = {
        "Continuum.amplitude": (0.2 * continuum, 4.0 * continuum),
        "Continuum.alpha": (-5.0, 5.0),
        # one free parameter for the iron: its amplitude.  No width or shift prior
        # is needed because the kernel is broad H-alpha.
        "FeII.amplitude": (1e-3, 3.0 * continuum),
        "OIII.dv_c": (-1000.0, 1000.0),
        "OIII.sigma_c": (30.0, 800.0),
        "OIII.amp_c0": amp,
        "OIII.amp_w0": ratio,
        "OIII.sigma_w0": (300.0, 4000.0),
        "NII.amp_c0": amp,
        "SII.amp_c0": amp, "SII.amp_c1": amp,
        "OI.amp_c0": amp,
        "Halpha.dv_w0": (-2500.0, 2500.0),
        "Halpha.sigma_w0": (300.0, 8000.0),
        "Halpha.dv_w1": (-6000.0, 6000.0),
        "Halpha.sigma_w1": (800.0, 25000.0),
    }
    for name in BALMER_NAMES:
        bounds[f"{name}.amp_c"] = amp
        bounds[f"{name}.amp_w0"] = ratio
        bounds[f"{name}.amp_w1"] = ratio
    for name in TWO_COMPONENT_LINES:
        bounds[f"{name}.amp_c"] = amp
        bounds[f"{name}.amp_w0"] = ratio
    for name in SINGLE_LINES:
        bounds[f"{name}.amplitude"] = amp

    log_params = [n for n in bounds
                  if n.endswith(("amp_c", "amp_c0", "amp_c1", "amp_w0", "amp_w1",
                                 "amplitude"))]
    return bounds, {n: log_uniform(*bounds[n]) for n in log_params}


bounds, custom_priors = make_priors(flux)
print(f"{len(bounds)} free parameters, each with an explicit prior:")
for q in free_qualified_names(model):
    lo, hi = bounds[q]
    kind = "log" if q in custom_priors else "linear"
    print(f"  {q:22s} ({lo:9.3f}, {hi:9.3f})  {kind}-uniform")
'''

fit_cell = '''\
fit = JNesty_Fit(model, wave, flux, error,
                 bounds_dict=bounds, custom_priors=custom_priors,
                 nlive=NLIVE, bound="multi", device="auto")

print(f"free parameters : {fit.ndim}")
print(f"device          : {fit.device}  [{jax_devices_summary()}]")
print(f"live points     : {fit.nlive}")
'''

prior_check = '''\
model.prepare(wave)
rng = np.random.default_rng(0)
u = rng.random((120, fit.ndim))
theta_prior = np.array([fit.prior_transform(row) for row in u])
draws = np.asarray(model.flux_batch(wave, theta_prior))

fig, ax = plt.subplots(figsize=(13.0, 3.8))
step = slice(None, None, 11)
ax.fill_between(wave[step], np.percentile(draws, 1, axis=0)[step],
                np.percentile(draws, 99, axis=0)[step], color="#2b6cb0",
                alpha=0.18, label="1-99% of 120 draws from the priors")
ax.plot(wave[step], flux[step], color="0.35", lw=0.6, label="data")
for centre in (LINE_WAVE["Hgamma"], LINE_WAVE["Hbeta"],
               LINE_WAVE["HeI_5876"], LINE_WAVE["Halpha"]):
    ax.axvline(centre, color="#b5364a", lw=0.6, ls=":")
ax.set_yscale("symlog", linthresh=1.0)
ax.set_xlabel("rest wavelength  [A]")
ax.set_ylabel("flux density  [1e-14 erg/s/cm$^2$/A]")
ax.set_title("The priors alone already bracket the spectrum (that is the point)")
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
'''

run_cell = '''\
started = time.perf_counter()
fit.fit(progress=False, dlogz=DLOGZ, max_iterations=200_000)
wall = time.perf_counter() - started

print(f"log Z          : {fit.log_evidence:.2f} +- {fit.log_evidence_err:.2f}")
print(f"iterations     : {fit.n_iterations}")
print(f"wall clock     : {wall/60:.1f} min")
'''

table_cell = '''\
PARAM_NOTES = {
    "Continuum.amplitude": "continuum flux density at 5500 A",
    "Continuum.alpha": "continuum slope (F_nu ~ nu^-alpha)",
    "FeII.amplitude": "iron strength (kernel tied to broad H-alpha)",
    "OIII.amp_c0": "[O III] 5007 narrow peak", "OIII.amp_w0": "[O III] wing / core",
    "OIII.dv_c": "shift of all narrow lines", "OIII.sigma_c": "width of all narrow lines",
    "OIII.sigma_w0": "width of the [O III] wing",
    "NII.amp_c0": "[N II] 6583 peak",
    "SII.amp_c0": "[S II] 6716 peak", "SII.amp_c1": "[S II] 6731 peak",
    "OI.amp_c0": "[O I] 6300 peak",
    "HeII.amplitude": "He II 4686 peak", "OIII4363.amplitude": "[O III] 4363 peak",
    "HeI.amp_c": "He I 5876 narrow peak", "HeI.amp_w0": "broad He I 5876 / narrow",
    "Halpha.dv_w0": "shift of broad component 0",
    "Halpha.sigma_w0": "width of broad component 0",
    "Halpha.dv_w1": "shift of broad component 1",
    "Halpha.sigma_w1": "width of broad component 1",
}
for name in BALMER_NAMES:
    PARAM_NOTES.setdefault(f"{name}.amp_c", f"narrow {name} peak")
    PARAM_NOTES.setdefault(f"{name}.amp_w0", f"broad {name} / narrow")
    PARAM_NOTES.setdefault(f"{name}.amp_w1", f"second broad {name} / narrow")

lo, med, hi = fit.get_quantiles((0.16, 0.5, 0.84))
print(f"{'parameter':22s} {'value':>10s} {'-':>8s} {'+':>8s}  meaning")
print("-" * 100)
for i, q in enumerate(fit.full_param_names):
    print(f"{q:22s} {med[i]:10.3f} {med[i]-lo[i]:8.3f} {hi[i]-med[i]:8.3f}  "
          f"{PARAM_NOTES.get(q, '')}")
'''

derived_cell = '''\
def derived_quantities(fit):
    """Line widths and integrated fluxes, computed per posterior sample."""
    samples = fit.samples_equal_weight
    index = {q: i for i, q in enumerate(fit.full_param_names)}
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))

    out = {}
    for line in BALMER_NAMES:
        wavec = LINE_WAVE[line]
        block = {}
        for label, comp in (("narrow", "c"), ("broad 0", "w0"), ("broad 1", "w1")):
            if comp == "c":
                amplitude = samples[:, index[f"{line}.amp_c"]]
                sigma = samples[:, index["OIII.sigma_c"]]
            else:
                amplitude = (samples[:, index[f"{line}.amp_c"]]
                             * samples[:, index[f"{line}.amp_{comp}"]])
                sigma = samples[:, index[f"Halpha.sigma_{comp}"]]
            # F = A sigma sqrt(2 pi) (lambda / c); A is in 1e-14 units and the
            # answer is quoted in 1e-17 erg/s/cm2, hence the 1e3.
            line_flux = amplitude * sigma * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3
            block[label] = {"fwhm": np.percentile(fwhm * sigma, [16, 50, 84]),
                            "flux": np.percentile(line_flux, [16, 50, 84])}
        broad = block["broad 0"]["flux"][1] + block["broad 1"]["flux"][1]
        block["broad/narrow"] = broad / block["narrow"]["flux"][1]
        out[line] = block

    # He I 5876: one broad component instead of two
    wavec = LINE_WAVE["HeI_5876"]
    narrow_amp = samples[:, index["HeI.amp_c"]]
    he1 = {}
    for label, amplitude, sigma in (
            ("narrow", narrow_amp, samples[:, index["OIII.sigma_c"]]),
            ("broad 0", narrow_amp * samples[:, index["HeI.amp_w0"]],
             samples[:, index["Halpha.sigma_w0"]])):
        line_flux = amplitude * sigma * np.sqrt(2.0 * np.pi) * wavec / 299792.458 * 1e3
        he1[label] = {"fwhm": np.percentile(fwhm * sigma, [16, 50, 84]),
                      "flux": np.percentile(line_flux, [16, 50, 84])}
    he1["broad/narrow"] = (he1["broad 0"]["flux"][1] / he1["narrow"]["flux"][1])
    out["He I 5876"] = he1

    # [S II] I(6716)/I(6731): ~1.4 in the low-density limit, ~0.4 at high density.
    ratio = samples[:, index["SII.amp_c0"]] / samples[:, index["SII.amp_c1"]]
    out["[S II] 6716/6731"] = {"ratio": np.percentile(ratio, [16, 50, 84])}
    return out


print(f"{'line':18s} {'component':10s} {'FWHM [km/s]':>20s}   "
      f"{'flux [1e-17 erg/s/cm2]':>24s}")
print("-" * 80)
for line, block in derived_quantities(fit).items():
    if "ratio" in block:
        r = block["ratio"]
        print(f"{line:18s} = {r[1]:.3f} (+{r[2]-r[1]:.3f}/-{r[1]-r[0]:.3f})   "
              "[density diagnostic]")
        continue
    for label in ("narrow", "broad 0", "broad 1"):
        if label not in block:
            continue
        fw, fl = block[label]["fwhm"], block[label]["flux"]
        print(f"{line:18s} {label:10s} {fw[1]:8.0f} (+{fw[2]-fw[1]:.0f}/-{fw[1]-fw[0]:.0f})"
              f"      {fl[1]:9.1f} (+{fl[2]-fl[1]:.1f}/-{fl[1]-fl[0]:.1f})")
    print(f"{'':18s} {'broad/narrow':10s} {block['broad/narrow']:8.2f}")
'''

quality_cell = '''\
model_best, _, theta_best = fit.get_best_fit()
predicted = np.asarray(model_best.flux(wave, theta_best))
residual = (flux - predicted) / error
print(f"pixels       : {wave.size:,}")
print(f"reduced chi2 : {np.sum(residual**2)/(wave.size - fit.ndim):.3f}")
print(f"residual rms : {np.std(residual):.2f} sigma")
print(f"worst pixel  : {np.max(np.abs(residual)):.1f} sigma")
'''

twin_cell = '''\
from astropy.modeling import Fittable1DModel


class TabulatedCurve(Fittable1DModel):
    """A curve already evaluated on a grid, presented as an astropy component.

    The iron is the one component without a like-for-like astropy twin: the CPU
    ``IronTemplate_tied`` convolves the template with an explicit kernel array,
    while the JAX model subtracts the intrinsic width in quadrature component by
    component. Rather than plot a slightly different iron, this wrapper carries the
    flux the fitted JAX model produces.
    """

    n_inputs = 1
    n_outputs = 1

    def __init__(self, wave_grid, flux_values, name=None, **kwargs):
        super().__init__(**kwargs)
        self._wave = np.asarray(wave_grid, dtype=float)
        self._flux = np.asarray(flux_values, dtype=float)
        if name is not None:
            self.name = name

    def evaluate(self, x):
        return np.interp(np.asarray(x, dtype=float), self._wave, self._flux)


def cpu_model_with_values(model_jax, wave_grid):
    """The astropy twin of the fitted model, at the best-fitting sample.

    ``galspec.plot.plot_fit`` walks an astropy compound model and draws every
    component, so the fitted JAX model is mirrored into its astropy equivalent.

    The values copied across are the *tie-resolved* ones (``param_dict()``): an
    astropy compound does not re-derive a tied parameter when it is evaluated --
    ties are a fitting-time constraint there -- so the twin has to be given the
    widths and shifts that the ties produced.

    The iron enters as a :class:`TabulatedCurve`: its flux is the JAX model's,
    found by subtracting the model with the iron amplitude set to zero.
    """
    resolved = model_jax.param_dict()
    theta = np.asarray(model_jax.get_free_params(), dtype=float)
    free = set(model_jax.free_param_names)
    free_qualified = [q for q, flat
                      in zip(model_jax.qualified_param_names, model_jax.param_names)
                      if flat in free]
    theta_no_iron = theta.copy()
    theta_no_iron[free_qualified.index("FeII.amplitude")] = 0.0
    iron_flux = (np.asarray(model_jax.flux(wave_grid, theta))
                 - np.asarray(model_jax.flux(wave_grid, theta_no_iron)))

    parts = list(build_model(galspec))                 # astropy leaves, no iron
    parts.insert(1, TabulatedCurve(wave_grid, iron_flux, name="FeII"))
    model_cpu = parts[0]
    for part in parts[1:]:
        model_cpu = model_cpu + part
    for name in ("Continuum", "FeII", "Halpha", "Hbeta", "Hgamma", "OIII", "NII",
                 "SII", "OI", "HeII", "OIII4363", "HeI"):
        if name in MODEL_LABELS:
            model_cpu[name].label = MODEL_LABELS[name]

    for flat, qualified in zip(model_jax.param_names, model_jax.qualified_param_names):
        value = resolved.get(flat)
        if value is None or qualified.startswith("FeII."):
            continue
        submodel, parameter = qualified.split(".", 1)
        getattr(model_cpu[submodel], parameter).value = float(value)
    return model_cpu


model_cpu = cpu_model_with_values(model_best, wave)
difference = np.max(np.abs(np.asarray(model_cpu(wave)) - predicted)) / np.max(predicted)
print(f"CPU/JAX parity: max difference {difference:.2e} of the peak flux")
assert difference < 1e-6, "the astropy twin does not reproduce the JAX model"
'''

plot_full = '''\
# residual_in_sigma divides by the error, so the panel shows "how many sigma" and
# a single bright pixel can no longer set the scale of the whole panel.
fig = plt.figure(figsize=(17.0, 8.0))
ax = fig.add_axes([0.055, 0.20, 0.82, 0.72])
axr = fig.add_axes([0.055, 0.07, 0.82, 0.13])
galspec.plot.plot_fit(
    wave, flux, model_cpu, ax=ax, axr=axr, mask_list=MASK_WINDOWS,
    error=error, residual_in_sigma=True,
    xlabel="Rest wavelength  [A]",
    ylabel=r"$F_\\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$ $\\AA^{-1}$]",
    legend_kwargs={"fontsize": 10.5, "ncol": 2, "loc": "center left",
                   "bbox_to_anchor": (1.005, 0.5), "frameon": False})
ax.set_title("IRAS 07546+3928: 4200-6800 A, two broad Gaussians per Balmer line, "
             "iron tied to broad H-alpha", fontsize=13)
'''

plot_zooms = '''\
zooms = [("H$\\\\gamma$ + H$\\\\beta$", (4230, 5100), (0.0, 2.4)),
         ("H$\\\\alpha$ + [N II] + [S II]", (6420, 6800), (0.0, 5.0)),
         ("He I 5876", (5750, 6020), (0.45, 1.35))]


def residual_limits(wave, flux, error, model, xlim):
    """Residual limits for one panel, in sigma, from that panel's own scatter."""
    inside = (wave >= xlim[0]) & (wave <= xlim[1])
    residual = ((flux - np.asarray(model(wave))) / error)[inside]
    scale = float(np.nanpercentile(np.abs(residual), 99.5))
    return (-1.25 * scale - 0.06, 1.25 * scale + 0.06)


for title, xlim, ylim in zooms:
    fig = plt.figure(figsize=(15.5, 7.0))
    ax = fig.add_axes([0.06, 0.20, 0.70, 0.72])
    axr = fig.add_axes([0.06, 0.07, 0.70, 0.13])
    # The legend sits outside the axes: drawn inside, an opaque frame covered the
    # [O II] 3727 peak in the blue-end panel -- the one feature that panel exists
    # to show.
    galspec.plot.plot_fit(
        wave, flux, model_cpu, ax=ax, axr=axr, xlim=xlim, ylim0=ylim,
        ylim1=residual_limits(wave, flux, error, model_cpu, xlim),
        mask_list=MASK_WINDOWS, error=error, residual_in_sigma=True,
        xlabel="Rest wavelength  [A]",
        ylabel=r"$F_\\lambda$  [$10^{-14}$ erg s$^{-1}$ cm$^{-2}$ $\\AA^{-1}$]",
        legend_kwargs={"fontsize": 9.0, "ncol": 1, "loc": "center left",
                       "bbox_to_anchor": (1.005, 0.5), "frameon": False})
    ax.set_title(f"{title}: {xlim[0]:.0f}-{xlim[1]:.0f} A", fontsize=12)
    plt.show()
'''

tied_iron_cell = '''\
# This is the model the fit uses, in isolation: the iron broadened by the *same*
# multi-Gaussian profile as the broad Balmer lines, tied component by component.
# The cell builds it, resolves the ties, compares the resulting iron profile with
# the single-Gaussian one, and measures what the extra work costs.
fe_tied = jxm.IronTemplate_tied(n_components=2, amplitude=0.2, sigma_0=2400.0,
                                sigma_1=6000.0, w_0=0.28, w_1=0.05,
                                template_name="park2022", name="FeII_tied")
# a standalone copy of the broad H-alpha line, so this demonstration cannot touch
# the model that the fit uses
halpha = jxm.Line_MultiGauss(n_components=3, amp_c=2.2, sigma_c=700.0,
                             wavec=LINE_WAVE["Halpha"], name="Halpha",
                             amp_w0=0.28, sigma_w0=2400.0,
                             amp_w1=0.05, sigma_w1=6000.0)
# compose first: a tie is resolved through the compound that holds both models
demo = halpha + fe_tied
fe_tied.dv_0.tied = jxm.tie.tie_MultiGauss_dv_w0("Halpha")
fe_tied.sigma_0.tied = jxm.tie.tie_MultiGauss_sigma_w0("Halpha")
fe_tied.w_0.tied = jxm.tie.tie_MultiGauss_amp_w0("Halpha")
fe_tied.dv_1.tied = jxm.tie.tie_MultiGauss_dv_w1("Halpha")
fe_tied.sigma_1.tied = jxm.tie.tie_MultiGauss_sigma_w1("Halpha")
fe_tied.w_1.tied = jxm.tie.tie_MultiGauss_amp_w1("Halpha")

print("tied iron parameters:", fe_tied.param_names)
print("free after tying   :", fe_tied.free_param_names)
print("(dv_*, sigma_*, w_* are all tied to the broad components of H-alpha)")

# iron profile: tied to the line profile versus one Gaussian of the same width
theta_demo = jnp.asarray(demo.get_free_params())       # ties resolved by the compound
theta_halpha = jnp.asarray(halpha.get_free_params())   # the line on its own
iron_tied = (np.asarray(demo.flux(wave, theta_demo))
             - np.asarray(halpha.flux(wave, theta_halpha)))
# the single-Gaussian description, for reference (the fit no longer uses it)
iron_plain = np.asarray(
    jxm.IronTemplate_new(amplitude=0.2, sigma=2500.0, dv=0.0,
                         name="FeII_plain").flux(wave, jnp.asarray([0.2, 2500.0, 0.0])))
print(f"tied kernel changes the iron profile by at most "
      f"{np.max(np.abs(iron_tied - iron_plain)) / np.max(iron_plain) * 100:.0f}% of its peak")

sel = (wave > 4200) & (wave < 5600)
fig, ax = plt.subplots(figsize=(11.0, 4.5))
ax.plot(wave[sel], iron_plain[sel], color="C0", lw=1.8,
        label="one Gaussian, sigma = 2500 km/s")
ax.plot(wave[sel], iron_tied[sel], color="C1", lw=1.8, ls="--",
        label="tied to the H-alpha broad profile")
ax.set_xlabel("Rest wavelength  [A]", fontsize=16)
ax.set_ylabel("Fe II flux", fontsize=16)
ax.legend(fontsize=10, frameon=False)
ax.set_title("Same template, two descriptions of its broadening", fontsize=12)
plt.show()

# What it costs.  Both compounds carry the same H-alpha line, so the difference is
# the iron cost.  Both are compiled and run once *before* timing: a GPU idles at low
# clock speed and ramps up under load, so timing the first model cold would credit
# the second one with a speed-up it does not have.
x = jnp.asarray(wave)
demo_plain = halpha.copy() + jxm.IronTemplate_new(amplitude=0.2, sigma=2500.0,
                                                  dv=0.0, name="FeII_plain")
evaluate_plain = jax.jit(lambda t: demo_plain.flux(x, t))
evaluate_tied = jax.jit(lambda t: demo.flux(x, t))
theta_plain = jnp.asarray(demo_plain.get_free_params())
for _ in range(3):                       # compile, then warm the clocks
    evaluate_plain(theta_plain).block_until_ready()
    evaluate_tied(theta_demo).block_until_ready()


def best_ms(evaluate, theta, reps=20, blocks=5):
    """Fastest of several blocks: the median is inflated by any stray GPU contention."""
    times = []
    for _ in range(blocks):
        start = time.perf_counter()
        for _ in range(reps):
            evaluate(theta).block_until_ready()
        times.append((time.perf_counter() - start) / reps * 1e3)
    return min(times)


# each model is measured twice, in alternating order: if the clock is still
# ramping, the two runs disagree and the faster one is the honest number
plain_first = best_ms(evaluate_plain, theta_plain)
tied_first = best_ms(evaluate_tied, theta_demo)
tied_second = best_ms(evaluate_tied, theta_demo)
plain_second = best_ms(evaluate_plain, theta_plain)
plain_ms = min(plain_first, plain_second)
tied_ms = min(tied_first, tied_second)
extra_min = (tied_ms - plain_ms) * 1e5 / 6e4        # per 1e5 evaluations
print(f"single-Gaussian iron: {plain_ms:.2f} ms per evaluation")
print(f"tied (2 components) : {tied_ms:.2f} ms per evaluation")
print(f"cost of the tied kernel: {tied_ms - plain_ms:+.2f} ms "
      f"({100 * (tied_ms - plain_ms) / plain_ms:+.0f}%), i.e. {extra_min:.1f} min "
      f"over 1e5 evaluations")
'''
histogram_cell = '''\
# Most of the time the marginals are all that is wanted, and a corner plot of thirty
# parameters is unreadable.  Each panel shows the histogram, the median and the 16-84
# percentile band, with the value printed on top.  The x range comes from the
# posterior, not from the prior: if it came from the prior, a parameter measured to
# 0.5% would be drawn as a spike one pixel wide.  When a prior bound is off the
# plotted range it is written in the corner instead of stretching the axis.
quoted = ["Halpha.amp_c", "Halpha.amp_w0", "Halpha.dv_w0", "Halpha.sigma_w0",
          "Halpha.sigma_w1", "Hbeta.amp_c", "OIII.sigma_c", "OIII.dv_c",
          "HeI.amp_c", "HeI.amp_w0", "SII.amp_c0", "SII.amp_c1"]
fig, axes = galspec.plot.plot_posterior_hist(
    fit, params=[q for q in quoted if q in fit.full_param_names], ncols=4, bins=40)
plt.show()
'''
corner_cell = '''\
import corner

wanted = ["OIII.dv_c", "OIII.sigma_c", "Halpha.dv_w0", "Halpha.sigma_w0",
          "Halpha.dv_w1", "Halpha.sigma_w1", "Continuum.amplitude", "Continuum.alpha",
          "FeII.amplitude", "Halpha.amp_c", "Halpha.amp_w0", "Halpha.amp_w1",
          "HeI.amp_c", "HeI.amp_w0"]
keep = [fit.full_param_names.index(name) for name in wanted
        if name in fit.full_param_names]
# param_labels turns "Halpha.sigma_w0" into "BH$\\alpha$:$\\sigma_{w0}$"
fig = corner.corner(fit.samples_equal_weight[:, keep],
                    labels=[fit.param_labels[i] for i in keep],
                    quantiles=(0.16, 0.5, 0.84), show_titles=True, title_fmt=".3g",
                    title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 9})
'''

save_cell = '''\
(EXAMPLE_DIR / "output").mkdir(exist_ok=True)
outfile = EXAMPLE_DIR / "output" / "iras07546_jax_posterior.npz"
np.savez_compressed(outfile,
                    samples=fit.samples_equal_weight,
                    parameter_names=np.array(fit.param_names),
                    qualified_names=np.array(fit.full_param_names),
                    log_evidence=fit.log_evidence,
                    log_evidence_error=fit.log_evidence_err,
                    wave=wave, flux=flux, error=error, predicted=predicted)
print(f"wrote {outfile}")

# The FITS file is the complete record of the fit: the samples (and their
# unit-cube coordinates), the evidence, the bounds, the sampler's convergence
# trajectory, the model itself and the data it was made against.
fits_out = EXAMPLE_DIR / "output" / "iras07546_jax_fit.fits"
save_jnesty_fit_to_fits(fit, fits_out)
print(f"wrote {fits_out}  ({fits_out.stat().st_size / 1e6:.1f} MB)")
'''

reload_cell = '''\
# Reading it back gives a JNesty_Fit with the same posterior, evidence, bounds and
# model -- and no sampler. This is the file worth keeping.
loaded = load_jnesty_fit_from_fits(fits_out)

print(f"log Z stored    {fit.log_evidence:12.2f} +/- {fit.log_evidence_err:.2f}")
print(f"log Z reloaded  {loaded.log_evidence:12.2f} +/- {loaded.log_evidence_err:.2f}")
print("same parameters:  ", list(loaded.param_names) == list(fit.param_names))
print("same posterior:   ",
      np.allclose(loaded.samples_equal_weight, fit.samples_equal_weight))

# everything the original object could do, it can do again
quantiles = loaded.get_quantiles((0.16, 0.5, 0.84))
print()
print(f"{'parameter':22s} {'16%':>10s} {'50%':>10s} {'84%':>10s}")
for index, name in enumerate(loaded.full_param_names[:6]):
    lo, mid, hi = quantiles[:, index]
    print(f"{name:22s} {lo:10.4g} {mid:10.4g} {hi:10.4g}")
'''

free_cell = '''\
# The sampler still holds buffers on the GPU sized by max_iterations, and an XLA
# context never gives its high-water mark back to the driver. Everything so far
# lives in host memory (and in the two files), so hand the card back to other
# jobs; the results stay usable.
fit.release_device_buffers()
print("sampler released; quantiles still available:", fit.get_quantiles().shape)
'''

cells = [
    ("md", """# Fitting IRAS 07546+3928 with JNesty on the GPU

This notebook fits one real spectrum with the GPU-native side of GalSpec: JAX
models (`galspec.jax_models`) sampled by **JNesty** (`galspec.jnesty_fit`). It is
the notebook form of `fit_iras07546_jax.py`; the script is what to use in a batch
pipeline, the notebook is for looking at what happens.

## What is fitted

**4200-6800 A** of the quasar **IRAS 07546+3928** (z = 0.0953) in the rest
frame -- 56 956 pixels, from H-gamma to the [S II] doublet. All of it is fitted at
once, so the continuum and the lines are constrained by the same posterior:

| wavelength | what | model |
| --- | --- | --- |
| 4340 A | H-gamma | narrow + **two** broad Gaussians |
| 4861 A | H-beta | narrow + **two** broad Gaussians |
| 6563 A | H-alpha | narrow + **two** broad Gaussians |
| 5876 A | He I | narrow + **one** broad component |
| 4959 / 5007 A | [O III] | narrow core + one broad wing |
| 6548 / 6584 A | [N II] | narrow, doublet ratio 2.96 (atomic physics) |
| 6716 / 6731 A | [S II] | narrow, **both peaks free** (the ratio is a density diagnostic) |
| 6300 / 6364 A | [O I] | narrow, doublet ratio 3.00 |
| 4686, 4363 A | He II, [O III] | single narrow Gaussian |
| all | continuum | power law + Fe II template |

Nothing outside that window is fitted: no H-delta or higher Balmer lines, no
[Ne III] 3869 or [O II] 3727, no [Ar III] 7136 or [O II] 7325. Stopping at
H-gamma on the blue side and at [S II] on the red side means **no masks are
needed** -- the Ca II H+K absorption of the host galaxy is below 4200 A, and the
atmospheric O2 A band is beyond 6800 A. The Na D window, which used to be masked,
now stays in the fit: He I 5876 sits inside it and is modelled instead.

## How it is fitted

* **No least-squares step.** Nothing is optimised before nested sampling starts,
  so the priors cannot inherit a preliminary answer. JNesty walks the prior
  directly.
* **No instrumental smoothing.** At 0.046 A per pixel the spectrum grossly
  oversamples the spectrograph: R = 2000 means ~150 km/s of instrument, while the
  narrow lines here are ~700 km/s. So the model is *not* convolved, exactly as in
  `wrk_sagan_iras07546.ipynb`; the fitted narrow width is the observed one.
* **Wide priors, taken from the data.** Line peaks from invisible to 3x the
  brightest pixel (log-uniform); broad components 300-8000 and 800-25000 km/s.
  Section 4 shows the prior spread against the data.
* **Physical ties.** Every narrow line shares one velocity shift and one width
  (measured from [O III]); the two broad components are each shared across the
  three Balmer lines; and the Fe II template is broadened by broad H-alpha itself.
  29 free numbers therefore describe twelve components.

Runtime: 8-15 minutes on one GPU with `nlive=400, dlogz=0.01` (fewer pixels and
fewer parameters than the previous 3680-7500 A version, so it is quicker). For a
quick look set `NLIVE = 200` and `DLOGZ = 0.1` in the settings cell and expect
wider error bars."""),

    ("md", """## 1. Imports and settings

Everything a user might want to change is in the next cell: the redshift, the
extinction, the wavelength window, and the two sampler settings."""),

    ("code", imports),
    ("code", settings),

    ("md", """## 2. The data

The FITS file that ships with GalSpec has flux only, no uncertainty, so the
uncertainty is measured from the data: subtract a running median, take the median
absolute deviation in a 30 A window (which ignores the few pixels belonging to
emission lines), convert to a Gaussian sigma, and smooth. That gives a
wavelength-dependent noise level, shown in the lower panel below, which rises
from ~0.3% of the flux at 4700 A to ~4% at the red end.

On top of that, **5% of the flux itself** is added in quadrature. This is the
photon noise of the source: the counts scale with the flux, so the uncertainty
does, and on the bright continuum it dominates everything else. It also covers
what the model does not describe exactly (flux calibration, the shape of the Fe
II template, departures from a pure power law), which is why 5% is the honest
number for a template fit rather than a fudge factor.

**No window is masked.** The three masks an earlier version needed are gone with
the wavelength cut: the host galaxy's Ca II H+K absorption is below 4200 A, the
Na D window is kept because He I 5876 inside it is now fitted as a line, and the
atmospheric O2 A band is beyond 6800 A.

Units: fluxes in `1e-14 erg/s/cm2/A`, velocities in km/s."""),

    ("code", load),
    ("code", plot_data),

    ("md", """## 3. The model

Twenty components, added together:

1. a **power-law continuum** anchored at 5500 A -- this is what fixes the
   continuum level and slope from the line-free regions of the whole spectrum;
2. an **Fe II template** -- the forest of iron lines that fakes a broad bump
   blueward of H-beta and is routinely mistaken for continuum. It uses
   `IronTemplate_tied`: the template is broadened by the *same* multi-Gaussian
   kernel as broad H-alpha, tied component by component, so the iron has a single
   free parameter, its amplitude. An earlier version fitted it with one velocity
   dispersion (`IronTemplate_new`); that is still available, and the two agree
   exactly when the kernel has one component. Tying it to the line is the more
   physical choice: the iron and the Balmer lines are emitted by the same gas, so
   they cannot have different kinematics;
3. **H-alpha, H-beta, H-gamma**, each a `Line_MultiGauss` with three Gaussians: a
   narrow core (`amp_c`, `dv_c`, `sigma_c`), a broad component (`amp_w0`,
   `dv_w0`, `sigma_w0`) and a second, broader one (`amp_w1`, `dv_w1`,
   `sigma_w1`). The wind amplitudes are *ratios to the core amplitude*, which is
   how the GalSpec models are written;
4. **He I 5876**, a narrow core plus **one** broad component. It is what used to
   be inside the Na D mask, and it has to be modelled now that the mask is gone;
5. **[O III] 4959/5007** as a doublet with a narrow core and one broad wing;
6. **[N II], [S II], [O I]** doublets, plus single narrow Gaussians for **He II
   4686** and **[O III] 4363**.

The ties are what make this affordable:

* one `dv_c` and one `sigma_c` for **all** narrow lines, taken from the [O III]
  core (forbidden lines come from the same low-density gas);
* `dv_w0`/`sigma_w0` shared by the three Balmer lines and by He I, and
  `dv_w1`/`sigma_w1` by the three Balmers: the two broad components are separate
  physical structures, so they get separate kinematics, but each is common to the
  lines. The Fe II kernel's two components are tied to these outright, shifts,
  widths and weights alike;
* doublet ratios fixed by atomic physics for [O III] (2.98), [N II] (2.96) and
  [O I] (3.00) -- but **not** for [S II], whose 6716/6731 ratio is a density
  diagnostic and therefore stays free.

Note there is no `* multi` polynomial here: the reference notebook multiplies one
in because it uses a least-squares fitter with no error model, whereas here the 5%
flux term already absorbs flux-calibration error, and a high-order polynomial on
top of a power law plus an Fe II template would be strongly degenerate with both."""),

    ("code", helpers),
    ("code", build),
    ("code", show_params),

    ("md", """### Why the iron is broadened by the line profile

That is the model the fit uses. The kernel is a multi-Gaussian whose shifts,
widths and weights are tied to the broad components of H-alpha, and the
normalisation keeps the iron scale separate from the line amplitudes. The cell
below builds the same construction standalone, compares the resulting iron
profile with the single-Gaussian one, and measures what the extra work costs.

The arithmetic is simple: subtract the template's own width in quadrature for
each component, then average the shifted profiles. It is **not free**: a
two-component kernel adds 0.1-0.45 ms per evaluation depending on how busy the
card is -- a bigger share of a small model than of a full spectrum, so the
percentage below depends on what else is in the model -- and that is up to a few
extra minutes over a whole fit. The
price is one smoothing and one interpolation per component, done on the 801-point
template rather than on the spectrum. Run the cell below and it measures the
number where you are.

What it *does* change is the physics: the iron gets the same velocity structure as
the Balmer lines, including any asymmetry, so it is the right model when the two
are known to share a broad-line region and the wrong one when they are not."""),

    ("code", tied_iron_cell),

    ("md", """## 4. The priors -- broad, and justified by the data

This is the part that decides whether a fit is honest. Every range below comes
from the spectrum itself, never from a previous fit:

| what | range | why |
| --- | --- | --- |
| line peaks | 1e-3 to 3x the brightest pixel, **log-uniform** | four decades: from invisible to implausibly strong |
| broad/narrow ratio | 1e-3 to 8, log-uniform | from no broad component to one that dominates |
| narrow widths | 30-800 km/s | unresolved up to the widest known narrow-line region |
| broad component 0 | 300-8000 km/s | FWHM up to ~19000 km/s |
| broad component 1 | 800-25000 km/s | FWHM up to ~59000 km/s, for a line wing |
| continuum level | 0.2-4x the line-free windows | cannot be negative, cannot swamp the lines |
| He I broad/narrow | 1e-3 to 8, log-uniform | same ratio prior as the Balmers |
| iron amplitude | 1e-3 to 3x the continuum level | the only free iron parameter |
| [S II] 6716/6731 | both peaks 1e-3 to 3x peak | the ratio is measured, not assumed |

One subtlety: bounds passed in `bounds_dict` are *linear*-uniform by design, and a
linear prior over four decades would put 99% of its probability at the bright
end. Amplitudes therefore get an explicit `log_uniform` callable through
`custom_priors`, so every factor of ten is equally likely."""),

    ("code", priors),

    ("md", """### Does the prior cover the data?

A useful sanity check, and the reason no least-squares seed is needed: draw
parameter sets from the prior, evaluate the model for each, and look at the
spread. If the data fall inside that band, the prior contains the answer. (The
band is wide on purpose -- that is what "broad prior" means.)"""),

    ("code", fit_cell),
    ("code", prior_check),

    ("md", """## 5. Run the sampler

`JNesty_Fit` compiles the log-likelihood and the prior transform with `jax.jit`,
batches likelihood evaluations on the GPU, and runs nested sampling until the
remaining evidence is below `dlogz`.

Nested sampling keeps `NLIVE` "live points" drawn from the prior and repeatedly
replaces the worst one with a point of higher likelihood: the volume shrinks, the
likelihood rises, and the integral of both is the evidence. The surviving points
are a weighted sample of the posterior, which is what we quote below.

JNesty seeds itself internally, so reruns agree statistically: two separate runs
of the equivalent script came out identical to the second decimal in `log Z`."""),

    ("code", run_cell),

    ("md", """## 6. What the fit says

The table is the answer: posterior median and the 16th/84th percentiles (a 68%
credible interval, the Bayesian counterpart of "1 sigma").

Read it with the ties in mind -- `OIII.dv_c` is the shift of *every* narrow line,
and `Halpha.sigma_w0` is the width of broad component 0 in *all three* Balmer
lines, in He I and in the Fe II kernel."""),

    ("code", table_cell),

    ("md", """### Widths and fluxes

Widths and fluxes are what a paper quotes, and they are combinations of
parameters, so they are computed **per posterior sample** rather than from the
medians -- that is how their error bars inherit the correlations in the fit.

`FWHM = 2.3548 sigma`. The integrated flux of a Gaussian of peak `A` and width
`sigma` at wavelength `lambda` is `A sigma sqrt(2 pi) lambda / c`, converted here
to the customary `1e-17 erg/s/cm2`.

`FWHM` for the narrow components is the *observed* width (no LSF was removed)."""),

    ("code", derived_cell),

    ("md", """## 7. How good is the fit?

`chi2/dof` near 1 means the model explains the data to within the error bars. Far
below 1 would mean the error bars were inflated; far above 1 means the model is
missing something.

The 5% flux term sets the scale of this number, and it is worth knowing which way
it cuts. A chi2/dof **above** 1 would mean the model cannot reach the error bar --
something is missing. What this fit gets is **0.32**, which says the opposite: the
5% term is *larger* than the true pixel-to-pixel noise (about 2-3% here), so the
residuals are 0.57 sigma on average and the posterior is correspondingly
conservative -- parameter errors are wider than the photon statistics alone would
give, which is the safe direction for a template fit that also carries flux
calibration and template shape uncertainty. Lower `PHOTON_FRACTION` to ~0.02 if
you want error bars set by pixel noise alone; raise it if you distrust the
spectrophotometry more than that.

If you need chi2/dof ~ 1 for its own sake, raise `PHOTON_FRACTION` -- but that
would be fitting the error bar, not the spectrum.

**The residuals are not white noise**, and a chi-square alone hides that: binned
over a few hundred pixels the mean residual wanders by more than pure noise would
allow, and neighbouring pixels are correlated. A 5% error bar cannot make that go
away -- it is the model, not the noise, that is coherent over tens of pixels. So
the rms below describes the pixel-to-pixel scatter, not the accuracy of the
model. Treat the error bars on individual parameters as statistical, and add a
systematic of order the worst binned residual when comparing with measurements
from other instruments."""),

    ("code", quality_cell),

    ("md", """## 8. The fit, drawn with GalSpec's own plotting function

`galspec.plot.plot_fit` takes the model and draws the data, the total model,
**every component**, the dashed sub-components of each multi-Gaussian line (so
the narrow core and the two broad Gaussians of each Balmer line can be told
apart) and the residual panel.

It needs an *astropy* compound model, so the fitted JAX model is mirrored into its
astropy equivalent first -- every component except the iron, which enters as the
curve the JAX model produces (the CPU tied-iron convolves with an explicit kernel
array, a different construction). The two are then compared: if they did not
agree, the plot would be a lie.

Two details of the figure are deliberate. The residual panel is drawn in **units
of sigma** (`residual_in_sigma=True`), so a residual of 3 means three error bars and
the stated rms can be checked by eye -- in raw flux units a single bright pixel
sets the scale of the whole panel. And the legend is placed **outside the axes**:
twenty components do not fit inside a panel without covering the data, and while
the frame was inside it sat exactly on top of a line peak in the zoom panels."""),

    ("code", twin_cell),
    ("code", plot_full),

    ("md", """### Zooms

The whole-spectrum figure is crowded; the same function, called on a wavelength
window, shows the components where they matter."""),

    ("code", plot_zooms),

    ("md", """### Correlations

Nested sampling gives the full posterior, not just error bars. An elongated blob
means two parameters trade off against each other -- a broader component with a
smaller peak, say -- and quoting them independently would be misleading."""),

    ("md", """### Marginal posteriors

The corner plot shows every pair; for the numbers people actually quote, one
histogram each is easier to read. `plot_posterior_hist` draws the marginal
posterior, marks the median and the 16-84 band, and prints the value on top.

The **x range comes from the posterior, not from the prior**. That matters: the
prior on a width spans 300-8000 km/s while the posterior spans a few km/s, so an
axis stretched to the prior would draw the measurement as a spike one pixel wide
and hide the shape that the panel exists to show. A parameter that has genuinely
filled its prior still looks flat, because the histogram itself is flat -- and
when a prior bound falls outside the plotted range its value is written in the
corner, so nothing is hidden by the zoom."""),

    ("code", histogram_cell),

    ("code", corner_cell),

    ("md", """## 9. What this does and does not show

* The blue end of the fitted range, 4200-4400 A, is the weakest part of the
  model: H-gamma is cut by the window edge on its blue wing, and there is no
  Balmer continuum, which rises towards its 3646 A limit and is not modelled at
  all. Trust H-gamma's broad flux less than H-alpha's.
* The two broad Gaussians are a *description* of a profile, not a physical
  decomposition. Real broad-line profiles have longer wings than any Gaussian, so
  read them as "a broad part and a broader part" and treat their individual
  fluxes as indicative. Because the iron kernel is tied to them, that caveat
  propagates to the iron profile too.
* Without the LSF folded in, the narrow widths include the instrument (~150 km/s
  FWHM, small next to ~700 km/s but not zero). Subtract it in quadrature if you
  need the gas kinematics alone.
* The narrow lines share one width by construction -- the standard assumption for
  forbidden-line regions, but it means a genuinely different narrow kinematics
  per line could not be detected.
* The [S II] ratio is measured, not assumed, because it is a density diagnostic:
  I(6716)/I(6731) near 1.3 means low-density gas, near 0.5 high-density. Turning
  it into an electron density needs a temperature and the atomic data, which is
  beyond a line-profile fit.
* The Fe II *shape* is still a template: tying its kernel to broad H-alpha fixes
  the broadening, but the relative strengths of the iron lines come from the
  reference object (Mrk 493). A real AGN with a different iron excitation would
  need a different template, not a different width.
* He I 5876 is fitted with one broad component; the line is weak, so its broad
  fraction is the least well constrained number in the table.
* The evidence (`log Z`) is only comparable between runs made with the same
  sampler.

## Where to go next

* `python fit_iras07546_jax.py` -- the same fit as a batch script.
* `python fit_iras07546_jax.py --quick` -- a rough answer in a few minutes.
* `docs/source/jax_gpu.rst` -- the CPU/GPU benchmark built on a fit like this.
* `tests/test_jax_parity.py` -- proves the JAX models equal the astropy ones.
* `tests/test_jax_io.py` -- proves the FITS round trip gives back the same fit."""),

    ("md", """## 10. Saving the fit so it can be reused

A nested-sampling run is expensive; the file should be the thing you keep. Two
files are written:

* `output/iras07546_jax_posterior.npz` -- a plain array dump, handy for a quick
  numpy read;
* `output/iras07546_jax_fit.fits` -- everything: the posterior samples, their
  unit-cube coordinates, the evidence and its error, the parameter names, bounds
  and priors, the sampler's convergence trajectory, the model description and the
  spectrum that was fitted.

The FITS layout is the one `galspec.dynesty_fit_io` already uses (`SAMPLES`,
`LOGWT`, `LOGL`, `LOGVOL`, `SAMPLES_EQ`, `PARAM_NAMES`, `PARAM_BOUNDS`,
`MODEL_PARAM`, `MODEL_STR`, ...), with three JNesty-only extensions: `SAMPLES_U`,
`DLZ_TRAJ` and `SCALE_TR`. The model is written by
`galspec.jax_models.serialization`, which records what a JAX model needs and an
astropy model does not -- the number of Gaussians per line, a template name or a
polynomial degree -- and the ties as *references* (`"OIII.dv_c"`), so they are
restored no matter how they were written in the code."""),

    ("code", save_cell),

    ("md", """### Reading it back

`load_jnesty_fit_from_fits` rebuilds the model and the fit, and every summary and
plot works on the result. The one thing it cannot rebuild is the prior
*transform*: `custom_priors` are Python callables and a closure cannot be written
to a file, so the file stores the resulting bounds and the names that had a
custom prior. A loaded fit is therefore for reading results, not for resuming the
sampler."""),

    ("code", reload_cell),

    ("md", """### Handing the GPU back

JNesty's device memory is dominated by one buffer that is allocated once at
`max_iterations` size (the dead-point history), plus the live set and the
compiled kernels. It does not grow from iteration to iteration, but an XLA
context never returns its high-water mark to the driver either, so a process that
has finished one fit keeps showing it in `nvidia-smi`. `release_device_buffers()`
drops the sampler and clears JAX's caches; the results, the model and the saved
files are untouched."""),

    ("code", free_cell),
]


def _backup_executed(path: Path) -> None:
    """Keep a copy of the notebook being replaced, if it has been executed.

    Regenerating the notebook discards its outputs, and an executed notebook is
    the result of a sampler run -- ten minutes of GPU.  A copy next to it (ignored
    by git) means that cannot be lost by accident.
    """
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text())
    except Exception:                                   # pragma: no cover - defensive
        return
    has_output = any(cell.get("outputs") for cell in existing.get("cells", []))
    if has_output:
        backup = path.with_name(path.stem + ".executed.ipynb")
        backup.write_text(path.read_text())
        print(f"kept the executed copy at {backup.name}")


def main() -> int:
    cells_out = []
    for index, (kind, source) in enumerate(cells):
        text = source if source.endswith("\n") else source + "\n"
        cell_id = f"cell-{index:02d}"
        if kind == "md":
            cells_out.append({"cell_type": "markdown", "id": cell_id, "metadata": {},
                              "source": text.splitlines(keepends=True)})
        else:
            cells_out.append({"cell_type": "code", "id": cell_id,
                              "execution_count": None, "metadata": {}, "outputs": [],
                              "source": text.splitlines(keepends=True)})
    nb = {
        "cells": cells_out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _backup_executed(OUT)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cells_out)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())