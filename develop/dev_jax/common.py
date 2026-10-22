"""Shared model definitions for the JAX backend demos, tests and benchmarks.

Builds the *same* physical models twice -- once with the astropy/GalSpec CPU
classes and once with :mod:`galspec.jax_models` -- so their fluxes and
likelihoods can be compared directly at identical parameter values.

The model definitions are taken from the existing examples:
``develop/dev_dynesty/demo_ir07546.py`` ([O III] window) and
``example/wrk_galspec_iras07546_dynesty.ipynb`` (full spectrum).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import galspec
from galspec import jax_models as jxm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_FILE = ROOT / "example" / "data" / "ir07546sed.fits"

LINE_WAVE = galspec.utils.line_wave_dict
LINE_LABEL = galspec.utils.line_label_dict


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_spectrum(oversample: int = 1, wave_min: float = 4900.0, wave_max: float = 5100.0):
    """Load IRAS 07546+3928 as in the dynesty demo (dereddened, rest frame)."""
    from astropy.io import fits
    from extinction import ccm89, remove

    with fits.open(DATA_FILE) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=float) * 1e14
    wave = header["CRVAL1"] + header["CD1_1"] * np.arange(flux.size)
    flux = remove(ccm89(wave, 0.176, r_v=3.1, unit="aa"), flux)
    wave = wave / (1.0 + 0.0953)
    use = (wave > wave_min) & (wave < wave_max)
    wave, flux = wave[use], flux[use]
    error = np.maximum(0.05 * np.abs(flux), 1e-4)
    if oversample > 1:
        dense_wave = np.linspace(wave[0], wave[-1], wave.size * oversample)
        flux = np.interp(dense_wave, wave, flux)
        error = np.interp(dense_wave, wave, error) * np.sqrt(oversample)
        wave = dense_wave
    return wave, flux, error


def load_full_spectrum():
    """Full 4200-6800 A rest-frame spectrum with the Na D gap masked out."""
    from astropy.io import fits
    from extinction import ccm89, remove

    with fits.open(DATA_FILE) as hdul:
        hdr, flux = hdul[0].header, np.asarray(hdul[0].data, float) * 1e14
    wave = hdr["CRVAL1"] + hdr["CD1_1"] * np.arange(flux.size)
    flux = remove(ccm89(wave, 0.176, r_v=3.1, unit="aa"), flux)
    wave = wave / (1.0 + 0.0953)
    keep = (wave > 4200) & (wave < 6800)
    wave, flux = wave[keep], flux[keep]
    error = np.maximum(0.05 * np.abs(flux), 1e-4)
    fit_mask = ~((wave > 5770) & (wave < 6000))
    return wave, flux, error, fit_mask


# --------------------------------------------------------------------------------------
# [O III] window model (demo_ir07546.py)
# --------------------------------------------------------------------------------------
O3_BOUNDS = {
    "amplitude_0": (0.2, 1.0),
    "alpha_0": (-2.0, 4.0),
    "amp_c0_1": (0.2, 3.5),
    "dv_c_1": (-350.0, 200.0),
    "sigma_c_1": (80.0, 600.0),
    "amp_w0_1": (0.01, 1.0),
    "dv_w0_1": (-900.0, 200.0),
    "sigma_w0_1": (120.0, 1400.0),
}


def build_o3_cpu():
    """CPU [O III] model, exactly as in ``develop/dev_dynesty/demo_ir07546.py``."""
    from astropy.modeling import models

    continuum = models.PowerLaw1D(
        amplitude=0.5, x_0=5000.0, alpha=1.0,
        fixed={"x_0": True}, name="Continuum")
    line = galspec.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.7, amp_c1=1.7 / 2.98,
        dv_c=-43.0, sigma_c=250.0,
        wavec0=LINE_WAVE["OIII_5007"],
        wavec1=LINE_WAVE["OIII_4959"], name="OIII",
        par_w={"amp_w0": 0.35, "dv_w0": -280.0, "sigma_w0": 340.0})
    line.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("OIII", 2.98)
    return continuum + line


def build_o3_jax():
    """JAX twin of :func:`build_o3_cpu`."""
    continuum = jxm.PowerLaw1D(
        amplitude=0.5, x_0=5000.0, alpha=1.0,
        fixed={"x_0": True}, name="Continuum")
    line = jxm.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.7, amp_c1=1.7 / 2.98,
        dv_c=-43.0, sigma_c=250.0,
        wavec0=LINE_WAVE["OIII_5007"],
        wavec1=LINE_WAVE["OIII_4959"], name="OIII",
        par_w={"amp_w0": 0.35, "dv_w0": -280.0, "sigma_w0": 340.0})
    line.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("OIII", 2.98)
    return continuum + line


# --------------------------------------------------------------------------------------
# Full-spectrum model (wrk_sagan_iras07546 / demo_ir07546_full_spectrum.py)
# --------------------------------------------------------------------------------------
def build_full_cpu():
    """CPU full additive spectrum model (no multiplicative polynomial)."""
    from astropy.modeling import models

    w, label = LINE_WAVE, LINE_LABEL
    pl = models.PowerLaw1D(amplitude=.55, x_0=5500, alpha=1,
                           fixed={"x_0": True}, name="Continuum")
    iron = galspec.IronTemplate_new(amplitude=.2, sigma=900, z=0, dv=0, name="Fe II")
    bha = galspec.Line_MultiGauss(n_components=2, amp_c=2.22, dv_c=280,
        sigma_c=830, wavec=w["Halpha"], name=label["Halpha"],
        amp_w0=.26, dv_w0=-185, sigma_w0=2400)
    bhb = galspec.Line_MultiGauss(n_components=2, amp_c=.8, dv_c=-17,
        sigma_c=850, wavec=w["Hbeta"], name=label["Hbeta"],
        amp_w0=.3, dv_w0=-120, sigma_w0=2700,
        bounds={"sigma_w0": (100, 4000)})
    bhg = galspec.Line_MultiGauss(n_components=1, amp_c=.4, dv_c=80,
        sigma_c=1200, wavec=w["Hgamma"], name=label["Hgamma"])
    bhe2 = galspec.Line_MultiGauss(n_components=1, amp_c=.06, dv_c=85,
        sigma_c=2300, wavec=w["HeII_4686"], name=label["HeII_4686"],
        bounds={"sigma_c": (100, 4000), "dv_c": (-500, 500)})
    o3 = galspec.Line_MultiGauss_doublet(n_components=3, amp_c0=1.7,
        amp_c1=.6, dv_c=-43, sigma_c=250, wavec0=w["OIII_5007"],
        wavec1=w["OIII_4959"], name="[O III]", amp_w0=.35,
        dv_w0=-280, sigma_w0=340, amp_w1=.05, dv_w1=500, sigma_w1=1000)
    s2 = galspec.Line_MultiGauss_doublet(n_components=1, amp_c0=.1,
        amp_c1=.1, wavec0=w["SII_6716"], wavec1=w["SII_6731"], name="[S II]")
    nha = galspec.Line_Gaussian(amplitude=1.1, wavec=w["Halpha"], name="narrow Ha")
    nhb = galspec.Line_Gaussian(amplitude=.4, wavec=w["Hbeta"], name="narrow Hb")
    nhg = galspec.Line_Gaussian(amplitude=.12, wavec=w["Hgamma"], name="narrow Hg")
    nhe2 = galspec.Line_Gaussian(amplitude=.08, wavec=w["HeII_4686"], name="narrow He2")
    no3 = galspec.Line_Gaussian(amplitude=.05, wavec=w["OIII_4363"], name="[O III] 4363")
    model = pl + iron + bha + nha + bhb + nhb + bhg + nhg + bhe2 + nhe2 + o3 + s2 + no3

    o3.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("[O III]", 2.98)
    s2.sigma_c.tied = galspec.tie_MultiGauss_sigma_c("[O III]")
    s2.dv_c.tied = galspec.tie_MultiGauss_dv_c("[O III]")
    for line in (nha, nhb, nhg, nhe2, no3):
        line.sigma.tied = galspec.tie_MultiGauss_sigma_c("[O III]")
        line.dv.tied = galspec.tie_MultiGauss_dv_c("[O III]")
    return model


def build_full_jax():
    """JAX twin of :func:`build_full_cpu`."""
    w, label = LINE_WAVE, LINE_LABEL
    pl = jxm.PowerLaw1D(amplitude=.55, x_0=5500, alpha=1,
                        fixed={"x_0": True}, name="Continuum")
    iron = jxm.IronTemplate_new(amplitude=.2, sigma=900, z=0, dv=0, name="Fe II")
    bha = jxm.Line_MultiGauss(n_components=2, amp_c=2.22, dv_c=280,
        sigma_c=830, wavec=w["Halpha"], name=label["Halpha"],
        amp_w0=.26, dv_w0=-185, sigma_w0=2400)
    bhb = jxm.Line_MultiGauss(n_components=2, amp_c=.8, dv_c=-17,
        sigma_c=850, wavec=w["Hbeta"], name=label["Hbeta"],
        amp_w0=.3, dv_w0=-120, sigma_w0=2700,
        bounds={"sigma_w0": (100, 4000)})
    bhg = jxm.Line_MultiGauss(n_components=1, amp_c=.4, dv_c=80,
        sigma_c=1200, wavec=w["Hgamma"], name=label["Hgamma"])
    bhe2 = jxm.Line_MultiGauss(n_components=1, amp_c=.06, dv_c=85,
        sigma_c=2300, wavec=w["HeII_4686"], name=label["HeII_4686"],
        bounds={"sigma_c": (100, 4000), "dv_c": (-500, 500)})
    o3 = jxm.Line_MultiGauss_doublet(n_components=3, amp_c0=1.7,
        amp_c1=.6, dv_c=-43, sigma_c=250, wavec0=w["OIII_5007"],
        wavec1=w["OIII_4959"], name="[O III]", amp_w0=.35,
        dv_w0=-280, sigma_w0=340, amp_w1=.05, dv_w1=500, sigma_w1=1000)
    s2 = jxm.Line_MultiGauss_doublet(n_components=1, amp_c0=.1,
        amp_c1=.1, wavec0=w["SII_6716"], wavec1=w["SII_6731"], name="[S II]")
    nha = jxm.Line_Gaussian(amplitude=1.1, wavec=w["Halpha"], name="narrow Ha")
    nhb = jxm.Line_Gaussian(amplitude=.4, wavec=w["Hbeta"], name="narrow Hb")
    nhg = jxm.Line_Gaussian(amplitude=.12, wavec=w["Hgamma"], name="narrow Hg")
    nhe2 = jxm.Line_Gaussian(amplitude=.08, wavec=w["HeII_4686"], name="narrow He2")
    no3 = jxm.Line_Gaussian(amplitude=.05, wavec=w["OIII_4363"], name="[O III] 4363")
    model = pl + iron + bha + nha + bhb + nhb + bhg + nhg + bhe2 + nhe2 + o3 + s2 + no3

    o3.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("[O III]", 2.98)
    s2.sigma_c.tied = galspec.tie_MultiGauss_sigma_c("[O III]")
    s2.dv_c.tied = galspec.tie_MultiGauss_dv_c("[O III]")
    for line in (nha, nhb, nhg, nhe2, no3):
        line.sigma.tied = galspec.tie_MultiGauss_sigma_c("[O III]")
        line.dv.tied = galspec.tie_MultiGauss_dv_c("[O III]")
    return model


# --------------------------------------------------------------------------------------
# Stellar continuum + template model (wrk_sagan_sdss0006.ipynb style)
# --------------------------------------------------------------------------------------
def build_stellar_cpu():
    """CPU stellar-continuum model with tied star parameters + a templated line."""
    from astropy.modeling import models

    pl = models.PowerLaw1D(amplitude=10, x_0=5000, alpha=1.7, name="PowerLaw",
                           fixed={"x_0": True, "alpha": False})
    stars = []
    for i, star_type in enumerate(["A", "F", "G", "K", "M"]):
        star = galspec.StarSpectrum(amplitude=1.0 if i == 0 else 0.5, sigma=250.0,
                                    Star_type=star_type, name=f"{star_type} Star")
        stars.append(star)
    for star in stars[1:]:
        star.sigma.tied = galspec.tie_StarSpectrum_sigma("A Star")
        star.delta_z.tied = galspec.tie_StarSpectrum_deltaz("A Star")

    vel = np.linspace(-6000.0, 6000.0, 601)
    tflux = np.exp(-0.5 * (vel / 900.0) ** 2)
    nha = galspec.Line_template(template_velc=vel, template_flux=tflux,
                                amplitude=2.0, dv=10.0, wavec=LINE_WAVE["Halpha"],
                                name="nHalpha")
    nii = galspec.Line_template(template_velc=vel, template_flux=tflux,
                                amplitude=0.6, dv=10.0, wavec=LINE_WAVE["NII_6583"],
                                name="NII_6583")
    nii.dv.tied = galspec.tie_template_dv("nHalpha")
    nii2 = galspec.Line_template(template_velc=vel, template_flux=tflux,
                                 amplitude=0.2, dv=10.0, wavec=LINE_WAVE["NII_6548"],
                                 name="NII_6548")
    nii2.dv.tied = galspec.tie_template_dv("nHalpha")
    nii2.amplitude.tied = galspec.tie_template_amplitude("NII_6583", ratio=2.96)
    model = pl
    for star in stars:
        model = model + star
    return model + nha + nii + nii2


def build_stellar_jax():
    """JAX twin of :func:`build_stellar_cpu`."""
    pl = jxm.PowerLaw1D(amplitude=10, x_0=5000, alpha=1.7, name="PowerLaw",
                        fixed={"x_0": True, "alpha": False})
    stars = []
    for i, star_type in enumerate(["A", "F", "G", "K", "M"]):
        star = jxm.StarSpectrum(amplitude=1.0 if i == 0 else 0.5, sigma=250.0,
                                Star_type=star_type, name=f"{star_type} Star")
        stars.append(star)
    for star in stars[1:]:
        star.sigma.tied = galspec.tie_StarSpectrum_sigma("A Star")
        star.delta_z.tied = galspec.tie_StarSpectrum_deltaz("A Star")

    vel = np.linspace(-6000.0, 6000.0, 601)
    tflux = np.exp(-0.5 * (vel / 900.0) ** 2)
    nha = jxm.Line_template(template_velc=vel, template_flux=tflux,
                            amplitude=2.0, dv=10.0, wavec=LINE_WAVE["Halpha"],
                            name="nHalpha")
    nii = jxm.Line_template(template_velc=vel, template_flux=tflux,
                            amplitude=0.6, dv=10.0, wavec=LINE_WAVE["NII_6583"],
                            name="NII_6583")
    nii.dv.tied = galspec.tie_template_dv("nHalpha")
    nii2 = jxm.Line_template(template_velc=vel, template_flux=tflux,
                             amplitude=0.2, dv=10.0, wavec=LINE_WAVE["NII_6548"],
                             name="NII_6548")
    nii2.dv.tied = galspec.tie_template_dv("nHalpha")
    nii2.amplitude.tied = galspec.tie_template_amplitude("NII_6583", ratio=2.96)
    model = pl
    for star in stars:
        model = model + star
    return model + nha + nii + nii2


# --------------------------------------------------------------------------------------
# Parameter synchronisation between the CPU and JAX twins
# --------------------------------------------------------------------------------------
def apply_cpu_ties(cpu_model, n_passes: int = 3):
    """Evaluate the astropy tied constraints, as the CPU fitters do.

    astropy only refreshes tied parameters when parameters are assigned, so a
    freshly constructed model still holds its initial values for tied
    parameters.  :class:`galspec.Dynesty_Fit` calls them explicitly; doing the
    same here makes the CPU model the correct reference for parity tests.
    """
    for _ in range(n_passes):
        for name in cpu_model.param_names:
            param = getattr(cpu_model, name)
            if callable(param.tied) and not param.fixed:
                value = param.tied(cpu_model)
                # the galspec tie classes may return another Parameter object
                param.value = float(getattr(value, "value", value))
    return cpu_model


def sync_params(cpu_model, jax_model):
    """Copy values, ``fixed`` flags and ties from an astropy model to its JAX twin.

    The CPU tied constraints are applied first so both models describe the same
    physical state.
    """
    apply_cpu_ties(cpu_model)
    cpu_leaves = list(cpu_model)
    jax_leaves = list(jax_model)
    if len(cpu_leaves) != len(jax_leaves):
        raise ValueError(f"submodel count mismatch: CPU {len(cpu_leaves)} vs JAX {len(jax_leaves)}")
    for cpu_leaf, jax_leaf in zip(cpu_leaves, jax_leaves):
        for name in jax_leaf.param_names:
            cpu_param = getattr(cpu_leaf, name)
            jax_leaf.set_value(name, float(cpu_param.value))
            jax_leaf.set_fixed(name, bool(cpu_param.fixed))
            tied = cpu_param.tied if callable(cpu_param.tied) else None
            jax_leaf.set_tied(name, tied)
    return jax_model


# --------------------------------------------------------------------------------------
# Realistic AGN model for the [O III] window: power law + Fe II template + [O III]
# doublet.  Eleven free parameters, and an expensive (convolved) template -- the
# kind of model an actual emission-line fit uses.
# --------------------------------------------------------------------------------------
AGN_BOUNDS = {
    "amplitude_0": (0.2, 1.0),        # power-law amplitude
    "alpha_0": (-2.0, 4.0),           # power-law index
    "amplitude_1": (0.01, 1.0),       # Fe II template amplitude
    "sigma_1": (300.0, 3000.0),       # Fe II velocity width
    "amp_c0_2": (0.2, 3.5),           # [O III] core, 5007
    "dv_c_2": (-350.0, 200.0),
    "sigma_c_2": (80.0, 600.0),
    "amp_w0_2": (0.01, 1.0),          # [O III] wing
    "dv_w0_2": (-900.0, 200.0),
    "sigma_w0_2": (120.0, 1400.0),
}


def build_agn_cpu():
    """CPU power law + Fe II template + [O III] doublet (11 free parameters)."""
    from astropy.modeling import models
    from galspec.line_profile import tie_MultiGauss_doublet_ratio

    continuum = models.PowerLaw1D(amplitude=0.5, x_0=5000.0, alpha=1.0,
                                  fixed={"x_0": True}, name="Continuum")
    iron = galspec.IronTemplate_new(amplitude=0.15, sigma=900.0, z=0.0, dv=0.0,
                                    name="Fe II")
    iron.z.fixed = True
    line = galspec.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.7, amp_c1=1.7 / 2.98, dv_c=-43.0, sigma_c=250.0,
        wavec0=LINE_WAVE["OIII_5007"], wavec1=LINE_WAVE["OIII_4959"],
        name="[O III]", amp_w0=0.35, dv_w0=-280.0, sigma_w0=340.0)
    line.amp_c1.tied = tie_MultiGauss_doublet_ratio("[O III]", 2.98)
    return continuum + iron + line


def build_agn_jax():
    """JAX twin of :func:`build_agn_cpu`."""
    from galspec.jax_models import IronTemplate, Line_MultiGauss_doublet, PowerLaw1D, tie

    continuum = PowerLaw1D(amplitude=0.5, x_0=5000.0, alpha=1.0,
                           fixed={"x_0": True}, name="Continuum")
    iron = IronTemplate(amplitude=0.15, sigma=900.0, z=0.0, dv=0.0, name="Fe II")
    iron.z.fixed = True
    line = Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.7, amp_c1=1.7 / 2.98, dv_c=-43.0, sigma_c=250.0,
        wavec0=LINE_WAVE["OIII_5007"], wavec1=LINE_WAVE["OIII_4959"],
        name="[O III]", amp_w0=0.35, dv_w0=-280.0, sigma_w0=340.0)
    line.amp_c1.tied = tie.tie_MultiGauss_doublet_ratio("[O III]", 2.98)
    return continuum + iron + line


def model_pairs():
    """All (name, cpu_model, jax_model) triples used by tests and benchmarks."""
    return [
        ("o3", build_o3_cpu(), build_o3_jax()),
        ("agn", build_agn_cpu(), build_agn_jax()),
        ("full", build_full_cpu(), build_full_jax()),
        ("stellar", build_stellar_cpu(), build_stellar_jax()),
    ]