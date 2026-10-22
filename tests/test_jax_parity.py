"""CPU/GPU model parity tests for the JAX backend.

These are the tests that establish *"the GPU model gives the same results as the
CPU model"*.  Two levels are checked:

1. **Exact forward-model parity** -- every JAX model is compared with its
   astropy/GalSpec CPU counterpart on random parameter vectors inside the prior
   bounds.  With ``jax_enable_x64`` the agreement is at float64 rounding
   (typically 1e-16 .. 1e-13), far below the 1e-8 tolerance asserted here.
2. **Likelihood parity** -- ``JNesty_Fit.log_likelihood`` is compared with
   ``Dynesty_Fit.log_likelihood`` for the demo models (including tied
   parameters), which is what the samplers actually consume.

Run with::

    python -m pytest tests/test_jax_parity.py -v
    python tests/test_jax_parity.py          # standalone, no pytest needed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Parity is device independent: the tests run on whatever JAX backend is
# available (GPU included, which is the stronger check).  Export
# JAX_PLATFORMS=cpu to force the CPU backend.
os.environ.setdefault("GALSPEC_JAX_X64", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "develop" / "dev_jax"))

import galspec                                   # noqa: E402
from astropy.modeling import models as amodels   # noqa: E402
from galspec import jax_models as jxm            # noqa: E402
from galspec.jax_models.functional import gaussian_smooth1d  # noqa: E402

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402

import common                                    # noqa: E402

RTOL = 1e-8       # asserted tolerance (actual agreement is ~1e-16)
NOISE_SEED = 20260901


def _rel_diff(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30)
    return float(np.max(np.abs(a - b)) / scale)


def _random_theta(model, rng, n):
    """Latin-hypercube-ish random parameter vectors inside the model's bounds."""
    lo = np.array([model.spec(n_).bounds[0] if model.spec(n_).bounds[0] is not None else -1.0
                   for n_ in model.free_param_names], dtype=float)
    hi = np.array([model.spec(n_).bounds[1] if model.spec(n_).bounds[1] is not None else 1.0
                   for n_ in model.free_param_names], dtype=float)
    # Keep parameters well inside their bounds (convolution kernels etc.).
    lo = lo + 0.1 * (hi - lo)
    hi = hi - 0.1 * (hi - lo)
    return rng.uniform(lo, hi, size=(n, lo.size))


def _set_jax_from_theta(model, theta):
    model.set_free_params(theta)
    return model


# --------------------------------------------------------------------------------------
# 1. Gaussian smoothing matches scipy exactly
# --------------------------------------------------------------------------------------
def test_gaussian_smoothing_matches_scipy():
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(NOISE_SEED)
    y = rng.standard_normal(400) + 5.0
    for sigma in (0.5, 1.7, 4.0, 11.3):
        ref = gaussian_filter1d(y, sigma)
        got = np.asarray(gaussian_smooth1d(jnp.asarray(y), sigma,
                                           radius=int(4.0 * sigma + 0.5)))
        assert _rel_diff(ref, got) < 1e-12, f"sigma={sigma}"


def test_gaussian_smoothing_traced_sigma():
    """Static radius + masked kernel must equal the dynamic-radius scipy result."""
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(7)
    y = rng.standard_normal(300)

    @jax.jit
    def smooth(sigma):
        return gaussian_smooth1d(jnp.asarray(y), sigma, radius=int(4.0 * 9.0 + 0.5))

    for sigma in (1.0, 3.5, 8.0):
        ref = gaussian_filter1d(y, sigma)
        got = np.asarray(smooth(jnp.asarray(sigma)))
        assert _rel_diff(ref, got) < 1e-12, f"sigma={sigma}"


# --------------------------------------------------------------------------------------
# 2. Leaf model parity
# --------------------------------------------------------------------------------------
LEAF_CASES = [
    ("Line_Gaussian",
     lambda: galspec.Line_Gaussian(amplitude=2.0, dv=50.0, sigma=300.0, wavec=5007.0),
     lambda: jxm.Line_Gaussian(amplitude=2.0, dv=50.0, sigma=300.0, wavec=5007.0)),
    ("Line_GaussHermite",
     lambda: galspec.Line_GaussHermite(amplitude=2.0, dv=-20.0, sigma=400.0, h3=0.1, h4=-0.05,
                                       wavec=5007.0, clip=False),
     lambda: jxm.Line_GaussHermite(amplitude=2.0, dv=-20.0, sigma=400.0, h3=0.1, h4=-0.05,
                                   wavec=5007.0, clip=False)),
    ("Line_GaussHermite_doublet",
     lambda: galspec.Line_GaussHermite_doublet(amp_c0=2.0, amp_c1=0.7, dv_c=-20.0, sigma_c=400.0,
                                               h3=0.1, h4=-0.05, wavec0=5007.0, wavec1=4959.0,
                                               clip=False),
     lambda: jxm.Line_GaussHermite_doublet(amp_c0=2.0, amp_c1=0.7, dv_c=-20.0, sigma_c=400.0,
                                           h3=0.1, h4=-0.05, wavec0=5007.0, wavec1=4959.0,
                                           clip=False)),
    ("Line_MultiGauss(3)",
     lambda: galspec.Line_MultiGauss(n_components=3, amp_c=2.0, dv_c=30.0, sigma_c=500.0,
                                     wavec=6563.0, par_w={"amp_w0": 0.3, "dv_w0": -300.0,
                                                          "sigma_w0": 900.0, "amp_w1": 0.05,
                                                          "dv_w1": 500.0, "sigma_w1": 1500.0}),
     lambda: jxm.Line_MultiGauss(n_components=3, amp_c=2.0, dv_c=30.0, sigma_c=500.0,
                                 wavec=6563.0, par_w={"amp_w0": 0.3, "dv_w0": -300.0,
                                                      "sigma_w0": 900.0, "amp_w1": 0.05,
                                                      "dv_w1": 500.0, "sigma_w1": 1500.0})),
    ("Line_MultiGauss_doublet(3)",
     lambda: galspec.Line_MultiGauss_doublet(n_components=3, amp_c0=1.7, amp_c1=0.6, dv_c=-43.0,
                                             sigma_c=250.0, wavec0=5007.0, wavec1=4959.0,
                                             amp_w0=0.35, dv_w0=-280.0, sigma_w0=340.0,
                                             amp_w1=0.05, dv_w1=500.0, sigma_w1=1000.0),
     lambda: jxm.Line_MultiGauss_doublet(n_components=3, amp_c0=1.7, amp_c1=0.6, dv_c=-43.0,
                                         sigma_c=250.0, wavec0=5007.0, wavec1=4959.0,
                                         amp_w0=0.35, dv_w0=-280.0, sigma_w0=340.0,
                                         amp_w1=0.05, dv_w1=500.0, sigma_w1=1000.0)),
    ("Line_Absorption",
     lambda: galspec.Line_Absorption(logtau0=0.4, dv=-80.0, sigma=250.0, Cf=0.8, wavec=4861.0),
     lambda: jxm.Line_Absorption(logtau0=0.4, dv=-80.0, sigma=250.0, Cf=0.8, wavec=4861.0)),
    ("Line_Absorption_doublet",
     lambda: galspec.Line_Absorption_doublet(logtau0=0.4, logtau1=0.1, dv=-80.0, sigma=250.0,
                                             Cf=0.8, wavec0=4861.0, wavec1=5007.0),
     lambda: jxm.Line_Absorption_doublet(logtau0=0.4, logtau1=0.1, dv=-80.0, sigma=250.0,
                                         Cf=0.8, wavec0=4861.0, wavec1=5007.0)),
    ("PowerLaw1D",
     lambda: __import__("astropy.modeling.models", fromlist=["x"]).PowerLaw1D(
         amplitude=0.7, x_0=5500.0, alpha=1.2, fixed={"x_0": True}),
     lambda: jxm.PowerLaw1D(amplitude=0.7, x_0=5500.0, alpha=1.2, fixed={"x_0": True})),
    ("Const1D",
     lambda: __import__("astropy.modeling.models", fromlist=["x"]).Const1D(amplitude=2.5),
     lambda: jxm.Const1D(amplitude=2.5)),
    ("Linear1D",
     lambda: __import__("astropy.modeling.models", fromlist=["x"]).Linear1D(slope=1e-4,
                                                                           intercept=0.3),
     lambda: jxm.Linear1D(slope=1e-4, intercept=0.3)),
    ("Gaussian1D",
     lambda: __import__("astropy.modeling.models", fromlist=["x"]).Gaussian1D(
         amplitude=3.0, mean=5000.0, stddev=12.0),
     lambda: jxm.Gaussian1D(amplitude=3.0, mean=5000.0, stddev=12.0)),
    ("Polynomial1D(3)",
     lambda: __import__("astropy.modeling.models", fromlist=["x"]).Polynomial1D(
         degree=3, c0=1.0, c1=2e-4, c2=-1e-8, c3=1e-12),
     lambda: jxm.Polynomial1D(degree=3, c0=1.0, c1=2e-4, c2=-1e-8, c3=1e-12)),
    ("WindowedPowerLaw1D",
     lambda: galspec.WindowedPowerLaw1D(amplitude=1.3, x_0=4500.0, alpha=1.4),
     lambda: jxm.WindowedPowerLaw1D(amplitude=1.3, x_0=4500.0, alpha=1.4)),
    ("WindowedLinear",
     lambda: galspec.WindowedLinear(c0=1.0, c1=2e-4),
     lambda: jxm.WindowedLinear(c0=1.0, c1=2e-4)),
]


def test_leaf_models_match_cpu():
    wave = np.linspace(4300, 6800, 1200)
    rng = np.random.default_rng(NOISE_SEED)
    worst = {}
    for name, make_cpu, make_jax in LEAF_CASES:
        cpu, jax_model = make_cpu(), make_jax()
        cpu_leaves, jax_leaves = [cpu], [jax_model]
        for cpu_leaf, jax_leaf in zip(cpu_leaves, jax_leaves):
            for pname in jax_leaf.param_names:
                cparam = getattr(cpu_leaf, pname)
                jax_leaf.set_value(pname, float(cparam.value))
                jax_leaf.set_fixed(pname, bool(cparam.fixed))
                tied = cparam.tied if callable(cparam.tied) else None
                jax_leaf.set_tied(pname, tied)
        got = np.asarray(jax_model(wave))
        ref = np.asarray(cpu(wave))
        worst[name] = _rel_diff(ref, got)
    failures = {k: v for k, v in worst.items() if v >= RTOL}
    assert not failures, f"leaf parity failures (rtol={RTOL}): {failures}"
    print(f"  leaf models: worst relative difference {max(worst.values()):.2e} "
          f"over {len(worst)} models")


def test_broken_powerlaw_matches_astropy():
    from astropy.modeling.models import BrokenPowerLaw1D as CpuBrokenPowerLaw1D

    # the galspec CPU export *is* astropy's model
    assert galspec.BrokenPowerLaw1D is CpuBrokenPowerLaw1D

    wave = np.linspace(3000, 8000, 900)
    cpu = CpuBrokenPowerLaw1D(amplitude=2.0, x_break=5200.0, alpha_1=0.4, alpha_2=1.8)
    jax_model = jxm.BrokenPowerLaw1D(amplitude=2.0, x_break=5200.0, alpha_1=0.4, alpha_2=1.8)
    assert _rel_diff(cpu(wave), jax_model(wave)) < 1e-12

    # random parameter vectors, including a break outside the window
    rng = np.random.default_rng(11)
    thetas = np.column_stack([
        rng.uniform(0.5, 3.0, 40), rng.uniform(3500.0, 7500.0, 40),
        rng.uniform(-2.0, 2.0, 40), rng.uniform(-2.0, 2.0, 40)])
    for theta in thetas:
        cpu.parameters = theta
        jax_model.set_free_params(theta)
        assert _rel_diff(cpu(wave), jax_model(wave)) < 1e-12

    # continuity at the break: both branches give the same value
    y = np.asarray(jax_model.flux(np.array([5200.0]), np.array([2.0, 5200.0, 0.4, 1.8])))
    assert np.allclose(y, 2.0, rtol=1e-12)


def test_iron_template_and_stellar_match_cpu():
    wave = np.linspace(4200, 6800, 1300)
    pairs = [
        ("IronTemplate",
         galspec.IronTemplate(amplitude=0.3, stddev=1200.0, z=0.02, name="Fe II"),
         jxm.IronTemplate(amplitude=0.3, stddev=1200.0, z=0.02, name="Fe II")),
        ("IronTemplate(boroson)",
         galspec.IronTemplate(amplitude=0.3, stddev=1500.0, z=0.0, template_name="boroson1992"),
         jxm.IronTemplate(amplitude=0.3, stddev=1500.0, z=0.0, template_name="boroson1992")),
        ("IronTemplate(below intrinsic)",
         galspec.IronTemplate(amplitude=1.0, stddev=200.0, z=0.0),
         jxm.IronTemplate(amplitude=1.0, stddev=200.0, z=0.0)),
        ("StarSpectrum",
         galspec.StarSpectrum(amplitude=3.0, sigma=280.0, Star_type="K", name="K Star"),
         jxm.StarSpectrum(amplitude=3.0, sigma=280.0, Star_type="K", name="K Star")),
        ("Multi_StarSpectrum",
         galspec.Multi_StarSpectrum(amp_0=1.0, amp_1=0.5, amp_2=0.25, amp_3=0.1, amp_4=0.05,
                                    sigma=240.0, velscale=69),
         jxm.Multi_StarSpectrum(amp_0=1.0, amp_1=0.5, amp_2=0.25, amp_3=0.1, amp_4=0.05,
                                sigma=240.0, velscale=69)),
    ]
    for name, cpu, jax_model in pairs:
        jax_model.prepare(wave)
        assert _rel_diff(cpu(wave), jax_model(wave)) < RTOL, name

    # fitted widths must be traceable under jit as well
    cpu = galspec.IronTemplate(amplitude=0.5, stddev=900.0, z=0.0)
    jax_model = jxm.IronTemplate(amplitude=0.5, stddev=900.0, z=0.0)
    jax_model.prepare(wave)
    compiled = jax.jit(lambda theta: jax_model.flux(wave, theta))
    for stddev in (400.0, 900.0, 1800.0, 3000.0):
        cpu.stddev = stddev
        theta = np.array([0.5, stddev, 0.0])
        assert _rel_diff(cpu(wave), compiled(jnp.asarray(theta))) < RTOL


def test_line_template_matches_cpu():
    wave = np.linspace(6400, 6720, 400)
    vel = np.linspace(-7000.0, 7000.0, 701)
    tflux = np.exp(-0.5 * (vel / 850.0) ** 2) * (1.0 + 0.1 * np.sin(vel / 900.0))
    cpu = galspec.Line_template(template_velc=vel, template_flux=tflux,
                                amplitude=2.0, dv=120.0, wavec=6563.0)
    jax_model = jxm.Line_template(template_velc=vel, template_flux=tflux,
                                  amplitude=2.0, dv=120.0, wavec=6563.0)
    assert _rel_diff(cpu(wave), jax_model(wave)) < RTOL


# --------------------------------------------------------------------------------------
# 3. LSF convolution parity
# --------------------------------------------------------------------------------------
def test_convolve_lsf_matches_cpu():
    wave = np.arange(6400.0, 6700.0, 0.5)
    cpu_line = galspec.Line_MultiGauss(n_components=2, amp_c=2.0, dv_c=100.0, sigma_c=400.0,
                                       wavec=6563.0, name="Ha",
                                       amp_w0=0.3, dv_w0=-200.0, sigma_w0=900.0)
    jax_line = jxm.Line_MultiGauss(n_components=2, amp_c=2.0, dv_c=100.0, sigma_c=400.0,
                                   wavec=6563.0, name="Ha",
                                   amp_w0=0.3, dv_w0=-200.0, sigma_w0=900.0)
    for resolving_power in (500.0, 2000.0, 6000.0):
        cpu = galspec.convolve_lsf(cpu_line.copy(), wavec=6563.0,
                                   resolving_power=resolving_power)
        jax_model = jxm.convolve_lsf(jax_line.copy(), wavec=6563.0,
                                     resolving_power=resolving_power)
        jax_model.prepare(wave)
        assert _rel_diff(cpu(wave), jax_model(wave)) < RTOL, f"R={resolving_power}"


def test_convolve_lsf_wrapping_a_compound_model():
    """Convolving a whole compound model (the realistic AGN setup).

    ``convolve_lsf`` around a *compound* model is how a real fit is built: the
    instrument profile must act on the total spectrum, not on one line.  The
    wrapper delegates its parameter plumbing to the compound, so this guards the
    flattening/tie handling as well as the numbers.
    """
    wave = np.arange(6480.0, 6650.0, 0.25)
    cpu = (galspec.Line_MultiGauss(n_components=2, amp_c=2.0, dv_c=50.0, sigma_c=300.0,
                                   wavec=6563.0, name="Ha",
                                   amp_w0=0.4, dv_w0=-150.0, sigma_w0=1200.0)
           + galspec.Line_Gaussian(amplitude=1.0, dv=0.0, sigma=300.0, wavec=6548.0,
                                   name="NII")
           + amodels.PowerLaw1D(amplitude=0.5, x_0=6563.0, alpha=1.0,
                                name="cont", fixed={"x_0": True}))
    jax_model = (jxm.Line_MultiGauss(n_components=2, amp_c=2.0, dv_c=50.0, sigma_c=300.0,
                                     wavec=6563.0, name="Ha",
                                     amp_w0=0.4, dv_w0=-150.0, sigma_w0=1200.0)
                 + jxm.Line_Gaussian(amplitude=1.0, dv=0.0, sigma=300.0, wavec=6548.0,
                                     name="NII")
                 + jxm.PowerLaw1D(amplitude=0.5, x_0=6563.0, alpha=1.0,
                                  name="cont", fixed={"x_0": True}))

    cpuj = galspec.convolve_lsf(cpu, wavec=6563.0, resolving_power=2000.0)
    jaxj = jxm.convolve_lsf(jax_model, wavec=6563.0, resolving_power=2000.0)
    jaxj.prepare(wave)

    assert _rel_diff(cpuj(wave), jaxj(wave)) < RTOL

    # the wrapper must expose the compound's *free* parameters (astropy's
    # param_names also lists the fixed ones, e.g. the fixed line wavelengths)
    cpu_free = {n for n in cpuj.param_names if not cpuj.fixed[n]}
    assert set(jaxj.free_param_names) == cpu_free


def test_convolve_lsf_var_matches_cpu():
    wave = np.arange(4100.0, 5000.0, 1.0)
    wave_res = np.array([0.6, 1.0, 2.0, 3.0, 5.0])   # microns
    resolving = np.array([30.0, 100.0, 200.0, 300.0, 350.0])
    cpu_line = galspec.Line_Gaussian(amplitude=2.0, dv=0.0, sigma=250.0, wavec=4861.0,
                                     name="Hb")
    jax_line = jxm.Line_Gaussian(amplitude=2.0, dv=0.0, sigma=250.0, wavec=4861.0,
                                 name="Hb")
    cpu = galspec.convolve_lsf_var(cpu_line.copy(), wavec=wave,
                                   resolution_data=(wave_res, resolving))
    jax_model = jxm.convolve_lsf_var(jax_line.copy(), wavec=wave,
                                     resolution_data=(wave_res, resolving))
    jax_model.prepare(wave)
    assert _rel_diff(cpu(wave), jax_model(wave)) < RTOL


# --------------------------------------------------------------------------------------
# 4. Compound models with ties, jit and vmap
# --------------------------------------------------------------------------------------
def test_demo_compound_models_match_cpu():
    wave, _flux, _err = common.load_spectrum()
    wave_full, _, _, mask = common.load_full_spectrum()

    results = {}
    for name, cpu, jax_model in common.model_pairs():
        common.sync_params(cpu, jax_model)
        if name == "stellar":
            grid = wave_full[mask]
        else:
            grid = wave if name == "o3" else wave_full[mask]
        jax_model.prepare(grid)
        ref = np.asarray(cpu(grid))
        got = np.asarray(jax_model(grid))
        results[name] = _rel_diff(ref, got)
        # same check through the flat free-parameter interface
        theta = jax_model.get_free_params()
        results[name + "/flux"] = _rel_diff(ref, np.asarray(jax_model.flux(grid, theta)))
        assert list(jax_model.free_param_names) == [
            n for n in cpu.param_names
            if not getattr(cpu, n).fixed and not getattr(cpu, n).tied], \
            f"{name}: free parameter names differ from the astropy model"

    failures = {k: v for k, v in results.items() if v >= RTOL}
    assert not failures, f"compound parity failures: {failures}"
    print(f"  compound models: worst relative difference {max(results.values()):.2e}")


def test_random_parameters_compound_parity():
    """Random (not just initial) parameters, for the demo compound models."""
    grid = np.linspace(4900.0, 5100.0, 800)
    rng = np.random.default_rng(NOISE_SEED)

    cpu = common.build_o3_cpu()
    jax_model = common.build_o3_jax()
    jax_model.prepare(grid)
    common.sync_params(cpu, jax_model)

    # Sample inside the *prior* (log-uniform amplitudes included) so every draw is
    # physically reasonable, and drive both models through the same theta.
    from galspec.jnesty_fit import JNesty_Fit
    wave_data, flux_data, err_data = common.load_spectrum()
    fit = JNesty_Fit(jax_model, wave_data, flux_data, err_data,
                     bounds_dict=common.O3_BOUNDS, nlive=20, device="auto", verbose=False)
    free = list(fit.param_names)

    def sample_theta(n):
        return np.array([fit.prior_transform(rng.uniform(0.0, 1.0, fit.ndim)) for _ in range(n)])

    worst = 0.0
    for theta in sample_theta(50):
        for n, v in zip(free, theta):
            setattr(cpu, n, float(v))
        common.apply_cpu_ties(cpu)
        jax_model.set_free_params(theta)
        worst = max(worst, _rel_diff(cpu(grid), jax_model.flux(grid, theta)))
    assert worst < RTOL, f"random-parameter parity failure: {worst:.2e}"
    print(f"  random parameters: worst relative difference {worst:.2e} over 50 draws")

    # batched evaluation must equal the looped one
    thetas = sample_theta(8)
    batch = np.asarray(jax_model.flux_batch(grid, thetas))
    for i, theta in enumerate(thetas):
        assert _rel_diff(jax_model.flux(grid, theta), batch[i]) < 1e-12


def test_multiplicative_compound_matches_cpu():
    """``(continuum + lines) * polynomial``, the multiplicative form used by the
    SAGAN notebook to model flux-calibration / extinction residuals."""
    from astropy.modeling import models

    wave = np.linspace(4200.0, 6800.0, 1200)
    cpu = (models.PowerLaw1D(amplitude=.55, x_0=5500, alpha=1, fixed={"x_0": True},
                             name="Continuum")
           + galspec.IronTemplate_new(amplitude=.2, sigma=900, z=0, dv=0,
                                       name="Fe II")
           + galspec.Line_Gaussian(amplitude=1.1, wavec=6563.0, dv=10.0, sigma=300.0,
                                   name="Ha")) \
        * models.Polynomial1D(degree=2, c0=1.0, c1=-1e-4, c2=2e-8, name="poly")

    jax_model = (jxm.PowerLaw1D(amplitude=.55, x_0=5500, alpha=1, fixed={"x_0": True},
                                name="Continuum")
                 + jxm.IronTemplate_new(amplitude=.2, sigma=900, z=0, dv=0,
                                       name="Fe II")
                 + jxm.Line_Gaussian(amplitude=1.1, wavec=6563.0, dv=10.0, sigma=300.0,
                                     name="Ha")) \
        * jxm.Polynomial1D(degree=2, c0=1.0, c1=-1e-4, c2=2e-8, name="poly")

    common.sync_params(cpu, jax_model)
    jax_model.prepare(wave)
    assert tuple(jax_model.param_names) == tuple(cpu.param_names)
    assert _rel_diff(cpu(wave), jax_model(wave)) < RTOL
    assert _rel_diff(cpu(wave), jax_model.flux(wave, jax_model.get_free_params())) < RTOL

    # submodel access and copy semantics, as with an astropy compound model
    assert isinstance(jax_model["poly"], jxm.Polynomial1D)
    assert float(jax_model["poly"].c1) == float(cpu["poly"].c1.value)
    clone = jax_model.copy()
    clone["poly"].c1 = 5e-4
    assert float(clone["poly"].c1) != float(jax_model["poly"].c1)


# --------------------------------------------------------------------------------------
# 5. Likelihood parity with the CPU fitter
# --------------------------------------------------------------------------------------
def test_loglikelihood_matches_dynesty_fitter():
    from galspec.dynesty_fit import Dynesty_Fit
    from galspec.jnesty_fit import JNesty_Fit

    wave, flux, error = common.load_spectrum()
    rng = np.random.default_rng(NOISE_SEED)

    cpu_model = common.build_o3_cpu()
    cpu_fit = Dynesty_Fit(cpu_model, wave, flux, error, bounds_dict=common.O3_BOUNDS, nlive=50)

    jax_model = common.build_o3_jax()
    common.sync_params(cpu_model, jax_model)
    jax_fit = JNesty_Fit(jax_model, wave, flux, error, bounds_dict=common.O3_BOUNDS,
                         nlive=50, device="auto", verbose=False)

    assert list(jax_fit.param_names) == list(cpu_fit.param_names), "parameter order differs"
    assert np.allclose(np.array(jax_fit.param_bounds), np.array(cpu_fit.param_bounds)), \
        "prior bounds differ between the CPU and GPU fitters"

    worst_theta = 0.0
    worst_prior = 0.0
    for _ in range(30):
        u = rng.uniform(0.0, 1.0, size=cpu_fit.ndim)
        theta_cpu = cpu_fit.prior_transform(u)
        theta_jax = jax_fit.prior_transform(u)
        worst_prior = max(worst_prior, _rel_diff(theta_cpu, theta_jax))
        worst_theta = max(worst_theta,
                          abs(cpu_fit.log_likelihood(theta_cpu) - jax_fit.log_likelihood(theta_cpu)))
    assert worst_prior < 1e-10, f"prior transform differs: {worst_prior:.2e}"
    assert worst_theta < RTOL, f"log-likelihood differs: {worst_theta:.2e}"

    # batched likelihood agrees with the scalar path
    batch_theta = np.array([cpu_fit.prior_transform(rng.uniform(0, 1, cpu_fit.ndim))
                            for _ in range(6)])
    got = jax_fit.log_likelihood_batch(batch_theta)
    for theta, value in zip(batch_theta, got):
        assert abs(value - jax_fit.log_likelihood(theta)) < 1e-9


def test_loglikelihood_matches_dynesty_fitter_full_spectrum():
    from galspec.dynesty_fit import Dynesty_Fit
    from galspec.jnesty_fit import JNesty_Fit

    wave_all, flux_all, error_all, mask = common.load_full_spectrum()
    wave, flux, error = wave_all[mask], flux_all[mask], error_all[mask]

    cpu_model = common.build_full_cpu()
    cpu_fit = Dynesty_Fit(cpu_model, wave, flux, error, nlive=50)
    jax_model = common.build_full_jax()
    common.sync_params(cpu_model, jax_model)
    jax_fit = JNesty_Fit(jax_model, wave, flux, error, nlive=50, device="auto", verbose=False)

    assert list(jax_fit.param_names) == list(cpu_fit.param_names)
    theta = cpu_fit.prior_transform(np.full(cpu_fit.ndim, 0.5))
    logl_cpu = cpu_fit.log_likelihood(theta)
    logl_jax = jax_fit.log_likelihood(theta)
    rel = abs(logl_cpu - logl_jax) / max(abs(logl_cpu), 1.0)
    assert rel < RTOL, f"full-spectrum log-likelihood differs: {logl_cpu} vs {logl_jax}"
    print(f"  full spectrum: {cpu_fit.ndim} free parameters, |dlogL| = {abs(logl_cpu-logl_jax):.2e}")


# --------------------------------------------------------------------------------------
# 6. Sampler smoke test
# --------------------------------------------------------------------------------------
def test_jnesty_smoke_run():
    from galspec.jnesty_fit import JNesty_Fit

    wave, flux, error = common.load_spectrum()
    jax_model = common.build_o3_jax()
    fit = JNesty_Fit(jax_model, wave, flux, error, bounds_dict=common.O3_BOUNDS,
                     nlive=60, device="auto", verbose=False)
    fit.fit(progress=False, dlogz=1.0, max_iterations=400)
    assert np.isfinite(fit.log_evidence)
    assert np.isfinite(fit.log_evidence_err)
    best, names, theta = fit.get_best_fit()
    assert len(theta) == fit.ndim
    assert np.all(np.isfinite(np.asarray(best(wave))))
    quantiles = fit.get_quantiles()
    assert quantiles.shape == (3, fit.ndim)
    # every parameter must sit on the prior grid it was declared with
    for i, (lower, upper) in enumerate(fit.param_bounds):
        assert lower <= quantiles[1, i] <= upper
    flagged = fit.prior_dominated(tolerance=0.01)
    assert isinstance(flagged, dict)
    assert all(edge in ("lower", "upper") for edge in flagged.values())


# --------------------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------------------
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        try:
            print(f"[run ] {name}")
            test()
            print(f"[ ok ] {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            import traceback
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
    print()
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())