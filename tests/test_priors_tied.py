"""Tests for priors centred on another parameter (``galspec.priors.gaussian_around``).

The feature is one line of science -- "let broad H-beta share broad H-alpha's
profile, but not exactly" -- and several ways to get it subtly wrong:

* the transform must be a *bijection* of the unit cube onto the parameter's box,
  otherwise the evidence is not comparable with a run that uses the plain box
  prior;
* it must be the *same* function in the JAX and the astropy fitter, or the two
  fitters disagree about the prior they claim to share;
* it must survive ``jax.jit`` / ``jax.vmap``, since that is how the sampler calls
  it;
* a log-scaled parameter (an amplitude) keeps its box in log10 units, so the
  truncation range has to be exponentiated or the tie silently lands in the wrong
  place;
* and it must *constrain* without *forcing*: a planted difference has to come back
  out of the posterior.

Run with::

    python -m pytest tests/test_priors_tied.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402
jax.config.update("jax_enable_x64", True)

from astropy.modeling import models                                 # noqa: E402
from scipy.special import ndtr                                       # noqa: E402

from galspec import jax_models as jxm                               # noqa: E402
from galspec.dynesty_fit import Dynesty_Fit                         # noqa: E402
from galspec.jnesty_fit import JNesty_Fit                           # noqa: E402
from galspec.priors import (CenteredPrior, gaussian_around,         # noqa: E402
                            resolve_prior_dependencies)


# --------------------------------------------------------------------------------------
# The transform on its own
# --------------------------------------------------------------------------------------
def test_centre_and_box_edges():
    """u = 0 and u = 1 reach the box; with a symmetric box u = 1/2 is the centre."""
    prior = gaussian_around("a", 500.0, lower=300.0, upper=2300.0)
    assert abs(prior.transform(0.0, 1300.0) - 300.0) < 1e-9
    assert abs(prior.transform(1.0, 1300.0) - 2300.0) < 1e-9
    assert abs(prior.transform(0.5, 1300.0) - 1300.0) < 1e-9
    # with the box far from the centre the mapping still reaches both ends
    wide = gaussian_around("a", 2000.0, lower=300.0, upper=8000.0)
    assert abs(wide.transform(0.0, 1300.0) - 300.0) < 1e-9
    assert abs(wide.transform(1.0, 1300.0) - 8000.0) < 1e-6


def test_transform_is_the_truncated_normal_cdf():
    """The inverse CDF round-trips: CDF(theta(u)) == u to ~1e-10."""
    centre, width, lower, upper = 1300.0, 260.0, 300.0, 8000.0
    prior = gaussian_around("a", 0.20, relative=True, lower=lower, upper=upper)
    p_lo, p_hi = ndtr((lower - centre) / width), ndtr((upper - centre) / width)
    us = np.linspace(1e-5, 1.0 - 1e-5, 2001)
    values = np.array([prior.transform(float(u), centre) for u in us])
    back = (ndtr((values - centre) / width) - p_lo) / (p_hi - p_lo)
    assert np.max(np.abs(back - us)) < 1e-9


def test_monotone_and_inside_the_box():
    """A fine sweep of u neither reverses nor leaves the box."""
    prior = gaussian_around("a", 0.5, relative=True, lower=100.0, upper=10000.0)
    values = np.array([prior.transform(float(u), 2500.0)
                       for u in np.linspace(0.0, 1.0, 1001)])
    assert np.all(np.diff(values) > 0.0)
    assert values[0] >= 100.0 and values[-1] <= 10000.0


def test_width_semantics():
    """Absolute, relative and scale_from are the three documented widths."""
    absolute = gaussian_around("a", 100.0, lower=0.0, upper=1e5)
    # +1 sigma sits at quantile Phi(1) = 0.8413 of the *truncated* distribution;
    # with the box far away that is the untruncated quantile
    assert abs(absolute.transform(0.8413447, 5000.0) - 5100.0) < 1.0

    relative = gaussian_around("a", 0.10, relative=True, lower=0.0, upper=1e5)
    assert abs(relative.transform(0.8413447, 5000.0) - 5500.0) < 1.0

    from_other = gaussian_around("a", 0.05, scale_from="b", lower=-1e4, upper=1e4)
    # width = 0.05 * 1300 = 65, so +1 sigma is the centre plus 65
    assert abs(from_other.transform(0.8413447, 0.0, 1300.0) - 65.0) < 0.2


def test_narrow_tie_reproduces_a_hard_tie():
    """As sigma -> 0 the parameter is the centre, which is what a tie means."""
    prior = gaussian_around("a", 1e-6, lower=300.0, upper=8000.0)
    values = np.array([prior.transform(float(u), 2483.0)
                       for u in np.linspace(0.0, 1.0, 101)])
    assert np.max(np.abs(values - 2483.0)) < 1e-3


def test_log_space_keeps_a_parameter_positive():
    """A log-space tie is multiplicative: u = 1/2 is the centre, never <= 0."""
    prior = gaussian_around("a", 0.2, log_space=True, lower=1.0, upper=1e5)
    assert abs(prior.transform(0.5, 1300.0) - 1300.0) < 1e-6
    values = np.array([prior.transform(float(u), 1300.0)
                       for u in np.linspace(0.0, 1.0, 501)])
    assert np.all(values > 0.0) and np.all(np.diff(values) > 0.0)


def test_transform_survives_jit_and_vmap():
    """The sampler calls it traced and batched."""
    prior = gaussian_around("a", 400.0, lower=0.0, upper=4000.0)
    single = jax.jit(lambda u, c: prior.transform(u, c))
    assert abs(float(single(jnp.asarray(0.5), jnp.asarray(2000.0))) - 2000.0) < 1e-6
    us = jnp.linspace(0.0, 1.0, 7)
    batched = jax.vmap(lambda u: prior.transform(u, 2000.0))(us)
    assert batched.shape == (7,)
    assert np.all(np.isfinite(np.asarray(batched)))
    # a width taken from another parameter stays traced: no float() of a tracer
    scaled = gaussian_around("a", 0.05, scale_from="b", lower=-6000.0, upper=6000.0)
    traced = jax.jit(lambda u, c, s: scaled.transform(u, c, s))
    assert abs(float(traced(jnp.asarray(0.5), jnp.asarray(100.0), jnp.asarray(2000.0)))
               - 100.0) < 1e-6


# --------------------------------------------------------------------------------------
# Dependency resolution
# --------------------------------------------------------------------------------------
def test_resolution_orders_dependencies_first():
    """A chain x <- y <- z is evaluated in that order."""
    names = ["z", "y", "x"]
    full = ["M.z", "M.y", "M.x"]
    priors = {
        "M.y": gaussian_around("M.x", 1.0),
        "M.z": gaussian_around("M.y", 1.0, scale_from="M.x"),
    }
    order, centered = resolve_prior_dependencies(names, full, priors)
    assert order.index(2) < order.index(1) < order.index(0)
    assert sorted(centered) == [0, 1]
    # z (index 0) is centred on y (1) and takes its width from x (2)
    assert centered[0][1] == 1 and centered[0][2] == 2
    # y (index 1) is centred on x (2) and has a fixed width
    assert centered[1][1] == 2 and centered[1][2] is None


def test_resolution_rejects_bad_dependencies():
    """Cycles, unknown names, ambiguity and self-reference are errors, not surprises."""
    names = ["a", "b"]
    full = ["M1.a", "M2.a"]
    with np.testing.assert_raises(ValueError):        # ambiguous bare name
        resolve_prior_dependencies(names, full, {"a": gaussian_around("a", 1.0)})
    with np.testing.assert_raises(ValueError):        # unknown parameter
        resolve_prior_dependencies(names, full, {"a": gaussian_around("M.nope", 1.0)})
    with np.testing.assert_raises(ValueError):        # centred on itself
        resolve_prior_dependencies(["a"], ["M1.a"], {"a": gaussian_around("M1.a", 1.0)})
    with np.testing.assert_raises(ValueError):        # two-parameter cycle
        resolve_prior_dependencies(
            ["a", "b"], ["M1.a", "M1.b"],
            {"a": gaussian_around("M1.b", 1.0), "b": gaussian_around("M1.a", 1.0)})


# --------------------------------------------------------------------------------------
# The two fitters must agree
# --------------------------------------------------------------------------------------
def _compound_pair(priors):
    jax_model = (jxm.Const1D(amplitude=1.0, name="Cont")
                 + jxm.PowerLaw1D(amplitude=1.0, x_0=5000.0, alpha=2.0, name="C"))
    ast_model = (models.Const1D(amplitude=1.0, name="Cont")
                 + models.PowerLaw1D(amplitude=1.0, x_0=5000.0, alpha=2.0, name="C"))
    wave = np.linspace(4000.0, 6000.0, 200)
    flux = np.full_like(wave, 1.0)
    err = np.full_like(wave, 0.01)
    return (JNesty_Fit(jax_model, wave, flux, err, custom_priors=priors,
                       nlive=20, device="cpu"),
            Dynesty_Fit(ast_model, wave, flux, err, custom_priors=priors))


def test_fitters_share_the_same_transform():
    """Same model, same custom_priors, same u -> same theta (the parity promise)."""
    priors = {"C.alpha": gaussian_around("C.x_0", 300.0)}
    jax_fit, cpu_fit = _compound_pair(priors)
    rng = np.random.default_rng(0)
    u = rng.random(jax_fit.ndim)
    theta_jax = np.asarray(jax_fit._prior.transform(jnp.asarray(u)))
    theta_cpu = cpu_fit.prior_transform(u)
    assert np.max(np.abs(theta_jax - theta_cpu)) < 1e-9


def test_log_scaled_parameter_uses_the_physical_box():
    """An amplitude's box lives in log10 units; the tie must undo that."""
    priors = {"C.amplitude": gaussian_around("Cont.amplitude", 0.2, relative=True)}
    jax_fit, cpu_fit = _compound_pair(priors)
    index = jax_fit.param_names.index("amplitude_1")
    prior, center, _ = jax_fit.centered_priors[index]
    assert prior.lower < 1.0 < prior.upper and prior.lower > 0.0
    u = np.full(jax_fit.ndim, 0.5)
    theta_jax = np.asarray(jax_fit._prior.transform(jnp.asarray(u)))
    theta_cpu = cpu_fit.prior_transform(u)
    assert np.max(np.abs(theta_jax - theta_cpu)) < 1e-9
    # at u = 1/2 the tied amplitude is the centre, which is the other amplitude
    assert abs(theta_jax[index] - theta_jax[0]) < 1e-6


# --------------------------------------------------------------------------------------
# Does it constrain without forcing?
# --------------------------------------------------------------------------------------
def test_planted_difference_is_recovered():
    """Two lines, amplitudes 1.0 and 1.3, tie width 0.1: the 1.3 must survive."""
    wave = np.linspace(4900.0, 5600.0, 1401)
    truth_1, truth_2, noise = 1.0, 1.3, 0.02
    model = (jxm.Gaussian1D(amplitude=1.0, mean=5000.0, stddev=20.0, name="L1",
                            fixed={"mean": True, "stddev": True})
             + jxm.Gaussian1D(amplitude=1.0, mean=5500.0, stddev=20.0, name="L2",
                              fixed={"mean": True, "stddev": True}))
    theta = np.asarray(model.get_free_params(), dtype=float)
    theta[model.free_param_names.index("amplitude_0")] = truth_1
    theta[model.free_param_names.index("amplitude_1")] = truth_2
    model.prepare(wave)
    flux = np.asarray(model.flux(wave, theta))
    rng = np.random.default_rng(42)
    flux = flux + rng.normal(0.0, noise, flux.shape)

    # L2 may differ from L1 by 10% of L1, which is 0.1: the 0.3 difference is 3 sigma
    priors = {"L2.amplitude": gaussian_around("L1.amplitude", 0.10, relative=True)}
    fit = JNesty_Fit(model, wave, flux, np.full_like(wave, noise),
                     custom_priors=priors, nlive=400, device="cpu")
    fit.fit(progress=False, dlogz=0.5, max_iterations=40000)
    lo, median, hi = fit.get_quantiles((0.16, 0.5, 0.84))
    names = fit.full_param_names
    first = names.index("L1.amplitude")
    second = names.index("L2.amplitude")
    # the data dominate the tie: both amplitudes come back
    assert abs(median[first] - truth_1) < 0.05
    assert abs(median[second] - truth_2) < 0.05
    # and the tie did not drag the second one onto the first
    assert median[second] - median[first] > 0.2

# --------------------------------------------------------------------------------------
# The plot has to say what the prior is, not just where its box ends
# --------------------------------------------------------------------------------------
def test_posterior_hist_names_the_centre_of_the_prior():
    """A centred prior is described as N(centre, width); a plain one keeps its box."""
    import matplotlib                                                    # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                      # noqa: E402
    from galspec.plot import plot_posterior_hist                         # noqa: E402

    rng = np.random.default_rng(3)
    samples = np.column_stack([rng.normal(1301.0, 7.0, 4000),
                               rng.normal(1300.0, 6.0, 4000)])

    class _Fit:
        samples_equal_weight = samples
        param_names = ["sigma_c_0", "sigma_c_1"]
        full_param_names = ["b_halpha.sigma_c", "b_hbeta.sigma_c"]
        param_labels = [r"BH$\alpha$:$\sigma_{c}$", r"BH$\beta$:$\sigma_{c}$"]
        param_bounds = {"sigma_c_0": (300.0, 8000.0), "sigma_c_1": (300.0, 8000.0)}
        centered_priors = {1: (gaussian_around("b_halpha.sigma_c", 0.20,
                                               relative=True, lower=300.0,
                                               upper=8000.0),
                               "b_halpha.sigma_c", None)}
        param_map = {"b_halpha.sigma_c": 0, "b_hbeta.sigma_c": 1}
        _log_scale_params = set()

    fig, axes = plot_posterior_hist(_Fit(), params=["b_hbeta.sigma_c"])
    texts = [text.get_text() for text in axes.ravel()[0].texts]
    plt.close(fig)
    assert any("$N$(" in text and "20%" in text for text in texts), texts

    fig, axes = plot_posterior_hist(_Fit(), params=["b_halpha.sigma_c"])
    texts = [text.get_text() for text in axes.ravel()[0].texts]
    plt.close(fig)
    assert any(text.startswith("prior [") for text in texts), texts
