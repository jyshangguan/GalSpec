"""Tests for the parameter-level prior API: ``model.<name>.prior = ...``.

The feature is one idea -- a parameter's prior is set *on the parameter*, the way
its ``value``, ``fixed`` and ``bounds`` already are -- and a handful of ways to get
it wrong:

* a prior must be normalisable, so ``Uniform(2, 1)`` or ``Normal`` over a negative
  width must be rejected at construction, not at sampling time;
* it must survive the trip through the sampler: the model's prior has to reach the
  nested-sampling unit-cube map unchanged;
* a centred prior (one parameter's prior depends on another's value) has to be
  drawn *after* the parameter it depends on -- and the order has to be found from
  the priors, not written down by hand;
* the ``nsigma`` limit must be a real hard limit that scales with the parent, not a
  suggestion;
* it must round-trip through a FITS file, since that is how a fit is archived;
* a fixed or tied parameter has no prior, and saying otherwise is an error, not a
  silent no-op.

Run with::

    python -m pytest tests/test_prior_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402
jax.config.update("jax_enable_x64", True)

from galspec import jax_models as jxm                               # noqa: E402
from galspec import priors as priors_                               # noqa: E402
from galspec.jnesty_fit import JNesty_Fit                           # noqa: E402
from galspec.priors import (CallablePrior, LogUniform, Normal,      # noqa: E402
                            Prior, SpikeAndSlab, Uniform, as_prior)


# --------------------------------------------------------------------------------------
# The distributions on their own
# --------------------------------------------------------------------------------------
def test_uniform_is_flat_and_normalisable():
    prior = Uniform(2.0, 5.0)
    assert prior.logpdf(2.0) == prior.logpdf(4.9)
    assert prior.logpdf(1.9) == -np.inf and prior.logpdf(5.1) == -np.inf
    assert np.isclose(np.exp(prior.logpdf(3.0)), 1.0 / 3.0)
    # the map is the inverse of the CDF: a quarter of the way along is a quarter
    # of the way through
    assert np.isclose(prior(0.25), 2.75)


def test_log_uniform_puts_equal_weight_per_decade():
    prior = LogUniform(1.0, 1000.0)
    # one decade in three: a third of the unit interval, wherever the decade is
    assert np.isclose(prior(1.0 / 3.0), 10.0)
    assert np.isclose(prior(2.0 / 3.0), 100.0)
    assert np.isclose(prior(0.0), 1.0) and np.isclose(prior(1.0), 1000.0)
    # which is the same as saying that the mass *per decade* is uniform: the
    # density in log10(x) is flat over the three decades, i.e. 1/3 each
    for value in (2.0, 10.0, 300.0):
        log10_density = np.exp(prior.logpdf(value)) * value * np.log(10.0)
        assert np.isclose(log10_density, 1.0 / 3.0)


def test_priors_reject_impossible_arguments():
    with pytest.raises(ValueError):
        Uniform(5.0, 2.0)
    with pytest.raises(ValueError):
        LogUniform(0.0, 10.0)                    # zero is not in the domain
    with pytest.raises(ValueError):
        Normal(10.0, sigma=-1.0)
    with pytest.raises(ValueError):
        Normal(10.0, sigma=1.0, nsigma=0.0)
    with pytest.raises(ValueError):
        SpikeAndSlab(p_off=1.5)


def test_spike_and_slab_keeps_a_real_share_of_the_prior_at_zero():
    """The "switch the component off" prior: exactly zero, not nearly zero.

    Half the unit interval maps onto the single value 0, so the sampler can report
    a probability that the component is absent.  Note the two different footings
    ``logpdf`` reports there -- log *mass* on the spike, log *density* on the slab
    -- which is exactly why this prior needs a nested sampler rather than a
    gradient sampler: a point mass has no gradient.
    """
    prior = SpikeAndSlab(p_off=0.5, lower=0.0, upper=1.0)
    assert prior(0.4) == 0.0                       # the spike: half the interval
    assert np.isclose(np.exp(prior.logpdf(0.0)), 0.5)   # ... and its mass
    assert 0.0 < prior(0.75) <= 1.0                # the slab: the other half
    assert np.isclose(prior(1.0), 1.0)
    assert np.isfinite(prior.logpdf(0.5))          # a density on the slab
    assert prior.logpdf(1.5) == -np.inf            # nowhere else
    # the map is monotone across the whole cube, spike included
    values = np.array([prior(u) for u in np.linspace(0.0, 1.0, 101)])
    assert np.all(np.diff(values) >= 0.0)


def test_as_prior_accepts_the_old_callables():
    """A plain callable is still a prior -- wrapped, so it can be stored and read."""
    wrapped = as_prior(lambda u: 10.0 ** (-3.0 + 4.0 * u))
    assert isinstance(wrapped, Prior)
    assert np.isclose(wrapped(0.5), 10.0 ** -1.0)
    # a bounds pair is a uniform prior
    assert np.isclose(as_prior((1.0, 3.0)).logpdf(2.0), -np.log(2.0))


# --------------------------------------------------------------------------------------
# A prior on a parameter
# --------------------------------------------------------------------------------------
def simple_model():
    """``PowerLaw1D + Line_MultiGauss``: a handful of free numbers, easy to reason about."""
    return (jxm.PowerLaw1D(amplitude=1.0, x_0=5000.0, alpha=0.0, name="Cont")
            + jxm.Line_MultiGauss(n_components=2, wavec=6563.0, name="Line",
                                  dv_c=0.0, sigma_c=500.0, amp_c=1.0,
                                  dv_w0=0.0, sigma_w0=2000.0, amp_w0=0.2))


def test_prior_is_set_on_the_parameter_and_read_back():
    model = simple_model()
    assert model.get_prior("Cont.amplitude") is None
    model.set_prior("Cont.amplitude", Uniform(0.1, 10.0))
    prior = model.get_prior("Cont.amplitude")
    assert isinstance(prior, Uniform) and (prior.lower, prior.upper) == (0.1, 10.0)
    # the same object is on the parameter itself, which is where the docs say it is
    assert model["Cont"].amplitude.prior is prior
    # and it shows up in the model's own list
    assert "Cont.amplitude" in model.priors
    # setting a new one replaces it; None clears it
    model.set_prior("Cont.amplitude", LogUniform(0.1, 10.0))
    assert isinstance(model.get_prior("Cont.amplitude"), LogUniform)
    model.set_prior("Cont.amplitude", None)
    assert model.get_prior("Cont.amplitude") is None


def test_prior_on_a_fixed_or_tied_parameter_is_an_error():
    model = simple_model()
    model["Cont"].alpha.fixed = True
    with pytest.raises(ValueError, match="fixed"):
        model.set_prior("Cont.alpha", Uniform(0.0, 1.0))
    with pytest.raises(KeyError):
        model.set_prior("Continuum.no_such_parameter", Uniform(0.0, 1.0))


def test_prior_reaches_the_sampler_unchanged():
    model = simple_model()
    model.set_prior("Cont.amplitude", LogUniform(0.1, 10.0))
    model.set_prior("Cont.alpha", Uniform(-2.0, 2.0))
    model.set_prior("Line.sigma_c", Uniform(100.0, 900.0))
    wave = np.linspace(6400.0, 6700.0, 64)
    fit = JNesty_Fit(model, wave, np.ones_like(wave), np.ones_like(wave) * 0.1,
                     nlive=10, device="cpu")
    names = list(fit.full_param_names)
    index = names.index("Cont.amplitude")
    assert fit._prior.custom[index] is model.get_prior("Cont.amplitude")
    # and the map really is the prior's own: the midpoint of a 0.1-10 log-uniform
    # is 1.0, which is what a linear box would get wrong
    u = np.full(fit.ndim, 0.5)
    theta = np.asarray(fit.prior_transform(u))
    assert np.isclose(theta[index], 1.0)
    # a linear prior's midpoint is just the middle of its box
    assert np.isclose(theta[names.index("Cont.alpha")], 0.0)


# --------------------------------------------------------------------------------------
# One parameter's prior depends on another
# --------------------------------------------------------------------------------------
def tied_model():
    """Broad H-alpha plus broad H-beta, the two profiles independent for now."""
    return (jxm.Line_MultiGauss(n_components=2, wavec=6563.0, name="Halpha",
                                dv_c=0.0, sigma_c=500.0, amp_c=10.0,
                                dv_w0=0.0, sigma_w0=2000.0, amp_w0=0.2)
            + jxm.Line_MultiGauss(n_components=2, wavec=4861.0, name="Hbeta",
                                  dv_c=0.0, sigma_c=500.0, amp_c=3.0,
                                  dv_w0=0.0, sigma_w0=2000.0, amp_w0=0.2))


def test_a_prior_can_be_written_in_terms_of_another_parameter():
    """The whole point: ``sigma_w0`` of H-beta is a Gaussian around H-alpha's."""
    model = tied_model()
    model.set_prior("Halpha.sigma_w0", Uniform(300.0, 8000.0))
    model.set_prior("Hbeta.sigma_w0",
                    Normal(mean="Halpha.sigma_w0", relative=0.2, nsigma=5))
    prior = model.get_prior("Hbeta.sigma_w0")
    assert prior.depends_on == ("Halpha.sigma_w0",)
    # the box it is truncated to is the parameter's own box, filled in from the
    # reference's range by the fitter
    wave = np.linspace(6400.0, 6700.0, 64)
    fit = JNesty_Fit(model, wave, np.ones_like(wave), np.ones_like(wave) * 0.1,
                     nlive=10, device="cpu")
    names = list(fit.full_param_names)
    # the dependency is drawn first, whatever order the parameters are in
    assert names.index("Halpha.sigma_w0") < names.index("Hbeta.sigma_w0")
    # the centre is resolved to the index of the parameter it reads, so the
    # sampler can look the value up without ever matching a string
    child, centre_index, width_from = fit.centered_priors[names.index("Hbeta.sigma_w0")]
    assert centre_index == names.index("Halpha.sigma_w0")
    assert width_from is None
    # and the offset is a fraction of the parent, at the same u: five times the
    # parent means five times the offset, i.e. the cap scales with the parent
    narrow = child.transform(0.9, 1000.0)
    wide = child.transform(0.9, 5000.0)
    assert wide > narrow > 1000.0
    assert np.isclose(wide / 5000.0, narrow / 1000.0)


def test_nsigma_is_a_hard_limit_that_scales_with_the_parent():
    """``nsigma`` means "never more than that many tie widths out", at any parent."""
    prior = Normal(mean="a", relative=0.2, nsigma=3, mini=0.0, maxi=1e6)
    for parent in (100.0, 1000.0, 10000.0):
        low = prior.transform(0.0, parent)
        high = prior.transform(1.0, parent)
        assert np.isclose(low, parent * (1.0 - 3 * 0.2))
        assert np.isclose(high, parent * (1.0 + 3 * 0.2))
    # beyond the limit the density is exactly zero -- not merely small
    limit = prior.transform(1.0, 1000.0)
    assert prior.logpdf(limit + 1.0, 1000.0) == -np.inf
    # and inside it, the same point is allowed
    assert np.isfinite(prior.logpdf(limit - 1.0, 1000.0))
    # a five-sigma limit that would reach zero or below is simply not truncating
    # there: the parameter's own box is what stops it
    tight = Normal(mean="a", relative=0.2, nsigma=5, mini=0.0, maxi=1e6)
    assert np.isclose(tight.transform(1.0, 1000.0), 2000.0)


def test_a_circular_dependency_is_reported_not_hung():
    model = tied_model()
    model.set_prior("Halpha.sigma_w0", Normal(mean="Hbeta.sigma_w0", relative=0.2))
    model.set_prior("Hbeta.sigma_w0", Normal(mean="Halpha.sigma_w0", relative=0.2))
    wave = np.linspace(6400.0, 6700.0, 64)
    with pytest.raises(ValueError, match="cic|cycl|order"):     # noqa: PT012
        JNesty_Fit(model, wave, np.ones_like(wave), np.ones_like(wave) * 0.1,
                   nlive=10, device="cpu")


# --------------------------------------------------------------------------------------
# Compatibility and persistence
# --------------------------------------------------------------------------------------
def test_a_model_prior_and_a_legacy_bound_agree():
    """A ``Uniform`` prior and a ``bounds_dict`` entry mean the same thing."""
    model = simple_model()
    model.set_prior("Line.sigma_c", Uniform(300.0, 800.0))
    wave = np.linspace(6400.0, 6700.0, 64)
    flux = np.ones_like(wave)
    error = np.ones_like(wave) * 0.1
    new = JNesty_Fit(model, wave, flux, error, nlive=10, device="cpu")
    legacy = JNesty_Fit(simple_model(), wave, flux, error, nlive=10, device="cpu",
                        bounds_dict={"Line.sigma_c": (300.0, 800.0)})
    assert new.param_names == legacy.param_names
    assert np.allclose(np.array(new.param_bounds), np.array(legacy.param_bounds))
    rng = np.random.default_rng(3)
    u = rng.uniform(size=(50, new.ndim))
    a = np.array([new.prior_transform(row) for row in u])
    b = np.array([legacy.prior_transform(row) for row in u])
    assert np.allclose(a, b)


def test_priors_survive_a_copy():
    model = simple_model()
    model.set_prior("Cont.amplitude", LogUniform(0.1, 10.0))
    clone = model.copy()
    assert isinstance(clone.get_prior("Cont.amplitude"), LogUniform)
    # and the copy is independent of the original
    clone.set_prior("Cont.amplitude", Uniform(1.0, 2.0))
    assert isinstance(model.get_prior("Cont.amplitude"), LogUniform)


def test_every_prior_round_trips_through_its_own_dict():
    """``to_dict``/``from_dict`` is what a FITS file stores, so it must be lossless."""
    priors = [
        Uniform(1.0, 2.0), LogUniform(1e-3, 10.0),
        SpikeAndSlab(p_off=0.3, lower=0.0, upper=1.0),
        Normal(mean="a", relative=0.2, nsigma=5, mini=0.0, maxi=100.0),
        Normal(mean="a", sigma=0.05, scale_from="b", nsigma=4),
        Normal(mean=1000.0, sigma=100.0),
        Normal(mean="a", sigma=0.2, log=True, nsigma=6),   # in log space
    ]
    for prior in priors:
        spec = prior.to_dict()
        rebuilt = Prior.from_dict(spec)
        assert type(rebuilt) is type(prior)
        assert rebuilt.depends_on == prior.depends_on
        if prior.support is not None:
            assert rebuilt.support == prior.support
    # a prior wrapped around user code cannot be written down, and says so rather
    # than being silently dropped
    wrapped = CallablePrior(lambda u: u, lower=0.0, upper=1.0)
    assert wrapped.serialisable is False
    with pytest.raises(TypeError, match="cannot be written"):
        wrapped.to_dict()


# --------------------------------------------------------------------------------------
# The transform is a bijection of the unit cube
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("prior, low, high", [
    (Uniform(300.0, 8000.0), 300.0, 8000.0),
    (LogUniform(1e-3, 10.0), 1e-3, 10.0),
    # a Gaussian is truncated to the box *and* to seven widths, whichever is
    # narrower -- here the seven widths are: 1000 -+ 7 x 100
    (Normal(mean=1000.0, sigma=100.0, mini=0.0, maxi=5000.0), 300.0, 1700.0),
])
def test_the_transform_is_monotone_and_reaches_both_ends(prior, low, high):
    u = np.linspace(0.0, 1.0, 101)
    values = np.array([prior(x) for x in u])
    assert np.isclose(values[0], low) and np.isclose(values[-1], high)
    assert np.all(np.diff(values) > 0.0)


def test_the_density_integrates_to_one_over_the_allowed_range():
    """The exposed log density is a *normalised* density, which is what a
    gradient sampler would need -- not just a shape."""
    prior = Normal(mean="a", relative=0.2, nsigma=5, mini=0.0, maxi=1e6)
    grid = np.linspace(0.0, 8000.0, 200001)          # exactly the 5-width range
    density = np.exp(prior.logpdf(grid, 4000.0))
    assert np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-6)
    # and it takes an array, so it can be drawn or passed to a sampler as a batch
    assert prior.logpdf(grid, 4000.0).shape == grid.shape
    assert prior.logpdf(8001.0, 4000.0) == -np.inf


def test_dv_and_sigma_use_the_same_machinery():
    """The two priors the user asked about, side by side, in one place."""
    sigma = Normal(mean="Halpha.sigma_c", relative=0.2, nsigma=5, mini=0.0, maxi=1e5)
    dv = Normal(mean="Halpha.dv_c", sigma=0.05, scale_from="Halpha.sigma_c",
                nsigma=5, mini=-2500.0, maxi=2500.0)
    # a width: +-5 x 20% of the parent's own value
    assert np.isclose(sigma.transform(1.0, 1000.0), 2000.0)
    # a shift: +-5 x 5% of the width the parent may have, not of the shift
    assert np.isclose(dv.transform(1.0, 0.0, 1000.0), 250.0)
    # the shift's own box still wins when it is narrower than the limit
    assert np.isclose(dv.transform(1.0, 0.0, 40000.0), 2500.0)