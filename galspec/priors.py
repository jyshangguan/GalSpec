''' Prior transforms for the Bayesian fitters.

`JNesty_Fit`, and `Dynesty_Fit` all accept a ``custom_priors`` dictionary that
replaces the default box prior of individual parameters.  A prior transform
maps the unit cube coordinate ``u`` of one parameter onto its physical value,
so the same callable works for both the JAX samplers (where it is traced) and
the numpy ones.  The functions here are the ones that are more than a one-line
lambda.

The one worth reading about is `component_toggle`: it turns an amplitude into a
**spike and slab** prior, i.e. a prior that gives the component a finite chance
of being *exactly* absent.  That is what lets a fixed-dimension sampler explore
"how many components does this profile need?" inside a single run.
'''

import numpy as np

__all__ = ['log_uniform', 'component_toggle', 'uniform_toggle',
           'log_uniform_toggle', 'toggle_prior_odds']


def log_uniform(lower, upper):
    ''' Log-uniform (scale-free) prior on ``(lower, upper)``, ``lower > 0``.

    Parameters
    ----------
    lower, upper : float
        Positive bounds of the parameter.

    Returns
    -------
    transform : callable
        ``transform(u)`` maps ``u`` in [0, 1] onto the parameter.

    Examples
    --------
    >>> amp = log_uniform(1e-3, 3.0)
    >>> amp(0.0), round(float(amp(0.5)), 3)
    (0.001, 0.0547...)
    '''
    lower, upper = float(lower), float(upper)
    if not 0.0 < lower < upper:
        raise ValueError('log_uniform needs 0 < lower < upper, got '
                         f'{lower} and {upper}')
    log_lo, log_hi = np.log10(lower), np.log10(upper)

    def transform(u):
        return 10.0 ** (log_lo + (log_hi - log_lo) * u)

    transform.lower, transform.upper = lower, upper
    return transform


def component_toggle(p_off=0.5, lower=0.0, upper=1.0, slab='linear'):
    ''' Spike-and-slab prior that lets a component be switched off.

    A fraction ``p_off`` of the unit interval maps onto **exactly zero** -- the
    component is absent, contributes nothing to the model, and needs no
    likelihood term of its own -- while the remaining ``1 - p_off`` maps onto
    the slab ``(lower, upper)``.  The zero is a *plateau*, not a delta function
    at a single point: an "off" state that owned a vanishing fraction of the
    prior volume would be one that nested sampling could never find.

    Because both states have real prior volume, the posterior probability that
    the component is on comes out of the sampler directly, and the Bayes factor
    for "on" is the posterior odds divided by the prior odds::

        BF = (P_on / P_off) / toggle_prior_odds(p_off)

    Parameters
    ----------
    p_off : float
        Prior probability that the component is off.  ``0.5`` gives
        even odds, which makes the posterior odds equal the Bayes factor.
        A ladder such as ``0.2, 0.5, 0.7`` for successive components expresses
        "higher-order components are less likely a priori", which shrinks them
        without prejudice to their width or shift.
    lower, upper : float
        Slab bounds for the amplitude when the component is on.  ``lower`` must
        be below the smallest amplitude that should count as "on".
    slab : {'linear', 'log'}
        Shape of the slab.  The default linear slab is the one to use for this
        amplitude: a log-uniform slab puts most of its density at the small
        amplitudes where an extra Gaussian can hide in the noise, and on mock
        data that produced spurious components at ~1% of the core with a
        Bayes factor of 400.  Use ``'log'`` only with ``lower`` well above
        zero, e.g. through `log_uniform_toggle`.

    Returns
    -------
    transform : callable
        ``transform(u)`` maps ``u`` in [0, 1] onto ``{0} U (lower, upper)``.
        It is written with ``jnp.where`` so it survives ``jax.jit`` and
        ``jax.vmap``; the array namespace is imported on the first call so that
        plain ``import galspec`` does not pay for JAX.
    '''
    if not 0.0 <= p_off < 1.0:
        raise ValueError(f'p_off must be in [0, 1), got {p_off}')
    lower, upper = float(lower), float(upper)
    if not lower < upper:
        raise ValueError(f'need lower < upper, got {lower} and {upper}')
    slab = str(slab).lower()
    if slab == 'log':
        if lower <= 0.0:
            raise ValueError('a log slab needs lower > 0; a component that '
                             'may be "on" at arbitrarily small amplitude is '
                             'not distinguishable from "off"')
        log_lo, log_hi = np.log10(lower), np.log10(upper)

        def slab_value(t):
            return 10.0 ** (log_lo + (log_hi - log_lo) * t)
    elif slab in ('linear', 'uniform'):
        width = upper - lower

        def slab_value(t):
            return lower + width * t
    else:
        raise ValueError(f'unknown slab {slab!r}; use "linear" or "log"')

    def transform(u):
        import jax.numpy as jnp
        # t runs 0 -> 1 across the slab; it is also evaluated on the spike,
        # where the where() discards it (t is negative there, never NaN).
        t = (u - p_off) / (1.0 - p_off)
        return jnp.where(u < p_off, 0.0, slab_value(t))

    transform.p_off = p_off
    transform.lower, transform.upper, transform.slab = lower, upper, slab
    return transform


def uniform_toggle(p_off=0.5, upper=1.0, lower=0.0):
    ''' Spike at zero plus a **linear** slab: the recommended toggle.

    Shorthand for ``component_toggle(p_off, lower, upper, slab='linear')``, for
    a component amplitude normalised by the line core.
    '''
    return component_toggle(p_off=p_off, lower=lower, upper=upper, slab='linear')


def log_uniform_toggle(p_off=0.5, lower=0.02, upper=1.0):
    ''' Spike at zero plus a **log-uniform** slab on ``(lower, upper)``.

    Scale-free in the amplitude, at the price of a soft floor: ``lower`` is the
    smallest amplitude that counts as "on", so ``lower=0.02`` means "wings
    weaker than 2% of the core peak are not modelled at all".  Keep ``lower``
    at a few per cent.  Measured on a 1%-noise mock with the slab running from
    1e-3: an absent second wing was reported at 1.2-1.5% of the core peak with
    a Bayes factor of 391 -- the log slab is densest exactly where a
    noise-absorbing component wants to sit.
    '''
    return component_toggle(p_off=p_off, lower=lower, upper=upper, slab='log')


def toggle_prior_odds(p_off):
    ''' Prior odds of "on" over "off" for a toggle prior: ``(1-p_off)/p_off``.

    Divide the posterior odds by this to get the Bayes factor, so that runs
    with different ``p_off`` can be compared on the same scale.
    '''
    if not 0.0 < p_off < 1.0:
        raise ValueError(f'p_off must be in (0, 1) to have finite odds, '
                         f'got {p_off}')
    return (1.0 - p_off) / p_off