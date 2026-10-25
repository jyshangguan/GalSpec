''' Prior distributions for the Bayesian fitters.

A prior says how likely each value of a parameter is before the data are seen.
Every prior here is one object with a single job: it turns a random number
``u`` between 0 and 1 into a parameter value.  That is all a nested sampler can
consume, and it is why a prior can double as the sampler's coordinate map: the
fraction of the unit interval that maps onto a value *is* the probability of
that value.

Attach a prior to a parameter of a JAX model, exactly like ``bounds`` or
``fixed``::

    import galspec.priors as priors

    b_halpha.sigma_w0.prior = priors.Uniform(800.0, 25000.0)
    b_halpha.amp_c.prior = priors.LogUniform(1e-3, 3.0)

    # a Gaussian centred on another parameter, capped at five tie widths
    b_hbeta.sigma_w0.prior = priors.Normal(mean="b_halpha.sigma_w0",
                                           relative=0.20, nsigma=5)

Two features are worth reading about:

`Normal` (and its older spelling `gaussian_around`) makes the prior of one
parameter a Gaussian centred on *the value of another parameter*, so a model can
say "similar, but not the same" -- broad H-beta sharing broad H-alpha's profile
shape, a narrow line sitting at the same velocity as its doublet partner.  Unlike
a tie, which forces equality, this keeps the constraint soft and lets the data
report the difference and its uncertainty.  ``nsigma`` then says how many
standard deviations of that difference are allowed at all: it is a hard stop, not
a wider Gaussian.

`SpikeAndSlab` (the documented spelling of `component_toggle`) turns an amplitude
into a prior that gives the component a finite chance of being *exactly* absent.
That is what lets a fixed-dimension sampler explore "how many components does
this profile need?" inside a single run.

`uniform_toggle` and `log_uniform` remain as the older callable-returning
factories: the fitters accept both.
'''

import numpy as np

__all__ = ['Prior', 'Uniform', 'TopHat', 'LogUniform', 'Normal',
           'ClippedNormal', 'SpikeAndSlab', 'CallablePrior', 'as_prior',
           'log_uniform', 'component_toggle', 'uniform_toggle',
           'log_uniform_toggle', 'toggle_prior_odds',
           'gaussian_around', 'CenteredPrior', 'resolve_prior_dependencies']


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
    >>> round(float(amp(0.0)), 4), round(float(amp(0.5)), 3)
    (0.001, 0.055)
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

# --------------------------------------------------------------------------------------
# Priors that read another parameter
# --------------------------------------------------------------------------------------
#: sqrt(2), written out so the transform stays a pure arithmetic expression.
_SQRT2 = 1.4142135623730951
#: How far from its centre a truncated Gaussian is resolved.  Beyond 7 sigma the
#: Gaussian's mass is 1.3e-12 and float64 has only a few units left in the
#: cumulative probability, so the tail is cut there rather than misrepresented.
_N_SIGMA = 7.0


def _erf(x):
    ''' Error function of the array namespace ``x`` belongs to.

    The same transform has to run on numpy arrays (the CPU fitters), on plain
    Python floats (callers who try one value) and on jax tracers (inside
    ``jax.jit``/``jax.vmap``), so the namespace is picked from the *type* of the
    input rather than from an ``import`` at module load.  That keeps
    ``import galspec`` free of JAX.
    '''
    if type(x).__module__.startswith('jax'):
        from jax.scipy.special import erf
    else:
        from scipy.special import erf
    return erf(x)


def _erfinv(x):
    ''' Inverse error function, in the namespace of ``x`` (see `_erf`). '''
    if type(x).__module__.startswith('jax'):
        from jax.scipy.special import erfinv
    else:
        from scipy.special import erfinv
    return erfinv(x)


def _clip(x, lower, upper):
    ''' ``clip`` in the namespace of ``x`` (see `_erf`). '''
    if type(x).__module__.startswith('jax'):
        import jax.numpy as jnp
        return jnp.clip(x, lower, upper)
    return np.clip(x, lower, upper)


def _normal_cdf(x):
    ''' Standard normal cumulative distribution function. '''
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _log(x):
    ''' Natural logarithm in the namespace of ``x`` (see `_erf`). '''
    if type(x).__module__.startswith('jax'):
        import jax.numpy as jnp
        return jnp.log(x)
    return np.log(x)


# --------------------------------------------------------------------------------------
# The prior objects
# --------------------------------------------------------------------------------------
class Prior:
    ''' Base class for the prior of a single parameter.

    A prior is called with a unit-cube coordinate and returns the parameter
    value, so the same object serves as the sampler's coordinate map.  It also
    reports its own log density (`logpdf`), which is what a gradient sampler or a
    diagnostic would ask for.

    Subclasses implement ``__call__``; a prior that reads *another parameter*
    (see `CenteredPrior`) has nothing to read from a bare ``u`` and raises there,
    because the fitter has to supply the other parameter's value first.
    '''

    #: Names of the other parameters this prior reads, if any.
    depends_on = ()

    #: ``False`` for priors that wrap a user callable and cannot be written to a
    #: FITS file.
    serialisable = True

    @property
    def support(self):
        ''' ``(lower, upper)`` in physical units, or ``None`` if the model's
        own bounds should be used. '''
        return None

    def __call__(self, u):
        ''' Map ``u`` in [0, 1] onto the parameter value. '''
        raise NotImplementedError(f'{type(self).__name__} has no unit-cube map')

    def logpdf(self, x):
        ''' Log density at ``x`` (numpy in, numpy or float out). '''
        raise NotImplementedError(f'{type(self).__name__} has no density')

    def with_bounds(self, lower, upper):
        ''' A copy restricted to ``[lower, upper]``, when that makes a difference. '''
        return self

    def _args(self):
        raise TypeError(f'{type(self).__name__} cannot be written to a FITS file')

    def to_dict(self):
        ''' Plain-data description, used to store the prior with a fit. '''
        return {'kind': type(self).__name__, 'args': dict(self._args())}

    @classmethod
    def from_dict(cls, data):
        ''' Rebuild a prior written by `to_dict`. '''
        kind = data['kind']
        args = dict(data.get('args') or {})
        target = _PRIOR_TYPES.get(kind)
        if target is None:
            raise ValueError(f'unknown prior kind {kind!r}; known kinds are '
                             f'{sorted(_PRIOR_TYPES)}')
        return target(**args)

    def __repr__(self):
        args = ', '.join(f'{key}={value!r}' for key, value in self._args().items()
                         if value is not None)
        return f'{type(self).__name__}({args})'


class Uniform(Prior):
    ''' Flat prior between ``lower`` and ``upper``.

    The same shape as the default: a parameter with no prior of its own is
    uniform over its bounds.  Give ``Uniform`` explicitly when the bounds worth
    documenting are the prior's, not the model's.

    Examples
    --------
    >>> prior = Uniform(800.0, 25000.0)
    >>> [round(float(prior(u)), 1) for u in (0.0, 0.5, 1.0)]
    [800.0, 12900.0, 25000.0]
    >>> round(float(prior.logpdf(1000.0)) + float(np.log(25000.0 - 800.0)), 12)
    0.0
    >>> prior.logpdf(10.0)
    -inf
    '''

    def __init__(self, lower, upper):
        lower, upper = float(lower), float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise ValueError(f'Uniform needs finite lower < upper, got {lower} and {upper}')
        self.lower, self.upper = lower, upper

    @property
    def support(self):
        return (self.lower, self.upper)

    def __call__(self, u):
        return self.lower + (self.upper - self.lower) * u

    def logpdf(self, x):
        scalar = np.ndim(x) == 0
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lower) & (x <= self.upper)
        value = np.where(inside, -np.log(self.upper - self.lower), -np.inf)
        return float(value) if scalar else value

    def _args(self):
        return {'lower': self.lower, 'upper': self.upper}


class TopHat(Uniform):
    ''' Prospector's name for a flat prior; identical to `Uniform`. '''


class LogUniform(Prior):
    ''' Flat prior in ``log10(parameter)``: every decade equally likely.

    The natural prior for a quantity known only by its order of magnitude -- a
    line amplitude that could be 1% or 100% of the spectrum -- because "twice as
    bright" and "half as bright" are equally probable.

    Examples
    --------
    >>> prior = LogUniform(1e-3, 3.0)
    >>> [round(float(prior(u)), 4) for u in (0.0, 0.5, 1.0)]
    [0.001, 0.0548, 3.0]
    >>> round(float(prior.logpdf(0.0548)) + float(np.log(0.0548 * np.log(3000.0))), 10)
    0.0
    '''

    def __init__(self, lower, upper):
        lower, upper = float(lower), float(upper)
        if not 0.0 < lower < upper:
            raise ValueError(f'LogUniform needs 0 < lower < upper, got {lower} and {upper}')
        self.lower, self.upper = lower, upper
        self._log_lo, self._log_hi = np.log10(lower), np.log10(upper)

    @property
    def support(self):
        return (self.lower, self.upper)

    def __call__(self, u):
        return 10.0 ** (self._log_lo + (self._log_hi - self._log_lo) * u)

    def logpdf(self, x):
        scalar = np.ndim(x) == 0
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lower) & (x <= self.upper)
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.where(inside,
                             -np.log(x * np.log(self.upper / self.lower)), -np.inf)
        return float(value) if scalar else value

    def _args(self):
        return {'lower': self.lower, 'upper': self.upper}


class SpikeAndSlab(Prior):
    ''' Prior that lets a component be switched off: a spike at zero plus a slab.

    A fraction ``p_off`` of the unit interval maps onto **exactly zero** -- the
    component is absent and contributes nothing to the model -- while the rest
    maps onto the slab ``(lower, upper)``.  Because both states own real prior
    volume, the posterior probability that the component is on comes out of the
    sampler directly, and the Bayes factor for "on" is the posterior odds divided
    by `toggle_prior_odds`.

    This is the object form of `component_toggle`, which remains the factory.

    Parameters
    ----------
    p_off : float
        Prior probability that the component is off.
    lower, upper : float
        Slab bounds for the parameter when the component is on.
    slab : {'linear', 'log'}
        Shape of the slab.  Prefer ``'linear'``: a log slab is densest exactly
        where a noise-absorbing component wants to sit (see `log_uniform_toggle`).
    '''

    def __init__(self, p_off=0.5, lower=0.0, upper=1.0, slab='linear'):
        self._toggle = component_toggle(p_off=p_off, lower=lower, upper=upper, slab=slab)
        self.p_off, self.lower, self.upper = float(p_off), float(lower), float(upper)
        self.slab = str(slab).lower()

    @property
    def support(self):
        return (min(self.lower, 0.0), self.upper)

    def __call__(self, u):
        return self._toggle(u)

    def logpdf(self, x):
        ''' Log *mass* on the spike, log density on the slab (see the class note). '''
        scalar = np.ndim(x) == 0
        x = np.asarray(x, dtype=float)
        slab = np.where((x > self.lower) & (x <= self.upper),
                        -np.log(1.0 - self.p_off) - np.log(self.upper - self.lower)
                        if self.slab in ('linear', 'uniform')
                        else -np.log(1.0 - self.p_off) - np.log(x * np.log(self.upper / self.lower)),
                        -np.inf)
        value = np.where(x == 0.0, np.log(self.p_off), slab)
        return float(value) if scalar else value

    def _args(self):
        return {'p_off': self.p_off, 'lower': self.lower, 'upper': self.upper,
                'slab': self.slab}


class CallablePrior(Prior):
    ''' Adapter for the older ``custom_priors`` style: a bare ``u -> value``.

    It behaves like any other prior but cannot be stored in a FITS file, so a
    reloaded fit reports it as missing.
    '''

    serialisable = False

    def __init__(self, func, lower=None, upper=None):
        if not callable(func):
            raise TypeError(f'CallablePrior needs a callable, got {type(func).__name__}')
        self.func = func
        self.lower, self.upper = lower, upper
        self._repr = repr(func)

    @property
    def support(self):
        if self.lower is None or self.upper is None:
            return None
        return (float(self.lower), float(self.upper))

    def __call__(self, u):
        return self.func(u)

    def _args(self):
        raise TypeError('a CallablePrior wraps Python code and cannot be written to '
                        'a FITS file; give it as a galspec.priors object instead')

    def __repr__(self):
        return f'CallablePrior({self._repr[:60]})'


def as_prior(obj):
    ''' Turn anything the fitters accept into a `Prior` instance.

    Accepts a `Prior` (returned unchanged), a ``(lower, upper)`` pair (read as
    `Uniform`, the same meaning a box has in ``bounds_dict``), one of the older
    factories' callables (`log_uniform`, `component_toggle`, ...) or any other
    ``u -> value`` callable, which is wrapped in `CallablePrior`.
    '''
    if obj is None:
        return None
    if isinstance(obj, Prior):
        return obj
    if (isinstance(obj, (tuple, list)) and len(obj) == 2
            and all(isinstance(bound, (int, float)) for bound in obj)):
        return Uniform(*obj)
    if callable(obj):
        if hasattr(obj, 'p_off') and hasattr(obj, 'slab'):
            return SpikeAndSlab(p_off=obj.p_off, lower=getattr(obj, 'lower', 0.0),
                                upper=getattr(obj, 'upper', 1.0), slab=obj.slab)
        return CallablePrior(obj, lower=getattr(obj, 'lower', None),
                             upper=getattr(obj, 'upper', None))
    raise TypeError('a prior must be a galspec.priors.Prior, a (lower, upper) pair, '
                    'or a callable taking the unit-cube coordinate, got '
                    f'{type(obj).__name__}')


class CenteredPrior(Prior):
    ''' Prior for one parameter, a Gaussian centred on other parameters' values.

    Use `Normal` (or the older `gaussian_around`) rather than constructing this
    directly; those factories take the same arguments and document them.

    The prior is **truncated to the parameter's own box**, which is what keeps it
    a proper density: the unit-cube coordinate is mapped, one to one, through the
    inverse CDF of the Gaussian renormalised over that range, so ``u = 0`` lands on
    its lower end and ``u = 1`` on its upper end.  Without the truncation the map
    would not be a bijection onto the parameter's range and the evidence would not
    be comparable with a run that uses the plain box prior.

    The range actually covered is the box *intersected* with ``nsigma`` standard
    deviations either side of the centre (seven by default: beyond that the
    Gaussian carries 1.3e-12 of its mass and float64 has only a few units left in
    the cumulative probability).  ``nsigma`` is a **hard** limit: a value further
    from the centre than ``nsigma`` widths has zero prior probability, so the
    posterior cannot go there.  For a tie narrow compared with its box -- the
    usual case, and the point of a tie -- the ``nsigma`` end is what ``u = 1``
    reaches, and the box never binds.

    The width may be read from a *third* parameter, which is what makes "the shift
    may differ by 5% of the line width" expressible: pass ``scale_from`` and the
    width becomes ``scale * abs(value of that parameter)``.
    '''

    def __init__(self, center, scale, relative=False, scale_from=None,
                 log_space=False, lower=None, upper=None, nsigma=None):
        if isinstance(center, str):
            if not center:
                raise ValueError('center parameter name is empty')
        elif not np.isscalar(center):
            raise TypeError('center must be a parameter name or a number, got '
                            f'{type(center).__name__}')
        elif not np.isfinite(float(center)):
            raise ValueError(f'center must be finite, got {center}')
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f'scale must be positive and finite, got {scale}')
        if scale_from is not None and not isinstance(scale_from, str):
            raise TypeError('scale_from must be a parameter name or None, got '
                            f'{type(scale_from).__name__}')
        if relative and log_space:
            raise ValueError('relative and log_space are exclusive: in log space '
                             'the scale is already a fraction (a factor e**scale)')
        if lower is not None and upper is not None and not lower < upper:
            raise ValueError(f'need lower < upper, got {lower} and {upper}')
        if nsigma is None:
            nsigma = _N_SIGMA
        nsigma = float(nsigma)
        if not np.isfinite(nsigma) or nsigma <= 0.0:
            raise ValueError(f'nsigma must be positive and finite, got {nsigma}')
        self.center = center
        self.scale = scale
        self.relative = bool(relative)
        self.scale_from = scale_from
        self.log_space = bool(log_space)
        self.lower, self.upper = lower, upper
        self.nsigma = nsigma

    @property
    def depends_on(self):
        ''' Names of the parameters this prior reads, centre first. '''
        names = []
        if isinstance(self.center, str):
            names.append(self.center)
        if self.scale_from is not None:
            names.append(self.scale_from)
        return tuple(names)

    def with_bounds(self, lower, upper):
        ''' A copy truncated to ``[lower, upper]`` (the parameter's own box). '''
        lower, upper = float(lower), float(upper)
        if not lower < upper:
            raise ValueError(f'need lower < upper, got {lower} and {upper}')
        twin = type(self).__new__(type(self))
        twin.__dict__.update(self.__dict__)
        twin.lower, twin.upper = lower, upper
        return twin

    @property
    def support(self):
        if self.lower is None or self.upper is None:
            return None
        return (float(self.lower), float(self.upper))

    def _args(self):
        args = {'center': self.center, 'scale': self.scale,
                'relative': self.relative, 'scale_from': self.scale_from,
                'log_space': self.log_space, 'lower': self.lower,
                'upper': self.upper}
        if self.nsigma != _N_SIGMA:
            args['nsigma'] = self.nsigma
        return args

    def _width(self, center, scale_value):
        ''' Resolved standard deviation for one evaluation. '''
        if scale_value is not None:
            # traced under jit: |x| for both jax and numpy, without float()
            width = self.scale * abs(scale_value)
        elif self.relative and not self.log_space:
            width = self.scale * abs(center)
        else:
            width = self.scale
        return width

    def transform(self, u, center, scale_value=None):
        ''' Map ``u`` in [0, 1] onto the parameter, given the resolved centre.

        Parameters
        ----------
        u : float or traced scalar
            Unit-cube coordinate of the parameter being set.
        center : float or traced scalar
            Value the Gaussian is centred on: the value of another parameter, or
            a constant when the centre is fixed.
        scale_value : float or traced scalar, optional
            Value of ``scale_from``, when a width source was requested.

        Returns
        -------
        value : same namespace as ``u``
        '''
        if self.lower is None or self.upper is None:
            raise RuntimeError('bounds not set: give gaussian_around() explicit '
                               'lower/upper, or let the fitter fill them in')
        is_jax = type(u).__module__.startswith('jax')
        if is_jax:
            import jax.numpy as jnp
            maximum, minimum, log, exp = jnp.maximum, jnp.minimum, jnp.log, jnp.exp
        else:
            maximum, minimum, log, exp = np.maximum, np.minimum, np.log, np.exp
        if self.log_space:
            # the Gaussian lives on ln(theta); the truncation range goes with it
            center = log(center)
            lo, hi = float(np.log(self.lower)), float(np.log(self.upper))
        else:
            lo, hi = self.lower, self.upper
        width = maximum(self._width(center, scale_value), 1e-9 * (hi - lo))
        # A truncated normal is numerically flat beyond ~7 sigma: the mass there is
        # 1.3e-12, and both the CDF and its inverse lose their precision in float64.
        # Truncating to the box *and* to +-nsigma keeps the map a bijection onto the
        # range it actually covers -- the box wins whenever it is the narrower of
        # the two, which is the case whenever the Gaussian is not negligible.
        lo = maximum(lo, center - self.nsigma * width)
        hi = minimum(hi, center + self.nsigma * width)
        p_lo = _normal_cdf((lo - center) / width)
        p_hi = _normal_cdf((hi - center) / width)
        p = _clip(p_lo + u * (p_hi - p_lo), 1e-15, 1.0 - 1e-15)
        value = center + width * _SQRT2 * _erfinv(2.0 * p - 1.0)
        if not is_jax:
            value = float(value)
        return exp(value) if self.log_space else value

    def __call__(self, u):
        ''' Unit-cube map for a prior whose centre is a *number*.

        A prior centred on another parameter cannot be called this way -- the
        other parameter's value is needed first -- so the fitters supply it
        through `transform`.  Calling it directly is how you check or plot a
        prior around a fixed value.
        '''
        if isinstance(self.center, str):
            raise TypeError(
                f'{type(self).__name__} is centred on the parameter '
                f'{self.center!r}, so its value is not a function of u alone; '
                'the fitter resolves the other parameter before calling it')
        if self.scale_from is not None:
            raise TypeError(
                f'{type(self).__name__} takes its width from the parameter '
                f'{self.scale_from!r}, so its value is not a function of u alone')
        return self.transform(u, float(self.center))

    def logpdf(self, x, center=None, scale_value=None):
        ''' Log density at ``x``, normalised over the range it is truncated to.

        Parameters
        ----------
        x : float or array
            Value(s) of the parameter.
        center : float, optional
            Resolved centre.  Defaults to ``self.center`` when that is a number.
        scale_value : float, optional
            Resolved value of ``scale_from``, when a width source was given.

        Returns
        -------
        logp : float or array
            ``-inf`` outside the truncation range; a scalar in, a scalar out.
        '''
        scalar = np.ndim(x) == 0
        if center is None:
            if isinstance(self.center, str):
                raise TypeError('give center= when the prior is centred on another '
                                'parameter')
            center = float(self.center)
        if self.support is None:
            raise RuntimeError('bounds not set: give explicit lower/upper, or let '
                               'the fitter fill them in')
        center = float(center)
        if self.log_space:
            with np.errstate(divide='ignore', invalid='ignore'):
                values = np.log(np.asarray(x, dtype=float))
            center = np.log(center)
            lo, hi = np.log(self.lower), np.log(self.upper)
        else:
            values = np.asarray(x, dtype=float)
            lo, hi = float(self.lower), float(self.upper)
        width = max(float(self._width(center, scale_value)), 1e-9 * (hi - lo))
        lo = max(lo, center - self.nsigma * width)
        hi = min(hi, center + self.nsigma * width)
        p_lo = float(_normal_cdf((lo - center) / width))
        p_hi = float(_normal_cdf((hi - center) / width))
        logp = (-0.5 * ((values - center) / width) ** 2
                - np.log(width * np.sqrt(2.0 * np.pi))
                - np.log(p_hi - p_lo))
        inside = (values >= lo) & (values <= hi)
        value = np.where(inside, logp, -np.inf)
        return float(value) if scalar else value

    def __repr__(self):
        center = self.center if isinstance(self.center, str) else f'{self.center:g}'
        width = f'{self.scale:g}'
        if self.scale_from is not None:
            width += f' x |{self.scale_from}|'
        elif self.relative and not self.log_space:
            width += ' x |center|'
        elif self.log_space:
            width += ' (log space)'
        cap = '' if self.nsigma == _N_SIGMA else f', max {self.nsigma:g} sigma'
        return f'{type(self).__name__}({center}, {width}{cap})'


def gaussian_around(center, scale, relative=False, scale_from=None,
                    log_space=False, lower=None, upper=None, nsigma=None):
    ''' Prior for one parameter: a Gaussian centred on another parameter's value.

    This is the older spelling of `Normal`; both do the same thing, and `Normal`
    is the one to write in new code (``sigma=``/``relative=`` in one keyword
    instead of ``scale=`` plus a flag).

    The concrete case this exists for: broad H-beta and H-gamma share the profile
    *shape* of broad H-alpha, but not exactly.  Tieing them forces equality; this
    keeps them similar while letting the data measure the difference.

    Parameters
    ----------
    center : str or float
        Parameter whose value is the centre, given by any name the fitters accept
        in ``bounds_dict`` (``"b_halpha.sigma_c"``, the flat ``"sigma_c_3"``, or a
        bare ``"sigma_c"`` when exactly one parameter has it).  A number instead
        makes the prior a plain truncated Gaussian around that constant.
    scale : float
        Standard deviation, in the units of the parameter.  Interpreted as a
        fraction when ``relative=True``, or as a multiplier of ``scale_from`` when
        that is given -- see the table below.
    relative : bool
        ``scale`` is a fraction of the *centre's* value: ``sigma = scale *
        abs(center)``.  Use it for quantities with a natural scale, e.g. a line
        width that may differ by 20%: ``gaussian_around("b_halpha.sigma_c", 0.2,
        relative=True)``.
    scale_from : str, optional
        Parameter whose value multiplies ``scale``, for a width that should follow
        a *different* quantity than the centre.  "The shift may differ by 5% of the
        line width" is ``gaussian_around("b_halpha.dv_c", 0.05,
        scale_from="b_halpha.sigma_c")``.  Takes precedence over ``relative``.
    log_space : bool
        Apply the Gaussian to ``ln(parameter)``, so ``scale`` is a fraction in the
        log (a factor ``exp(scale)``).  Keeps a positive parameter positive for any
        scale; the right choice for amplitude ratios, where "twice as bright" and
        "half as bright" should be equally likely.  Cannot be combined with
        ``relative``.
    lower, upper : float, optional
        Truncation range.  Leave them out and the fitter fills in the parameter's
        own prior box, which is what keeps the prior proper and the evidence
        comparable with a run that does not use a centred prior.
    nsigma : float, optional
        How many standard deviations of the difference are allowed *at all*.
        Seven by default, which is where the Gaussian stops being representable
        in float64 rather than a physical choice.  ``nsigma=5`` says "do not
        allow an offset wider than five tie widths": the prior is then exactly
        zero beyond that, the sampler cannot go there, and a posterior that piles
        up against the limit means the data wanted to go further.

    Returns
    -------
    prior : CenteredPrior
        Attach it to the parameter with ``param.prior = ...``.

    Notes
    -----
    The density is ``N(theta; center, sigma)`` renormalised over the truncation
    range, so the unit cube maps onto ``[lower, upper]`` without loss.  The
    sampling cost of a centred prior is small: it is the same bijection of the
    unit cube that the box prior uses, only with a different shape.

    Comparing evidences between a tied model and a centred-prior model is
    meaningful but has to be read correctly: softening a tie always costs an
    Occam penalty of about ``ln(sigma * sqrt(2*pi) / posterior_width)`` per
    parameter, so the Bayes factor answers "was the hard tie good enough?", not
    "are the two profiles the same?".  The difference itself is in the posterior.

    Examples
    --------
    >>> width = gaussian_around("b_halpha.sigma_c", 0.20, relative=True,
    ...                         lower=300.0, upper=8000.0, nsigma=5.0)
    >>> width.depends_on
    ('b_halpha.sigma_c',)
    >>> shift = gaussian_around("b_halpha.dv_c", 0.05,
    ...                         scale_from="b_halpha.sigma_c",
    ...                         lower=-2500.0, upper=2500.0)
    >>> shift.depends_on
    ('b_halpha.dv_c', 'b_halpha.sigma_c')
    >>> round(width.transform(0.5, 1301.0), 1)     # u = 0.5 lands on the centre
    1301.0
    >>> round(width.transform(1.0, 1301.0), 1)     # u = 1 lands on centre + 5 sigma
    2602.0
    '''
    return CenteredPrior(center=center, scale=scale, relative=relative,
                         scale_from=scale_from, log_space=log_space,
                         lower=lower, upper=upper, nsigma=nsigma)


class Normal(CenteredPrior):
    ''' Gaussian prior centred on a value you give, or on another parameter.

    Parameters
    ----------
    mean : str or float
        The centre: another parameter's name, or a number.  A name may be
        written as ``"b_halpha.sigma_w0"`` (qualified), ``"sigma_w0_2"`` (flat)
        or ``"sigma_w0"`` when only one parameter has that name; the fitter
        resolves it and evaluates this prior after the parameter it names.
    sigma : float, optional
        Standard deviation in the units of the parameter.  Give this or
        ``relative``, not both.
    relative : float, optional
        Standard deviation as a fraction of the centre's magnitude, e.g. ``0.2``
        for "may differ by 20%".
    scale_from : str, optional
        Take the width from a *different* parameter's value instead: the shift
        "may differ by 5% of the line width" is ``Normal(mean="...dv_c",
        sigma=0.05, scale_from="...sigma_c")``.  Multiplies ``sigma`` or
        ``relative``.
    nsigma : float, optional
        Hard limit on the offset, in standard deviations.  ``nsigma=5`` forbids
        an offset wider than five widths.
    log : bool
        Put the Gaussian on ``ln(parameter)``, so the width is a multiplicative
        factor.  Keeps a positive quantity positive.
    mini, maxi : float, optional
        Explicit truncation range.  Left out, the parameter's own bounds are
        used.

    Examples
    --------
    >>> prior = Normal(mean="b_halpha.sigma_w0", relative=0.20, nsigma=5.0,
    ...                mini=800.0, maxi=25000.0)
    >>> prior.depends_on
    ('b_halpha.sigma_w0',)
    >>> round(prior.transform(0.5, 4000.0), 1)      # u = 0.5 lands on the centre
    4000.0
    >>> round(prior.transform(1.0, 4000.0), 1)      # u = 1 is the 5 sigma limit
    8000.0
    '''

    def __init__(self, mean, sigma=None, relative=None, scale_from=None,
                 nsigma=None, log=False, mini=None, maxi=None):
        if (sigma is None) == (relative is None):
            raise ValueError('give exactly one of sigma= (absolute width) or '
                             'relative= (fraction of the centre)')
        if sigma is not None:
            scale, is_relative = float(sigma), False
        else:
            scale, is_relative = float(relative), True
        super().__init__(center=mean, scale=scale, relative=is_relative,
                         scale_from=scale_from, log_space=bool(log),
                         lower=mini, upper=maxi, nsigma=nsigma)

    def _args(self):
        args = {'mean': self.center}
        if self.relative and not self.log_space:
            args['relative'] = self.scale
        else:
            args['sigma'] = self.scale
        if self.scale_from is not None:
            args['scale_from'] = self.scale_from
        if self.log_space:
            args['log'] = True
        if self.nsigma != _N_SIGMA:
            args['nsigma'] = self.nsigma
        if self.lower is not None:
            args['mini'] = self.lower
        if self.upper is not None:
            args['maxi'] = self.upper
        return args


class ClippedNormal(Normal):
    ''' `Normal` with an explicit truncation range, as in prospector.

    ``ClippedNormal(mean, sigma, mini, maxi)`` is the same distribution as
    ``Normal(mean, sigma=..., mini=..., maxi=...)``; the name exists because that
    is the spelling people coming from prospector look for.
    '''

    def __init__(self, mean, sigma=None, mini=None, maxi=None, relative=None,
                 scale_from=None, nsigma=None, log=False):
        if mini is None or maxi is None:
            raise ValueError('ClippedNormal needs mini= and maxi=')
        super().__init__(mean=mean, sigma=sigma, relative=relative,
                         scale_from=scale_from, nsigma=nsigma, log=log,
                         mini=mini, maxi=maxi)


#: Kinds `Prior.from_dict` can rebuild; the keys are the class names.
_PRIOR_TYPES = {
    'Uniform': Uniform,
    'TopHat': TopHat,
    'LogUniform': LogUniform,
    'SpikeAndSlab': SpikeAndSlab,
    'CenteredPrior': CenteredPrior,
    'Normal': Normal,
    'ClippedNormal': ClippedNormal,
}


def resolve_prior_dependencies(names, full_names, custom_priors):
    ''' Order parameters so a centred prior is evaluated after what it reads.

    Both fitters build the parameter vector from the unit cube in one pass, one
    element at a time.  A centred prior breaks that: it needs the *value* of
    another parameter, which may itself be centred on a third one.  This resolves
    each centre and width source to a parameter index, sorts the parameters so
    that every dependency comes first, and reports exactly what each centred prior
    has to read.

    Parameters
    ----------
    names : sequence of str
        Flattened parameter names, in the order the fitters use.
    full_names : sequence of str
        Qualified names of the same parameters (``"Halpha.sigma_c"``), used to
        resolve the readable spellings.
    custom_priors : mapping
        The fitters' user-supplied priors, keyed by flattened name.

    Returns
    -------
    order : list of int
        Parameter indices, dependencies first, covering every index exactly once.
    centered : dict
        ``{index: (prior, center_ref, scale_ref)}`` where the refs are either an
        integer index into ``names`` or a float constant (``scale_ref`` may be
        ``None``).

    Raises
    ------
    ValueError
        If a centre or width source does not exist, if a bare name matches more
        than one parameter, or if the dependencies form a cycle.
    '''
    names = list(names)
    full_names = list(full_names) if full_names is not None else list(names)
    index_of = {name: index for index, name in enumerate(names)}

    def resolve(name, what):
        if not isinstance(name, str):
            raise TypeError(f'{what} must be a parameter name or a number, got '
                            f'{type(name).__name__}')
        # flat name, then qualified name, then a unique bare match
        if name in index_of:
            return index_of[name]
        if name in full_names:
            return full_names.index(name)
        bare = name.rsplit('.', 1)[-1]
        hits = [index for index, full in enumerate(full_names)
                if full == name or full.rsplit('.', 1)[-1] == bare]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            candidates = [full_names[index] for index in hits]
            raise ValueError(
                f'{what} {name!r} is ambiguous: it matches {candidates}. '
                'Spell it with its model prefix.')
        raise ValueError(f'{what} {name!r} is not a free parameter; '
                         f'known names are {names[:6]}... ')

    centered = {}
    dependencies = {index: [] for index in range(len(names))}
    for index, name in enumerate(names):
        prior = custom_priors.get(name)
        if prior is None and index < len(full_names):
            prior = custom_priors.get(full_names[index])
        if not isinstance(prior, CenteredPrior):
            continue
        center = prior.center
        if isinstance(center, str):
            center_ref = resolve(center, 'center')
            if center_ref == index:
                raise ValueError(f'{name!r} cannot be centred on itself')
            dependencies[index].append(center_ref)
        else:
            center_ref = float(center)
        scale_ref = None
        if prior.scale_from is not None:
            scale_ref = resolve(prior.scale_from, 'scale_from')
            if scale_ref == index:
                raise ValueError(f'{name!r} cannot take its width from itself')
            dependencies[index].append(scale_ref)
        centered[index] = (prior, center_ref, scale_ref)

    # Kahn's algorithm; independent parameters keep their original order, so a
    # model without centred priors transforms exactly as it did before.
    dependencies = {index: deps for index, deps in dependencies.items() if deps}
    order = [index for index in range(len(names)) if index not in dependencies]
    done = set(order)
    while dependencies:
        promoted = [index for index in sorted(dependencies)
                    if all(dep in done for dep in dependencies[index])]
        if not promoted:
            cycle = ', '.join(names[index] for index in sorted(dependencies))
            raise ValueError(f'centred priors form a cycle through: {cycle}')
        for index in promoted:
            order.append(index)
            done.add(index)
            del dependencies[index]
    return order, centered
