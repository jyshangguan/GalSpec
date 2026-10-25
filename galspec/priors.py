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

The second is `gaussian_around`: it makes the prior of one parameter a Gaussian
centred on *the value of another parameter*, so a model can say "similar, but not
the same" -- broad H-beta sharing broad H-alpha's profile shape, a narrow line
sitting at the same velocity as its doublet partner, and so on.  Unlike a tie,
which forces equality, this keeps the constraint soft and lets the data report the
difference and its uncertainty.
'''

import numpy as np

__all__ = ['log_uniform', 'component_toggle', 'uniform_toggle',
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


class CenteredPrior:
    ''' Prior for one parameter, a Gaussian centred on other parameters' values.

    Use `gaussian_around` rather than constructing this directly; the factory
    takes the same arguments and documents them.

    The prior is **truncated to the parameter's own box**, which is what keeps it
    a proper density: the unit-cube coordinate is mapped, one to one, through the
    inverse CDF of the Gaussian renormalised over that range, so ``u = 0`` lands on
    its lower end and ``u = 1`` on its upper end.  Without the truncation the map
    would not be a bijection onto the parameter's range and the evidence would not
    be comparable with a run that uses the plain box prior.

    The range actually covered is the box *intersected* with seven standard
    deviations either side of the centre: beyond that the Gaussian carries 1.3e-12
    of its mass and float64 has only a few units left in the cumulative
    probability, so the tail is cut where it can still be represented.  For a tie
    narrow compared with its box -- the usual case, and the point of a tie -- the
    seven-sigma end is what ``u = 1`` reaches, and the box never binds.

    The width may be read from a *third* parameter, which is what makes "the shift
    may differ by 5% of the line width" expressible: pass ``scale_from`` and the
    width becomes ``scale * abs(value of that parameter)``.
    '''

    def __init__(self, center, scale, relative=False, scale_from=None,
                 log_space=False, lower=None, upper=None):
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
        self.center = center
        self.scale = scale
        self.relative = bool(relative)
        self.scale_from = scale_from
        self.log_space = bool(log_space)
        self.lower, self.upper = lower, upper

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
        twin = CenteredPrior(self.center, self.scale, relative=self.relative,
                             scale_from=self.scale_from,
                             log_space=self.log_space,
                             lower=lower, upper=upper)
        return twin

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
        # Truncating to the box *and* to +-7 sigma keeps the map a bijection onto the
        # range it actually covers -- the box wins whenever it is the narrower of
        # the two, which is the case whenever the Gaussian is not negligible.
        lo = maximum(lo, center - _N_SIGMA * width)
        hi = minimum(hi, center + _N_SIGMA * width)
        p_lo = _normal_cdf((lo - center) / width)
        p_hi = _normal_cdf((hi - center) / width)
        p = _clip(p_lo + u * (p_hi - p_lo), 1e-15, 1.0 - 1e-15)
        value = center + width * _SQRT2 * _erfinv(2.0 * p - 1.0)
        if not is_jax:
            value = float(value)
        return exp(value) if self.log_space else value

    def __repr__(self):
        center = self.center if isinstance(self.center, str) else f'{self.center:g}'
        width = f'{self.scale:g}'
        if self.scale_from is not None:
            width += f' x |{self.scale_from}|'
        elif self.relative and not self.log_space:
            width += ' x |center|'
        elif self.log_space:
            width += ' (log space)'
        return f'CenteredPrior({center}, {width})'


def gaussian_around(center, scale, relative=False, scale_from=None,
                    log_space=False, lower=None, upper=None):
    ''' Prior for one parameter: a Gaussian centred on another parameter's value.

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

    Returns
    -------
    prior : CenteredPrior
        Pass it in the fitters' ``custom_priors`` dictionary, keyed by the
        parameter it constrains.

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
    ...                         lower=300.0, upper=8000.0)
    >>> width.depends_on
    ('b_halpha.sigma_c',)
    >>> shift = gaussian_around("b_halpha.dv_c", 0.05,
    ...                         scale_from="b_halpha.sigma_c",
    ...                         lower=-2500.0, upper=2500.0)
    >>> shift.depends_on
    ('b_halpha.dv_c', 'b_halpha.sigma_c')
    >>> round(width.transform(0.5, 1301.0), 1)     # u = 0.5 lands on the centre
    1301.0
    '''
    return CenteredPrior(center=center, scale=scale, relative=relative,
                         scale_from=scale_from, log_space=log_space,
                         lower=lower, upper=upper)


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
