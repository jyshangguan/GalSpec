"""JAX continuum models, mirroring the astropy/GalSpec CPU equivalents.

Every class here reproduces the ``evaluate`` method of its CPU counterpart
bit-for-bit (to float64 rounding), which is what the parity test suite checks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

from .core import JaxModel, ParamSpec

__all__ = [
    "PowerLaw1D",
    "BrokenPowerLaw1D",
    "Const1D",
    "Linear1D",
    "Polynomial1D",
    "WindowedPowerLaw1D",
    "WindowedLinear",
    "Gaussian1D",
]


def _domain_pair(value, default: Tuple[float, float]) -> Tuple[float, float]:
    """Normalise an astropy-style ``domain``/``window`` argument to two floats.

    Accepts ``None`` (use ``default``), any 2-sequence, or a numpy array -- the
    serialiser stores these as arrays and hands them back that way, so the
    constructor has to take them without complaint.
    """
    if value is None:
        return tuple(float(v) for v in default)          # type: ignore[return-value]
    pair = tuple(float(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"expected two numbers, got {value!r}")
    return pair                                          # type: ignore[return-value]


class PowerLaw1D(JaxModel):
    """``amplitude * (x / x_0) ** (-alpha)`` -- matches ``astropy.models.PowerLaw1D``."""

    model_kind = "PowerLaw1D"
    PARAMS = (
        ParamSpec("amplitude", 1.0),
        ParamSpec("x_0", 1.0),
        ParamSpec("alpha", 1.0),
    )

    def evaluate(self, x, p):
        return p["amplitude"] * (x / p["x_0"]) ** (-p["alpha"])


class BrokenPowerLaw1D(JaxModel):
    """Continuously broken power law.

    ``f(x) = A (x/x_break)**(-alpha_1)`` for ``x < x_break`` and
    ``A (x/x_break)**(-alpha_2)`` otherwise, matching
    ``astropy.modeling.models.BrokenPowerLaw1D`` exactly (same sign convention:
    a *positive* ``alpha`` makes the flux decrease with wavelength).
    """

    model_kind = "BrokenPowerLaw1D"
    PARAMS = (
        ParamSpec("amplitude", 1.0, bounds=(0.0, None)),
        ParamSpec("x_break", 5000.0, bounds=(0.0, None)),
        ParamSpec("alpha_1", 1.0, bounds=(-5.0, 5.0)),
        ParamSpec("alpha_2", 1.0, bounds=(-5.0, 5.0)),
    )

    def evaluate(self, x, p):
        alpha = jnp.where(x < p["x_break"], p["alpha_1"], p["alpha_2"])
        return p["amplitude"] * (x / p["x_break"]) ** (-alpha)


class Const1D(JaxModel):
    """Constant continuum -- matches ``astropy.models.Const1D``."""

    model_kind = "Const1D"
    PARAMS = (ParamSpec("amplitude", 1.0),)

    def evaluate(self, x, p):
        return p["amplitude"] * jnp.ones_like(x)


class Linear1D(JaxModel):
    """``slope * x + intercept`` -- matches ``astropy.models.Linear1D``."""

    model_kind = "Linear1D"
    PARAMS = (
        ParamSpec("slope", 1.0),
        ParamSpec("intercept", 0.0),
    )

    def evaluate(self, x, p):
        return p["slope"] * x + p["intercept"]


class Polynomial1D(JaxModel):
    """Polynomial continuum ``sum_i c_i u**i`` -- matches ``astropy.models.Polynomial1D``.

    The degree is fixed at construction time (like the astropy model), which
    keeps the parameter vector length static for ``jit``.

    ``domain`` and ``window`` implement astropy's linear mapping of the abscissa
    before the polynomial is evaluated::

        u = window[0] + (x - domain[0]) * (window[1] - window[0])
                            / (domain[1] - domain[0])

    They are not decoration.  A local continuum over 4920-5080 A evaluated on the
    raw wavelength needs ``c3 ~ 1e-11`` for the cubic term to contribute at the
    flux level of the data (``x**3 ~ 1e11``), so no single prior shape fits every
    order and the sampler has to find four coefficients spanning eleven decades.
    Mapping that window onto ``[-1, 1]`` puts every coefficient at the scale of the
    flux itself: ``Polynomial1D(degree=3, domain=(4920, 5080), window=(-1, 1))``,
    after which ``Uniform(-2, 2)`` is a sensible prior for all four.  The defaults
    ``domain = window = (-1, -1)`` are astropy's, and make the map the identity, so
    an existing model keeps its exact behaviour.
    """

    model_kind = "Polynomial1D"

    def __init__(self, degree: int = 1, domain=None, window=None,
                 name: Optional[str] = None, fixed: Optional[dict] = None,
                 bounds: Optional[dict] = None, **kwargs):
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = degree
        self.domain = _domain_pair(domain, (-1.0, -1.0))
        self.window = _domain_pair(window, (-1.0, -1.0))
        # astropy's defaults are domain = window = (-1, -1), and an unchanged
        # domain maps onto itself, i.e. the polynomial is evaluated on the raw
        # abscissa.  Only a *degenerate* domain that would actually be mapped is
        # an error.
        if self.domain != self.window and self.domain[0] == self.domain[1]:
            raise ValueError(f"domain must span a range, got {self.domain}")
        self._param_specs = tuple(
            ParamSpec(f"c{i}", 1.0 if i == 0 else 0.0) for i in range(degree + 1))
        super().__init__(name=name, fixed=fixed, bounds=bounds, **kwargs)

    @property
    def PARAMS(self):  # type: ignore[override]
        return self._param_specs

    def _mapped(self, x):
        """The abscissa the polynomial is written in (astropy's ``domain`` map)."""
        if self.domain == self.window:
            return x
        (d0, d1), (w0, w1) = self.domain, self.window
        return w0 + (x - d0) * (w1 - w0) / (d1 - d0)

    def evaluate(self, x, p):
        u = self._mapped(x)
        out = jnp.zeros_like(u)
        for i in range(self.degree + 1):
            out = out + p[f"c{i}"] * u ** i
        return out


class WindowedPowerLaw1D(JaxModel):
    """Power law evaluated only inside ``[x_min, x_max)`` -- ``galspec.WindowedPowerLaw1D``."""

    model_kind = "WindowedPowerLaw1D"
    PARAMS = (
        ParamSpec("amplitude", 1.0),
        ParamSpec("x_0", 4500.0, fixed=True),
        ParamSpec("alpha", -1.0),
        ParamSpec("x_min", 4000.0, fixed=True),
        ParamSpec("x_max", 5000.0, fixed=True),
    )

    def evaluate(self, x, p):
        y = p["amplitude"] * (x / p["x_0"]) ** (-p["alpha"])
        mask = (x >= p["x_min"]) & (x < p["x_max"])
        return jnp.where(mask, y, 0.0)


class WindowedLinear(JaxModel):
    """Linear model active inside ``[x_min, x_max]`` -- ``galspec.WindowedLinear``."""

    model_kind = "WindowedLinear"
    PARAMS = (
        ParamSpec("c0", 1.0),
        ParamSpec("c1", 0.0),
        ParamSpec("x_min", 4000.0, fixed=True),
        ParamSpec("x_max", 5000.0, fixed=True),
    )

    def evaluate(self, x, p):
        y = p["c0"] + p["c1"] * x
        mask = (x >= p["x_min"]) & (x <= p["x_max"])
        return jnp.where(mask, y, 0.0)


class Gaussian1D(JaxModel):
    """``amplitude * exp(-0.5 ((x - mean)/stddev)**2)`` -- ``astropy.models.Gaussian1D``."""

    model_kind = "Gaussian1D"
    PARAMS = (
        ParamSpec("amplitude", 1.0),
        ParamSpec("mean", 0.0),
        ParamSpec("stddev", 1.0, bounds=(0.0, None)),
    )

    def evaluate(self, x, p):
        return p["amplitude"] * jnp.exp(-0.5 * ((x - p["mean"]) / p["stddev"]) ** 2)