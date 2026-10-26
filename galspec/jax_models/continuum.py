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
    """Polynomial continuum ``sum_i c_i x**i`` -- matches ``astropy.models.Polynomial1D``.

    The degree is fixed at construction time (like the astropy model), which
    keeps the parameter vector length static for ``jit``.
    """

    model_kind = "Polynomial1D"

    def __init__(self, degree: int = 1, name: Optional[str] = None,
                 fixed: Optional[dict] = None, bounds: Optional[dict] = None, **kwargs):
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = degree
        self._param_specs = tuple(
            ParamSpec(f"c{i}", 1.0 if i == 0 else 0.0) for i in range(degree + 1))
        super().__init__(name=name, fixed=fixed, bounds=bounds, **kwargs)

    @property
    def PARAMS(self):  # type: ignore[override]
        return self._param_specs

    def evaluate(self, x, p):
        out = jnp.zeros_like(x)
        for i in range(self.degree + 1):
            out = out + p[f"c{i}"] * x ** i
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