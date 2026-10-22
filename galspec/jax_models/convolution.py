"""JAX instrument-line-spread-function convolution wrappers.

Two flavours, mirroring :mod:`galspec.convolution` and
:mod:`galspec.convolution_var`:

``convolve_lsf(model, wavec, resolving_power)``
    Gaussian LSF with constant resolving power, i.e. constant sigma in
    wavelength.  Reproduces ``scipy.ndimage.gaussian_filter1d`` exactly through
    :func:`galspec.jax_models.functional.gaussian_smooth1d`.

``convolve_lsf_var(model, wavec, resolution_data)``
    Wavelength-dependent resolving power (e.g. JWST NIRSpec prism).  The CPU
    implementation precomputes one normalised kernel per pixel; here those
    weights are frozen into constant arrays (gather indices + weights) and
    applied with a single ``take_along_axis`` product -- parameter independent,
    hence fully ``jit``/``vmap``-able, and numerically identical.

Both wrappers are *transparent*: they expose the wrapped model's parameters, so
ties, bounds, priors and ``set_prior_limits`` hooks keep working unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np

from .core import JaxCompoundModel, JaxModel
from .functional import gaussian_smooth1d, interp_extrapolate

__all__ = [
    "JaxConvolvedModel",
    "JaxVariableConvolvedModel",
    "convolve_lsf",
    "convolve_lsf_var",
]


class _WrapperModel(JaxModel):
    """Shared plumbing: a leaf model that wraps another model."""

    #: the wrapped model
    inner: JaxModel

    def _init_wrapper(self, inner: JaxModel, name: Optional[str]) -> None:
        # Share the inner model's mutable state so ties/bounds/values stay in sync.
        self.inner = inner
        self.name = name or getattr(inner, "name", None) or type(inner).__name__
        self._belongs_to = None
        self._compound_inner = isinstance(inner, JaxCompoundModel)
        if self._compound_inner:
            # A compound model keeps its parameters on its leaves, so there is no
            # single value dictionary to share; everything is delegated instead.
            return
        self._values = inner._values
        self._specs = inner._specs
        self._ties = inner._ties
        self._provided = frozenset()
        self._prepared = False

    # -- delegation ----------------------------------------------------------------
    @property
    def param_names(self) -> Tuple[str, ...]:
        return self.inner.param_names

    @property
    def free_param_names(self) -> Tuple[str, ...]:
        return tuple(self.inner.free_param_names)

    def _is_free(self, name: str) -> bool:
        return self.inner._is_free(name)

    def spec(self, name):
        return self.inner.spec(name)

    def get_free_params(self):
        return self.inner.get_free_params()

    def set_free_params(self, theta):
        self.inner.set_free_params(theta)
        return self

    def param_dict(self, theta=None):
        return self.inner.param_dict(theta)

    def _lookup_qualified(self, qualified: str):
        return self.inner._lookup_qualified(qualified)

    @property
    def qualified_param_names(self):
        return self.inner.qualified_param_names

    def set_value(self, name: str, value: float) -> None:
        self.inner.set_value(name, value)

    def set_fixed(self, name: str, flag: bool) -> None:
        self.inner.set_fixed(name, flag)

    def set_bounds(self, name: str, bounds) -> None:
        self.inner.set_bounds(name, bounds)

    def set_tied(self, name: str, func) -> None:
        self.inner.set_tied(name, func)

    def set_prior_limits(self, limits: dict) -> None:
        if hasattr(self.inner, "set_prior_limits"):
            self.inner.set_prior_limits(limits)

    def prepare(self, x) -> None:
        self.inner.prepare(x)
        self._prepared = True

    def _smooth(self, flux, x):
        """Apply the instrument profile. Base class: identity."""
        return flux

    def __call__(self, x):
        """Evaluate with the current parameters, deriving the kernel if needed.

        The convolution kernel size must be a static shape for ``jit``, so
        :meth:`prepare` normally runs first (``JNesty_Fit`` does it).  An eager
        ``model(wave)`` call prepares itself, which keeps the wrapper usable like
        the astropy ``convolve_lsf`` helpers.
        """
        if getattr(self, "_prepared", False) is False:
            self.prepare(np.asarray(x, dtype=float))
        return self._smooth(self.inner(jnp.asarray(x)), jnp.asarray(x))

    def flux(self, x, theta):
        """Evaluate the wrapped model for a parameter vector, then smooth it."""
        return self._smooth(self.inner.flux(x, theta), x)

    def evaluate(self, x, p):
        if self._compound_inner:
            raise RuntimeError(
                "a convolve_lsf wrapper around a compound model must be evaluated "
                "with flux()/__call__(), not as a leaf of another compound model; "
                "convolve the whole model instead")
        return self._smooth(self.inner.evaluate(x, p), x)

    def __getattr__(self, item: str):
        if item.startswith("_") or item in ("name", "PARAMS", "model_kind"):
            raise AttributeError(item)
        inner = self.__dict__.get("inner")
        if inner is not None and item in inner.__dict__.get("_specs", {}):
            return getattr(inner, item)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {item!r}")


class JaxConvolvedModel(_WrapperModel):
    """Applies a constant Gaussian LSF (uniform grid) after evaluating ``inner``."""

    model_kind = "JaxConvolvedModel"

    def __init__(self, inner: JaxModel, sigma_x: float, name: Optional[str] = None):
        if sigma_x < 0:
            raise ValueError("sigma_x must be >= 0")
        self._init_wrapper(inner, name)
        self.sigma_x = float(sigma_x)
        self._sigma_pix: Optional[float] = None
        self._radius = 0

    def prepare(self, x) -> None:
        xv = np.asarray(x, dtype=float)
        if xv.size < 2:
            raise ValueError("x must have at least 2 points")
        dx = np.diff(xv)
        if not np.allclose(dx, np.median(dx), rtol=1e-6, atol=0.0):
            raise ValueError("convolve_lsf requires a uniformly spaced wavelength grid")
        self._sigma_pix = self.sigma_x / float(np.median(dx))
        self._radius = int(4.0 * self._sigma_pix + 0.5) if self._sigma_pix > 0 else 0
        super().prepare(x)

    def _smooth(self, flux, x):
        if self._sigma_pix is None:
            raise RuntimeError(
                "convolve_lsf wrapper needs prepare(wave) before evaluation "
                "(JNesty_Fit calls it automatically)")
        if self._radius <= 0:
            return flux
        return gaussian_smooth1d(flux, self._sigma_pix, radius=self._radius)


class JaxVariableConvolvedModel(_WrapperModel):
    """Applies a wavelength-dependent Gaussian LSF after evaluating ``inner``."""

    model_kind = "JaxVariableConvolvedModel"

    def __init__(self, inner: JaxModel, wavec: Sequence[float], resolution_data,
                 name: Optional[str] = None, cfg=None):
        self._init_wrapper(inner, name)
        self._wavec = np.asarray(wavec, dtype=float)
        if self._wavec.ndim != 1 or self._wavec.size == 0:
            raise ValueError("wavec must be a non-empty 1-D array")
        self._indices, self._weights = _build_variable_kernels(
            self._wavec, resolution_data, cfg=cfg)
        self._needs_interp = False

    def prepare(self, x) -> None:
        xv = np.asarray(x, dtype=float)
        same_grid = (xv.size == self._wavec.size
                     and np.allclose(xv, self._wavec, rtol=1e-6, atol=0.0))
        self._needs_interp = not same_grid
        super().prepare(x)

    def _smooth(self, flux, x):
        if self._needs_interp:
            flux = interp_extrapolate(self._wavec, x, flux)
        gathered = jnp.take_along_axis(flux[None, :], self._indices, axis=1)
        out = jnp.sum(gathered * self._weights, axis=1)
        if self._needs_interp:
            out = interp_extrapolate(x, self._wavec, out)
        return out


def _build_variable_kernels(wavec: np.ndarray, resolution_data, cfg=None):
    """Freeze the CPU ``_VariableKernel`` weights into padded constant arrays."""
    from ..convolution_var import GaussianConv1DConfig, ResolutionCurve, _VariableKernel

    cfg = cfg or GaussianConv1DConfig()
    if isinstance(resolution_data, ResolutionCurve):
        rc = resolution_data
    elif isinstance(resolution_data, (tuple, list)) and len(resolution_data) == 2:
        rc = ResolutionCurve(resolution_data[0], resolution_data[1])
    else:
        raise TypeError("resolution_data must be a (wave, R) tuple or a ResolutionCurve")

    kernel = _VariableKernel(wavec, rc, cfg)

    n = len(wavec)
    max_width = max(len(idx) for idx in kernel.indices)
    indices = np.zeros((n, max_width), dtype=np.int32)
    weights = np.zeros((n, max_width), dtype=np.float64)
    for i, (idx, w) in enumerate(zip(kernel.indices, kernel.weights)):
        indices[i, :len(idx)] = idx
        weights[i, :len(idx)] = w
    return jnp.asarray(indices), jnp.asarray(weights)


def convolve_lsf(model: JaxModel, wavec: Union[float, jnp.ndarray], resolving_power: float,
                 name: Optional[str] = None) -> JaxConvolvedModel:
    """Convolve ``model`` with a Gaussian LSF of constant resolving power.

    Mirrors :func:`galspec.convolve_lsf` (FWHM-to-sigma factor 2.3548).  The
    model must be evaluated on a uniformly spaced wavelength grid.
    """
    wavec_value = float(np.asarray(getattr(wavec, "value", wavec)))
    sigma_x = wavec_value / (float(resolving_power) * 2.3548)
    return JaxConvolvedModel(model, sigma_x=sigma_x, name=name)


def convolve_lsf_var(model: JaxModel, wavec: Sequence[float], resolution_data,
                     name: Optional[str] = None, cfg=None) -> JaxVariableConvolvedModel:
    """Convolve ``model`` with a wavelength-dependent Gaussian LSF.

    Mirrors :func:`galspec.convolve_lsf_var`; ``resolution_data`` is either a
    ``(wave, R)`` tuple or a :class:`galspec.convolution_var.ResolutionCurve`.
    """
    return JaxVariableConvolvedModel(model, np.asarray(wavec, dtype=float),
                                     resolution_data, name=name, cfg=cfg)