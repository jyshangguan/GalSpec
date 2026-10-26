"""Pure-``jnp`` numerical helpers shared by the JAX model classes.

Everything in this module is written so that it can be ``jit``/``vmap``-compiled:
no host-side branching on traced values, no dynamic shapes.  Where the CPU
GalSpec code uses :mod:`scipy.ndimage`, the numerically identical formula is
reproduced here (see :func:`gaussian_smooth1d`).
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from ..constants import ls_km

__all__ = [
    "LS_KM",
    "velocity",
    "gaussian_kernel1d",
    "gaussian_smooth1d",
    "interp_extrapolate",
]

LS_KM = float(ls_km)


def velocity(x, wavec):
    """Convert a wavelength offset into a velocity offset [km/s].

    Mirrors ``v = (x - wavec) / wavec * ls_km`` used throughout GalSpec.
    """
    return (x - wavec) / wavec * LS_KM


def gaussian_kernel1d(sigma, radius: int, truncate: float = 4.0):
    """Unit-sum Gaussian kernel with a *static* radius.

    Reproduces :func:`scipy.ndimage.gaussian_filter1d`'s kernel
    (``_gaussian_kernel1d``): tap ``d`` is kept when ``|d| <= floor(truncate *
    sigma + 0.5)`` and the truncated kernel is renormalised.  Because ``sigma``
    is a traced value under ``jit`` the radius cannot be data dependent, so the
    mask is evaluated with :func:`jax.numpy.where`; the weights are therefore
    identical to SciPy's for any ``sigma`` whose radius is ``<= radius``.

    Parameters
    ----------
    sigma : scalar
        Standard deviation in pixels (traced allowed).
    radius : int
        Static kernel half-width.
    truncate : float
        Truncation in units of ``sigma`` (SciPy default 4.0).
    """
    sigma = jnp.maximum(jnp.abs(jnp.asarray(sigma)), 1e-12)
    d = jnp.arange(-int(radius), int(radius) + 1)  # dtype follows the active precision
    radius_sigma = jnp.floor(truncate * sigma + 0.5)
    w = jnp.where(jnp.abs(d) <= radius_sigma, jnp.exp(-0.5 * (d / sigma) ** 2), 0.0)
    return w / jnp.sum(w)


def gaussian_smooth1d(y, sigma_pix, radius: Optional[int] = None, truncate: float = 4.0):
    """Exact float64 reproduction of :func:`scipy.ndimage.gaussian_filter1d`.

    SciPy's ``mode='reflect'`` is a *half-sample symmetric* extension (the edge
    sample is repeated), which corresponds to ``jnp.pad(..., mode='symmetric')``
    -- not to ``mode='reflect'``.  Verified to agree to ~2e-16 in float64.

    Parameters
    ----------
    y : array_like, shape (n,)
        Signal to smooth.
    sigma_pix : scalar
        Gaussian sigma in pixels (may be traced under ``jit``).
    radius : int, optional
        Static kernel half-width.  Defaults to
        ``int(truncate * sigma + 0.5)`` for scalar sigma; ``sigma_pix`` must be a
        Python scalar in that case.  Pass an explicit radius when ``sigma_pix``
        is traced.
    truncate : float
        Truncation in units of sigma (SciPy default 4.0).
    """
    y = jnp.asarray(y)
    if radius is None:
        try:
            sigma_host = float(sigma_pix)
        except (TypeError, ValueError) as exc:  # traced value without a radius
            raise ValueError(
                "sigma_pix is a traced value; pass an explicit static `radius` "
                "(see JaxModel.sigma_pix_max)."
            ) from exc
        radius = int(truncate * abs(sigma_host) + 0.5)

    if int(radius) <= 0:
        return y

    w = gaussian_kernel1d(sigma_pix, int(radius), truncate=truncate)
    padded = jnp.pad(y, int(radius), mode="symmetric")
    return jnp.convolve(padded, w, mode="valid")


def interp_extrapolate(x, xp, fp):
    """Linear interpolation with linear extrapolation (``interp1d(fill_value="extrapolate")``).

    ``jnp.interp`` clamps outside the tabulated range, whereas GalSpec's
    ``Multi_StarSpectrum`` extrapolates linearly; this helper reproduces the
    latter.
    """
    x = jnp.asarray(x)
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    y = jnp.interp(x, xp, fp)

    slope_lo = (fp[1] - fp[0]) / (xp[1] - xp[0])
    slope_hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = jnp.where(x < xp[0], fp[0] + slope_lo * (x - xp[0]), y)
    y = jnp.where(x > xp[-1], fp[-1] + slope_hi * (x - xp[-1]), y)
    return y