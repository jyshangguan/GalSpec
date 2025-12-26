# path: astropy_gaussconv_decorate_call_and_deriv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ['convolve_lsf']


Mode = Literal["reflect", "constant", "nearest", "mirror", "wrap"]
_TAG = "_gaussconv1d_callswap"


@dataclass(frozen=True)
class GaussianConv1DConfig:
    mode: Mode = "reflect"
    cval: float = 0.0
    truncate: float = 4.0
    uniform_rtol: float = 1e-2


def _x_to_value_and_unit(x: Any) -> Tuple[np.ndarray, Any]:
    unit = getattr(x, "unit", None)
    xv = np.asarray(getattr(x, "value", x), dtype=float)
    if xv.ndim != 1:
        raise ValueError("x must be 1D.")
    if xv.size < 2:
        raise ValueError("x must have at least 2 points.")
    if not np.all(np.isfinite(xv)):
        raise ValueError("x must be finite.")
    if np.any(np.diff(xv) <= 0):
        raise ValueError("x must be strictly increasing (sorted, no duplicates).")
    return xv, unit


def _is_uniform(xv: np.ndarray, rtol: float) -> Tuple[bool, float]:
    dx = np.diff(xv)
    dx0 = float(np.median(dx))
    if dx0 <= 0:
        return False, dx0
    return bool(np.allclose(dx, dx0, rtol=rtol, atol=0.0)), dx0


def _smooth_1d(y: Any, sigma_pix: float, *, cfg: GaussianConv1DConfig) -> Any:
    unit = getattr(y, "unit", None)
    yv = np.asarray(getattr(y, "value", y), dtype=float)
    out = yv if sigma_pix <= 0 else gaussian_filter1d(
        yv, sigma=sigma_pix, mode=cfg.mode, cval=cfg.cval, truncate=cfg.truncate
    )
    return out if unit is None else out * unit


def _smooth_derivs(derivs: Any, sigma_pix: float, *, cfg: GaussianConv1DConfig) -> Any:
    """
    Convolve each derivative vector along the sample axis (last axis).
    Supports:
      - (npar, npts) ndarray
      - tuple/list of 1D arrays
    """
    if sigma_pix <= 0:
        return derivs

    if isinstance(derivs, (list, tuple)):
        out = [_smooth_1d(d, sigma_pix, cfg=cfg) for d in derivs]
        return type(derivs)(out)

    d = np.asarray(derivs)
    if d.ndim == 1:
        return _smooth_1d(d, sigma_pix, cfg=cfg)
    if d.ndim == 2:
        return gaussian_filter1d(d, sigma=sigma_pix, axis=-1, mode=cfg.mode, cval=cfg.cval, truncate=cfg.truncate)

    raise ValueError(f"Unsupported fit_deriv shape: {d.shape}")


def decorate_gaussian_convolution_1d_inplace(
    model: Any,
    *,
    sigma_x: float,
    cfg: GaussianConv1DConfig = GaussianConv1DConfig(),
) -> Any:
    """
    In-place decorator: after calling this, `model(x)` returns gaussian_filter1d(model_orig(x)).
    Also fixes LevMar by convolving `fit_deriv` if present.
    Requires uniform x for stable physical sigma_x (raises if non-uniform).
    """
    sigma_x = float(sigma_x)
    if sigma_x < 0:
        raise ValueError("sigma_x must be >= 0.")
    if getattr(model, _TAG, None):
        return model

    orig_cls = model.__class__
    orig_call = orig_cls.__call__
    orig_fit_deriv = getattr(orig_cls, "fit_deriv", None)

    def _sigma_pix_from_x(x: Any) -> float:
        xv, _ = _x_to_value_and_unit(x)
        ok, dx = _is_uniform(xv, rtol=cfg.uniform_rtol)
        if not ok:
            raise ValueError("x must be uniformly spaced for this decorator (uniform grid required).")
        return 0.0 if sigma_x <= 0 else sigma_x / dx

    def __call__(self, x, *args, **kwargs):
        y0 = orig_call(self, x, *args, **kwargs)
        return _smooth_1d(y0, _sigma_pix_from_x(x), cfg=cfg)

    if callable(orig_fit_deriv):
        @staticmethod
        def fit_deriv(x, *params):
            d0 = orig_fit_deriv(x, *params)
            return _smooth_derivs(d0, _sigma_pix_from_x(x), cfg=cfg)
        cls_dict = {"__call__": __call__, "fit_deriv": fit_deriv}
    else:
        cls_dict = {"__call__": __call__}

    ConvolvedCls = type(f"{orig_cls.__name__}__GaussConv1D", (orig_cls,), cls_dict)
    model.__class__ = ConvolvedCls
    setattr(model, _TAG, {"orig_cls": orig_cls, "sigma_x": sigma_x, "cfg": cfg})
    return model


def convolve_lsf(model: Any, wavec: float, resolving_power: float):
    """
    Create a new model class that represents the convolution of the input model
    with a Gaussian LSF of given resolving power at wavec.

    Parameters
    ----------
    model : Fittable1DModel
        The base model to be convolved.
    wavec : float
        The central wavelength (in same units as x) where the resolving power is defined.
    resolving_power : float
        The resolving power R = lambda / delta_lambda.

    Returns
    -------
    Fittable1DModel
        A new model class representing the convolved model.
    """
    sigma_x = wavec / (resolving_power * 2.3548)  # FWHM to sigma conversion
    return decorate_gaussian_convolution_1d_inplace(model=model, sigma_x=sigma_x)