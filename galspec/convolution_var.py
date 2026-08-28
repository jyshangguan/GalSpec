"""
Variable LSF convolution for GalSpec

Implements wavelength-dependent resolving power convolution for instruments
like JWST NIRSpec prism where resolving power varies significantly with wavelength.

This module extends the functionality of convolution.py to support variable
resolving power while maintaining compatibility with the existing API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple, Dict, List, Optional, Sequence, Union
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from astropy.io import fits
from astropy.modeling import Parameter

__all__ = [
    'convolve_lsf_var',
    'ResolutionCurve',
    'find_variable_convolved_submodels',
    'refresh_variable_convolved_submodels_inplace',
]

Mode = Literal["reflect", "constant", "nearest", "mirror", "wrap"]
_TAG_VAR = "_gaussconv1d_var_callswap"

# Import configuration from base convolution module
import sys
if hasattr(sys.modules.get('galspec.convolution'), 'GaussianConv1DConfig'):
    from galspec.convolution import GaussianConv1DConfig
else:
    @dataclass(frozen=True)
    class GaussianConv1DConfig:
        mode: Mode = "reflect"
        cval: float = 0.0
        truncate: float = 4.0
        uniform_rtol: float = 1e-6


class ResolutionCurve:
    """
    Interpolate wavelength-dependent resolving power from arrays.

    This class handles the wavelength-dependent resolving power curve for
    instruments like JWST NIRSpec prism, providing interpolation to any
    wavelength in the valid range.

    Parameters
    ----------
    wavelength : array-like
        Wavelength values (in microns or Angstroms)
    resolution : array-like
        Resolving power values at each wavelength
    wave_unit : str, optional
        Unit of wavelength ('micron' or 'angstrom').
        If None, will be detected from values (default: None)
    interpolation : str, optional
        Type of interpolation ('linear', 'log', 'loglog')
    extrapolate : bool or str, optional
        If True, extrapolate beyond data range. If 'bounds', use edge values.

    Examples
    --------
    >>> import numpy as np
    >>> wave = np.array([0.5, 1.0, 2.0, 3.0, 5.0])  # microns
    >>> R = np.array([50, 100, 200, 300, 350])
    >>> rc = ResolutionCurve(wave, R, wave_unit='micron')
    >>> R_at_2 = rc.get_resolution(2.0)  # Get R at 2.0 microns
    >>> sigma = rc.get_sigma_x(2.0)  # Get sigma_x at 2.0 microns
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        resolution: np.ndarray,
        wave_unit: Optional[str] = None,
        interpolation: str = 'loglog',
        extrapolate: str = 'bounds'
    ):
        # Store raw data
        self._wave_raw = np.asarray(wavelength, dtype=float)
        self._R_raw = np.asarray(resolution, dtype=float)

        if len(self._wave_raw) != len(self._R_raw):
            raise ValueError(f"wavelength and resolution must have same length, "
                           f"got {len(self._wave_raw)} and {len(self._R_raw)}")

        if len(self._wave_raw) < 2:
            raise ValueError(f"Need at least 2 wavelength points, got {len(self._wave_raw)}")

        # Sort by wavelength (in case it's not sorted)
        sort_idx = np.argsort(self._wave_raw)
        self._wave_raw = self._wave_raw[sort_idx]
        self._R_raw = self._R_raw[sort_idx]

        # Auto-detect wavelength unit if not specified
        if wave_unit is None:
            median_wave = np.median(self._wave_raw)
            if median_wave < 100:
                wave_unit = 'micron'  # Likely microns (< 100)
            else:
                wave_unit = 'angstrom'  # Likely Angstroms (> 100)

        # Store wavelength range and unit info
        self._wave_unit = wave_unit
        self.wave_min = float(self._wave_raw.min())
        self.wave_max = float(self._wave_raw.max())
        self.wave_array = self._wave_raw.copy()

        # Create interpolation function
        self._interpolation = interpolation
        self._extrapolate = extrapolate

        if interpolation == 'linear':
            self._interp_func = interp1d(
                self._wave_raw, self._R_raw,
                kind='linear',
                bounds_error=False,
                fill_value=(self._R_raw[0], self._R_raw[-1]) if extrapolate == 'bounds' else np.nan
            )
        elif interpolation == 'log':
            self._interp_func = interp1d(
                np.log(self._wave_raw), self._R_raw,
                kind='linear',
                bounds_error=False,
                fill_value=(self._R_raw[0], self._R_raw[-1]) if extrapolate == 'bounds' else np.nan
            )
        elif interpolation == 'loglog':
            self._interp_func = interp1d(
                np.log(self._wave_raw), np.log(self._R_raw),
                kind='linear',
                bounds_error=False,
                fill_value=(np.log(self._R_raw[0]), np.log(self._R_raw[-1])) if extrapolate == 'bounds' else np.nan
            )
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation}")

    def _normalize_wavelength(self, wave: np.ndarray) -> np.ndarray:
        """Convert input wavelength to the unit used in the resolution file."""
        wave = np.asarray(wave, dtype=float)

        # If input has units, extract the value
        if hasattr(wave, 'unit'):
            wave = wave.value
            # Try to convert to the unit we expect
            target_unit = 'um' if self._wave_unit == 'micron' else 'AA'
            try:
                wave = wave.to(target_unit).value
            except Exception:
                # If conversion fails, assume it's already in the right unit
                pass

        return wave

    def get_resolution(self, wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get resolving power at given wavelength(s).

        Parameters
        ----------
        wavelength : float or array-like
            Wavelength value(s) in Angstroms or microns (auto-detected)

        Returns
        -------
        R : float or array-like
            Resolving power at the requested wavelength(s)

        Examples
        --------
        >>> rc = ResolutionCurve('galspec/data/NIRSpec_prism_resolution.fits')
        >>> R_2micron = rc.get_resolution(2.0)  # 2.0 microns
        >>> R_6563 = rc.get_resolution(6563)  # 6563 Angstroms (H-alpha)
        """
        wave = self._normalize_wavelength(wavelength)

        # Convert to our internal unit if needed
        if self._wave_unit == 'micron' and np.median(wave) > 100:
            # Input is likely in Angstroms, convert to microns
            wave_input = wave / 1e4
        elif self._wave_unit == 'angstrom' and np.median(wave) < 100:
            # Input is likely in microns, convert to Angstroms
            wave_input = wave * 1e4
        else:
            wave_input = wave

        # Apply interpolation
        if self._interpolation == 'loglog':
            result = np.exp(self._interp_func(np.log(wave_input)))
        elif self._interpolation == 'log':
            result = self._interp_func(np.log(wave_input))
        else:
            result = self._interp_func(wave_input)

        # Handle scalar input
        if np.isscalar(wavelength):
            return float(result)
        return result

    def get_sigma_x(
        self,
        wavelength: Union[float, np.ndarray],
        fwhm_to_sigma: float = 2.3548
    ) -> Union[float, np.ndarray]:
        """
        Get Gaussian sigma_x for LSF convolution at given wavelength(s).

        The sigma_x is calculated as: sigma_x = wavelength / (R * fwhm_to_sigma)

        Parameters
        ----------
        wavelength : float or array-like
            Wavelength value(s) in same unit as used in resolution file
        fwhm_to_sigma : float, optional
            Conversion factor from FWHM to sigma (default: 2.3548)

        Returns
        -------
        sigma_x : float or array-like
            Gaussian sigma in wavelength units

        Examples
        --------
        >>> rc = ResolutionCurve('galspec/data/NIRSpec_prism_resolution.fits')
        >>> sigma = rc.get_sigma_x(2.0)  # at 2.0 microns
        """
        wave = self._normalize_wavelength(wavelength)

        # Get resolution
        R = self.get_resolution(wave)

        # Calculate sigma_x
        # Need to ensure wave is in the same unit as expected by the resolution curve
        if self._wave_unit == 'micron' and np.median(wave) > 100:
            wave_for_calc = wave / 1e4
        elif self._wave_unit == 'angstrom' and np.median(wave) < 100:
            wave_for_calc = wave * 1e4
        else:
            wave_for_calc = wave

        sigma_x = wave_for_calc / (R * fwhm_to_sigma)

        if np.isscalar(wavelength):
            return float(sigma_x)
        return sigma_x


class _VariableKernel:
    """
    Precomputed variable-width Gaussian convolution kernels.

    The convolution weights depend only on the wavelength grid and the
    resolution curve, not on the model flux.  This class builds the per-pixel
    index/weight pairs once and then applies them to arbitrary flux vectors
    with a small Python loop.
    """

    def __init__(
        self,
        wave: np.ndarray,
        resolution_curve: ResolutionCurve,
        cfg: GaussianConv1DConfig
    ) -> None:
        wavev = np.asarray(getattr(wave, "value", wave), dtype=float)
        if wavev.ndim != 1 or len(wavev) == 0:
            raise ValueError("wave must be a non-empty 1D array")

        self.wave = wavev
        self.n = len(wavev)

        # Average wavelength spacing used by the original implementation.
        dw = float(np.median(np.diff(wavev)))
        if not np.isfinite(dw) or dw <= 0:
            raise ValueError("wave must be increasing and finite")

        sigma_x = np.asarray(resolution_curve.get_sigma_x(wavev), dtype=float)

        # Match the unit conversion performed by the original per-pixel loop.
        wave_median = float(np.median(wavev))
        if resolution_curve._wave_unit == "micron" and wave_median > 100:
            sigma_x = sigma_x * 1e4
        elif resolution_curve._wave_unit == "angstrom" and wave_median < 100:
            sigma_x = sigma_x / 1e4

        sigma_pix = sigma_x / dw
        truncate = cfg.truncate

        indices: List[np.ndarray] = []
        weights: List[np.ndarray] = []

        for i in range(self.n):
            if sigma_pix[i] <= 0.5:
                indices.append(np.array([i], dtype=int))
                weights.append(np.array([1.0]))
                continue

            kernel_width = int(truncate * sigma_pix[i])
            i_start = max(0, i - kernel_width)
            i_end = min(self.n, i + kernel_width + 1)
            idx = np.arange(i_start, i_end, dtype=int)

            w = np.exp(-0.5 * ((wavev[idx] - wavev[i]) / sigma_x[i]) ** 2)
            weight_sum = float(np.sum(w))
            if weight_sum > 0:
                w = w / weight_sum
            else:
                # Degenerate kernel: behave like the original fallback, which
                # keeps the central flux value unchanged.
                w = (idx == i).astype(float)

            indices.append(idx)
            weights.append(w)

        self.indices = tuple(indices)
        self.weights = tuple(weights)

    def apply(self, yv: np.ndarray) -> np.ndarray:
        """Convolve a flux array using the precomputed kernels."""
        if yv.shape != (self.n,):
            raise ValueError(
                f"flux has shape {yv.shape}, expected {(self.n,)}"
            )

        result = np.empty_like(yv, dtype=float)
        for i in range(self.n):
            result[i] = np.dot(self.weights[i], yv[self.indices[i]])
        return result


def _smooth_1d_variable(
    y: Any,
    wave: np.ndarray,
    resolution_curve: Optional[ResolutionCurve] = None,
    *,
    cfg: GaussianConv1DConfig,
    kernel_cache: Optional[_VariableKernel] = None
) -> Any:
    """
    Apply variable-width Gaussian convolution (simplified implementation).

    This is a basic implementation that applies local Gaussian convolution
    at each wavelength point using the wavelength-dependent sigma from the
    resolution curve.

    For production use, consider optimizations like:
    - Vectorized operations
    - Kernel caching for similar wavelength ranges
    - Numba/C acceleration

    Parameters
    ----------
    y : array-like
        Flux values to convolve
    wave : np.ndarray
        Wavelength array (must be same length as y)
    resolution_curve : ResolutionCurve, optional
        Resolution curve object
    cfg : GaussianConv1DConfig
        Convolution configuration
    kernel_cache : _VariableKernel, optional
        Precomputed variable convolution kernel.  When provided, it avoids
        rebuilding the per-pixel kernels on every call.

    Returns
    -------
    convolved : array-like
        Convolved flux values
    """
    unit = getattr(y, 'unit', None)
    yv = np.asarray(getattr(y, 'value', y), dtype=float)

    if kernel_cache is None:
        if resolution_curve is None:
            raise ValueError(
                "resolution_curve and kernel_cache cannot both be None"
            )
        wavev = np.asarray(getattr(wave, 'value', wave), dtype=float)
        if yv.shape != wavev.shape:
            raise ValueError(
                f"y and wave must have same shape: {yv.shape} vs {wavev.shape}"
            )
        kernel_cache = _VariableKernel(wavev, resolution_curve, cfg)
    elif yv.shape != (kernel_cache.n,):
        raise ValueError(
            f"y has shape {yv.shape}, expected {(kernel_cache.n,)}"
        )

    result = kernel_cache.apply(yv)

    return result if unit is None else result * unit


def _smooth_derivs_variable(
    derivs: Any,
    wave: np.ndarray,
    resolution_curve: Optional[ResolutionCurve] = None,
    *,
    cfg: GaussianConv1DConfig,
    kernel_cache: Optional[_VariableKernel] = None
) -> Any:
    """
    Convolve derivative vectors with variable LSF.

    For variable convolution, derivatives become more complex as the
    convolution itself varies with wavelength. This implementation provides
    a basic approach that convolves each derivative vector.

    Parameters
    ----------
    derivs : array-like
        Derivative vectors (typically for fit_deriv)
    wave : np.ndarray
        Wavelength array
    resolution_curve : ResolutionCurve, optional
        Resolution curve object
    cfg : GaussianConv1DConfig
        Convolution configuration
    kernel_cache : _VariableKernel, optional
        Precomputed variable convolution kernel.

    Returns
    -------
    convolved_derivs : array-like
        Convolved derivative vectors
    """
    if kernel_cache is None:
        if resolution_curve is None:
            raise ValueError(
                "resolution_curve and kernel_cache cannot both be None"
            )
        wavev = np.asarray(getattr(wave, 'value', wave), dtype=float)
        kernel_cache = _VariableKernel(wavev, resolution_curve, cfg)

    if isinstance(derivs, (list, tuple)):
        out = [
            _smooth_1d_variable(
                d, wave, resolution_curve, cfg=cfg, kernel_cache=kernel_cache
            )
            for d in derivs
        ]
        return type(derivs)(out)

    d = np.asarray(derivs)
    if d.ndim == 1:
        return _smooth_1d_variable(
            d, wave, resolution_curve, cfg=cfg, kernel_cache=kernel_cache
        )
    if d.ndim == 2:
        # Convolve each parameter's derivative
        result = np.zeros_like(d)
        for i in range(d.shape[0]):
            result[i] = _smooth_1d_variable(
                d[i], wave, resolution_curve, cfg=cfg, kernel_cache=kernel_cache
            )
        return result

    raise ValueError(f"Unsupported fit_deriv shape: {d.shape}")


def convolve_lsf_var(
    model: Any,
    wavec: np.ndarray,
    resolution_data: Union[str, Path, Tuple[np.ndarray, np.ndarray], ResolutionCurve],
    class_label: Optional[str] = None,
    cfg: GaussianConv1DConfig = GaussianConv1DConfig(),
) -> Any:
    """
    Convolve model with wavelength-dependent Gaussian LSF.

    This function creates a new model class that represents the convolution
    of the input model with a Gaussian LSF with wavelength-dependent resolving
    power. Unlike `convolve_lsf` which uses constant resolving power, this
    function supports variable resolving power for instruments like JWST
    NIRSpec prism.

    Parameters
    ----------
    model : Fittable1DModel
        The base model to be convolved.
    wavec : np.ndarray
        The wavelength array where the model will be evaluated. This defines
        the grid for convolution and should match the wavelength array used
        in fitting/evaluation.
    resolution_data : str, Path, tuple, or ResolutionCurve
        Either:
        - Tuple of (wavelength_array, resolution_array)
        - ResolutionCurve object

        Wavelength arrays can be in microns or Angstroms (will be auto-detected).
    class_label : str, optional
        An optional label to append to the generated class name.
    cfg : GaussianConv1DConfig, optional
        Convolution configuration (mode, truncate, etc.)

    Returns
    -------
    Fittable1DModel
        A new model class representing the convolved model. The returned
        model has metadata stored in model.meta about the convolution.

    Examples
    --------
    >>> import numpy as np
    >>> import galspec
    >>> from astropy.modeling import models
    >>>
    >>> # Create a model
    >>> gauss = models.Gaussian1D(amplitude=1.0, mean=6562.8, stddev=10.0)
    >>>
    >>> # Define wavelength grid
    >>> wave = np.linspace(6400, 6700, 1000)
    >>>
    >>> # Apply variable LSF convolution from arrays directly
    >>> wave_res = np.array([0.6, 1.0, 2.0, 3.0, 5.0])  # microns
    >>> R_res = np.array([30, 100, 200, 300, 350])
    >>> model_conv = galspec.convolve_lsf_var(
    ...     gauss,
    ...     wavec=wave,
    ...     resolution_data=(wave_res, R_res)
    ... )
    >>>
    >>> # Or using ResolutionCurve object
    >>> rc = galspec.ResolutionCurve(wave_res, R_res, wave_unit='micron')
    >>> model_conv2 = galspec.convolve_lsf_var(
    ...     gauss,
    ...     wavec=wave,
    ...     resolution_data=rc
    ... )
    >>>
    >>> # Use the convolved model
    >>> flux = model_conv(wave)

    Notes
    -----
    The variable LSF convolution is computationally more expensive than
    constant LSF convolution (O(n*m) vs O(n) where n is the number of
    wavelength points and m is the average kernel size). For typical
    spectra (n ~ 1000-10000), expect the convolution to take 0.1-5 seconds.

    The convolution preserves total flux to within the accuracy specified
    by the truncate parameter (default: 4 sigma).

    See Also
    --------
    convolve_lsf : Constant resolving power convolution
    ResolutionCurve : Class for loading and interpolating resolution curves
    """
    # Normalize wavec to numpy array
    if isinstance(wavec, Parameter):
        wavec = wavec.value
    wavec = np.asarray(wavec, dtype=float)

    if wavec.ndim != 1:
        raise ValueError("wavec must be 1D array")

    # Load or create resolution curve
    if isinstance(resolution_data, ResolutionCurve):
        rc = resolution_data
    elif isinstance(resolution_data, (tuple, list)) and len(resolution_data) == 2:
        wave_res, R_res = resolution_data
        # Create ResolutionCurve from arrays
        rc = ResolutionCurve(wave_res, R_res)
    else:
        raise TypeError(
            f"resolution_data must be tuple (wave, R) or ResolutionCurve, "
            f"got {type(resolution_data)}"
        )

    # Store reference to wavec and resolution curve
    # We'll use these in the wrapped __call__ method
    _wavec_ref = wavec.copy()
    _rc_ref = rc

    # Get original class and methods
    orig_cls = model.__class__
    orig_call = orig_cls.__call__
    orig_fit_deriv = getattr(orig_cls, "fit_deriv", None)

    # Check if already convolved
    if getattr(model, _TAG_VAR, None):
        return model

    # Build the variable convolution kernel once.  The kernel depends only on
    # the wavelength grid and resolution curve, so it can be reused by every
    # subsequent __call__/fit_deriv invocation (important for MCMC).
    _kernel_cache = _VariableKernel(_wavec_ref, _rc_ref, cfg)

    # Define the wrapped __call__ method
    def __call__(self, x, *args, **kwargs):
        y0 = orig_call(self, x, *args, **kwargs)
        # Interpolate y0 onto wavec grid if needed
        x_vals = np.asarray(getattr(x, 'value', x), dtype=float)

        # Check if we need to interpolate
        if len(x_vals) == len(_wavec_ref) and np.allclose(x_vals, _wavec_ref, rtol=1e-6):
            # Same grid, no interpolation needed
            y_to_convolve = y0
            wave_to_use = _wavec_ref
        else:
            # Need to interpolate onto our stored grid
            from scipy.interpolate import interp1d
            y_unit = getattr(y0, 'unit', None)

            yv = np.asarray(getattr(y0, 'value', y0), dtype=float)

            # Create interpolation function
            # Handle non-finite values
            finite_mask = np.isfinite(yv)
            if np.sum(finite_mask) < 2:
                # Not enough valid points for interpolation
                return y0

            interp_func = interp1d(
                x_vals[finite_mask],
                yv[finite_mask],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            y_to_convolve = interp_func(_wavec_ref)
            if y_unit is not None:
                y_to_convolve = y_to_convolve * y_unit
            wave_to_use = _wavec_ref

        # Apply variable convolution
        result = _smooth_1d_variable(
            y_to_convolve,
            wave_to_use,
            _rc_ref,
            cfg=cfg,
            kernel_cache=_kernel_cache,
        )

        # Interpolate back to original grid if needed
        if len(x_vals) != len(_wavec_ref):
            result_unit = getattr(result, 'unit', None)
            rv = np.asarray(getattr(result, 'value', result), dtype=float)

            interp_back = interp1d(
                _wavec_ref,
                rv,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            result = interp_back(x_vals)
            if result_unit is not None:
                result = result * result_unit

        return result

    # Define wrapped fit_deriv if present
    if callable(orig_fit_deriv):
        @staticmethod
        def fit_deriv(x, *params):
            d0 = orig_fit_deriv(x, *params)
            return _smooth_derivs_variable(
                d0,
                _wavec_ref,
                _rc_ref,
                cfg=cfg,
                kernel_cache=_kernel_cache,
            )
        cls_dict = {"__call__": __call__, "fit_deriv": fit_deriv}
    else:
        cls_dict = {"__call__": __call__}

    # Create new class
    label_suffix = class_label if class_label else "Var"
    ConvolvedCls = type(
        f"{orig_cls.__name__}_{label_suffix}_GaussConv1DVar",
        (orig_cls,),
        cls_dict
    )

    # Assign new class to model
    model.__class__ = ConvolvedCls

    # Store metadata and tag
    setattr(model, _TAG_VAR, {
        "orig_cls": orig_cls,
        "wavec": _wavec_ref,
        "resolution_curve": _rc_ref,
        "cfg": cfg
    })

    # Store metadata in model.meta
    model.meta = dict(getattr(model, "meta", {}) or {})
    model.meta["lsf_convolved"] = True
    model.meta["lsf_variable"] = True
    model.meta["lsf_wave_range"] = (float(_wavec_ref.min()), float(_wavec_ref.max()))
    model.meta["lsf_resolution_wave_range"] = (_rc_ref.wave_min, _rc_ref.wave_max)

    # Store resolution data info
    if isinstance(resolution_data, (str, Path)):
        model.meta["lsf_resolution_file"] = str(resolution_data)
    else:
        model.meta["lsf_resolution_file"] = "ResolutionCurve_object"

    # Register class in module namespace (for pickling)
    import galspec.convolution_var as conv_var_module
    setattr(conv_var_module, ConvolvedCls.__name__, ConvolvedCls)
    ConvolvedCls.__module__ = conv_var_module.__name__

    return model


def _looks_variable_convolved(m: Any, *, tag_attr: str) -> bool:
    """Check if a model has been convolved with variable LSF."""
    meta = getattr(m, "meta", {}) or {}
    if meta.get("lsf_variable") is True:
        return True
    if hasattr(m, tag_attr):
        return True
    return "__GaussConv1DVar" in m.__class__.__name__


def iter_submodels(root: Any) -> Sequence[Any]:
    """
    Iterate submodels in a compound tree if possible; otherwise yield root only.
    """
    traverse = getattr(root, "traverse_postorder", None)
    if callable(traverse):
        return list(traverse())
    return [root]


@dataclass(frozen=True)
class ConvolvedNode:
    """One submodel detected as convolved."""
    model: Any
    name: Optional[str]
    cls_name: str
    meta: Dict[str, Any]


def find_variable_convolved_submodels(
    model: Any,
    *,
    tag_attr: str = "_gaussconv1d_var_callswap",
) -> List[ConvolvedNode]:
    """
    Return submodels in (possibly compound) `model` that have been
    convolved with variable LSF.

    Detection uses, in order:
      1) model.meta["lsf_variable"] == True  (recommended tag)
      2) hasattr(model, tag_attr)             (private decorator tag)
      3) "__GaussConv1DVar" in class name     (fallback)

    Parameters
    ----------
    model : Any
        Model to search (can be compound)
    tag_attr : str, optional
        Name of the private tag attribute

    Returns
    -------
    List[ConvolvedNode]
        List of convolved submodels with metadata

    Examples
    --------
    >>> import galspec
    >>> submodels = galspec.find_variable_convolved_submodels(compound_model)
    >>> for sm in submodels:
    ...     print(f"{sm.name}: {sm.meta.get('lsf_resolution_file')}")
    """
    out: List[ConvolvedNode] = []
    for node in iter_submodels(model):
        if _looks_variable_convolved(node, tag_attr=tag_attr):
            meta = dict(getattr(node, "meta", {}) or {})
            out.append(
                ConvolvedNode(
                    model=node,
                    name=getattr(node, "name", None),
                    cls_name=node.__class__.__name__,
                    meta=meta,
                )
            )
    return out


def refresh_variable_convolved_submodels_inplace(
    model: Any,
    *,
    tag_attr: str = "_gaussconv1d_var_callswap",
) -> int:
    """
    Rebuild variable-convolved wrapper classes in-place.

    This function is similar to refresh_convolved_submodels_inplace from
    convolution.py, but for variable LSF convolution. It uses the stored
    decorator tag to restore and re-apply the decorator, generating fresh
    wrapper classes in the current runtime.

    This is useful after unpickling models that were convolved with
    variable LSF, as the dynamic wrapper classes may not be properly
    restored.

    Parameters
    ----------
    model : Any
        Model to refresh (can be compound)
    tag_attr : str, optional
        Name of the private tag attribute

    Returns
    -------
    int
        Number of submodels refreshed

    Examples
    --------
    >>> import galspec
    >>> n_refreshed = galspec.refresh_variable_convolved_submodels_inplace(model)
    >>> print(f"Refreshed {n_refreshed} submodels")
    """
    refreshed = 0

    for node in find_variable_convolved_submodels(model, tag_attr=tag_attr):
        m = node.model
        tag = getattr(m, tag_attr, None)

        if not isinstance(tag, dict):
            continue

        orig_cls = tag.get("orig_cls")
        wavec = tag.get("wavec")
        resolution_curve = tag.get("resolution_curve")
        cfg = tag.get("cfg", GaussianConv1DConfig())

        if orig_cls is None or wavec is None or resolution_curve is None:
            continue

        try:
            # Restore original class
            m.__class__ = orig_cls

            # Remove old tag
            try:
                delattr(m, tag_attr)
            except Exception:
                pass

            # Re-apply variable convolution decorator
            # This requires re-running the decoration logic
            _wavec_ref = wavec
            _rc_ref = resolution_curve
            _kernel_cache = _VariableKernel(_wavec_ref, _rc_ref, cfg)

            orig_call = orig_cls.__call__
            orig_fit_deriv = getattr(orig_cls, "fit_deriv", None)

            def __call__(self, x, *args, **kwargs):
                y0 = orig_call(self, x, *args, **kwargs)
                x_vals = np.asarray(getattr(x, 'value', x), dtype=float)

                if len(x_vals) == len(_wavec_ref) and np.allclose(x_vals, _wavec_ref, rtol=1e-6):
                    y_to_convolve = y0
                    wave_to_use = _wavec_ref
                else:
                    from scipy.interpolate import interp1d
                    y_unit = getattr(y0, 'unit', None)
                    yv = np.asarray(getattr(y0, 'value', y0), dtype=float)
                    finite_mask = np.isfinite(yv)
                    if np.sum(finite_mask) < 2:
                        return y0
                    interp_func = interp1d(
                        x_vals[finite_mask], yv[finite_mask],
                        kind='linear', bounds_error=False, fill_value='extrapolate'
                    )
                    y_to_convolve = interp_func(_wavec_ref)
                    if y_unit is not None:
                        y_to_convolve = y_to_convolve * y_unit
                    wave_to_use = _wavec_ref

                result = _smooth_1d_variable(
                    y_to_convolve,
                    wave_to_use,
                    _rc_ref,
                    cfg=cfg,
                    kernel_cache=_kernel_cache,
                )

                if len(x_vals) != len(_wavec_ref):
                    result_unit = getattr(result, 'unit', None)
                    rv = np.asarray(getattr(result, 'value', result), dtype=float)
                    interp_back = interp1d(
                        _wavec_ref, rv, kind='linear',
                        bounds_error=False, fill_value='extrapolate'
                    )
                    result = interp_back(x_vals)
                    if result_unit is not None:
                        result = result * result_unit

                return result

            if callable(orig_fit_deriv):
                @staticmethod
                def fit_deriv(x, *params):
                    d0 = orig_fit_deriv(x, *params)
                    return _smooth_derivs_variable(
                        d0,
                        _wavec_ref,
                        _rc_ref,
                        cfg=cfg,
                        kernel_cache=_kernel_cache,
                    )
                cls_dict = {"__call__": __call__, "fit_deriv": fit_deriv}
            else:
                cls_dict = {"__call__": __call__}

            label_suffix = "Var"
            ConvolvedCls = type(
                f"{orig_cls.__name__}_{label_suffix}_GaussConv1DVar",
                (orig_cls,),
                cls_dict
            )

            m.__class__ = ConvolvedCls
            setattr(m, tag_attr, {
                "orig_cls": orig_cls,
                "wavec": _wavec_ref,
                "resolution_curve": _rc_ref,
                "cfg": cfg
            })

            refreshed += 1

        except Exception as e:
            # Best-effort refresh; don't break caller if one submodel fails
            import warnings
            warnings.warn(f"Failed to refresh variable convolved submodel: {e}")
            continue

    return refreshed
