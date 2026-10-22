"""JAX stellar-continuum models -- mirror :mod:`galspec.stellar_continuum`.

The CPU models rebin the stellar spectra to a logarithmic wavelength scale and
broaden them with ``scipy.ndimage.gaussian_filter``.  The JAX versions reuse the
same rebinned data (via :func:`galspec.stellar_continuum.get_star`) and the exact
Gaussian convention of :func:`galspec.jax_models.functional.gaussian_smooth1d`.

Two CPU behaviours need host precomputation, both handled in :meth:`prepare`:

* the ``wave_temp`` selection window, which depends on the fitted grid, and
* the static Gaussian kernel radius, which depends on the largest allowed
  ``sigma`` (a fitted parameter cannot set a static shape).
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

from ..constants import ls_km
from ..stellar_continuum import get_star
from .core import JaxModel, ParamSpec
from .functional import gaussian_smooth1d, interp_extrapolate

__all__ = ["StarSpectrum", "Multi_StarSpectrum"]

_DEFAULT_SIGMA_MAX = 6000.0


def _prepare_radius(sigma_max: float, dln_lam: float) -> int:
    sigma_pix_max = (sigma_max / ls_km) / dln_lam
    return int(4.0 * abs(sigma_pix_max) + 0.5)


class StarSpectrum(JaxModel):
    """Single-star continuum template -- ``galspec.StarSpectrum``.

    Parameters
    ----------
    Star_type : {'A', 'F', 'G', 'K', 'M'}
        Which stellar template to use.
    velscale : float, optional
        Velocity scale for the template rebinning (``None`` = template's own).
    sigma_max : float, optional
        Upper bound on the fitted velocity dispersion used to size the static
        convolution kernel; tightened automatically from the prior by
        :class:`galspec.jnesty_fit.JNesty_Fit`.

    Notes
    -----
    Unlike the CPU class, ``delta_z`` is honoured when passed to the
    constructor (the CPU ``__init__`` silently drops it).
    """

    model_kind = "StarSpectrum"
    PARAMS = (
        ParamSpec("amplitude", 1.0, bounds=(0.0, None)),
        ParamSpec("sigma", 200.0, bounds=(20.0, 6000.0)),
        ParamSpec("delta_z", 0.0, bounds=(-0.001, 0.001), fixed=False),
    )

    def __init__(self, Star_type: str = "A", velscale: Optional[float] = None,
                 sigma_max: Optional[float] = None, name: Optional[str] = None,
                 fixed: Optional[dict] = None, bounds: Optional[dict] = None, **kwargs):
        _lam, _flux, lam_rebin, f_rebin_norm = get_star(Star_type, velscale)
        self.Star_type = Star_type
        self.velscale = velscale
        self._wave_temp = jnp.asarray(lam_rebin)
        self._flux_temp = jnp.asarray(f_rebin_norm)
        self._ln_lam = jnp.asarray(np.log(lam_rebin))
        self._dln_lam = float(np.log(lam_rebin[1]) - np.log(lam_rebin[0]))

        upper = bounds.get("sigma", (None, None))[1] if bounds else None
        if sigma_max is None:
            sigma_max = float(upper) if upper is not None else _DEFAULT_SIGMA_MAX
        elif upper is not None and np.isfinite(upper):
            sigma_max = max(float(sigma_max), float(upper))
        self.sigma_max = float(sigma_max)
        self._radius = _prepare_radius(self.sigma_max, self._dln_lam)

        # Populated by prepare(); defaults to the full template.
        self._sel_wave = self._wave_temp
        self._sel_flux = self._flux_temp
        super().__init__(name=name, fixed=fixed, bounds=bounds, **kwargs)

    def set_prior_limits(self, limits: dict) -> None:
        upper = limits.get("sigma", (None, None))[1]
        if upper is not None and np.isfinite(upper):
            self.sigma_max = float(upper)
            self._radius = _prepare_radius(self.sigma_max, self._dln_lam)

    def prepare(self, x) -> None:
        """Select the template samples inside the fitted range (CPU behaviour)."""
        x = np.asarray(x, dtype=float)
        wave = np.asarray(self._wave_temp)
        sel = (wave >= x.min() - 10.0) & (wave <= x.max() + 10.0)
        if not np.any(sel):
            raise ValueError("fitted wavelength range does not overlap the stellar template")
        self._sel_wave = jnp.asarray(wave[sel])
        self._sel_flux = jnp.asarray(np.asarray(self._flux_temp)[sel])

    def evaluate(self, x, p):
        sigma_pix = jnp.maximum(p["sigma"] / ls_km / self._dln_lam, 1e-6)
        smoothed = gaussian_smooth1d(self._sel_flux, sigma_pix, radius=self._radius)
        model_x = self._sel_wave / (1.0 + p["delta_z"])
        return p["amplitude"] * jnp.interp(x, model_x, smoothed)


class Multi_StarSpectrum(JaxModel):
    """Linear combination of stellar templates -- ``galspec.Multi_StarSpectrum``."""

    model_kind = "Multi_StarSpectrum"

    def __init__(self, Star_types: Sequence[str] = ("A", "F", "G", "K", "M"),
                 velscale: Optional[float] = 69.0, sigma_max: Optional[float] = None,
                 name: Optional[str] = None, fixed: Optional[dict] = None,
                 bounds: Optional[dict] = None, **kwargs):
        Star_types = tuple(Star_types)
        if len(Star_types) != 5:
            raise ValueError("Multi_StarSpectrum needs exactly 5 star types (amp_0..amp_4)")
        self.Star_types = Star_types
        self.velscale = velscale

        wave = None
        fluxes = []
        for star_type in Star_types:
            _lam, _flux, lam_rebin, f_rebin_norm = get_star(star_type, velscale)
            if wave is None:
                wave = np.asarray(lam_rebin, dtype=float)
            elif not np.allclose(wave, np.asarray(lam_rebin, dtype=float)):
                raise ValueError("stellar templates are not on a common wavelength grid")
            fluxes.append(np.asarray(f_rebin_norm, dtype=float))

        self._wave_temp = jnp.asarray(wave)
        self._flux_stack = jnp.asarray(np.vstack(fluxes))
        self._dln_lam = float(np.log(wave[1]) - np.log(wave[0]))

        upper = bounds.get("sigma", (None, None))[1] if bounds else None
        if sigma_max is None:
            sigma_max = float(upper) if upper is not None else _DEFAULT_SIGMA_MAX
        elif upper is not None and np.isfinite(upper):
            sigma_max = max(float(sigma_max), float(upper))
        self.sigma_max = float(sigma_max)
        self._radius = _prepare_radius(self.sigma_max, self._dln_lam)

        self._param_specs = tuple(
            [ParamSpec(f"amp_{i}", 1.0, bounds=(0.0, None)) for i in range(5)]
            + [ParamSpec("sigma", 200.0, bounds=(20.0, 6000.0))]
        )
        super().__init__(name=name, fixed=fixed, bounds=bounds, **kwargs)

    @property
    def PARAMS(self):  # type: ignore[override]
        return self._param_specs

    def set_prior_limits(self, limits: dict) -> None:
        upper = limits.get("sigma", (None, None))[1]
        if upper is not None and np.isfinite(upper):
            self.sigma_max = float(upper)
            self._radius = _prepare_radius(self.sigma_max, self._dln_lam)

    def evaluate(self, x, p):
        sigma_pix = jnp.maximum(p["sigma"] / ls_km / self._dln_lam, 1e-6)

        # All templates share one kernel width, so smooth the stack at once.
        smoothed = jnp.stack([
            gaussian_smooth1d(self._flux_stack[i], sigma_pix, radius=self._radius)
            for i in range(5)
        ])
        flux = jnp.zeros_like(x)
        for i in range(5):
            flux = flux + p[f"amp_{i}"] * interp_extrapolate(x, self._wave_temp, smoothed[i])
        return flux