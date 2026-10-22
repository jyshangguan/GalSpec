"""JAX Fe II template model -- mirrors :class:`galspec.iron_template.IronTemplate`."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from ..constants import ls_km
from ..utils import package_path, splitter
from .core import JaxModel, ParamSpec
from .functional import gaussian_smooth1d

__all__ = ["IronTemplate"]

#: Default upper bound on the template velocity dispersion [km/s] used to size the
#: static convolution kernel.  Raise via ``sigma_max`` if a prior extends beyond it.
_DEFAULT_SIGMA_MAX = 5000.0


def _load_template(template_name: str):
    """Read the Fe II template exactly as the CPU model does (normalised to 1)."""
    from astropy.table import Table

    if template_name == "park2022":
        table = Table.read(f"{package_path}{splitter}data{splitter}irontemplate_park2022.dat",
                           format="cds")
        wave = np.asarray(table["Wave"].data, dtype=float)
        flux = np.asarray(table["Flux"].data, dtype=float)
        stddev_intr = 800.0 / 2.3548          # Mrk 493 intrinsic dispersion
    elif template_name == "boroson1992":
        table = Table.read(
            f"{package_path}{splitter}data{splitter}irontemplate_boroson1992.ipac", format="ipac")
        wave = np.asarray(table["Spectral_axis"].data, dtype=float)
        flux = np.asarray(table["Intensity"].data, dtype=float)
        stddev_intr = 900.0 / 2.3548          # I Zw 1 intrinsic dispersion
    else:
        raise KeyError(f"Cannot recognize the iron template model ({template_name})!")

    flux = flux / np.max(flux)
    # CPU normalises by the maximum inside 4500 < wave < 5500; keep identical.
    window = (wave > 4500) & (wave < 5500)
    flux = flux / np.max(flux[window])
    return wave, flux, stddev_intr


class IronTemplate(JaxModel):
    """Fe II emission template -- ``galspec.IronTemplate``.

    The CPU model broadens the template with
    ``gaussian_filter1d(template, sigma_pix)`` where
    ``sigma_pix = sqrt(stddev**2 - stddev_intr**2) / vchan``; here the kernel
    radius is static (:class:`~galspec.jax_models.core.JaxModel.prepare`) so the
    model stays ``jit``-able.

    Parameters
    ----------
    template_name : {'park2022', 'boroson1992'}
        Which template to load.
    sigma_max : float, optional
        Upper bound on the fitted ``stddev`` used to size the static kernel
        radius.  ``JNesty_Fit`` tightens this automatically from the prior.
    """

    model_kind = "IronTemplate"
    PARAMS = (
        ParamSpec("amplitude", 1.0, bounds=(0.0, None)),
        ParamSpec("stddev", 900.0 / 2.3548, bounds=(900.0 / 2.3548, None)),
        ParamSpec("z", 0.0, bounds=(0.0, 10.0)),
    )

    def __init__(self, template_name: str = "park2022", sigma_max: Optional[float] = None,
                 name: Optional[str] = None, fixed: Optional[dict] = None,
                 bounds: Optional[dict] = None, **kwargs):
        wave, flux, stddev_intr = _load_template(template_name)
        self.template_name = template_name
        self._wave_temp = jnp.asarray(wave)
        self._flux_temp = jnp.asarray(flux)
        self._stddev_intr = float(stddev_intr)
        self._vchan = float((wave[1] - wave[0]) / wave[0] * ls_km)

        # Size the static kernel from the tightest available upper bound.
        upper = bounds.get("stddev", (None, None))[1] if bounds else None
        if sigma_max is None:
            sigma_max = float(upper) if upper is not None else _DEFAULT_SIGMA_MAX
        elif upper is not None and np.isfinite(upper):
            sigma_max = max(float(sigma_max), float(upper))
        self.sigma_max = float(max(sigma_max, self._stddev_intr * 1.001))
        self._update_radius()
        super().__init__(name=name, fixed=fixed, bounds=bounds, **kwargs)

    # -- static kernel sizing ------------------------------------------------------
    def _update_radius(self) -> None:
        sig_max_pix = np.sqrt(max(self.sigma_max ** 2 - self._stddev_intr ** 2, 0.0)) / self._vchan
        # sigma_pix_max reproducibility requires the floor(4*sigma+0.5) convention.
        self._radius = int(4.0 * sig_max_pix + 0.5)

    def set_prior_limits(self, limits: dict) -> None:
        """Refine the static kernel radius from the fitter's prior bounds."""
        upper = limits.get("stddev", (None, None))[1]
        if upper is not None and np.isfinite(upper):
            self.sigma_max = float(upper)
            self._update_radius()

    def evaluate(self, x, p):
        stddev = p["stddev"]
        sig_pix = jnp.sqrt(jnp.clip(stddev ** 2 - self._stddev_intr ** 2, 0.0)) / self._vchan
        sigma_safe = jnp.maximum(sig_pix, 1e-6)
        smoothed = gaussian_smooth1d(self._flux_temp, sigma_safe, radius=self._radius)
        flux_conv = jnp.where(stddev < self._stddev_intr, self._flux_temp, smoothed)
        return p["amplitude"] * jnp.interp(x, self._wave_temp * (1.0 + p["z"]), flux_conv)