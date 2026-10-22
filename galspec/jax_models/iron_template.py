"""JAX Fe II template models.

Two models, both mirroring the CPU side of GalSpec:

``IronTemplate``
    The JAX counterpart of :class:`galspec.iron_template.IronTemplate_new`:
    ``amplitude``, ``sigma`` (velocity dispersion), ``dv`` (velocity shift) and a
    ``z`` that defaults to *fixed*.  This is the parameterisation to fit with --
    ``sigma`` and ``dv`` in km/s are the quantities a paper quotes, and the
    redshift is known.

``IronTemplate_tied``
    The same template broadened not by one Gaussian but by a **multi-Gaussian**
    profile that is fitted simultaneously -- normally the profile of the broad
    Balmer lines, tied component by component.  A one-component tied model with
    ``dv = 0`` and unit weight reproduces ``IronTemplate`` exactly.

Why the broadened template is built the way it is
-------------------------------------------------
The template itself already contains the intrinsic broadening of the reference
object (Mrk 493, ~340 km/s).  What we want is the template as it would look after
passing through a Gaussian of width ``sigma``, so the extra variance to add is
``sigma**2 - sigma_intr**2`` -- the quadrature subtraction
:class:`~galspec.iron_template.IronTemplate_new` also does.  For a *sum* of
Gaussians the same works component by component, which is why no deconvolution is
needed here (the CPU ``IronTemplate_tied`` instead deconvolves the template with
an FFT and a Wiener filter, then convolves with an explicit kernel array -- that
needs a host-side kernel and is not differentiable through the kernel
parameters).

Cost
----
The convolution happens on the *template* grid -- 801 channels for ``park2022``,
5 418 for ``boroson1992`` -- not on the 80 000-pixel spectrum, and only the final
result is interpolated onto the data grid.  A two-component kernel therefore costs
one extra smoothing and one extra interpolation per component: measured at
**+0.1 to +0.45 ms per evaluation** on the example hardware (it moves with how
busy the card is), i.e. 0.2-7 minutes over a fit that needs 1e5-1e6 evaluations.  The percentage is large for a model
made of one line plus the iron and small for a whole spectrum, because
``jnp.interp`` onto the data grid dominates either way.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from ..constants import ls_km
from ..utils import package_path, splitter
from .core import JaxModel, ParamSpec
from .functional import gaussian_smooth1d

__all__ = ["IronTemplate", "IronTemplate_new", "IronTemplate_tied"]

#: Default upper bound on the fitted dispersion [km/s] used to size the static
#: convolution kernel.  ``JNesty_Fit`` tightens it from the prior automatically.
_DEFAULT_SIGMA_MAX = 5000.0

#: Smallest dispersion the model treats as "narrower than the template itself".
_MIN_SIGMA = 1e-6


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


class _IronBase(JaxModel):
    """Shared template loading, kernel sizing and the final interpolation."""

    model_kind = "IronTemplate"
    PARAMS = (
        ParamSpec("amplitude", 1.0, bounds=(0.0, None)),
        ParamSpec("sigma", 900.0 / 2.3548, bounds=(900.0 / 2.3548, None)),
        ParamSpec("z", 0.0, fixed=True),
        ParamSpec("dv", 0.0, bounds=(-2000.0, 2000.0)),
    )

    def __init__(self, template_name: str = "park2022", sigma_max: Optional[float] = None,
                 name: Optional[str] = None, fixed: Optional[dict] = None,
                 bounds: Optional[dict] = None, **kwargs):
        if "stddev" in kwargs:                      # pre-0.3 spelling of ``sigma``
            import warnings

            warnings.warn("IronTemplate parameter 'stddev' was renamed 'sigma' to match "
                          "galspec.IronTemplate_new; passing stddev= still works but is "
                          "deprecated", DeprecationWarning, stacklevel=3)
            kwargs["sigma"] = kwargs.pop("stddev")
            if fixed is not None and "stddev" in fixed:
                fixed = dict(fixed, sigma=fixed.pop("stddev"))
            if bounds is not None and "stddev" in bounds:
                bounds = dict(bounds, sigma=bounds.pop("stddev"))
        wave, flux, stddev_intr = _load_template(template_name)
        self.template_name = template_name
        self._wave_temp = jnp.asarray(wave)
        self._flux_temp = jnp.asarray(flux)
        self._stddev_intr = float(stddev_intr)
        self._vchan = float((wave[1] - wave[0]) / wave[0] * ls_km)

        upper = bounds.get("sigma", (None, None))[1] if bounds else None
        if sigma_max is None:
            sigma_max = float(upper) if upper is not None else _DEFAULT_SIGMA_MAX
        elif upper is not None and np.isfinite(upper):
            sigma_max = max(float(sigma_max), float(upper))
        self.sigma_max = float(max(sigma_max, self._stddev_intr * 1.001))
        self._update_radius()
        super().__init__(name=name, fixed=fixed, bounds=bounds, **kwargs)

    # -- static kernel sizing ------------------------------------------------------
    def _sigma_pixels(self, sigma_max: Optional[float] = None) -> float:
        sigma_max = self.sigma_max if sigma_max is None else sigma_max
        return float(np.sqrt(max(sigma_max ** 2 - self._stddev_intr ** 2, 0.0)) / self._vchan)

    def _update_radius(self) -> None:
        # floor(4*sigma + 0.5) is the convention scipy uses, so the kernel matches.
        self._radius = int(4.0 * self._sigma_pixels() + 0.5)

    def set_prior_limits(self, limits: dict) -> None:
        """Refine the static kernel radius from the fitter's prior bounds."""
        upper = limits.get("sigma", (None, None))[1]
        if upper is not None and np.isfinite(upper):
            self.sigma_max = float(upper)
            self._update_radius()

    # -- evaluation ----------------------------------------------------------------
    def _smoothed_template(self, sigma):
        """Template broadened to a total dispersion ``sigma`` [km/s]."""
        sig_pix = jnp.sqrt(jnp.clip(sigma ** 2 - self._stddev_intr ** 2, 0.0)) / self._vchan
        smoothed = gaussian_smooth1d(self._flux_temp, jnp.maximum(sig_pix, _MIN_SIGMA),
                                     radius=self._radius)
        return jnp.where(sigma < self._stddev_intr, self._flux_temp, smoothed)


class IronTemplate(_IronBase):
    """Fe II emission with a single Gaussian broadening -- ``galspec.IronTemplate_new``.

    Parameters
    ----------
    template_name : {'park2022', 'boroson1992'}
        Which template to load.
    sigma_max : float, optional
        Upper bound on the fitted ``sigma``, used to size the static convolution
        kernel.  ``JNesty_Fit`` tightens this automatically from the prior, so it
        only needs setting when the model is evaluated outside a fit.
    """

    model_kind = "IronTemplate"

    def evaluate(self, x, p):
        wave_shifted = self._wave_temp * (1.0 + p["z"] + p["dv"] / ls_km)
        return p["amplitude"] * jnp.interp(x, wave_shifted,
                                           self._smoothed_template(p["sigma"]))


class IronTemplate_tied(_IronBase):
    """Fe II emission broadened by a fitted multi-Gaussian profile.

    The kernel is a sum of Gaussian components, each with its own velocity shift,
    width and weight, so it can be tied to the broad Balmer lines component by
    component::

        fe = jxm.IronTemplate_tied(n_components=2, template_name="park2022", name="FeII")
        fe.dv_0.tied = jxm.tie.tie_MultiGauss_dv_w0("Halpha")
        fe.sigma_0.tied = jxm.tie.tie_MultiGauss_sigma_w0("Halpha")
        fe.w_0.tied = jxm.tie.tie_MultiGauss_amp_w0("Halpha")
        fe.dv_1.tied = jxm.tie.tie_MultiGauss_dv_w1("Halpha")
        ...

    The weights are normalised to unit sum, so ``amplitude`` keeps the meaning it
    has in :class:`IronTemplate` (the template scale) whatever the line
    amplitudes do.  Left untied, the components are free iron-profile parameters,
    which is a legitimate model too.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components in the broadening kernel (2 matches the
        worked example's two broad Balmer components).
    template_name : {'park2022', 'boroson1992'}
        Which template to load.
    sigma_max : float, optional
        Upper bound on ``sigma_k``, used to size the static kernel.
    """

    model_kind = "IronTemplate_tied"

    def __init__(self, n_components: int = 2, template_name: str = "park2022",
                 sigma_max: Optional[float] = None, name: Optional[str] = None,
                 fixed: Optional[dict] = None, bounds: Optional[dict] = None, **kwargs):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = int(n_components)
        super().__init__(template_name=template_name, sigma_max=sigma_max, name=name,
                         fixed=fixed, bounds=bounds, **kwargs)

    def _build_specs(self, **kwargs):
        from .core import _apply_kwargs

        specs = [
            ParamSpec("amplitude", 1.0, bounds=(0.0, None)),
            ParamSpec("z", 0.0, fixed=True),
        ]
        for index in range(self.n_components):
            specs += [
                ParamSpec(f"dv_{index}", 0.0, bounds=(-2000.0, 2000.0)),
                ParamSpec(f"sigma_{index}", 2000.0, bounds=(self._stddev_intr, None)),
                ParamSpec(f"w_{index}", 1.0, bounds=(0.0, None)),
            ]
        return _apply_kwargs(specs, kwargs, type(self).__name__)

    def set_prior_limits(self, limits: dict) -> None:
        """Size the kernel from the largest ``sigma_k`` bound."""
        upper = 0.0
        for index in range(self.n_components):
            candidate = limits.get(f"sigma_{index}", (None, None))[1]
            if candidate is not None and np.isfinite(candidate):
                upper = max(upper, float(candidate))
        if upper > 0.0:
            self.sigma_max = upper
            self._update_radius()

    def evaluate(self, x, p):
        weights = jnp.stack([p[f"w_{index}"] for index in range(self.n_components)])
        total_weight = jnp.where(jnp.sum(weights) > 0, jnp.sum(weights), 1.0)
        # Each component is projected onto the data separately rather than summed on
        # the template grid first: a shift resampled on the 150 km/s template grid
        # costs ~1e-3 of the peak in interpolation error, and projecting each
        # component keeps a one-component kernel *identical* to IronTemplate.
        flux = jnp.zeros_like(jnp.asarray(x, dtype=jnp.float64))
        for index in range(self.n_components):
            template = self._smoothed_template(p[f"sigma_{index}"])
            wave_k = self._wave_temp * (1.0 + p["z"] + p[f"dv_{index}"] / ls_km)
            flux = flux + weights[index] / total_weight * jnp.interp(x, wave_k, template)
        return p["amplitude"] * flux


#: The JAX models follow the CPU naming, so ``IronTemplate_new`` is an alias.
IronTemplate_new = IronTemplate