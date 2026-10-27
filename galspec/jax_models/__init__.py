"""JAX / GPU spectral-model backend for GalSpec.

This subpackage is an optional, self-contained re-implementation of the most
used GalSpec models in pure :mod:`jax.numpy`, so the same forward model can run
on a GPU (or on the CPU under ``jit``/``vmap``) and be sampled with
:class:`galspec.jnesty_fit.JNesty_Fit`.

It is *additive*: nothing in the existing astropy-based API changes, and
``import galspec`` does not import JAX.

Numerical parity
----------------
All models are written to reproduce their astropy/GalSpec CPU counterparts to
float64 rounding.  The parity test suite is ``tests/test_jax_parity.py``:

.. code-block:: bash

    python -m pytest tests/test_jax_parity.py -v

Precision
---------
``jax_enable_x64`` is switched on at import so results match numpy to ~1e-13.
Set the environment variable ``GALSPEC_JAX_X64=0`` *before* importing to run in
float32 (faster on consumer GPUs, ~1e-6 agreement with the CPU models).

Quick start
-----------
.. code-block:: python

    from galspec.jax_models import PowerLaw1D, Line_MultiGauss_doublet, tie
    from galspec.jnesty_fit import JNesty_Fit

    continuum = PowerLaw1D(amplitude=0.5, x_0=5000.0, alpha=1.0, fixed={"x_0": True},
                           name="Continuum")
    o3 = Line_MultiGauss_doublet(n_components=2, amp_c0=1.7, amp_c1=1.7 / 2.98,
                                 dv_c=-43.0, sigma_c=250.0, wavec0=5006.84,
                                 wavec1=4958.91, name="OIII",
                                 par_w={"amp_w0": 0.35, "dv_w0": -280.0, "sigma_w0": 340.0})
    o3.amp_c1.tied = tie.tie_MultiGauss_doublet_ratio("OIII", 2.98)
    model = continuum + o3

    fit = JNesty_Fit(model, wave, flux, error, nlive=200, bound="multi")
    fit.fit(dlogz=0.5)
"""

from __future__ import annotations

import os

import jax

# Match numpy precision by default; opt out with GALSPEC_JAX_X64=0 for speed.
if os.environ.get("GALSPEC_JAX_X64", "1") != "0":  # pragma: no cover - process-wide setting
    jax.config.update("jax_enable_x64", True)

from .core import JaxCompoundModel, JaxModel, JaxParameter, ParamSpec  # noqa: E402
from .functional import LS_KM, gaussian_kernel1d, gaussian_smooth1d, velocity  # noqa: E402

from .continuum import (  # noqa: E402
    BrokenPowerLaw1D,
    Const1D,
    Gaussian1D,
    Linear1D,
    Polynomial1D,
    PowerLaw1D,
    WindowedLinear,
    WindowedPowerLaw1D,
)
from .line_profile import (  # noqa: E402
    Line_Absorption,
    Line_Absorption_doublet,
    Line_Exponential,
    Line_GaussHermite,
    Line_GaussHermite_doublet,
    Line_Gaussian,
    Line_MultiGauss,
    Line_MultiGauss_doublet,
    Line_template,
    Line_template_doublet,
)
from .iron_template import IronTemplate, IronTemplate_new, IronTemplate_tied  # noqa: E402
from .stellar import Multi_StarSpectrum, StarSpectrum  # noqa: E402
from .convolution import (  # noqa: E402
    JaxConvolvedModel,
    JaxVariableConvolvedModel,
    convolve_lsf,
    convolve_lsf_var,
)

# Tie constraints: the CPU implementations are generic and work unchanged on the
# JAX models, so re-export the very same objects for a familiar API.
from .. import line_profile as _cpu_line_profile  # noqa: E402
from .. import stellar_continuum as _cpu_stellar  # noqa: E402

tie = _cpu_line_profile
tie_StarSpectrum_deltaz = _cpu_stellar.tie_StarSpectrum_deltaz
tie_StarSpectrum_sigma = _cpu_stellar.tie_StarSpectrum_sigma

#: Every model class shipped by this backend, for discovery/help.
MODELS = {
    cls.model_kind: cls
    for cls in (
        PowerLaw1D, BrokenPowerLaw1D, Const1D, Linear1D, Polynomial1D,
        WindowedPowerLaw1D, WindowedLinear, Gaussian1D,
        Line_Gaussian, Line_GaussHermite, Line_GaussHermite_doublet,
        Line_MultiGauss, Line_MultiGauss_doublet, Line_Absorption,
        Line_Absorption_doublet, Line_template, Line_template_doublet,
        Line_Exponential,
        IronTemplate, IronTemplate_tied, StarSpectrum, Multi_StarSpectrum,
    )
}

__all__ = [
    # core
    "JaxModel", "JaxCompoundModel", "JaxParameter", "ParamSpec",
    "velocity", "gaussian_smooth1d", "gaussian_kernel1d", "LS_KM",
    # continuum
    "PowerLaw1D", "BrokenPowerLaw1D", "Const1D", "Linear1D", "Polynomial1D",
    "WindowedPowerLaw1D", "WindowedLinear", "Gaussian1D",
    # lines
    "Line_Gaussian", "Line_GaussHermite", "Line_GaussHermite_doublet",
    "Line_MultiGauss", "Line_MultiGauss_doublet", "Line_Absorption",
    "Line_Absorption_doublet", "Line_template", "Line_template_doublet",
    "Line_Exponential",
    # templates / stellar
    "IronTemplate", "IronTemplate_new", "IronTemplate_tied",
    "StarSpectrum", "Multi_StarSpectrum",
    # convolution
    "convolve_lsf", "convolve_lsf_var",
    "JaxConvolvedModel", "JaxVariableConvolvedModel",
    # ties (CPU implementations, reused)
    "tie", "tie_StarSpectrum_deltaz", "tie_StarSpectrum_sigma",
    # misc
    "MODELS",
]