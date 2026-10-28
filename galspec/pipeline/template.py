"""The starter configuration that ``python -m galspec.pipeline --template`` writes.

The point of a template is that the first ``--dry-run`` succeeds: a file full of
placeholders teaches nothing except where the syntax errors are.  So this one
describes a *working* fit -- the two-line mock spectrum that ships with GalSpec --
with every section short enough to read and the alternatives for a real sample
written next to it as comments.

Nothing here is imported by the pipeline proper; :mod:`galspec.pipeline.__main__`
calls :func:`template_text` and :func:`write_template`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .errors import PipelineError

#: The spectrum the generated file points at, when GalSpec is running from a
#: checkout.  An installed package has no ``example/`` directory, and the template
#: then carries a placeholder path instead.
EXAMPLE_SPECTRUM = (Path(__file__).resolve().parents[2]
                    / "example" / "data" / "mock_two_line_spectrum.fits")

PLACEHOLDER = "path/to/your/spectrum.fits"


def _data_line(spectrum: Optional[Path]) -> str:
    if spectrum is not None and Path(spectrum).is_file():
        return (f'file = "{Path(spectrum).resolve()}"'
                "   # the mock that ships with GalSpec: this file fits it as it is")
    return (f'file = "{PLACEHOLDER}"'
            "   # <- edit this first: one FITS file per source")


def template_text(spectrum: Optional[Path] = EXAMPLE_SPECTRUM) -> str:
    """Return the starter configuration as text.

    Parameters
    ----------
    spectrum : path, optional
        The spectrum the ``[data] file`` line should point at.  Defaults to the
        mock spectrum in ``example/data``; when that does not exist the template
        gets a placeholder instead.
    """
    available = spectrum is not None and Path(spectrum).is_file()
    opening = (
        "This file is a working example: [data] points at the two-line mock\n"
        "# spectrum that ships with GalSpec, so every command below succeeds before\n"
        "# you change anything.  Replace it first, then work downwards.\n"
        if available else
        "Every command below works once [data] file names a real spectrum with\n"
        "# WAVELENGTH, FLUX and ERROR arrays.\n"
    )

    return f'''\
# GalSpec pipeline configuration -- written by
#     python -m galspec.pipeline --template
#
# A TOML file describes the data, the model and the sampler:
#     python -m galspec.pipeline {Path("config.toml")} --dry-run   # read, do not fit
#     python -m galspec.pipeline config.toml                       # fit
#     python -m galspec.pipeline config.toml --workers 4           # ... or a sample
#
# {opening}#
# The full reference is docs/source/config_fit.rst.  A 17-component, 27-parameter
# worked example is example/fit_iras07546_config.toml.


# ======================================================================================
# [run] -- where the outputs go, and how many processes
# ======================================================================================
[run]
output_dir = "out"          # resolved relative to this file, not the working directory
workers = 1                 # >1 fits the sources in that many spawned processes
# force = false             # refit even when a source's outputs already exist
# progress = true           # the sampler bar: unset = on a terminal only (workers = 1)


# ======================================================================================
# [data] -- which spectra, and what to do to them before fitting
# ======================================================================================
[data]
{_data_line(spectrum)}

# Wavelength, flux and error are all mandatory, found as columns of a binary table
# or as image HDUs with these EXTNAMEs (matched case-insensitively).
wave_key = "WAVELENGTH"
flux_key = "FLUX"
error_key = "ERROR"

# The narrow-line profile can travel inside the spectrum's own file: a table HDU
# whose first column is a velocity in km/s and whose second is the normalised flux.
# Components whose model takes a profile (Line_template, Line_template_doublet) then
# use it, and a component's own template_file = "..." overrides it.
# narrow_template_key = "NARROW"

# A spectrum is fitted in the rest frame with the Galaxy's reddening removed.  Both
# numbers are read from the header when it has them; when it does not, they are
# assumed to be zero AND THE RUN WARNS.  Declaring them here is silent.
z_default = 0.0                 # the mock is at rest
z_keys = ["Z", "REDSHIFT", "ZBEST"]
e_bv_keys = ["E(B-V)", "EBV", "EBV_GAL", "A_V", "AV"]
extinction = false              # true: de-redden with the header's colour excess
# extinction_rv = 3.1           # R_V of the reddening law

# Rest-frame window to fit, and windows to drop (sky residuals, telluric bands).
# wave_min = 4000.0
# wave_max = 5000.0
# mask = [[5577.0, 5585.0]]

min_pixels = 50                 # refuse a source with fewer pixels after the cuts
continuum_percentile = 50.0     # percentile behind the continuum_level token

# A sample instead of one spectrum -- exactly one of these forms, not several:
# files = ["a.fits", "b.fits"]          # a list, relative to this file
# glob = "spectra/*.fits"               # a shell pattern; ** recurses
# catalog = "sample.csv"                # columns: id, file, and optionally z
# catalog_z_column = "z"                # the column holding the redshift, if any


# ======================================================================================
# [fit] -- how hard to work
# ======================================================================================
[fit]
nlive = 500                 # live points: the main cost knob
dlogz = 0.01                # stop when the remaining evidence is below this
max_iterations = 200000     # safety cap
device = "auto"             # auto / cpu / gpu
share_compilation = true    # compile once for a sample; changes no number
# bound = "multi"           # none / single / multi


# ======================================================================================
# [output] -- the files, per source
# ======================================================================================
[output]
thin = 1                    # save every n-th posterior sample
fit_plot = true             # data, best model, residuals
corner = false              # a corner plot is unreadable above ~10 parameters
histogram = false
# corner_params = ["Line1.sigma", "Line2.sigma"]   # Submodel.parameter, or amp_c_3


# ======================================================================================
# [model] -- one [[model.component]] per additive term, SUMMED IN THIS ORDER
# ======================================================================================
# The order is the order of the sum, and the flat parameter names carry the index:
# the first component's parameters end in _0, this file's two lines are Line1 and
# Line2.  Ties and priors use the qualified spelling (Line1.sigma), which does not
# depend on that index.
[model]
# Every free parameter gets a default prior from its name (docs/source/config_fit.rst
# has the table: a logarithmic amplitude between a thousandth of the continuum and
# three times the spectrum's peak, dv within a few thousand km/s, sigma 10-8000 km/s,
# ...), so a component may be written with nothing but its type and its line -- the
# minimal model of this file is four lines and is written out below.  The blocks
# that follow spell the priors out instead, to show the syntax: a 150 km/s line
# whose width you want bounded is the usual reason to write one.  --dry-run lists
# which prior each free parameter ended up with.
#
# The values (amplitude, dv, sigma) are optional too: nested sampling draws from the
# prior, so they only say where the model starts.  They are written out here because
# they document the model.
require_priors = false      # true: every prior must be spelled out in this file

# ---- the whole model could be this short --------------------------------------------
# with the default priors and the per-kind defaults of the models themselves:
#
# [[model.component]]
# type = "Line_MultiGauss"
# n_components = 2
# wavec = "@Halpha"

# ---- a flat continuum ---------------------------------------------------------------
# type is any kind galspec.jax_models knows; the remaining keys are that model's
# parameters, plus constructor arguments such as n_components or template_file.
[[model.component]]
type = "Linear1D"
name = "Continuum"
slope = 0.0
intercept = "0.001*peak_flux"       # numbers may be expressions over the spectrum

# per parameter: a uniform prior over a box, in the same units as the value.
# Tokens: peak_flux, max_flux, min_flux, median_flux, continuum_level, rms,
#         n_pixels, z
[model.component.priors.intercept]
kind = "uniform"
lower = "-0.05*peak_flux"
upper = "0.05*peak_flux"

[model.component.priors.slope]
kind = "uniform"
lower = "-1e-5"
upper = "1e-5"

# ---- an emission line ---------------------------------------------------------------
# Line_Gaussian fixes its centre by default; fixed = {{ wavec = false }} frees it, and
# the box below is what keeps the two lines apart.
[[model.component]]
type = "Line_Gaussian"
name = "Line1"
wavec = 4400.0
amplitude = "0.5*peak_flux"
dv = 0.0
sigma = 150.0
fixed = {{ wavec = false }}

[model.component.bounds]
amplitude = ["0.01*peak_flux", "3*peak_flux"]
dv = [-500.0, 500.0]
sigma = [50.0, 2500.0]
wavec = [4390.0, 4410.0]

[model.component.priors.amplitude]
kind = "loguniform"         # equal probability per decade: the right prior for a
lower = "0.01*peak_flux"    # quantity spanning a factor of a hundred
upper = "3*peak_flux"

[model.component.priors.dv]
kind = "normal"
mean = 0.0
sigma = 150.0

[model.component.priors.sigma]
kind = "uniform"
lower = 50.0
upper = 2500.0

[model.component.priors.wavec]
kind = "uniform"
lower = 4390.0
upper = 4410.0

# ---- a second line that shares the first one's velocity -----------------------------
# A tie is a reference ("Line1.dv") or a named kind carrying a ratio, e.g.
#     amp_c1 = {{ kind = "doublet_ratio", ref = "OIII", ratio = 2.98 }}
# or  sigma   = {{ ref = "Line1.sigma", scale = 1.5 }}
# Line names resolve to a rest wavelength: wavec = "@Halpha", "@OIII_5007", ...
[[model.component]]
type = "Line_Gaussian"
name = "Line2"
wavec = 4600.0
amplitude = "0.5*peak_flux"
sigma = 150.0

[model.component.bounds]
amplitude = ["0.01*peak_flux", "3*peak_flux"]
sigma = [50.0, 2000.0]

[model.component.ties]
dv = "Line1.dv"

[model.component.priors.amplitude]
kind = "loguniform"
lower = "0.01*peak_flux"
upper = "3*peak_flux"

[model.component.priors.sigma]
kind = "uniform"
lower = 50.0
upper = 2000.0
'''


def write_template(path, *, force: bool = False,
                   spectrum: Optional[Path] = EXAMPLE_SPECTRUM) -> Path:
    """Write the starter configuration to ``path``.

    Parameters
    ----------
    path : path-like
        Where to write.
    force : bool
        Overwrite an existing file.  Without it an existing file is an error: the
        command must not eat a configuration somebody already wrote.
    spectrum : path, optional
        Passed to :func:`template_text`.

    Returns
    -------
    Path
        The file that was written.
    """
    target = Path(path).expanduser()
    if target.exists() and not force:
        raise PipelineError(f"{target} already exists; pass --force to overwrite it")
    text = template_text(spectrum)
    if target.parent and not target.parent.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target