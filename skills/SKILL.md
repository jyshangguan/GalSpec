---
name: galspec
description: Use this skill when the user asks to use GalSpec to fit a spectrum, do spectral fitting with GalSpec, or fit AGN or galaxy emission-line or absorption-line spectra with GalSpec.
version: 1.0.0
---

# GalSpec Spectral Fitting

GalSpec is a local Python package for fitting astronomical spectra, especially AGN
and galaxy spectra with emission and absorption lines.

GalSpec should be available as the Python package `galspec`.

If code or local package inspection is needed, first locate the installed package with Python, for example:
`python -c “import inspect, pathlib, galspec; print(pathlib.Path(inspect.getfile(galspec)).resolve())”`

This skill helps with:
- fitting spectra with GalSpec
- building spectral models
- choosing line-profile components
- fitting AGN broad and narrow lines
- fitting absorption troughs
- using LSQ, MCMC, or nested sampling
- tying parameters between components
- applying instrumental LSF convolution
- measuring physical parameters from fitted spectra

## When to use this skill

Use this skill when the user asks for things like:
- “Use GalSpec to fit the spectrum”
- “Fit this spectrum with GalSpec”
- “Use GalSpec for spectral fitting”
- “Fit the AGN spectrum with GalSpec”
- “Fit the emission lines with GalSpec”
- “Use GalSpec to model the absorption lines”
- “Use GalSpec to fit a Type 1 AGN spectrum”
- “Use GalSpec to fit BAL features”

Also use this skill when the user asks to:
- fit AGN or galaxy spectra
- decompose continuum, broad lines, narrow lines, and absorption
- fit Balmer lines, Fe II, or forbidden-line doublets
- run MCMC or Bayesian spectral fitting with GalSpec

## Supported use cases

- Type 1 AGN
- BAL QSOs
- Type 2 AGN
- galaxies with stellar continua
- spectra with emission and/or absorption lines

## Required routing before planning

Before proposing a fitting plan, first identify the task type and read the
relevant documentation.

### Read `fitting_strategies/narrow_line_template.md` first if the request mentions:
- `[S II]`, `S II`
- `[N II]`, `N II`
- `[O III]`, `O III`
- `doublet`
- `narrow line template`
- `LSF`
- `line spread function`
- `instrumental profile`
- `instrumental broadening`
- `6716`, `6731`, or `6716/6731`

### Read `fitting_strategies/type1_agn.md` first if the request mentions:
- `Type 1 AGN`
- `broad line AGN`
- `broad Hα`
- `broad Hbeta`
- `broad Hβ`
- `Balmer`
- `BAL`
- `BAL QSO`
- `absorption trough`
- `broad absorption line`

### Read `function_reference/line_profile_models.md` first if the request is about:
- a simple single emission line
- choosing a line-profile class
- line model syntax
- basic component behavior

### Read `function_reference/parameter_tying.md` first if the request is about:
- tying parameters
- shared velocities or widths
- doublet ratios
- linking absorption parameters across lines

Only after reading the relevant file should you propose a model or workflow.

## General workflow

When this skill is used:

1. Determine the science case and spectral region.
2. Read the relevant routing document above before making a plan.
3. Inspect the spectrum setup:
   - wavelength range
   - flux and uncertainty arrays
   - rest frame or observed frame
   - redshift
   - instrumental resolution or LSF information
4. Choose model components appropriate for the task.
5. Start with an LSQ fit to obtain reasonable initial parameters.
6. Refine with MCMC or dynesty if uncertainty estimation is needed.
7. Clearly report:
   - model components
   - parameter ties
   - assumptions
   - priors or bounds
   - measurements derived from the fit

## Key modeling guidance

- Use `Line_MultiGauss_doublet` for forbidden-line doublets such as
  `[S II]`, `[N II]`, and `[O III]`.
- Use `Line_template` for narrow-line template fitting.
- Use `Line_MultiGauss` for broad permitted lines such as Hα and Hβ.
- Use `Line_Absorption` for absorption components.
- Apply LSF convolution when required by the instrument or data product.
- Prefer LSQ initialization before MCMC unless the user explicitly asks
  otherwise.
- Tie parameters where physically justified to reduce degeneracy.

## Quick reference

### Basic example

```python
import numpy as np
import galspec
from galspec.utils import line_wave_dict
from astropy.modeling import fitting

# Load your spectrum
wave, flux, ferr = load_your_spectrum()

# Continuum + one Gaussian line
cont = galspec.WindowedPowerLaw1D(
    amplitude=1.0,
    x_0=5000,
    alpha=-1.0,
    x_min=4500,
    x_max=5500,
)

line = galspec.Line_Gaussian(
    amplitude=1.0,
    dv=0,
    sigma=500,
    wavec=line_wave_dict["Halpha"],
)

model = cont + line

# LSQ fit
fitter = fitting.LevMarLSQFitter()
model_lsq = fitter(model, wave, flux, weights=1 / ferr**2)

# MCMC fit
mcmc = galspec.MCMC_Fit(model_lsq, wave, flux, ferr, nwalkers=50, nsteps=1000)
samples, model_fit, param_names = mcmc.fit()