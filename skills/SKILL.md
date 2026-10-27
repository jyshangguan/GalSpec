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
- tying parameters, including soft ties via `priors.Normal(mean="other.param")`
- deciding how many broad/narrow components the data need
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

### Read `function_reference/gpu_jax_fitting.md` first if the request mentions:
- `JAX`, `jax`, `GPU`, `CUDA`, `nvidia-smi`
- `jax_models`, `jxm`, `JNesty`, `nested sampling`, `evidence`, `log Z`
- "make it fast", "large number of components", "many parameters"
- `batch_size`, `queue_size`, `jit`
- "how many components", "is a fourth component needed", `SpikeAndSlab`,
  `toggle prior`, component count, gating
- a *soft* tie: `priors.Normal(mean="other.param", relative=..., nsigma=...)`,
  "centre the prior on X", Bayes factors, the pull of one component against another
- "does this line need a wing/second component", "the forbidden lines have wings",
  a gated extra component on a single line

The astropy twins (`galspec.*`) and the JAX twins (`galspec.jax_models.*`) share
class names and keyword arguments, so a model can be moved between backends
with no rewriting; only the sampler and the array handling differ.

### Always worth a minute, whatever the request
Before adding any component, bin the residual of the current model in velocity
around the line and look at its shape.  For a doublet, scale one line by its
theoretical ratio and overlay the two: a feature that appears in both lines in
that ratio is real; one that appears in a single line is a local blend, a
continuum error, or noise.  Forty lines of code, no sampler, and it decides
whether the extra component is justified at all.

### Read `function_reference/line_profile_models.md` first if the request is about:
- a simple single emission line
- choosing a line-profile class
- line model syntax
- basic component behavior

### Read `function_reference/parameter_tying.md` first if the request is about:
- hard ties (`tied=`, `tie_*` helpers) between parameters of one or several models
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
6. Refine with MCMC or dynesty if uncertainty estimation is needed.  If a GPU
   is available, use the JAX twins plus `JNesty_Fit` instead -- the model is
   identical and the sampler is 10-100x faster
   (`function_reference/gpu_jax_fitting.md`).
7. Clearly report:
   - model components
   - parameter ties
   - assumptions
   - priors or bounds
   - measurements derived from the fit

## Key modeling guidance

- Use `Line_MultiGauss_doublet` for forbidden-line doublets such as
  `[S II]`, `[N II]`, and `[O III]`.
- Use `Line_template` for narrow-line template fitting, and
  `Line_template_doublet` for a doublet of templated lines (it is
  `Line_MultiGauss_doublet` with the Gaussian replaced by the measured profile:
  `amp_c0`/`amp_c1`/`dv_c`/`wavec0`/`wavec1`, and no width).
- Before freezing a narrow-line template, **check that the other narrow lines
  actually share its width**. Measure the width on at least one Balmer line
  locally and compare; on IRAS 07546+3928 the [O III]-anchored profile is 16%
  narrower than narrow Hα, and a frozen template then under-fits Hα/Hβ/[N II] and
  loses evidence against a Gaussian model with one free common width.
- Use `Line_MultiGauss` for broad permitted lines such as Hα and Hβ.
- Use `Line_Absorption` for absorption components.
- Apply LSF convolution when required by the instrument or data product.
- Set the error budget from the data, not from the pipeline: measure the
  line-free scatter, scale the formal errors, add a percentage floor, and check
  whether the residual scatter is *correlated* (telluric bands, sky
  subtraction) -- correlated residuals need a wavelength-dependent floor, not a
  larger constant one.
- If the residuals are dominated by residual telluric absorption, fix the
  errors, do not add physics: atmospheric features sit at fixed **observed**
  wavelengths, so convert before blaming the object.
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