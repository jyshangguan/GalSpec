# SAGAN Spectral Fitting Guide

## Overview

SAGAN (Spectral Analysis for Galaxies and AGN) is a comprehensive Python package for fitting astronomical spectra, particularly designed for AGN and galaxy spectroscopy. It provides Bayesian inference tools using MCMC (emcee) and nested sampling (dynesty), with support for complex line profiles, iron templates, stellar continua, and line spread function convolution.

### Key Features

- **Bayesian Inference**: Robust parameter estimation with uncertainty quantification using MCMC and nested sampling
- **Multi-component Modeling**: Decompose spectra into continuum, broad/narrow emission lines, and absorption components
- **Velocity-resolved Analysis**: Study kinematics through velocity shifts and dispersions
- **Instrument Convolution**: Apply LSF convolution with constant or wavelength-dependent resolution
- **Comprehensive Models**: Line profiles, continuum, iron templates, stellar populations

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [General Fitting Strategy](#general-fitting-strategy)
4. [Fitting Workflow](#fitting-workflow)
5. [Choosing Between LSQ, MCMC, and Dynesty](#choosing-between-lsq-mcmc-and-dynesty)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Related Documentation](#related-documentation)

## Installation

```bash
# Basic requirements
pip install numpy scipy matplotlib astropy

# For MCMC fitting
pip install emcee corner

# For nested sampling
pip install dynesty

# For parallel processing
pip install multiprocess

# For Milky Way extinction correction
pip install extinction sfdmap

# Optional: for resolution degradation
pip install spectres

# Install SAGAN
cd /path/to/SAGAN
pip install -e .
```

## Data Preparation

Proper data preparation is critical for reliable spectral fitting. This section covers the essential steps.

### 1. Milky Way Extinction Correction

Before fitting, correct for Galactic dust attenuation using SFD dust maps:

```python
from extinction import ccm89, remove
import sagan

# Option 1: Using ReadSpectrum class (recommended)
readspec = sagan.ReadSpectrum(ra, dec, z)
wave_obs, flux_obs, ferr_obs = load_spectrum()  # Your data loading

# Get MW-corrected spectrum in rest frame
wave, flux, ferr = readspec.unredden_res()

# Option 2: Manual correction with known A_V
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

coord = SkyCoord(ra=ra, dec=dec, unit='deg')
sfd = SFDQuery()
ebv = sfd(coord)
av = 2.742 * ebv  # R_V = 3.1 for Milky Way

# Apply Cardelli extinction law
flux_corrected = remove(ccm89(wave_obs, Av=av, Rv=3.1), flux_obs)
```

**Important**: Always verify the extinction correction by checking the continuum slope. Over-correction can introduce artificial features.

### 2. Redshift Determination

Accurate redshift is essential for rest-frame analysis. Use one of these approaches:

#### Interactive Approach (Recommended)
Ask the user to provide redshift from reliable sources:

```python
# Priority order for redshift sources:
# 1. User-provided table (most reliable)
# 2. Simbad query (by object name or coordinates)
# 3. SDSS spectroscopic redshift
# 4. Literature values

print("Please provide the redshift for the target.")
print("Options:")
print("1. User-provided table value")
print("2. Query Simbad (requires object name or coordinates)")
print("3. Use catalog/SDSS value")
```

#### Programmatic Approach

```python
# Option 1: From user table (e.g., target_info.ipac)
from astropy.table import Table
tb = Table.read('target_info.ipac', format='ipac')
z_source = tb['zred'][0]  # User-provided value

# Option 2: Query Simbad
from astroquery.simbad import Simbad
result = Simbad.query_object('J0925+6409')
z_source = result['Z_VALUE'][0]  # Spectroscopic redshift

# Option 3: Refine using narrow lines (if initial z available)
z_init = z_source  # Use provided value as starting point
wave_rest = wave_obs / (1 + z_init)
# Fit narrow lines to measure small dv corrections
# dv_measured = fit_narrow_lines(...)
# z_refined = (1 + z_init) * (1 + dv_measured / ls_km) - 1
```

**Recommendation**: Prefer user-provided redshift from table, use Simbad as backup, and only refine with narrow lines if needed (high-redshift objects where small dv corrections matter).

### 3. Spectral Quality Checks

Before fitting, verify your data quality:

```python
# Continuum S/N
snr = np.median(flux) / np.median(ferr)
print(f"Continuum S/N: {snr:.1f}")
# Should be > 10 per resolution element for reliable fitting

# Bad pixel identification
bad_pixels = (ferr <= 0) | ~np.isfinite(flux) | ~np.isfinite(ferr)
print(f"Bad pixels: {np.sum(bad_pixels)} / {len(flux)}")

# Wavelength calibration check
# Look for sky lines at known wavelengths:
# 5577.3 Å [O I], 6300.3 Å [O I], 6363.8 Å [O I]
from sagan.utils import line_wave_dict
sky_lines = [5577.3, 6300.3, 6363.8]
for line in sky_lines:
    # Check if line is at expected wavelength
    pass

# Flux calibration verification
# Compare absolute flux with standard stars or catalog values
```

### 4. Normalization Strategy

For emission line fitting, normalize to local continuum:

```python
from astropy.modeling import models, fitting

# Fit continuum with polynomial
cont = models.Polynomial1D(degree=2)
fitter = fitting.LevMarLSQFitter()

# Select continuum regions (avoid emission lines)
continuum_mask = (wave > 4800) & (wave < 4900) | (wave > 5100) & (wave < 5200)
continuum_fit = fitter(cont, wave[continuum_mask], flux[continuum_mask])

# Normalize by continuum
flux_norm = flux / continuum_fit(wave)
ferr_norm = ferr / continuum_fit(wave)
```

**Tip**: For complex continua, consider using a higher-order polynomial or spline fitting.

### 5. Masking Strategy

Define appropriate masks for your data:

```python
# Create weight array (1 = use, 0 = mask)
weight = np.ones_like(wave)

# Mask bad pixels
weight[bad_pixels] = 0

# Mask atmospheric absorption bands
weight[(wave > 6860) & (wave < 6960)] = 0  # B-band
weight[(wave > 7580) & (wave < 7700)] = 0  # A-band

# Mask specific bad regions
weight[(wave > 5398) & (wave < 5403)] = 0  # Example: detector defect
weight[(wave > 6097) & (wave < 6103)] = 0

# Define fitting windows
line_windows = [(4200, 5400), (6100, 7000)]  # Example regions
weight_line = np.zeros_like(wave)
for window in line_windows:
    weight_line[(wave >= window[0]) & (wave <= window[1])] = 1
```

## General Fitting Strategy

### Core Principles

1. **Start Simple, Build Complexity**
   - Begin with a basic model (continuum + emission lines)
   - Incrementally add components (absorption, iron template, stellar continuum)
   - Validate each addition before proceeding

2. **Chi-square Minimization First**
   - Use LSQ fitting for initial parameter estimates
   - Verify the model visually before MCMC
   - LSQ provides good starting points for Bayesian methods

3. **Physical Parameter Constraints**
   - Use physically motivated bounds for all parameters
   - Link related parameters (e.g., absorption dv across lines)
   - Fix well-constrained parameters when appropriate

4. **Wavelength Window Selection**
   - Choose windows around key features
   - Include sufficient continuum on both sides of lines
   - Avoid regions with strong skylines or cosmic rays

### Model Selection Guidelines

For different object types, use appropriate components:

| Object Type | Continuum | Narrow Lines | Broad Lines | Absorption | Iron Template | Stellar Continuum |
|-------------|-----------|--------------|-------------|------------|---------------|-------------------|
| Type 1 AGN | Power law | Line_template | Line_MultiGauss | Line_Absorption | IronTemplate | Optional |
| Type 2 AGN | Power law | Line_Gaussian | None | Optional | Optional | StarSpectrum |
| BAL QSO | Power law | Line_template | Line_MultiGauss | Line_Absorption | IronTemplate | Optional |
| Galaxy | BlackBody/Polynomial | Line_Gaussian | None | Optional | None | StarSpectrum |

## Fitting Workflow

### Step 1: Data Preparation
```
Load spectrum → MW extinction correction → Rest-frame conversion → Quality checks → Normalization
```

### Step 2: Initial Model Building
```python
import sagan
from sagan.utils import line_wave_dict
from astropy.modeling import models, fitting

# Define continuum
cont = sagan.WindowedPowerLaw1D(
    amplitude=1.0,
    x_0=5000,
    alpha=-1.0,
    x_min=4500,
    x_max=5500,
    name='continuum'
)

# Define emission line
line = sagan.Line_Gaussian(
    amplitude=1.0,
    dv=0,
    sigma=500,
    wavec=line_wave_dict['Halpha'],
    name='Halpha'
)

# Combine components
model = cont + line
```

### Step 3: LSQ Fitting
```python
# LSQ fit for initial parameters
fitter = fitting.LevMarLSQFitter()
model_lsq = fitter(model, wave, flux, weights=weight/ferr**2, maxiter=10000)

# Visual check
import matplotlib.pyplot as plt
plt.step(wave, flux, where='mid', label='Data')
plt.plot(wave, model_lsq(wave), label='Fit', color='red')
plt.legend()
plt.show()
```

### Step 4: Convolution (if needed)
```python
# Apply LSF convolution
resolving_power = 1800  # Instrument-specific
model_convolved = sagan.convolve_lsf(
    model_lsq,
    wavec=line_wave_dict['Halpha'],
    resolving_power=resolving_power
)
```

### Step 5: Bayesian Fitting

#### Option A: MCMC (emcee)
```python
# Initialize MCMC
mcmc = sagan.MCMC_Fit(
    model_convolved,
    wave,
    flux,
    ferr,
    nwalkers=100,    # ≥ 2*ndim
    nsteps=6000,
    nburn=5000
)

# Run fitting
samples, model_fit, param_names = mcmc.fit(progress=True)

# Check convergence
chain, tau = mcmc.check_convergence()
# Good convergence: nsteps / tau_max > 50

# Get best fit
model_best, par_names, theta_best = mcmc.get_best_fit()
```

#### Option B: Dynesty (Nested Sampling)
```python
# Define parameter bounds
bounds_dict = {
    'Halpha.amplitude': (0.1, 100),
    'Halpha.sigma': (100, 2000),
    'continuum.alpha': (-2, 0)
}

# Initialize Dynesty
dynesty_fit = sagan.Dynesty_Fit(
    model_convolved,
    wave,
    flux,
    ferr,
    bounds_dict=bounds_dict,
    sample_method='rwalk',
    nlive=500
)

# Run fitting
results = dynesty_fit.fit()

# Get best fit
model_best, par_names, theta_best = dynesty_fit.get_best_fit()

# Evidence for model comparison
print(f"Log evidence: {results.logz:.2f} ± {results.logzerr:.2f}")
```

### Step 6: Model Validation
```python
# Calculate residuals
residuals = flux - model_best(wave)
chi2 = np.sum((residuals / ferr)**2)
dof = len(wave) - len(theta_best)
reduced_chi2 = chi2 / dof

print(f"χ²/dof = {reduced_chi2:.2f}")

# Visual check
ax, axr = sagan.plot.plot_fit_new(wave, flux, model_best, error=ferr)
plt.show()
```

## Choosing Between LSQ, MCMC, and Dynesty

### LSQ (Levenberg-Marquardt)
- **When to use**: Initial parameter estimation, quick fits
- **Pros**: Fast, easy to use
- **Cons**: No uncertainty quantification, can get stuck in local minima
- **Typical use**: First step before MCMC/Dynesty

### MCMC (emcee)
- **When to use**: Full posterior exploration, parameter uncertainties
- **Pros**: Robust uncertainty estimates, handles correlated parameters
- **Cons**: Computationally expensive, requires convergence checking
- **Typical use**: Final analysis, publication-quality results

### Dynesty (Nested Sampling)
- **When to use**: Model comparison, evidence calculation, multi-modal posteriors
- **Pros**: Provides Bayesian evidence, efficient for high-dimensional problems
- **Cons**: Requires explicit bounds for all parameters
- **Typical use**: Model selection, complex parameter spaces

### Decision Tree
```
Need quick initial guess?
├─ Yes → LSQ
└─ No
    Need model comparison (evidence)?
    ├─ Yes → Dynesty
    └─ No
        Need full posterior with uncertainties?
        └─ Yes → MCMC
```

## Best Practices

### 1. Parameter Initialization
- Use literature values or visual inspection for initial guesses
- Start with simpler models before adding complexity
- Check that initial model resembles data qualitatively

### 2. Convergence Checking (MCMC)
```python
chain, tau = mcmc.check_convergence()
if chain.shape[0] / tau.max() < 50:
    print("Warning: Chain may not be fully converged")
    print("Consider increasing nsteps")

# Visual check
import matplotlib.pyplot as plt
plt.plot(chain[:, :, 0].T, color='k', alpha=0.1)
plt.xlabel('Step')
plt.ylabel('Parameter value')
plt.show()
```

### 3. Model Comparison
```python
# Calculate BIC
chi2 = np.sum(((flux - model_fit(wave)) / ferr)**2)
n_params = len(model_fit.free_parameters)
n_data = len(wave)
bic = chi2 + n_params * np.log(n_data)

# For nested sampling, compare evidence
print(f"Log Z: {results.logz:.2f}")
```

### 4. Handling Bounds
- **MCMC**: Can use None for unbounded (but not recommended)
- **Dynesty**: MUST specify bounds for all parameters
- Use physical constraints: σ > 0, 0 < Cf < 1, etc.

### 5. Multi-core Fitting
```python
# Use for complex models (many components)
samples, model_fit, par_names = mcmc.fit_ncores(ncores=8)
```

### 6. Saving Results
```python
# Save MCMC chain
np.save('mcmc_samples.npy', samples)
np.save('mcmc_par_names.npy', par_names)

# Save fitted model
import pickle
with open('fitted_model.pkl', 'wb') as f:
    pickle.dump(model_fit, f)

# Or use SAGAN's built-in save/load
mcmc.save_samples('samples.npz')
sagan.save_mcmc(mcmc, 'model.pkl')
```

## Troubleshooting

### Problem 1: Poor Convergence
**Symptoms**: Autocorrelation time not stabilizing, walkers stuck in separate modes

**Solutions**:
- Increase `nwalkers` (≥ 2 × number of parameters)
- Increase `nsteps` (run longer)
- Improve initial guesses
- Check parameter bounds are reasonable

### Problem 2: Absorption Trough Not Fitted
**Symptoms**: Absorption model at wrong velocity or depth

**Solutions**:
- Verify initial `dv` places trough near observed absorption
- Check `tau_0` and `Cf` bounds allow sufficient depth
- Consider multiple absorption components if complex trough

### Problem 3: Iron Template Misfit
**Symptoms**: Iron template doesn't match features

**Solutions**:
- Try both `park2022` and `boroson1992` templates
- Adjust `sigma` for proper width
- Consider using `IronTemplate_new` with free `dv`
- Ensure template covers wavelength range of interest

### Problem 4: Parameter Correlations
**Symptoms**: Strong degeneracies in corner plot

**Solutions**:
- Fix well-constrained parameters
- Use stronger priors (bounds)
- Tie related parameters across components
- Consider if model is overparameterized

### Problem 5: Convolution Issues
**Symptoms**: Model narrower/broader than data

**Solutions**:
- Verify `resolving_power` matches instrument
- Ensure convolution applied AFTER absorption
- Check wavelength range for convolution
- For variable R, use `convolve_lsf_var`

## Related Documentation

- **[Type 1 AGN Fitting Strategy](strategy_types/type1_agn.md)** - Specific guidance for fitting Type 1 AGN and BAL QSOs
- **[Function Reference](function_reference.md)** - Complete documentation of all SAGAN functions and classes

## Quick Start Template

```python
#!/usr/bin/env python
"""SAGAN spectral fitting template."""

import numpy as np
import matplotlib.pyplot as plt
import sagan
from sagan.utils import line_wave_dict
from astropy.modeling import models, fitting

# ===== 1. Load Data =====
wave, flux, ferr = load_your_spectrum()

# ===== 2. Data Preparation =====
# MW extinction correction (if needed)
# Rest-frame conversion
# Normalization

# ===== 3. Define Model =====
cont = sagan.WindowedPowerLaw1D(amplitude=1.0, x_0=5000, alpha=-1.0,
                                x_min=4500, x_max=5500)
line = sagan.Line_Gaussian(amplitude=1.0, dv=0, sigma=500,
                           wavec=line_wave_dict['Halpha'])
model = cont + line

# ===== 4. Fit (LSQ) =====
fitter = fitting.LevMarLSQFitter()
model_lsq = fitter(model, wave, flux, weights=1/ferr**2)

# ===== 5. Fit (MCMC) =====
mcmc = sagan.MCMC_Fit(model_lsq, wave, flux, ferr,
                      nwalkers=50, nsteps=1000)
samples, model_mcmc, par_names = mcmc.fit()

# ===== 6. Plot Results =====
ax, axr = sagan.plot.plot_fit_new(wave, flux, model_mcmc, error=ferr)
plt.show()

# ===== 7. Corner Plot =====
import corner
fig = corner.corner(samples, labels=par_names, show_titles=True)
plt.show()
```

---

**Version**: 1.0
**Last Updated**: 2025-03-20
**Author**: Based on SAGAN package by Shangguan et al.
