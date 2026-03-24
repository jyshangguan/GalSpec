# Global Continuum Fitting and Subtraction Guide

**Purpose**: Model and subtract the GLOBAL continuum from AGN spectra to isolate emission lines for accurate flux measurements and line profile analysis.

**When to use**: Stage 1 of AGN spectral analysis - before fitting emission lines. Must be completed before any emission line fitting.

**Scope**: This guide covers GLOBAL continuum fitting - modeling the continuum across the ENTIRE wavelength range simultaneously, using a single set of parameters for all continuum components.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Continuum Fitting Matters](#why-continuum-fitting-matters)
3. [Types of Continuum in AGN Spectra](#types-of-continuum-in-agn-spectra)
4. [Choosing Continuum Windows](#choosing-continuum-windows)
5. [Step-by-Step Workflow](#step-by-step-workflow)
6. [Building the Continuum Model](#building-the-continuum-model)
7. [Fitting the Continuum](#fitting-the-continuum)
8. [Visualization with plot_fit()](#visualization-with-plot_fit)
9. [Quality Checks](#quality-checks)
10. [Generating Continuum-Subtracted Spectrum](#generating-continuum-subtracted-spectrum)
11. [Saving Results](#saving-results)
12. [Troubleshooting](#troubleshooting)
13. [Complete Example](#complete-example)

---

## Overview

**Global continuum fitting** is the **critical first step** in AGN spectral analysis. This approach models the continuum across the ENTIRE wavelength range simultaneously, using a single set of parameters for all continuum components.

### What is "Global" Continuum Fitting?

**Global continuum fitting** means:
- Fitting ONE continuum model to ALL wavelength regions at once
- Using a SINGLE set of parameters (AGN power-law, stellar, iron) that apply globally
- Modeling the continuum across the full spectral range (e.g., 4000-8000 Å)
- The continuum is **NOT** allowed to vary independently in different wavelength regions

**Contrast with local continuum fitting**:
- Local: Fit separate continuum in each region (e.g., around Hα, Hβ)
- Global: Single continuum model for entire spectrum ✓

**Advantages of global fitting**:
- Physically motivated (continuum components are global properties)
- Fewer free parameters (more robust)
- Consistent continuum shape across all wavelengths
- Better constraints from combining all continuum windows

### Continuum Components

The global continuum consists of:
- **AGN power-law**: Accretion disk emission (single power-law across all wavelengths)
- **Host galaxy starlight**: Stellar continuum (single velocity dispersion, tied redshift)
- **Blended Fe II**: Iron template (single amplitude and width)

Accurate global continuum modeling and subtraction is essential for:
- Measuring emission line fluxes
- Analyzing line profiles (widths, asymmetries)
- Detecting weak emission lines
- Calculating equivalent widths

### Key Principle

> **Always fit and subtract the GLOBAL continuum BEFORE fitting emission lines.**
>
> The continuum-subtracted spectrum provides a clean baseline for line fitting, eliminating degeneracies between continuum shape and line fluxes.
>
> **Fit once, subtract everywhere** - the same continuum model applies to all wavelength regions.

---

## Why Continuum Fitting Matters

### Without Proper Continuum Subtraction

```python
# Wrong: Fitting lines directly to spectrum
line = Line_Gaussian(amplitude=50, dv=100, sigma=400, wavec=6562.819)
model = cont + line  # Continuum and line fit simultaneously
# Problem: Line amplitude and continuum shape are degenerate!
```

**Issues**:
- Line fluxes biased by continuum slope
- Uncertain continuum level under lines
- Poor constraints on weak lines
- Incorrect equivalent widths

### With Proper Continuum Subtraction

```python
# Correct: Subtract continuum first
flux_subtracted = flux - continuum_fit
line = Line_Gaussian(amplitude=?, dv=?, sigma=?, wavec=6562.819)
# Line amplitude directly measurable from continuum-subtracted spectrum
```

**Benefits**:
- Line fluxes measured relative to known baseline
- No continuum-line degeneracy
- Weak lines more detectable
- Accurate equivalent widths

---

## Types of Continuum in AGN Spectra

Type 1 AGN spectra have **three main continuum components**:

### 1. AGN Power-Law Continuum

**Source**: Thermal emission from accretion disk

**Model**: `WindowedPowerLaw1D`

```python
agn_cont = WindowedPowerLaw1D(
    amplitude=?,      # Flux at reference wavelength
    x_0=5500,         # Reference wavelength (Å)
    alpha=-1.5,       # Power law index (F_ν ∝ ν^α)
    x_min=4200,       # Blue limit
    x_max=8000,       # Red limit
    name='AGN_powerlaw'
)
```

**Typical values**:
- `alpha`: -1.0 to -2.0 (steeper for higher luminosity AGN)
- `amplitude`: 5-50 (1e-17 erg/cm²/s/Å, varies with object)

**Equation**: `F(λ) = amplitude × (λ/x_0)^(-alpha)`

**CRITICAL**: `x_min` and `x_max` should cover the **full wavelength range** of your spectrum to allow proper subtraction across all regions.

### 2. Stellar Continuum (Host Galaxy)

**Source**: Integrated starlight from host galaxy

**Model**: `Multi_StarSpectrum` (recommended) or `StarSpectrum`

#### Recommended: Multi_Component Stellar Model

For realistic host galaxy modeling, use **multiple stellar types** combined with **tied redshift**:

```python
# Option 1: All 5 stellar types (most flexible)
stellar = sagan.Multi_StarSpectrum(
    n_components=5,
    star_types=['A', 'F', 'G', 'K', 'M'],  # All available types
    amplitudes=[0.05, 0.1, 0.4, 0.3, 0.1],  # Initial relative contributions
    velscale=200,
    delta_z=0,          # Shared redshift for ALL components (km/s)
    sigma=150,          # Shared velocity dispersion (km/s)
    name='stellar'
)

# Tie delta_z (redshift) across all stellar components
# All components share the same systemic velocity
stellar.delta_z_0.fixed = True  # Fix to systemic redshift
stellar.delta_z_0.value = 0     # Set to zero (already in rest frame)
```

**Parameter tying for redshift**:
- All stellar components share the SAME `delta_z` parameter
- This ensures all stars have the same systemic velocity
- The redshift is tied to the AGN rest frame (delta_z = 0)

#### Alternative: Single Stellar Type

If the host galaxy has a dominant stellar population:

```python
# Option 2: Single stellar type (simpler)
stellar = sagan.StarSpectrum(
    amplitude=0.2,       # Relative contribution (0.1-0.5 typical)
    Star_type='G',       # Most common for AGN hosts
    velscale=200,        # Velocity scale (km/s)
    delta_z=0,           # Redshift offset (km/s)
    sigma=150,           # Velocity dispersion (km/s)
    name='stellar'
)
```

**Available stellar templates**:
- `A`: HD 97633 (A0V star) - Young population
- `F`: HD 89254 - Intermediate-age
- `G`: HD 140027 - **Most common for AGN hosts**
- `K`: HD 49520 - Older population
- `M`: HD 44478 - Old population

**Typical values**:
- `amplitudes` (Multi_StarSpectrum): Adjust based on host galaxy properties
  - G and K stars typically dominate (older bulge)
  - A and F stars contribute less (younger populations)
  - M stars contribute little (very old, faint)
- `Star_type` (StarSpectrum): 'G' or 'K' (older stellar populations)
- `sigma`: 100-300 km/s (bulge velocity dispersion)
- `delta_z`: 0 (tied to systemic redshift)

**When to use Multi_StarSpectrum vs StarSpectrum**:
- **Multi_StarSpectrum**: Host galaxies with mixed stellar populations (recommended)
- **StarSpectrum**: Host dominated by single population or low S/N

**When to omit stellar component**: For high-luminosity AGN where host contribution is negligible (<5%)

#### ⚠️ CRITICAL: Stellar Template Wavelength Coverage

**The stellar templates have limited wavelength coverage**:
- **Coverage range**: ~3500-9000 Å (exact range: 3466-9000 Å)
- **Issue**: SDSS spectra can extend below 3500 Å (down to ~3800 Å observed frame, ~3330 Å at z=0.14)

**Solution**: Filter your spectrum to the template range before fitting:

```python
# Filter data to stellar template wavelength range
wave_min_fit = 3500
wave_max_fit = 9000
fit_mask = (wave_rest >= wave_min_fit) & (wave_rest <= wave_max_fit)

wave_rest = wave_rest[fit_mask]
flux_rest = flux_rest[fit_mask]
ferr_rest = ferr_rest[fit_mask]

# Update continuum windows to match filtered data
cont_mask = np.zeros(len(wave_rest), dtype=bool)
for wmin, wmax in CONTINUUM_WINDOWS:
    if wmin >= wave_rest.min() and wmax <= wave_rest.max():
        cont_mask |= (wave_rest > wmin) & (wave_rest < wmax)
```

**Note**: The `Multi_StarSpectrum.evaluate()` method now uses `fill_value="extrapolate"` to handle wavelengths slightly outside the template range, but it's still recommended to filter to 3500-9000 Å for best results.

**Check your wavelength range**:
```python
print(f"Wavelength range: {wave_rest.min():.1f} - {wave_rest.max():.1f} A")
if wave_rest.min() < 3500:
    print("WARNING: Spectrum extends below 3500 A - filtering recommended")
if wave_rest.max() > 9000:
    print("WARNING: Spectrum extends above 9000 A - filtering recommended")
```

#### Parameter Tying for Redshift

**CRITICAL**: When using `Multi_StarSpectrum`, all stellar components must share the same redshift:

```python
# After creating the model, tie delta_z parameters
stellar.delta_z_0.fixed = True  # Fix first component's delta_z
stellar.delta_z_0.value = 0     # Set to zero (rest frame)

# All other components automatically share this value
# because Multi_StarSpectrum uses a single delta_z parameter
```

**Why tie redshift?**
- All stars in the host galaxy share the same systemic velocity
- The stellar absorption features must align with the AGN rest frame
- Prevents unphysical solutions where different stellar types have different velocities

### 3. Iron Template (Fe II Emission)

**Source**: Blended Fe II emission lines from broad line region

**Model**: `IronTemplate`

```python
iron = IronTemplate(
    amplitude=?,      # Relative strength (0.3-1.0 typical)
    stddev=800/2.3548, # Velocity dispersion (km/s)
    z=0,              # Redshift (already in rest frame)
    template_name='park2022',  # or 'boroson1992'
    name='iron'
)
```

**Available templates**:
- `park2022`: Based on Mrk 493 (Park et al. 2022) - **Recommended**
- `boroson1992`: Based on I Zw 1 (Boroson & Green 1992)

**Typical values**:
- `template_name`: 'park2022' (modern template)
- `stddev`: 800/2.3548 km/s (intrinsic width from template)
- `amplitude`: 0.3-1.0 (30-100% of AGN continuum)

**When to omit**: For weak Fe II emitters or low S/N spectra

---

## Global vs. Local Continuum Fitting

This guide covers **GLOBAL continuum fitting**. It's important to understand the distinction:

### Global Continuum Fitting (This Guide)

**Definition**: Fit a single continuum model to ALL wavelength regions simultaneously.

**Characteristics**:
- One AGN power-law with a single amplitude and alpha for the entire spectrum
- One stellar component with a single velocity dispersion for all wavelengths
- One iron template with a single amplitude and width
- All continuum windows fit together in a single model
- Parameters are GLOBAL - they don't vary with wavelength

**Advantages**:
- Physically motivated (continuum components are intrinsic properties)
- Fewer free parameters (more robust, less degenerate)
- Consistent continuum shape across all wavelengths
- Better S/N from combining all continuum windows
- Standard approach for AGN spectral analysis

**Example**:
```python
# One model fits ALL wavelengths at once
model_cont = agn_cont + stellar + iron  # Single parameters for entire spectrum
fitter(model_cont, wave_cont, flux_cont)  # All windows fit simultaneously
```

### Local Continuum Fitting (Different Approach)

**Definition**: Fit separate continuum in different wavelength regions.

**Characteristics**:
- Different continuum parameters for different wavelength ranges
- Typically used for specific line regions (e.g., around Hα only)
- Continuum may vary independently in each region

**When to use**:
- Very high S/N spectra where continuum shape changes with wavelength
- Complex continua that cannot be described by simple power-law
- Diagnostic fits to specific regions

**Example**:
```python
# Separate fits for Hα and Hβ regions
model_ha = fit_continuum_around(wave, flux, region=(6400, 6700))
model_hb = fit_continuum_around(wave, flux, region=(4800, 5100))
# Different parameters in each region
```

### Why This Guide Uses Global Fitting

**For AGN spectral analysis**, global continuum fitting is **strongly recommended** because:

1. **Physical**: AGN continuum from accretion disk follows a single power-law
2. **Stellar**: Host galaxy has a single velocity dispersion
3. **Robust**: Fewer parameters prevent overfitting
4. **Standard**: Used in most AGN studies (e.g., SDSS, BOSS)

**Only consider local fitting if**:
- You have very high S/N (>50 per pixel) and see evidence for continuum curvature
- Continuum windows show systematic residuals with global fit
- You're fitting a specific small region for diagnostic purposes

### Summary

| Aspect | Global Fitting | Local Fitting |
|--------|---------------|---------------|
| **Parameters** | Single set for all wavelengths | Different per region |
| **Physical** | ✓ (intrinsic properties) | ✗ (mathematical convenience) |
| **Robustness** | ✓ (fewer parameters) | ✗ (more parameters) |
| **Standard** | ✓ (widely used) | ✗ (rarely needed) |
| **This guide** | ✓ (primary method) | - (not covered) |

---

## Choosing Continuum Windows

Continuum windows are **line-free regions** used to constrain the continuum fit. These regions must:
- Exclude strong emission lines
- Cover the full wavelength range
- Have sufficient S/N (>20 preferred)

### Standard Continuum Windows for Type 1 AGN

**Optical region (4000-8000 Å)**:

| Window | Wavelength (Å) | Lines Excluded | Notes |
|--------|---------------|----------------|-------|
| Window 1 | 4200–4300 | Hδ (4101), Hγ (4345) | Blue continuum |
| Window 2 | 4430–4560 | Hβ (4861), [O III] (4959/5007) | |
| Window 3 | 5060–5400 | Hβ (4861), [O III] (4959/5007) | Between Hβ and [O I] |
| Window 4 | 5600–5700 | [O I] (6300), Hα (6563) | |
| Window 5 | 6180–6230 | Hα (6563), [N II] (6548/6583) | Before Hα complex |
| Window 6 | 6800–7000 | [S II] (6716/6731) | After [S II] doublet |
| Window 7 | 7500–8000 | Hα (6563), [N II], [S II] | Far red continuum |

**UV region (if available)**:
- 3200-3400 Å: Mg II (2800) region gap
- 3650-3750 Å: Avoid Balmer jump (3646 Å)

**NIR region (if available)**:
- 8400-8800 Å: Ca II triplet (8498/8542/8662) gaps
- 9000-9500 Å: Beyond Paschen lines

### CRITICAL: Check Each Window

Before fitting, **verify each window** for your spectrum:

```python
windows = [(4200, 4300), (4430, 4560), (5060, 5400),
           (5600, 5700), (6180, 6230), (6800, 7000), (7500, 8000)]

for wmin, wmax in windows:
    # Check if window is within wavelength range
    if wmin < wave.min() or wmax > wave.max():
        print(f"Window {wmin}-{wmax} OUTSIDE range - SKIP")
        continue

    # Check for contamination
    mask = (wave > wmin) & (wave < wmax)
    median_flux = np.median(flux[mask])
    std_flux = np.std(flux[mask])

    # Flag if high variance (possible line contamination)
    if std_flux / median_flux > 0.3:
        print(f"WARNING: Window {wmin}-{wmax} may be contaminated!")
```

**Modify windows if needed**:
- Narrow windows if they contain weak line wings
- Skip windows entirely if they fall outside your wavelength range
- Add custom windows for your specific spectrum

---

## Step-by-Step Workflow

### Step 1: Prepare Spectrum

```python
from sagan.utils import ReadSpectrum
from astropy.io import fits

hdu = fits.open('spectrum.fits')
spec = ReadSpectrum(is_sdss=True, hdu=hdu)
wave_rest, flux_rest, ferr_rest = spec.unredden_res()
```

**Requirements**:
- Spectrum must be **MW extinction corrected** (`unredden_res()` does this)
- Spectrum must be in **rest frame** (divided by 1+z)
- Flux errors must be available

### Step 2: Define Continuum Windows

```python
# Define windows (rest frame, Angstroms)
CONTINUUM_WINDOWS = [
    (4200, 4300),  # Window 1
    (4430, 4560),  # Window 2
    (5060, 5400),  # Window 3
    (5600, 5700),  # Window 4
    (6180, 6230),  # Window 5
    (6800, 7000),  # Window 6
    (7500, 8000),  # Window 7
]

# Create mask for continuum regions
cont_mask = np.zeros(len(wave_rest), dtype=bool)
for wmin, wmax in CONTINUUM_WINDOWS:
    # Only include if within wavelength range
    if wmin >= wave_rest.min() and wmax <= wave_rest.max():
        cont_mask |= (wave_rest > wmin) & (wave_rest < wmax)

# Extract continuum data
wave_cont = wave_rest[cont_mask]
flux_cont = flux_rest[cont_mask]
ferr_cont = ferr_rest[cont_mask]
```

### Step 3: Build Continuum Model

See [Building the Continuum Model](#building-the-continuum-model) below.

### Step 4: Fit Continuum Model

See [Fitting the Continuum](#fitting-the-continuum) below.

### Step 5: Visualize Fit

See [Visualization with plot_fit()](#visualization-with-plot_fit) below.

### Step 6: Quality Checks

See [Quality Checks](#quality-checks) below.

### Step 7: Generate Continuum-Subtracted Spectrum

See [Generating Continuum-Subtracted Spectrum](#generating-continuum-subtracted-spectrum) below.

### Step 8: Save Results

See [Saving Results](#saving-results) below.

---

## Building the Continuum Model

### Full Model (All Three Components with Multi-Star Stellar)

```python
import sagan
from sagan.continuum import WindowedPowerLaw1D

# Estimate initial amplitude from data
median_flux = np.median(flux_cont)

# Determine wavelength range (USE FULL RANGE)
wave_min = int(np.floor(wave_rest.min()))
wave_max = int(np.ceil(wave_rest.max()))

# 1. AGN Power-law continuum
agn_cont = WindowedPowerLaw1D(
    amplitude=median_flux,
    x_0=5500,
    alpha=-1.5,
    x_min=wave_min,      # CRITICAL: Use full spectrum range
    x_max=wave_max,      # CRITICAL: Use full spectrum range
    name='AGN_powerlaw'
)

# 2. Stellar emission (MULTI-COMPONENT - RECOMMENDED)
# Use all 5 stellar types with tied redshift
stellar = sagan.Multi_StarSpectrum(
    n_components=5,
    star_types=['A', 'F', 'G', 'K', 'M'],
    amplitudes=[0.05, 0.1, 0.4, 0.3, 0.1],  # Relative contributions
    velscale=200,
    delta_z=0,          # Shared redshift (all components)
    sigma=150,          # Shared velocity dispersion
    name='stellar'
)

# Fix redshift to systemic value (critical for multi-component fit)
stellar.delta_z_0.fixed = True  # All stars share systemic redshift
stellar.delta_z_0.value = 0     # Already in rest frame

# 3. Iron template
iron = sagan.IronTemplate(
    amplitude=0.5,
    stddev=800/2.3548,
    z=0,
    template_name='park2022',  # Recommended
    name='iron'
)

# Combine all components
model_cont = agn_cont + stellar + iron
```

### Alternative: Single Stellar Type

```python
# For simpler modeling or low S/N spectra
stellar = sagan.StarSpectrum(
    amplitude=0.2,
    Star_type='G',       # Most common
    velscale=200,
    delta_z=0,
    sigma=150,
    name='stellar'
)

model_cont = agn_cont + stellar + iron
```

### Simplified Models

#### For High-Luminosity AGN (Host Negligible)

```python
# Only AGN power-law + iron (omit stellar)
model_cont = agn_cont + iron
```

#### For Weak Fe II Emitters

```python
# Only AGN power-law + stellar (omit iron)
model_cont = agn_cont + stellar
```

#### For Pure AGN (Bright, No Host, No Iron)

```python
# Only AGN power-law
model_cont = agn_cont
```

### Initial Parameter Guidelines

| Parameter | Initial Value | How to Estimate |
|-----------|---------------|-----------------|
| `AGN amplitude` | median(flux_cont) | Median flux in continuum windows |
| `AGN alpha` | -1.5 | Typical for Type 1 AGN |
| `Stellar amplitudes` | [0.05, 0.1, 0.4, 0.3, 0.1] | A, F, G, K, M contributions |
| `Stellar Star_type` | ['A', 'F', 'G', 'K', 'M'] | Use all for full model |
| `Stellar sigma` | 150 km/s | Typical bulge dispersion |
| `Stellar delta_z` | 0 (fixed) | Tied to systemic redshift |
| `Iron amplitude` | 0.5 | 50% of AGN continuum |
| `Iron stddev` | 800/2.3548 | Intrinsic width from template |

**CRITICAL: Parameter Tying for Multi-StarSpectrum**
```python
# After creating the model, ensure redshift is tied
stellar.delta_z_0.fixed = True  # Fix to systemic value
stellar.delta_z_0.value = 0     # Set to zero (rest frame)

# This ensures ALL stellar types share the same velocity
# Prevents unphysical solutions with different stellar velocities
```

### ⚠️ CRITICAL: Compound Model Parameter Access

When astropy combines models with the `+` operator, it adds **numerical suffixes** to parameter names. This is a common source of errors!

#### Parameter Naming Rule

**Format**: `{parameter_name}_{model_index}` where `model_index` starts at 0

**Example for `model_cont = agn_cont + multi_stellar + iron`:**

| Model | Index | Parameters |
|-------|-------|------------|
| AGN Power-law | 0 | `amplitude_0`, `alpha_0`, `x_0_0`, `x_min_0`, `x_max_0` |
| Multi_StarSpectrum | 1 | `amp_0_1`, `amp_1_1`, `amp_2_1`, `amp_3_1`, `amp_4_1`, `sigma_1` |
| IronTemplate | 2 | `amplitude_2`, `stddev_2`, `z_2` |

#### Accessing Fitted Parameters

```python
# After fitting: model_cont_fit = fitter(model_cont, wave_cont, flux_cont)

# AGN Power-law (first model, index 0)
print(f"AGN amplitude: {model_cont_fit.amplitude_0.value}")
print(f"AGN alpha: {model_cont_fit.alpha_0.value}")

# Multi_StarSpectrum (second model, index 1)
print(f"A-type amplitude: {model_cont_fit.amp_0_1.value}")  # NOT amp_0!
print(f"F-type amplitude: {model_cont_fit.amp_1_1.value}")
print(f"G-type amplitude: {model_cont_fit.amp_2_1.value}")
print(f"K-type amplitude: {model_cont_fit.amp_3_1.value}")
print(f"M-type amplitude: {model_cont_fit.amp_4_1.value}")
print(f"Velocity dispersion: {model_cont_fit.sigma_1.value}")

# IronTemplate (third model, index 2)
print(f"Iron amplitude: {model_cont_fit.amplitude_2.value}")
```

#### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `AttributeError: "amp_0" not found` | Used `model.amp_0` instead of `model.amp_0_1` | Add suffix `_1` for second model |
| `AttributeError: "sigma" not found` | Forgot suffix entirely | Use `sigma_1` for second model |
| `AttributeError: "amplitude" not found` | Ambiguous - multiple models have `amplitude` | Use `amplitude_0`, `amplitude_1`, `amplitude_2` |

#### Print All Parameters (Debugging)

If you're unsure about parameter names:

```python
# Print all parameter names
print("All parameters:")
for name in model_cont_fit.param_names:
    print(f"  {name}")

# Or use model representation
print(model_cont_fit)
```

**Output example**:
```
All parameters:
  amplitude_0
  x_0_0
  alpha_0
  x_min_0
  x_max_0
  amp_0_1
  amp_1_1
  amp_2_1
  amp_3_1
  amp_4_1
  sigma_1
  amplitude_2
  stddev_2
  z_2
```

#### Accessing Sub-Models

You can also access individual sub-models by index:

```python
# Access sub-models directly
agn_model = model_cont_fit[0]  # AGN power-law
stellar_model = model_cont_fit[1]  # Multi_StarSpectrum
iron_model = model_cont_fit[2]  # IronTemplate

# Now parameters don't have suffixes
print(f"AGN alpha: {agn_model.alpha.value}")
print(f"Stellar sigma: {stellar_model.sigma.value}")
print(f"Iron amplitude: {iron_model.amplitude.value}")

# Evaluate individual components at a wavelength
flux_agn = agn_model(5500)
flux_stellar = stellar_model(5500)
flux_iron = iron_model(5500)
```

---

## Fitting the Continuum

### Basic Fit

```python
from astropy.modeling import fitting

fitter = fitting.LevMarLSQFitter()
model_cont_fit = fitter(model_cont, wave_cont, flux_cont,
                        weights=1/ferr_cont**2, maxiter=10000)
```

### Assess Fit Quality

```python
# Calculate chi2
chi2 = np.sum(((flux_cont - model_cont_fit(wave_cont)) / ferr_cont)**2)
dof = len(wave_cont) - len(model_cont_fit.parameters)
chi2_dof = chi2 / dof

print(f"Fit statistics:")
print(f"  chi2 = {chi2:.1f}")
print(f"  DOF = {dof}")
print(f"  chi2/DOF = {chi2_dof:.3f}")
```

**Quality criteria**:
- χ²/DOF < 2.0: Good fit
- χ²/DOF 2.0-3.0: Acceptable
- χ²/DOF > 3.0: Poor fit - check model or windows

### Print Fitted Parameters

**IMPORTANT**: When using compound models, parameter names have numerical suffixes (see [Compound Model Parameter Access](#-critical-compound-model-parameter-access)).

```python
print("\nFitted parameters:")
print(f"  AGN Power-law:")
print(f"    amplitude = {model_cont_fit.amplitude_0.value:.6f}")
print(f"    alpha = {model_cont_fit.alpha_0.value:.6f}")

# Check if stellar component exists and determine type
if len(model_cont_fit) > 1:
    print(f"  Stellar:")
    # Check if it's Multi_StarSpectrum (has amp_0_1, amp_1_1, etc.)
    if 'amp_0_1' in model_cont_fit.param_names:
        # Multi_StarSpectrum
        print(f"    amp_0 (A-type) = {model_cont_fit.amp_0_1.value:.6f}")
        print(f"    amp_1 (F-type) = {model_cont_fit.amp_1_1.value:.6f}")
        print(f"    amp_2 (G-type) = {model_cont_fit.amp_2_1.value:.6f}")
        print(f"    amp_3 (K-type) = {model_cont_fit.amp_3_1.value:.6f}")
        print(f"    amp_4 (M-type) = {model_cont_fit.amp_4_1.value:.6f}")
        print(f"    sigma = {model_cont_fit.sigma_1.value:.2f} km/s")
    else:
        # Single StarSpectrum
        print(f"    amplitude = {model_cont_fit.amplitude_1.value:.6f}")
        print(f"    sigma = {model_cont_fit.sigma_1.value:.2f} km/s")

if len(model_cont_fit) > 2:
    print(f"  Iron:")
    print(f"    amplitude = {model_cont_fit.amplitude_2.value:.6f}")
```

**Alternative: Use sub-model access** (cleaner, no suffixes needed):

```python
print("\nFitted parameters (using sub-model access):")
print(f"  AGN Power-law:")
print(f"    amplitude = {model_cont_fit[0].amplitude.value:.6f}")
print(f"    alpha = {model_cont_fit[0].alpha.value:.6f}")

if len(model_cont_fit) > 1:
    print(f"  Stellar:")
    stellar_model = model_cont_fit[1]
    if hasattr(stellar_model, 'amp_0'):  # Multi_StarSpectrum
        print(f"    amp_0 (A-type) = {stellar_model.amp_0.value:.6f}")
        print(f"    amp_1 (F-type) = {stellar_model.amp_1.value:.6f}")
        print(f"    amp_2 (G-type) = {stellar_model.amp_2.value:.6f}")
        print(f"    amp_3 (K-type) = {stellar_model.amp_3.value:.6f}")
        print(f"    amp_4 (M-type) = {stellar_model.amp_4.value:.6f}")
        print(f"    sigma = {stellar_model.sigma.value:.2f} km/s")
    else:  # Single StarSpectrum
        print(f"    amplitude = {stellar_model.amplitude.value:.6f}")
        print(f"    sigma = {stellar_model.sigma.value:.2f} km/s")

if len(model_cont_fit) > 2:
    print(f"  Iron:")
    print(f"    amplitude = {model_cont_fit[2].amplitude.value:.6f}")
```

### Check Component Contributions

```python
# Evaluate components at reference wavelength (5500 A)
flux_agn = model_cont_fit[0](5500)
flux_stellar = model_cont_fit[1](5500) if len(model_cont_fit) > 1 else 0
flux_iron = model_cont_fit[2](5500) if len(model_cont_fit) > 2 else 0
flux_total = flux_agn + flux_stellar + flux_iron

print(f"\nComponent contributions at 5500 A:")
print(f"  AGN power-law:   {flux_agn/flux_total*100:.1f}%")
print(f"  Stellar:         {flux_stellar/flux_total*100:.1f}%")
print(f"  Iron template:   {flux_iron/flux_total*100:.1f}%")
```

---

## Visualization with plot_fit()

### CRITICAL: Use SAGAN's plot_fit() Function

**Always use `sagan.plot.plot_fit_new()` for continuum visualization** - this is the standard SAGAN plotting function.

```python
from sagan import plot as sagan_plot

# Create weight array: 1 for continuum windows, 0 elsewhere
weight = np.zeros_like(wave_rest, dtype=float)
weight[cont_mask] = 1.0

# Plot using SAGAN's plot_fit_new()
ax, axr = sagan_plot.plot_fit_new(
    wave_rest,
    flux_rest,
    model_cont_fit,
    weight=weight,          # Highlights continuum windows
    error=ferr_rest,        # Show error bars
    xlabel='Wavelength (Å, rest frame)',
    ylabel='Flux (1e-17 erg/cm²/s/Å)',
    legend_kwargs={'fontsize': 12, 'ncol': 2}
)

# Add title
ax.set_title(f'{TARGET_NAME} - Type 1 AGN Continuum Fit', fontsize=14)

plt.savefig('continuum_fit_diagnostic.png', dpi=150)
plt.close()
```

### What plot_fit_new() Shows

**Top panel**:
- Full spectrum (black step line)
- Continuum windows highlighted (gray weight line)
- Total continuum fit (red line)
- Individual components (colored lines):
  - AGN power-law
  - Stellar component
  - Iron template

**Bottom panel**:
- Residuals (data - model)
- Error bars
- Zero line (dashed)

### Interpreting the Plot

**Good fit characteristics**:
- Continuum fit (red) tracks data in continuum windows
- Residuals scatter around zero with no systematic trends
- Individual components show reasonable shapes

**Bad fit indicators**:
- Continuum fit systematically above/below data in windows
- Residuals show trends (sloped, curved)
- Individual components have unreasonable shapes

---

## Quality Checks

### 1. Numerical Checks

```python
# Check chi2/DOF
assert chi2_dof < 3.0, f"chi2/DOF too large: {chi2_dof:.2f}"

# Check component contributions are reasonable
assert 0.1 < flux_agn/flux_total < 1.0, "AGN contribution unreasonable"
assert 0.0 < flux_stellar/flux_total < 0.8, "Stellar contribution unreasonable"
assert 0.0 < flux_iron/flux_total < 1.5, "Iron contribution unreasonable"

# Check AGN power-law index
assert -3.0 < model_cont_fit.alpha_0.value < 0.5, f"AGN alpha unreasonable: {model_cont_fit.alpha_0.value:.2f}"

print("✓ All numerical checks passed")
```

### 2. Visual Checks

Examine the `plot_fit_new()` output:

- [ ] Continuum fit tracks data in continuum windows
- [ ] No systematic trends in residuals
- [ ] Individual components look reasonable
- [ ] No emission lines contaminating continuum windows

### 3. Continuum-Subtracted Spectrum Checks

After subtraction (see below), verify:

```python
# Check continuum windows are flat
for wmin, wmax in CONTINUUM_WINDOWS:
    if wmin >= wave_rest.min() and wmax <= wave_rest.max():
        mask = (wave_rest > wmin) & (wave_rest < wmax)
        median_sub = np.median(flux_subtracted[mask])
        std_sub = np.std(flux_subtracted[mask])

        # Should be near zero
        assert abs(median_sub) < 1.0, f"Window {wmin}-{wmax}: median = {median_sub:.2f}"

        # Should have small scatter
        assert std_sub < 2.0, f"Window {wmin}-{wmax}: std = {std_sub:.2f}"

print("✓ Continuum-subtracted spectrum checks passed")
```

---

## Generating Continuum-Subtracted Spectrum

### Subtract Continuum

```python
# Evaluate continuum model at ALL wavelengths (not just windows)
continuum_all = model_cont_fit(wave_rest)

# Subtract continuum
flux_subtracted = flux_rest - continuum_all
```

### Verify Continuum-Subtracted Spectrum

```python
print("\nContinuum-subtracted spectrum in continuum windows:")
for i, (wmin, wmax) in enumerate(CONTINUUM_WINDOWS, 1):
    if wmin >= wave_rest.min() and wmax <= wave_rest.max():
        mask = (wave_rest > wmin) & (wave_rest < wmax)
        median_sub = np.median(flux_subtracted[mask])
        std_sub = np.std(flux_subtracted[mask])
        print(f"  Window {i} ({wmin}-{wmax} A): median = {median_sub:+.3f}, std = {std_sub:.3f}")
```

**Expected results**:
- Median near 0 (±0.5) in all continuum windows
- Standard deviation ~1-2 (measurement noise)
- No systematic offsets between windows

### Plot Continuum-Subtracted Spectrum

```python
fig, ax = plt.subplots(figsize=(12, 5))

# Plot continuum-subtracted spectrum
ax.plot(wave_rest, flux_subtracted, 'k-', linewidth=0.5, alpha=0.7)

# Highlight continuum windows
for wmin, wmax in CONTINUUM_WINDOWS:
    if wmin >= wave_rest.min() and wmax <= wave_rest.max():
        ax.axvspan(wmin, wmax, color='blue', alpha=0.1)

# Zero line
ax.axhline(0, color='r', linestyle='--', alpha=0.5)

ax.set_xlabel('Wavelength (Å, rest frame)')
ax.set_ylabel('Flux (1e-17 erg/cm²/s/Å)')
ax.set_title(f'{TARGET_NAME} - Continuum-Subtracted Spectrum')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('continuum_subtracted_spectrum.png', dpi=150)
plt.close()
```

**What to look for**:
- Continuum windows near zero (blue shaded regions)
- Emission lines clearly visible (positive flux)
- Absorption lines visible (negative flux)
- No remaining continuum slope

---

## Saving Results

### 1. Save Continuum Parameters (ASCII)

```python
params_file = 'continuum_fit_params.txt'
with open(params_file, 'w') as f:
    f.write("# Type 1 AGN Continuum Fit Parameters\n")
    f.write(f"# Target: {TARGET_NAME}\n")
    f.write(f"# Redshift: {spec.z:.5f}\n\n")

    f.write("[AGN_powerlaw]\n")
    f.write(f"amplitude = {model_cont_fit.amplitude_0.value:.6f}\n")
    f.write(f"alpha = {model_cont_fit.alpha_0.value:.6f}\n")
    f.write(f"x_0 = 5500.0\n")
    f.write(f"x_min = {wave_min}\n")
    f.write(f"x_max = {wave_max}\n\n")

    if 'stellar' in model_cont_fit.submodel_names:
        f.write("[Stellar]\n")
        f.write(f"amplitude = {model_cont_fit.amplitude_1.value:.6f}\n")
        f.write(f"Star_type = G\n")
        f.write(f"sigma = {model_cont_fit.sigma_1.value:.2f}\n\n")

    if 'iron' in model_cont_fit.submodel_names:
        f.write("[Iron]\n")
        f.write(f"amplitude = {model_cont_fit.amplitude_2.value:.6f}\n")
        f.write(f"template = park2022\n")
        f.write(f"stddev = 800/2.3548\n\n")

    f.write("[Fit_statistics]\n")
    f.write(f"chi2 = {chi2:.2f}\n")
    f.write(f"DOF = {dof}\n")
    f.write(f"chi2_per_DOF = {chi2_dof:.3f}\n")
```

### 2. Save Continuum-Subtracted Spectrum (FITS)

```python
from astropy.table import Table

tbl = Table([
    wave_rest,
    flux_subtracted,
    ferr_rest,
    continuum_all
], names=['WAVELENGTH', 'FLUX_SUB', 'ERROR', 'CONTINUUM'])

tbl.meta['TARGET'] = TARGET_NAME
tbl.meta['REDSHIFT'] = float(spec.z)
tbl.meta['CONT_MODEL'] = 'AGN_powerlaw + stellar + iron'

tbl.write('continuum_subtracted_spectrum.fits', format='fits', overwrite=True)
```

### 3. Save Continuum-Subtracted Spectrum (ASCII)

```python
np.savetxt('continuum_subtracted_spectrum.txt',
           np.column_stack([wave_rest, flux_subtracted, ferr_rest, continuum_all]),
           header='wavelength(A) flux_subtracted error continuum',
           fmt='%.6f')
```

---

## Troubleshooting

### Problem: chi2/DOF > 3.0

**Possible causes**:
1. Poor continuum windows (line contamination)
2. Wrong model components
3. Incorrect initial parameters

**Solutions**:
```python
# 1. Check continuum windows
# Plot spectrum and inspect each window
fig, ax = plt.subplots()
ax.plot(wave_rest, flux_rest, 'k-')
for wmin, wmax in CONTINUUM_WINDOWS:
    ax.axvspan(wmin, wmax, color='blue', alpha=0.1)
plt.show()

# 2. Try different stellar types
for star_type in ['G', 'K', 'F']:
    stellar = StarSpectrum(amplitude=0.2, Star_type=star_type, ...)
    # Re-fit and compare chi2

# 3. Remove problematic component
# If iron amplitude → 0, try fitting without it
model_cont = agn_cont + stellar  # Omit iron
```

### Problem: Stellar amplitude → 0

**Cause**: Host galaxy contribution is negligible

**Solution**: Omit stellar component
```python
model_cont = agn_cont + iron  # No stellar
```

### Problem: Iron amplitude → 0

**Cause**: Weak Fe II emission or degeneracy with other components

**Solution**: Omit iron component
```python
model_cont = agn_cont + stellar  # No iron
```

### Problem: AGN alpha > 0 (positive slope)

**Cause**: Improper initial values or degenerate with stellar component

**Solutions**:
```python
# 1. Constrain alpha to negative values
agn_cont.alpha.bounds = (-3.0, 0.0)

# 2. Fix stellar contribution if well-constrained
stellar.amplitude.fixed = True

# 3. Try omitting stellar component
model_cont = agn_cont + iron
```

### Problem: ValueError - "A value in x_new is below/above interpolation range"

**Cause**: Spectrum wavelength extends beyond stellar template coverage (~3500-9000 Å)

**Error message**:
```
ValueError: A value (3331.73) in x_new is below the interpolation range's minimum value (3466.0).
```

**Solutions**:
```python
# Filter spectrum to stellar template range BEFORE fitting
wave_min_fit = 3500
wave_max_fit = 9000
fit_mask = (wave_rest >= wave_min_fit) & (wave_rest <= wave_max_fit)

wave_rest = wave_rest[fit_mask]
flux_rest = flux_rest[fit_mask]
ferr_rest = ferr_rest[fit_mask]

# Also update AGN power-law x_min to match
agn_cont = WindowedPowerLaw1D(
    amplitude=median_flux,
    x_0=5500,
    alpha=-1.5,
    x_min=3500,  # Match stellar template lower limit
    x_max=wave_max,
    name='AGN_powerlaw'
)
```

**Note**: As of SAGAN v1.0+, `Multi_StarSpectrum.evaluate()` uses `fill_value="extrapolate"` to handle wavelengths slightly outside the template range, but filtering is still recommended for best results.

### Problem: AttributeError - Parameter name not found

**Cause**: Incorrect parameter name when accessing compound model parameters

**Error messages**:
```
AttributeError: 'amp_0' not found
AttributeError: "amp_0" not found. Did you mean: 'amp_0_1'?
AttributeError: 'sigma' not found
```

**Solution**: Use numerical suffixes for compound model parameters (see [Compound Model Parameter Access](#-critical-compound-model-parameter-access))

```python
# WRONG - forgot suffix
print(model_cont_fit.amp_0.value)  # Error!

# WRONG - used wrong suffix
print(model_cont_fit.amp_0_0.value)  # Error! This would be for first model

# CORRECT - use suffix for model index
print(model_cont_fit.amp_0_1.value)  # Multi_StarSpectrum is second model (index 1)

# ALTERNATIVE - use sub-model access (no suffix needed)
print(model_cont_fit[1].amp_0.value)  # Access second sub-model directly
```

**Quick reference for `model = agn + stellar + iron`:**
| Parameter | Correct Access | Model Index |
|-----------|---------------|-------------|
| AGN amplitude | `amplitude_0` | 0 |
| AGN alpha | `alpha_0` | 0 |
| Stellar amp_0 | `amp_0_1` | 1 |
| Stellar sigma | `sigma_1` | 1 |
| Iron amplitude | `amplitude_2` | 2 |

### Problem: Stellar sigma at lower bound (20 km/s)

**Cause**:
1. Initial sigma too low for host galaxy
2. Spectral resolution doesn't resolve stellar dispersion
3. Data quality insufficient to constrain dispersion

**Solutions**:
```python
# 1. Increase initial sigma and lower bound
stellar = Multi_StarSpectrum(
    amp_0=0.05, amp_1=0.05, amp_2=0.05, amp_3=0.05, amp_4=0.05,
    sigma=150,           # Start with realistic value
    sigma_bounds=(50, 600),  # Raise lower bound from 20
    velscale=200,
    Star_types=['A', 'F', 'G', 'K', 'M']
)

# 2. If still at lower bound, fix it
if stellar.sigma.value < 25:
    stellar.sigma.fixed = True
    stellar.sigma.value = 150  # Set to typical bulge value

# 3. If stellar contribution is small (<10%), omit stellar component
# Check contribution
flux_stellar = model_cont_fit[1](5500)
flux_total = model_cont_fit[0](5500) + flux_stellar + model_cont_fit[2](5500)
if flux_stellar / flux_total < 0.1:
    print("Stellar contribution < 10% - consider omitting stellar component")
    model_cont = agn_cont + iron  # Refit without stellar
```

### Problem: Residuals show systematic trends

**Cause**: Missing continuum component or wrong continuum windows

**Solutions**:
```python
# 1. Check for line contamination in windows
# Plot residuals vs wavelength
residuals = flux_cont - model_cont_fit(wave_cont)
fig, ax = plt.subplots()
ax.scatter(wave_cont, residuals, s=5)
ax.axhline(0, color='r', linestyle='--')
plt.show()

# 2. Add Balmer pseudo-continuum if fitting blue region
from sagan.continuum import BalmerPseudoContinuum
balmer = BalmerPseudoContinuum(i_ref=1.0, sigma=1000, dv=0)
model_cont = agn_cont + stellar + iron + balmer
```

---

## Complete Example

```python
#!/usr/bin/env python
"""
Fit Type 1 AGN GLOBAL continuum and subtract
Target: SDSS-J000111.15-100155.5

This script demonstrates GLOBAL continuum fitting:
- Single continuum model for entire wavelength range
- Multi-component stellar model with tied redshift
- All continuum windows fit simultaneously
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
from astropy.io import fits
from astropy.table import Table
import sys

sys.path.insert(0, '/path/to/SAGAN')
import sagan
from sagan.utils import ReadSpectrum
from sagan.continuum import WindowedPowerLaw1D
from sagan import plot as sagan_plot

# ========================================
# Configuration
# ========================================
TARGET_NAME = 'SDSS-J000111.15-100155.5'
FITS_FILE = 'data/SDSS-J000111.15-100155.5.fits'

# Continuum windows (rest frame)
CONTINUUM_WINDOWS = [
    (4200, 4300), (4430, 4560), (5060, 5400),
    (5600, 5700), (6180, 6230), (6800, 7000), (7500, 8000)
]

# ========================================
# 1. Load Spectrum
# ========================================
hdu = fits.open(FITS_FILE)
spec = ReadSpectrum(is_sdss=True, hdu=hdu)
wave_rest, flux_rest, ferr_rest = spec.unredden_res()

# ========================================
# 2. Define Continuum Windows
# ========================================
cont_mask = np.zeros(len(wave_rest), dtype=bool)
for wmin, wmax in CONTINUUM_WINDOWS:
    if wmin >= wave_rest.min() and wmax <= wave_rest.max():
        cont_mask |= (wave_rest > wmin) & (wave_rest < wmax)

wave_cont = wave_rest[cont_mask]
flux_cont = flux_rest[cont_mask]
ferr_cont = ferr_rest[cont_mask]

# ========================================
# 3. Build GLOBAL Continuum Model
# ========================================
wave_min = int(np.floor(wave_rest.min()))
wave_max = int(np.ceil(wave_rest.max()))

# AGN Power-law (GLOBAL - applies to all wavelengths)
agn_cont = WindowedPowerLaw1D(
    amplitude=np.median(flux_cont),
    x_0=5500, alpha=-1.5,
    x_min=wave_min, x_max=wave_max,  # CRITICAL: Full range
    name='AGN_powerlaw'
)

# Multi-component stellar model with tied redshift
stellar = sagan.Multi_StarSpectrum(
    n_components=5,
    star_types=['A', 'F', 'G', 'K', 'M'],
    amplitudes=[0.05, 0.1, 0.4, 0.3, 0.1],  # Initial guess
    velscale=200,
    delta_z=0,          # Shared redshift (GLOBAL)
    sigma=150,          # Shared dispersion (GLOBAL)
    name='stellar'
)

# CRITICAL: Fix redshift to systemic value
stellar.delta_z_0.fixed = True
stellar.delta_z_0.value = 0

# Iron template (GLOBAL - applies to all wavelengths)
iron = sagan.IronTemplate(
    amplitude=0.5, stddev=800/2.3548, z=0,
    template_name='park2022',
    name='iron'
)

# Combine: GLOBAL continuum = AGN + stellar + iron
model_cont = agn_cont + stellar + iron

# ========================================
# 4. Fit
# ========================================
fitter = fitting.LevMarLSQFitter()
model_cont_fit = fitter(model_cont, wave_cont, flux_cont,
                        weights=1/ferr_cont**2, maxiter=10000)

# ========================================
# 5. Visualize with plot_fit_new()
# ========================================
weight = np.zeros_like(wave_rest, dtype=float)
weight[cont_mask] = 1.0

ax, axr = sagan_plot.plot_fit_new(
    wave_rest, flux_rest, model_cont_fit,
    weight=weight, error=ferr_rest,
    xlabel='Wavelength (Å, rest frame)',
    ylabel='Flux (1e-17 erg/cm²/s/Å)'
)
ax.set_title(f'{TARGET_NAME} - Continuum Fit')
plt.savefig('continuum_fit_diagnostic.png', dpi=150)
plt.close()

# ========================================
# 6. Generate Continuum-Subtracted Spectrum
# ========================================
continuum_all = model_cont_fit(wave_rest)
flux_subtracted = flux_rest - continuum_all

# ========================================
# 7. Save Results
# ========================================
# Parameters
with open('continuum_fit_params.txt', 'w') as f:
    f.write(f"[AGN_powerlaw]\n")
    f.write(f"amplitude = {model_cont_fit.amplitude_0.value:.6f}\n")
    f.write(f"alpha = {model_cont_fit.alpha_0.value:.6f}\n\n")
    f.write(f"[Stellar]\n")
    f.write(f"amplitude = {model_cont_fit.amplitude_1.value:.6f}\n")
    f.write(f"sigma = {model_cont_fit.sigma_1.value:.2f}\n\n")
    f.write(f"[Iron]\n")
    f.write(f"amplitude = {model_cont_fit.amplitude_2.value:.6f}\n")

# Spectrum
tbl = Table([
    wave_rest, flux_subtracted, ferr_rest, continuum_all
], names=['WAVELENGTH', 'FLUX_SUB', 'ERROR', 'CONTINUUM'])

tbl.write('continuum_subtracted_spectrum.fits',
          format='fits', overwrite=True)

print("✓ Continuum fitting complete!")
```

---

## References

- **Type 1 AGN Fitting Strategy**: `type1_agn.md`
- **Function Reference**: `../function_reference/continuum_models.md`
- **SAGAN Source**: `sagan/continuum.py`
- **Plotting Functions**: `sagan/plot.py`
