# Continuum Fitting and Subtraction Guide

**Purpose**: Model and subtract the continuum from AGN spectra to isolate emission lines for accurate flux measurements and line profile analysis.

**When to use**: Stage 1 of AGN spectral analysis - before fitting emission lines. Must be completed before any emission line fitting.

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

Continuum fitting is the **critical first step** in AGN spectral analysis. The continuum represents the underlying continuum emission from:
- The accretion disk (AGN power-law)
- Host galaxy starlight (stellar continuum)
- Blended Fe II emission (iron template)

Accurate continuum modeling and subtraction is essential for:
- Measuring emission line fluxes
- Analyzing line profiles (widths, asymmetries)
- Detecting weak emission lines
- Calculating equivalent widths

### Key Principle

> **Always fit and subtract the continuum BEFORE fitting emission lines.**
>
> The continuum-subtracted spectrum provides a clean baseline for line fitting, eliminating degeneracies between continuum shape and line fluxes.

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

**Model**: `StarSpectrum`

```python
stellar = StarSpectrum(
    amplitude=?,      # Relative contribution (0.1-0.5 typical)
    Star_type='G',    # 'A', 'F', 'G', 'K', or 'M'
    velscale=200,     # Velocity scale (km/s)
    delta_z=0,        # Redshift offset (km/s)
    sigma=150,        # Velocity dispersion (km/s)
    name='stellar'
)
```

**Available stellar templates**:
- `A`: HD 97633 (A0V star)
- `F`: HD 89254
- `G`: HD 140027 (most common for AGN hosts)
- `K`: HD 49520
- `M`: HD 44478

**Typical values**:
- `Star_type`: 'G' or 'K' (older stellar populations)
- `sigma`: 100-300 km/s (bulge velocity dispersion)
- `amplitude`: 0.1-0.5 (10-50% of total flux)

**When to omit**: For high-luminosity AGN where host contribution is negligible (<5%)

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

### Full Model (All Three Components)

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

# 2. Stellar emission
stellar = sagan.StarSpectrum(
    amplitude=0.2,       # Initial: 20% contribution
    Star_type='G',       # Most common for AGN hosts
    velscale=200,
    delta_z=0,
    sigma=150,           # Typical stellar dispersion
    name='stellar'
)

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
| `Stellar amplitude` | 0.2 | 20% contribution (adjust if obvious) |
| `Stellar Star_type` | 'G' | Most common; try 'K' if poor fit |
| `Stellar sigma` | 150 km/s | Typical bulge dispersion |
| `Iron amplitude` | 0.5 | 50% of AGN continuum |
| `Iron stddev` | 800/2.3548 | Intrinsic width from template |

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

```python
print("\nFitted parameters:")
print(f"  AGN Power-law:")
print(f"    amplitude = {model_cont_fit.amplitude_0.value:.6f}")
print(f"    alpha = {model_cont_fit.alpha_0.value:.6f}")

if 'stellar' in model_cont_fit.submodel_names:
    print(f"  Stellar:")
    print(f"    amplitude = {model_cont_fit.amplitude_1.value:.6f}")
    print(f"    sigma = {model_cont_fit.sigma_1.value:.2f} km/s")

if 'iron' in model_cont_fit.submodel_names:
    print(f"  Iron:")
    print(f"    amplitude = {model_cont_fit.amplitude_2.value:.6f}")
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

### Problem: Stellar sigma at lower bound (20 km/s)

**Cause**: Spectral resolution doesn't resolve stellar dispersion

**Solutions**:
```python
# 1. Fix sigma to minimum
stellar.sigma.fixed = True
stellar.sigma.value = 20

# 2. Use simpler stellar model
# Omit stellar if contribution is small (<10%)
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
Fit Type 1 AGN continuum and subtract
Target: SDSS-J000111.15-100155.5
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
# 3. Build Model
# ========================================
wave_min = int(np.floor(wave_rest.min()))
wave_max = int(np.ceil(wave_rest.max()))

agn_cont = WindowedPowerLaw1D(
    amplitude=np.median(flux_cont),
    x_0=5500, alpha=-1.5,
    x_min=wave_min, x_max=wave_max,  # CRITICAL: Full range
    name='AGN_powerlaw'
)

stellar = sagan.StarSpectrum(
    amplitude=0.2, Star_type='G',
    velscale=200, delta_z=0, sigma=150,
    name='stellar'
)

iron = sagan.IronTemplate(
    amplitude=0.5, stddev=800/2.3548, z=0,
    template_name='park2022',
    name='iron'
)

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
