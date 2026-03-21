# Narrow Line Template Plotting Guide

**Purpose**: Standardize diagnostic and validation plots for narrow line template generation in SAGAN.

**Why standardize?** Consistent plotting ensures:
- Clear visual validation of fit quality at each step
- Easy comparison between different fitting approaches
- Reusable code patterns for the SAGAN community
- Proper documentation of scientific results

---

## Table of Contents

1. [Standard Plot Types](#standard-plot-types)
2. [Type A: Diagnostic Plot (3-panel)](#type-a-diagnostic-plot-3-panel)
3. [Type B: Validation Plot (4-panel)](#type-b-validation-plot-4-panel)
4. [Standard Plotting Parameters](#standard-plotting-parameters)
5. [Quality Metrics to Display](#quality-metrics-to-display)
6. [Code Templates](#code-templates)
7. [Complete Examples](#complete-examples)

---

## Standard Plot Types

There are **two standard plot types** for narrow line template work:

| Plot Type | Purpose | When to Use |
|-----------|---------|-------------|
| **Type A: Diagnostic (3-panel)** | Intermediate fitting steps | After each fit (1-comp, 2-comp, etc.) |
| **Type B: Validation (4-panel)** | Final template validation | After generating template |

### Workflow

```
Load Spectrum → Fit [S II] (Type A) → Add Component (Type A) → Generate Template → Validate (Type B)
```

---

## Type A: Diagnostic Plot (3-panel)

**Purpose**: Evaluate intermediate fitting steps to diagnose issues and guide model improvement.

### Layout

```
┌─────────────────────────────────────┐
│  Panel 1: Main Fit (top)            │
│  - Data with error bars             │
│  - Model overplot                   │
│  - Line position markers            │
│  - Grid, legend                     │
├─────────────────────────────────────┤
│  Panel 2: Raw Residuals (middle)    │
│  - Residuals vs wavelength          │
│  - Horizontal zero line             │
│  - Grid                             │
├─────────────────────────────────────┤
│  Panel 3: Normalized Residuals      │
│  - Residuals/σ vs wavelength        │
│  - Zero line                        │
│  - ±3σ lines (red, dotted)          │
│  - Grid                             │
└─────────────────────────────────────┘
```

### When to Use

- After **every fitting step** (1-component, 2-component, etc.)
- When comparing different models (BIC comparison)
- For **all narrow line fitting**, not just template generation
- Anytime you need to check fit quality

### Key Elements

**Panel 1 (top): Main Fit**
- Data: `errorbar(wave, flux, yerr=ferr, fmt='o', markersize=3, capsize=2, alpha=0.7)`
- Model: `plot(wave_plot, model(wave_plot), linewidth=2)`
- Line positions: `axvline(line_wave, linestyle='--', alpha=0.3)`
- Grid: `grid(True, alpha=0.3)`
- Legend with clear labels

**Panel 2 (middle): Raw Residuals**
- Residuals: `plot(wave, residuals, 'r-', lw=1)`
- Zero line: `axhline(0, color='k', linestyle=':', alpha=0.5)`
- Grid: `grid(True, alpha=0.3)`

**Panel 3 (bottom): Normalized Residuals**
- Normalized: `plot(wave, residuals/ferr, 'b-', lw=1)`
- Zero line: `axhline(0, color='k', linestyle=':', alpha=0.5)`
- ±3σ lines: `axhline(±3, color='r', linestyle=':', alpha=0.5)`
- X-label: `'Wavelength (Å, rest frame)'`
- Grid: `grid(True, alpha=0.3)`

### Interpretation Guide

| Residual Pattern | Diagnosis | Action |
|------------------|-----------|--------|
| Random scatter around 0 | Good fit | Proceed |
| Systematic bumps | Missing component | Add component |
| Large deviations (>3σ) | Outliers or bad model | Check data/model |
| Offset from zero | Continuum problem | Adjust continuum |

---

## Type B: Validation Plot (4-panel)

**Purpose**: Validate the final template by comparing original Gaussian fit with template-based fit.

### Layout

```
┌──────────────────────┬──────────────────────┐
│ Panel 1: Template    │ Panel 2: Gaussian    │
│ Shape (velocity)     │ Fit                  │
│ - Velocity space     │ - Data + model       │
│ - Center line        │ - Line positions     │
│ - Half-max line      │ - χ² annotation      │
├──────────────────────┼──────────────────────┤
│ Panel 3: Template    │ Panel 4: Residuals   │
│ Fit                  │ Comparison           │
│ - Data + template    │ - Gaussian resids    │
│ - Amplitude ratios   │ - Template resids    │
│                      │ - Zero line          │
└──────────────────────┴──────────────────────┘
```

### When to Use

- **Only at the end** of template generation workflow
- After generating the single-line template
- Before saving template file
- As documentation of template quality

### Key Elements

**Panel 1 (top-left): Template Shape**
- X-axis: Velocity (km/s), centered at 0
- Y-axis: Normalized flux (0 to 1)
- Template: `plot(velc_temp, flux_temp, 'k-', lw=2)`
- Center line: `axvline(0, color='r', linestyle='--', alpha=0.5)`
- Half-max line: `axhline(0.5, color='b', linestyle=':', alpha=0.5)`
- FWHM annotation in title
- Grid: `grid(True, alpha=0.3)`

**Panel 2 (top-right): Original Gaussian Fit**
- Data with error bars
- Gaussian model fit
- Line position markers
- Title includes number of components
- χ² annotation (optional)
- Grid

**Panel 3 (bottom-left): Template-Based Fit**
- Data with error bars
- Template fit
- Line position markers
- Amplitude ratios (e.g., [S II] 6716/6731)
- X-label: `'Wavelength (Å, rest frame)'`
- Grid

**Panel 4 (bottom-right): Residuals Comparison**
- Original fit residuals: `'r-', lw=1, alpha=0.7`
- Template fit residuals: `'g--', lw=1, alpha=0.7`
- Zero line: `axhline(0, color='k', linestyle=':', alpha=0.5)`
- X-label: `'Wavelength (Å, rest frame)'`
- Legend
- Grid

---

## Standard Plotting Parameters

### Figure Size

```python
# Type A: Diagnostic (3-panel)
figsize = (12, 10)

# Type B: Validation (4-panel)
figsize = (14, 10)
```

### Data Points

```python
# Error bars
markersize = 3
capsize = 2
alpha = 0.7
color = 'k'  # black
fmt = 'o'
```

### Model Lines

```python
# Model overplot
linewidth = 2
colors = {
    'data': 'k',      # black
    'model': 'r',     # red
    'template': 'g'   # green
}
```

### Grid

```python
# Grid settings
alpha = 0.3
linestyle = '-'  # default grid
```

### Reference Lines

```python
# Line position markers
alpha = 0.3
linestyle = '--'
color = 'b' or 'g'  # blue/green for different lines

# Zero lines (residuals)
alpha = 0.5
linestyle = ':'
color = 'k'  # black

# ±3σ lines
alpha = 0.5
linestyle = ':'
color = 'r'  # red
```

### Output

```python
# Save figure
plt.savefig(filename, dpi=150, bbox_inches='tight')
```

---

## Quality Metrics to Display

### For Diagnostic Plots (Type A)

**In the output text:**
```python
print(f'  χ² = {chi2:.1f}')
print(f'  χ²/DOF = {chi2/dof:.2f}')
print(f'  BIC = {bic:.1f}')
print(f'  Parameters: {n_params}')
```

**For model comparison:**
```python
print(f'  ΔBIC = {bic_2 - bic_1:.1f}')
if bic_2 < bic_1:
    print('  ✓ 2-component model preferred')
else:
    print('  ✓ 1-component model preferred')
```

### For Validation Plots (Type B)

**Panel 1 title:**
```python
ax.set_title(f'Narrow Line Template (FWHM={fwhm_temp:.1f} km/s)')
```

**Panel 4 legend or text:**
```python
print(f'  [S II] 6716/6731 ratio: {ratio:.3f}')
print(f'  χ² original: {chi2_original:.1f}')
print(f'  χ² template: {chi2_template:.1f}')
print(f'  Δχ² = {chi2_template - chi2_original:.1f}')
```

### Quality Thresholds

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| χ²/DOF | < 2 | 2-3 | > 3 |
| ΔBIC | < -10 | -10 to 0 | > 0 (worse) |
| Template peak | |v| < 10 | 10-50 | > 50 |
| FWHM | < 500 | 500-800 | > 800 |
| Δχ² (validation) | |Δχ²| < 15 | 15-30 | > 30 |

---

## Code Templates

### Template 1: Diagnostic Plot (Type A)

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_diagnostic(wave, flux, ferr, model, title,
                   line_waves=None, filename='diagnostic.png'):
    """
    Create a 3-panel diagnostic plot for narrow line fitting.

    Parameters
    ----------
    wave : array
        Wavelength (rest frame)
    flux : array
        Flux values
    ferr : array
        Flux errors
    model : astropy Model
        Fitted model
    title : str
        Plot title
    line_waves : list, optional
        List of line wavelengths to mark
    filename : str, optional
        Output filename
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: Main fit
    ax = axes[0]
    ax.errorbar(wave, flux, yerr=ferr, fmt='o', markersize=3,
                color='k', capsize=2, label='Data', alpha=0.7)
    wave_plot = np.linspace(wave[0], wave[-1], 200)
    ax.plot(wave_plot, model(wave_plot), 'r-', lw=2, label='Model')

    # Mark line positions if provided
    if line_waves is not None:
        colors = ['b', 'g', 'c', 'm', 'y']
        for i, lw in enumerate(line_waves):
            ax.axvline(lw, color=colors[i % len(colors)],
                      linestyle='--', alpha=0.3, label=f'Line {i+1}')

    ax.set_ylabel('Flux')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Raw residuals
    resid = flux - model(wave)
    ax = axes[1]
    ax.plot(wave, resid, 'r-', lw=1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)

    # Panel 3: Normalized residuals
    ax = axes[2]
    ax.plot(wave, resid/ferr, 'b-', lw=1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axhline(3, color='r', linestyle=':', alpha=0.5)
    ax.axhline(-3, color='r', linestyle=':', alpha=0.5)
    ax.set_ylabel('Normalized Residuals')
    ax.set_xlabel('Wavelength (Å, rest frame)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'✓ Plot saved to: {filename}')
    plt.close()
```

### Template 2: Validation Plot (Type B)

```python
def plot_validation(wave, flux, ferr,
                   model_gaussian, model_template,
                   velc_temp, flux_temp,
                   chi2_gaussian, chi2_template,
                   title='Template Validation',
                   line_waves=None,
                   filename='validation.png'):
    """
    Create a 4-panel validation plot for template quality.

    Parameters
    ----------
    wave : array
        Wavelength (rest frame)
    flux : array
        Flux values
    ferr : array
        Flux errors
    model_gaussian : astropy Model
        Original Gaussian fit
    model_template : astropy Model
        Template-based fit
    velc_temp : array
        Template velocity array (km/s)
    flux_temp : array
        Template normalized flux
    chi2_gaussian : float
        χ² from Gaussian fit
    chi2_template : float
        χ² from template fit
    title : str
        Overall title
    line_waves : list, optional
        List of line wavelengths to mark
    filename : str
        Output filename
    """
    from scipy.interpolate import interp1d

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate FWHM
    f_interp = interp1d(velc_temp, flux_temp, kind='cubic',
                        bounds_error=False, fill_value=0)
    velc_fine = np.linspace(velc_temp[0], velc_temp[-1], 10000)
    flux_fine = f_interp(velc_fine)
    crossings = np.where(np.diff(np.sign(flux_fine - 0.5)))[0]
    if len(crossings) >= 2:
        fwhm = velc_fine[crossings[-1]] - velc_fine[crossings[0]]
    else:
        fwhm = np.nan

    # Panel 1: Template shape
    ax = axes[0, 0]
    ax.plot(velc_temp, flux_temp, 'k-', lw=2)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Center')
    ax.axhline(0.5, color='b', linestyle=':', alpha=0.5, label='Half max')
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Template Shape (FWHM={fwhm:.1f} km/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Gaussian fit
    ax = axes[0, 1]
    ax.errorbar(wave, flux, yerr=ferr, fmt='o', markersize=3,
                color='k', capsize=2, label='Data', alpha=0.7)
    wave_plot = np.linspace(wave[0], wave[-1], 200)
    ax.plot(wave_plot, model_gaussian(wave_plot), 'r-', lw=2,
            label='Gaussian fit')

    if line_waves is not None:
        colors = ['b', 'g', 'c', 'm', 'y']
        for i, lw in enumerate(line_waves):
            ax.axvline(lw, color=colors[i % len(colors)],
                      linestyle='--', alpha=0.3)

    ax.set_ylabel('Flux')
    ax.set_title(f'Gaussian Fit (χ²={chi2_gaussian:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Template fit
    ax = axes[1, 0]
    ax.errorbar(wave, flux, yerr=ferr, fmt='o', markersize=3,
                color='k', capsize=2, label='Data', alpha=0.7)
    ax.plot(wave_plot, model_template(wave_plot), 'g-', lw=2,
            label='Template fit')

    if line_waves is not None:
        colors = ['b', 'g', 'c', 'm', 'y']
        for i, lw in enumerate(line_waves):
            ax.axvline(lw, color=colors[i % len(colors)],
                      linestyle='--', alpha=0.3)

    ax.set_ylabel('Flux')
    ax.set_xlabel('Wavelength (Å, rest frame)')
    ax.set_title(f'Template Fit (χ²={chi2_template:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Residuals comparison
    ax = axes[1, 1]
    resid_gaussian = flux - model_gaussian(wave)
    resid_template = flux - model_template(wave)
    ax.plot(wave, resid_gaussian, 'r-', lw=1, label='Gaussian resids',
            alpha=0.7)
    ax.plot(wave, resid_template, 'g--', lw=1, label='Template resids',
            alpha=0.7)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Wavelength (Å, rest frame)')
    ax.set_title(f'Δχ² = {chi2_template - chi2_gaussian:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'✓ Plot saved to: {filename}')
    plt.close()
```

---

## Complete Examples

### Example 1: Full [S II] Template Generation Workflow

```python
#!/usr/bin/env python
"""
Generate narrow line template from [S II] doublet with standard plots
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
from astropy.io import fits
import sys
sys.path.insert(0, '/Users/shangguan/Softwares/my_modules/SAGAN')

import sagan
from sagan.utils import line_wave_dict
from sagan.continuum import WindowedPowerLaw1D

# Load data
fits_file = 'data/SDSS-J000111.15-100155.5.fits'
hdul = fits.open(fits_file)
data = hdul['COADD'].data
loglam = data['loglam']
flux_obs = data['flux']
ivar = data['ivar']

mask = ivar > 0
loglam = loglam[mask]
flux_obs = flux_obs[mask]
ivar = ivar[mask]

wave_obs = 10**loglam
ferr_obs = 1.0 / np.sqrt(ivar)

specobj = hdul['SPECOBJ'].data
z = specobj[0]['Z']
wave_rest = wave_obs / (1 + z)

# Select [S II] region
sii_min, sii_max = 6700, 6745
s2_region = (wave_rest > sii_min) & (wave_rest < sii_max)
wave_s2 = wave_rest[s2_region]
flux_s2 = flux_obs[s2_region]
ferr_s2 = ferr_obs[s2_region]

sii_6716 = line_wave_dict['SII_6716']
sii_6731 = line_wave_dict['SII_6731']

# ========================================
# Step 1: Fit with 1 component
# ========================================
print('\n' + '='*70)
print('Step 1: Fit [S II] with 1 Gaussian component')
print('='*70)

cont_s2_1 = WindowedPowerLaw1D(
    amplitude=5.0, x_0=6720.0, alpha=-0.5,
    x_min=sii_min, x_max=sii_max, name='Cont_SII_1'
)

sii_doublet_1 = sagan.Line_MultiGauss_doublet(
    n_components=1,
    amp_c0=5.0, amp_c1=4.0,
    dv_c=0.0, sigma_c=100.0,
    wavec0=sii_6716,
    wavec1=sii_6731,
    name='SII_doublet_1'
)

model_init_1 = cont_s2_1 + sii_doublet_1
fitter = fitting.LevMarLSQFitter()
model_fit_1 = fitter(model_init_1, wave_s2, flux_s2,
                     weights=1/ferr_s2**2, maxiter=10000)

# Calculate statistics
chi2_1 = np.sum(((flux_s2 - model_fit_1(wave_s2)) / ferr_s2)**2)
dof_1 = len(wave_s2) - len(model_fit_1.parameters)
bic_1 = chi2_1 + len(model_fit_1.parameters) * np.log(len(wave_s2))

print(f'\n1-component fit results:')
print(f'  χ² = {chi2_1:.1f}, DOF = {dof_1}')
print(f'  χ²/DOF = {chi2_1/dof_1:.2f}')
print(f'  BIC = {bic_1:.1f}')

# Create Diagnostic Plot (Type A)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

ax = axes[0]
ax.errorbar(wave_s2, flux_s2, yerr=ferr_s2, fmt='o', markersize=3,
            color='k', capsize=2, label='Data', alpha=0.7)
wave_plot = np.linspace(wave_s2[0], wave_s2[-1], 200)
ax.plot(wave_plot, model_fit_1(wave_plot), 'r-', lw=2, label='1-component fit')
ax.axvline(sii_6716, color='b', linestyle='--', alpha=0.3, label='[S II] 6716')
ax.axvline(sii_6731, color='g', linestyle='--', alpha=0.3, label='[S II] 6731')
ax.set_ylabel('Flux')
ax.set_title('Step 1: 1-Component Fit')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals
resid_1 = flux_s2 - model_fit_1(wave_s2)
ax = axes[1]
ax.plot(wave_s2, resid_1, 'r-', lw=1)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.set_ylabel('Residuals')
ax.grid(True, alpha=0.3)

# Normalized residuals
ax = axes[2]
ax.plot(wave_s2, resid_1/ferr_s2, 'b-', lw=1)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.axhline(3, color='r', linestyle=':', alpha=0.5)
ax.axhline(-3, color='r', linestyle=':', alpha=0.5)
ax.set_ylabel('Normalized Residuals')
ax.set_xlabel('Wavelength (Å, rest frame)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sii_fit_1comp.png', dpi=150, bbox_inches='tight')
print('\n✓ Diagnostic plot saved to: sii_fit_1comp.png')
plt.close()

# ========================================
# Step 2: Fit with 2 components
# ========================================
print('\n' + '='*70)
print('Step 2: Fit [S II] with 2 Gaussian components')
print('='*70)

cont_s2_2 = WindowedPowerLaw1D(
    amplitude=model_fit_1.amplitude_0.value,
    x_0=model_fit_1.x_0_0.value,
    alpha=model_fit_1.alpha_0.value,
    x_min=sii_min, x_max=sii_max, name='Cont_SII_2'
)

sii_doublet_2 = sagan.Line_MultiGauss_doublet(
    n_components=2,
    amp_c0=model_fit_1.amp_c0_1.value,
    amp_c1=model_fit_1.amp_c1_1.value,
    dv_c=model_fit_1.dv_c_1.value,
    sigma_c=model_fit_1.sigma_c_1.value,
    amp_w0=0.5,
    dv_w0=-200.0,
    sigma_w0=150.0,
    wavec0=sii_6716,
    wavec1=sii_6731,
    name='SII_doublet_2'
)

model_init_2 = cont_s2_2 + sii_doublet_2
model_fit_2 = fitter(model_init_2, wave_s2, flux_s2,
                     weights=1/ferr_s2**2, maxiter=10000)

# Calculate statistics
chi2_2 = np.sum(((flux_s2 - model_fit_2(wave_s2)) / ferr_s2)**2)
dof_2 = len(wave_s2) - len(model_fit_2.parameters)
bic_2 = chi2_2 + len(model_fit_2.parameters) * np.log(len(wave_s2))

print(f'\n2-component fit results:')
print(f'  χ² = {chi2_2:.1f}, DOF = {dof_2}')
print(f'  χ²/DOF = {chi2_2/dof_2:.2f}')
print(f'  BIC = {bic_2:.1f}')

# Compare BIC
print(f'\nBIC comparison:')
print(f'  ΔBIC = {bic_2 - bic_1:.1f}')
if bic_2 < bic_1:
    print(f'  ✓ 2-component model is preferred')
    model_final = model_fit_2
    n_components = 2
else:
    print(f'  ✓ 1-component model is preferred')
    model_final = model_fit_1
    n_components = 1

# Create another Diagnostic Plot for 2-component
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

ax = axes[0]
ax.errorbar(wave_s2, flux_s2, yerr=ferr_s2, fmt='o', markersize=3,
            color='k', capsize=2, label='Data', alpha=0.7)
wave_plot = np.linspace(wave_s2[0], wave_s2[-1], 200)
ax.plot(wave_plot, model_fit_2(wave_plot), 'r-', lw=2, label='2-component fit')
ax.axvline(sii_6716, color='b', linestyle='--', alpha=0.3)
ax.axvline(sii_6731, color='g', linestyle='--', alpha=0.3)
ax.set_ylabel('Flux')
ax.set_title('Step 2: 2-Component Fit')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals
resid_2 = flux_s2 - model_fit_2(wave_s2)
ax = axes[1]
ax.plot(wave_s2, resid_2, 'r-', lw=1)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.set_ylabel('Residuals')
ax.grid(True, alpha=0.3)

# Normalized residuals
ax = axes[2]
ax.plot(wave_s2, resid_2/ferr_s2, 'b-', lw=1)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.axhline(3, color='r', linestyle=':', alpha=0.5)
ax.axhline(-3, color='r', linestyle=':', alpha=0.5)
ax.set_ylabel('Normalized Residuals')
ax.set_xlabel('Wavelength (Å, rest frame)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sii_fit_2comp.png', dpi=150, bbox_inches='tight')
print('\n✓ Diagnostic plot saved to: sii_fit_2comp.png')
plt.close()

# ========================================
# Step 3: Generate template
# ========================================
print('\n' + '='*70)
print('Generating SINGLE-LINE narrow template')
print('='*70)

# Get the doublet component
sii_doublet_fit = model_final[1]  # Second component is the doublet

# Create velocity array
velc_temp = np.linspace(-800, 800, 2000)

# Generate template using gen_template() method
flux_temp = sii_doublet_fit.gen_template(velc_temp, normalized=True)

print(f'\nTemplate properties:')
print(f'  Velocity range: {velc_temp[0]:.0f} to {velc_temp[-1]:.0f} km/s')
print(f'  Peak at velocity: {velc_temp[np.argmax(flux_temp)]:.2f} km/s')

# Measure FWHM
from scipy.interpolate import interp1d
f_interp = interp1d(velc_temp, flux_temp, kind='cubic',
                    bounds_error=False, fill_value=0)
velc_fine = np.linspace(velc_temp[0], velc_temp[-1], 10000)
flux_fine = f_interp(velc_fine)
crossings = np.where(np.diff(np.sign(flux_fine - 0.5)))[0]
if len(crossings) >= 2:
    fwhm_temp = velc_fine[crossings[-1]] - velc_fine[crossings[0]]
    print(f'  Template FWHM: {fwhm_temp:.1f} km/s')

# ========================================
# Step 4: Verification fit
# ========================================
print('\n' + '='*70)
print('Verification: Fitting [S II] doublet with template')
print('='*70)

dv_measured = model_final.dv_c_1.value

cont_test = WindowedPowerLaw1D(
    amplitude=model_final.amplitude_0.value,
    x_0=model_final.x_0_0.value,
    alpha=model_final.alpha_0.value,
    x_min=sii_min, x_max=sii_max, name='Cont'
)

sii_6716_temp = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=model_final.amp_c0_1.value,
    dv=dv_measured,
    wavec=sii_6716,
    name='SII_6716'
)

sii_6731_temp = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=model_final.amp_c1_1.value,
    dv=dv_measured,
    wavec=sii_6731,
    name='SII_6731'
)

model_template_fit = cont_test + sii_6716_temp + sii_6731_temp
model_fit_template = fitter(model_template_fit, wave_s2, flux_s2,
                            weights=1/ferr_s2**2, maxiter=10000)

print(f'\nTemplate fit results:')
print(f'  [S II] 6716 amplitude: {model_fit_template.amplitude_1.value:.4f}')
print(f'  [S II] 6731 amplitude: {model_fit_template.amplitude_2.value:.4f}')
ratio = model_fit_template.amplitude_1.value / model_fit_template.amplitude_2.value
print(f'  [S II] 6716/6731 ratio: {ratio:.3f}')

# Compare χ²
resid_original = flux_s2 - model_final(wave_s2)
resid_template = flux_s2 - model_fit_template(wave_s2)
chi2_original = np.sum((resid_original / ferr_s2)**2)
chi2_template = np.sum((resid_template / ferr_s2)**2)

print(f'\nχ² comparison:')
print(f'  Original {n_components}-component Gaussian: χ² = {chi2_original:.1f}')
print(f'  Template-based fit: χ² = {chi2_template:.1f}')
print(f'  Difference: {chi2_template - chi2_original:.1f}')

# ========================================
# Step 5: Create Validation Plot (Type B)
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Template shape
ax = axes[0, 0]
ax.plot(velc_temp, flux_temp, 'k-', lw=2)
ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Center')
ax.axhline(0.5, color='b', linestyle=':', alpha=0.5, label='Half max')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Normalized Flux')
ax.set_title(f'Narrow Line Template (FWHM={fwhm_temp:.1f} km/s)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Best Gaussian fit
ax = axes[0, 1]
ax.errorbar(wave_s2, flux_s2, yerr=ferr_s2, fmt='o', markersize=3,
            color='k', capsize=2, label='Data', alpha=0.7)
wave_plot = np.linspace(wave_s2[0], wave_s2[-1], 200)
ax.plot(wave_plot, model_final(wave_plot), 'r-', lw=2,
        label=f'{n_components}-component fit')
ax.axvline(sii_6716, color='b', linestyle='--', alpha=0.3)
ax.axvline(sii_6731, color='g', linestyle='--', alpha=0.3)
ax.set_ylabel('Flux')
ax.set_title(f'Gaussian Fit (χ²={chi2_original:.1f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Template fit
ax = axes[1, 0]
ax.errorbar(wave_s2, flux_s2, yerr=ferr_s2, fmt='o', markersize=3,
            color='k', capsize=2, label='Data', alpha=0.7)
ax.plot(wave_plot, model_fit_template(wave_plot), 'g-', lw=2, label='Template fit')
ax.axvline(sii_6716, color='b', linestyle='--', alpha=0.3)
ax.axvline(sii_6731, color='g', linestyle='--', alpha=0.3)
ax.set_ylabel('Flux')
ax.set_xlabel('Wavelength (Å, rest frame)')
ax.set_title(f'Template Fit (χ²={chi2_template:.1f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Residuals comparison
ax = axes[1, 1]
ax.plot(wave_s2, resid_original, 'r-', lw=1, label='Gaussian resids', alpha=0.7)
ax.plot(wave_s2, resid_template, 'g--', lw=1, label='Template resids', alpha=0.7)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.set_ylabel('Residuals')
ax.set_xlabel('Wavelength (Å, rest frame)')
ax.set_title(f'Δχ² = {chi2_template - chi2_original:.1f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Template Validation', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('narrow_template_validation.png', dpi=150, bbox_inches='tight')
print('\n✓ Validation plot saved to: narrow_template_validation.png')
plt.close()

# ========================================
# Step 6: Save template
# ========================================
print('\n' + '='*70)
print('Saving narrow line template')
print('='*70)

np.savetxt('narrow_template.txt',
           np.column_stack([velc_temp, flux_temp]),
           header='velocity_kms normalized_flux')

print('\n✓ Template saved:')
print('  - narrow_template.txt')
print('\nTemplate properties:')
print(f'  - Single-line profile shape')
print(f'  - FWHM: {fwhm_temp:.1f} km/s')
print(f'  - Centered at dv=0')
print('\nUsage example:')
print('  nha = Line_template(template_velc=velc_temp,')
print('                       template_flux=flux_temp,')
print('                       amplitude=50.0,')
print('                       dv=10,')
print('                       wavec=line_wave_dict["Halpha"])')
print('='*70)
```

---

## Summary

### Standard Workflow Checklist

For every narrow line template generation:

- [ ] **Type A Diagnostic Plot** after each fitting step
  - [ ] 3-panel layout (fit, raw residuals, normalized residuals)
  - [ ] Error bars on data (markersize=3, capsize=2)
  - [ ] Line position markers (alpha=0.3)
  - [ ] ±3σ lines on normalized residuals
  - [ ] Grid (alpha=0.3)
  - [ ] Quality metrics printed (χ², BIC, DOF)

- [ ] **Type B Validation Plot** at the end
  - [ ] 4-panel layout (template shape, Gaussian fit, template fit, residuals comparison)
  - [ ] FWHM measured and displayed
  - [ ] χ² comparison between models
  - [ ] Amplitude ratios displayed
  - [ ] All standard styling elements

- [ ] **Output files**
  - [ ] Diagnostic plots: `*_fit_1comp.png`, `*_fit_2comp.png`, etc.
  - [ ] Validation plot: `*_validation.png`
  - [ ] Template data: `*.txt` (ASCII format)
  - [ ] All plots at dpi=150

---

## References

- Example template script: `/Users/shangguan/Softwares/my_modules/SAGAN/skills/fitting_strategies/example/narrow_line_template_template.py`
- Narrow line template guide: `narrow_line_template.md`
- SAGAN plotting utilities: `sagan/plot.py`
- SAGAN diagnostic plotting functions: `plot_narrow_line_diagnostic()`, `plot_narrow_line_template_validation()`
