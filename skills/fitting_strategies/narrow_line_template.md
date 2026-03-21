# Narrow Line Template Generation Guide

**Purpose**: Extract the intrinsic narrow line profile shape from a clean emission line for use in all subsequent narrow line fitting.

**When to use**: Stage 1 of Type 1 AGN fitting, before fitting broad line complexes.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Use a Template?](#why-use-a-template)
3. [Choosing the Template Source](#choosing-the-template-source)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Quality Checks](#quality-checks)
6. [Using the Template](#using-the-template)
7. [Troubleshooting](#troubleshooting)
8. [Complete Example](#complete-example)

---

---

## ⚠️ CRITICAL WARNING: Do NOT Use [N II] for Narrow Line Template

**NEVER use [N II] λλ6548,6583 to generate the narrow line template.**

### Why [N II] is Unsuitable

The [N II] doublet is **strongly blended with the broad Hα component**:

```
Hα rest wavelength:        6562.8 Å
[N II] 6548:               6548.0 Å  (Δλ = 14.8 Å → ~680 km/s)
[N II] 6583:               6583.5 Å  (Δλ = 20.7 Å → ~950 km/s)
```

### Problems with Using [N II]

1. **Broad Hα contamination**: Broad Hα typically has FWHM = 2000-5000 km/s, extending well beyond the [N II] lines
2. **Asymmetric blending**: The broad Hα wing is often redshifted or asymmetric, contaminating [N II] differently
3. **Degenerate fits**: When fitting [N II] + broad Hα simultaneously, the narrow line width becomes poorly constrained
4. **Incorrect template**: The extracted "narrow" profile will include broad line flux, giving widths that are too large

### The Correct Approach

**PRIMARY CHOICE: [S II] λλ6716,6731 doublet** ⭐⭐⭐
- **Well-separated from Hα** (Δλ = 153-167 Å → ~7000 km/s from Hα center)
- **No broad component contamination** - both are forbidden lines
- **Both lines have similar strength** - provides strong constraints
- **Clean continuum on both sides** - easier continuum fitting
- **Doublet ratio validates fit** - [S II] 6716/6731 ratio ~0.5-1.0

**SECONDARY CHOICE: [O III] λλ4959,5007 doublet** ⭐⭐
- **Well-separated from Hβ** (Hβ at 4861 Å, Δλ = 98-145 Å)
- **Very strong in many AGN** - high S/N
- **Note**: May have blue wing from outflow (may require masking wing region)

**NEVER USE:**
- ❌ [N II] λλ6548,6583 (blended with broad Hα)
- ❌ Hα or any broad permitted line
- ❌ Any line blended with broad components

---

## Overview

The narrow line template is a **single-line profile shape** extracted from a clean emission line (typically [S II] λλ6716,6731 or [O III] λλ4959,5007). This template captures the intrinsic narrow line width and any substructure (e.g., core + wing components).

### Key Principle

> **All narrow forbidden lines share the same profile shape.**

Once extracted from [S II], this shape is used for **ALL** narrow lines ([O III], [N II], [S II], [O I], He II, etc.) via the `Line_template` class.

### Why This Works

- Narrow forbidden lines originate from same spatial region (NLR)
- Have similar kinematics (FWHM ~100-500 km/s)
- Share instrumental broadening
- Any substructure (e.g., blueshifted wing) affects all lines similarly

---

## Why Use a Template?

### Without Template

```python
# Fit each narrow line independently
nha = Line_Gaussian(amplitude=?, dv=?, sigma=?, wavec=6562.819)
no3 = Line_Gaussian(amplitude=?, dv=?, sigma=?, wavec=5006.843)
nn2 = Line_Gaussian(amplitude=?, dv=?, sigma=?, wavec=6583.460)
# ... 7+ narrow lines × 3 parameters each = 21+ free parameters
```

**Problems:**
- Too many free parameters
- Degeneracies between broad and narrow components
- Narrow line widths poorly constrained
- Fit can diverge or give unphysical results

### With Template

```python
# Extract template once
velc_temp, flux_temp = extract_template_from_SII()

# Use for ALL narrow lines
nha = Line_template(template_velc=velc_temp, template_flux=flux_temp,
                   amplitude=?, dv=?, wavec=6562.819)
no3 = Line_template(template_velc=velc_temp, template_flux=flux_temp,
                   amplitude=?, dv=?, wavec=5006.843)
# ... 7 amplitudes + 1 shared dv = 8 free parameters
```

**Benefits:**
- >60% reduction in parameters
- No degeneracies (shape is fixed)
- All narrow lines share kinematics (tied dv)
- More robust, physically meaningful results

---

## Choosing the Template Source

### Priority Order ⭐⭐⭐

**IMPORTANT**: Follow this priority order when selecting the template source:

1. **[S II] λλ6716,6731 doublet** (PRIMARY CHOICE - Always use if available)
2. **[O III] λλ4959,5007 doublet** (SECONDARY CHOICE - Use only if [S II] unavailable/weak)
3. **Fixed-width Gaussian** (FALLBACK - Only if S/N < 20 for both above)

### ⭐⭐⭐ Option 1: [S II] λλ6716,6731 Doublet (PRIMARY)

**Why this is the BEST choice:**

- **Well-isolated from broad Hα**: Separated by ~7000 km/s (Δλ = 153-167 Å)
- **Pure forbidden lines**: No broad component contamination possible
- **Doublet provides strong constraints**: Two lines with similar widths
- **Clean continuum**: Line-free regions on both sides for robust continuum fitting
- **Doublet ratio validates fit**: [S II] 6716/6731 flux ratio ~0.5-1.0 (density-sensitive)

**When to use:** ALWAYS, if available with S/N > 20

**Wavelength range (rest frame):** 6700-6745 Å

**Potential issues:**
- Sometimes weak in low-luminosity AGN
- May be contaminated by broad Hα wing if Hα is extremely broad (FWHM > 10,000 km/s)

**Check for contamination:**
```python
# Fit [S II] region and check residuals
# If residuals show broad component under [S II], mask the region or use [O III]
```

### ⭐⭐ Option 2: [O III] λλ4959,5007 Doublet (SECONDARY)

**Why this is the SECOND choice:**

- **Very strong in most AGN**: Higher S/N than [S II]
- **Well-separated from Hβ**: Hβ at 4861 Å, [O III] at 4959/5007 Å (Δλ = 98-145 Å)
- **Doublet provides constraints**: Fixed theoretical ratio (1:2.98)

**When to use:** If [S II] is unavailable, has S/N < 20, or is contaminated

**Wavelength range (rest frame):** 4940-5020 Å

**Potential issues:**
- **Blue wing from outflow**: [O III] often has blueshifted wing from NLR outflow
- **May require masking**: Exclude the wing region (-500 to -1000 km/s) when generating template
- **Iron contamination**: Possible Fe II emission in this region (rare)

**Masking procedure for blue wing:**
```python
# If strong blue wing present, mask it when fitting
mask_wing = (velc > -1000) & (velc < -100)  # Mask wing region
wave_fit = wave[~mask_wing]
flux_fit = flux[~mask_wing]
```

### Option 3: Fixed-Width Gaussian (FALLBACK - Low S/N)

**When to use:** S/N < 20 for both [S II] and [O III]

**Approach:** Use instrumental resolution as the width

```python
# Calculate instrumental LSF width
ls_km = 299792.458  # Speed of light in km/s
resolving_power = 2000  # SDSS spectrograph
lsf_sigma = ls_km / (resolving_power * 2.3548)  # ~64 km/s

# Create fixed-width Gaussian
velc_temp = np.linspace(-500, 500, 1000)
flux_temp = np.exp(-0.5 * (velc_temp / lsf_sigma)**2)
flux_temp = flux_temp / np.max(flux_temp)
```

**Note:** This is based on instrumental resolution only, not empirical measurement from the spectrum.

### ❌ NEVER USE: [N II] λλ6548,6583

**See WARNING section above** - [N II] is blended with broad Hα and will produce incorrect templates.



## Step-by-Step Workflow

### Step 1: Prepare the Spectrum

```python
# Load spectrum (already extinction-corrected and in rest frame)
wave, flux, ferr = load_spectrum(target)

# Define [S II] region
s2_region = (wave > 6700) & (wave < 6745)
wave_s2 = wave[s2_region]
flux_s2 = flux[s2_region]
ferr_s2 = ferr[s2_region]
```

### Step 2: Build the Model

```python
from sagan.continuum import WindowedPowerLaw1D
from sagan.utils import line_wave_dict

# Power-law continuum
cont_s2 = WindowedPowerLaw1D(
    amplitude=15.0,   # Estimate from continuum
    x_0=6720.,       # Reference wavelength
    alpha=0.5,       # Small for local fit
    x_min=6700,
    x_max=6745,
    name='Cont_SII'
)

# [S II] doublet - start with 1 component
sii_doublet = sagan.Line_MultiGauss_doublet(
    n_components=1,
    amp_c0=8.0,      # [S II] 6716 amplitude
    amp_c1=6.0,      # [S II] 6731 amplitude
    dv_c=0,          # Velocity shift
    sigma_c=100,     # Width (km/s)
    wavec0=line_wave_dict['SII_6716'],
    wavec1=line_wave_dict['SII_6731'],
    name='SII_doublet'
)

# Combine
model_init = cont_s2 + sii_doublet
```

### Step 3: Fit and Evaluate

```python
from astropy.modeling import fitting

fitter = fitting.LevMarLSQFitter()
model_fit = fitter(model_init, wave_s2, flux_s2,
                   weights=1/ferr_s2**2, maxiter=10000)

# Calculate BIC
from sagan.utils import calculate_bic
bic, chi2, n_params = calculate_bic(model_fit, wave_s2, flux_s2, ferr_s2)
```

### Step 4: Check Residuals - Add Components if Needed

```python
# Plot residuals
import matplotlib.pyplot as plt
residual = flux_s2 - model_fit(wave_s2)

plt.figure(figsize=(10, 4))
plt.step(wave_s2, residual, 'k-')
plt.axhline(0, color='r', linestyle='--')
plt.show()
```

**If residuals show systematic bumps:**
- Try 2-component model: `n_components=2`
- Add wing parameters: `amp_w0`, `dv_w0`, `sigma_w0`
- Compare BIC: add only if ΔBIC < -10

### Step 5: Generate Template ⭐

#### For [S II] or [O III] Doublet (Using `Line_MultiGauss_doublet`)

```python
# Get the doublet component
sii_doublet_fit = model_fit[1]  # Second component is the doublet

# Create velocity array
velc_temp = np.linspace(-800, 800, 2000)  # km/s

# Generate template - SIMPLE!
# Note: gen_template() is ONLY for Line_MultiGauss_doublet models
flux_temp = sii_doublet_fit.gen_template(velc_temp, normalized=True)
```

The `gen_template()` method:
- Extracts single-line profile shape from the doublet model
- Includes all components (core + wings)
- Normalizes to peak = 1
- Centers at dv = 0

**Important**: `gen_template()` is a method of `Line_MultiGauss_doublet` class and is **only available** for doublet models.

#### For Single Line (Using `Line_Gaussian` or `Line_MultiGauss`)

If you fit a single narrow line (not a doublet), you can directly generate the template from the best-fit model:

```python
# Fit a single line, e.g., [O III] 4363
from sagan.line_profile import Line_Gaussian

line_model = Line_Gaussian(
    amplitude=5.0,
    dv=0,      # <-- CRITICAL: set dv=0 for template
    sigma=100,
    wavec=line_wave_dict['OIII_4363']
)

model_init = cont + line_model
model_fit = fitter(model_init, wave, flux, weights=1/ferr**2)

# Generate template directly from fitted model
# (Make sure dv=0 in the model!)
velc_temp = np.linspace(-800, 800, 2000)

# Evaluate at wavelengths corresponding to velocities
wave_for_template = line_wave_dict['OIII_4363'] * (1 + velc_temp / ls_km)
flux_temp = model_fit[1](wave_for_template)  # Second component is the line
flux_temp = flux_temp / np.max(flux_temp)  # Normalize
```

**Key point**: Whether using `gen_template()` for doublets or evaluating directly for single lines, always ensure the template is centered at **dv = 0** (shape only, no velocity shift).

### Step 6: Quality Checks and Visualization

#### 6.1. Check Peak Position

```python
peak_idx = np.argmax(flux_temp)
peak_vel = velc_temp[peak_idx]
assert abs(peak_vel) < 10, f"Peak not at 0! Peak at {peak_vel:.2f} km/s"
print(f"✓ Peak centered at: {peak_vel:.2f} km/s")
```

#### 6.2. Measure FWHM

```python
from scipy.interpolate import interp1d
f_interp = interp1d(velc_temp, flux_temp, kind='cubic',
                    bounds_error=False, fill_value=0)
velc_fine = np.linspace(velc_temp[0], velc_temp[-1], 10000)
flux_fine = f_interp(velc_fine)
crossings = np.where(np.diff(np.sign(flux_fine - 0.5)))[0]
if len(crossings) >= 2:
    fwhm = velc_fine[crossings[-1]] - velc_fine[crossings[0]]
    print(f"✓ Template FWHM: {fwhm:.1f} km/s")
    # FWHM can be up to 800 km/s (if wing components present)
    assert fwhm < 800, f"FWHM {fwhm:.1f} too large!"
```

#### 6.3. Plot Fitting Result (REQUIRED)

**Always use `sagan.plot.plot_fit()` or the specialized narrow line plotting functions:**

> **Standard Plotting**: For standardized narrow line template plotting, see `NARROW_LINE_TEMPLATE_PLOTTING_GUIDE.md` for the two standard plot types:
> - **Type A (Diagnostic Plot)**: 3-panel plot for intermediate fitting steps
> - **Type B (Validation Plot)**: 4-panel plot for final template validation
>
> **Reusable Functions**: `sagan/plot.py` provides two specialized functions:
> - `plot_narrow_line_diagnostic()`: Creates Type A diagnostic plots
> - `plot_narrow_line_template_validation()`: Creates Type B validation plots

For simple fitting visualization:

```python
from sagan import plot

# Plot the [S II] region fit
ax = plot.plot_fit(wave_s2, flux_s2, model_fit, ferr_s2)
ax.set_title('[S II] Region - Best Fit')
ax.axvline(line_wave_dict['SII_6716'], color='r', linestyle='--', alpha=0.3, label='[S II] 6716')
ax.axvline(line_wave_dict['SII_6731'], color='b', linestyle='--', alpha=0.3, label='[S II] 6731')
ax.legend()
plt.savefig('sii_fitting_result.png', dpi=150)
print("✓ Fitting result plot saved to: sii_fitting_result.png")
```

#### 6.4. Plot Template Profile

```python
import matplotlib.pyplot as plt

# Plot the template profile
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(velc_temp, flux_temp, 'k-', lw=2)
ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Center (v=0)')
ax.axhline(0.5, color='b', linestyle=':', alpha=0.5, label='Half max')
ax.axhline(1.0, color='g', linestyle=':', alpha=0.5, label='Peak')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Narrow Line Template Profile')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('narrow_template_profile.png', dpi=150)
print("✓ Template profile plot saved to: narrow_template_profile.png")
```

#### 6.5. Verification Fit (Optional)

```python
# Fit [S II] with template to verify it matches
dv_measured = model_fit.dv_c_1.value

sii_6716 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=model_fit.amp_c0_1.value,
    dv=dv_measured,
    wavec=line_wave_dict['SII_6716']
)

sii_6731 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=model_fit.amp_c1_1.value,
    dv=dv_measured,
    wavec=line_wave_dict['SII_6731']
)

model_verify = cont_s2 + sii_6716 + sii_6731
model_verify_fit = fitter(model_verify, wave_s2, flux_s2,
                          weights=1/ferr_s2**2, maxiter=10000)

# Compare χ²
chi2_orig = np.sum(((flux_s2 - model_fit(wave_s2)) / ferr_s2)**2)
chi2_temp = np.sum(((flux_s2 - model_verify_fit(wave_s2)) / ferr_s2)**2)
print(f"✓ χ² comparison: Δχ² = {chi2_temp - chi2_orig:.1f}")
```

### Step 7: Save Template

```python
# Save as ASCII text
np.savetxt('narrow_template.txt',
           np.column_stack([velc_temp, flux_temp]),
           header='velocity_kms normalized_flux')
```

---

## Quality Checks

### Checklist

Before proceeding to Stage 2, verify:

- [ ] **Plot created with `plot.plot_fit()`** showing the [S II] fitting result
- [ ] **Template profile plot** saved showing the normalized profile shape
- [ ] Peak at v ≈ 0 km/s (|v_peak| < 10 km/s)
- [ ] FWHM < 800 km/s (can be up to 800 if wing components present)
- [ ] Profile is smooth (no wiggles or artifacts)
- [ ] Verification fit χ² similar to original (|Δχ²| < 15, if performed)
- [ ] [S II] 6716/6731 ratio reasonable (0.5-1.5, for doublet case)

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Peak not at 0** | peak_vel > 10 km/s | Check that `gen_template()` was used (not manual evaluation) |
| **FWHM too large** | FWHM > 800 km/s | Check if broad line contamination present |
| **χ² mismatch** | |Δχ²| > 15 | Check template was generated correctly |
| **No plot created** | Missing visualization | Always use `sagan.plot.plot_fit()` to verify fit quality |
| **Template not used** | Doublet appears in template | Ensure using `gen_template()`, not evaluating doublet model |

---

## Using the Template

### Loading the Template

```python
# Load template from ASCII file
velc_temp, flux_temp = np.loadtxt('narrow_template.txt', unpack=True)
```

### Applying to Narrow Lines

```python
# All narrow lines use the SAME template
nha = sagan.Line_template(
    template_velc=velc_temp,    # ← Same for all
    template_flux=flux_temp,    # ← Same for all
    amplitude=50.0,             # ← Different for each
    dv=10,                      # ← Initial guess (will be fitted)
    wavec=line_wave_dict['Halpha'],
    name='nHalpha'
)

no3 = sagan.Line_template(
    template_velc=velc_temp,    # ← Same
    template_flux=flux_temp,    # ← Same
    amplitude=44.0,
    dv=10,
    wavec=line_wave_dict['OIII_5007'],
    name='OIII_5007'
)

nn2 = sagan.Line_template(
    template_velc=velc_temp,    # ← Same
    template_flux=flux_temp,    # ← Same
    amplitude=9.0,
    dv=10,
    wavec=line_wave_dict['NII_6583'],
    name='NII_6583'
)
```

### Tying Velocities

**Critical**: Tie all narrow line velocities to ONE parameter:

```python
# Tie all to nHalpha (skip nHalpha itself!)
narrow_lines = ['OIII_5007', 'NII_6583', 'NII_6548', 'SII_6716', 'SII_6731']
for ln in narrow_lines:
    model[ln].dv.tied = sagan.tie_template_dv('nHalpha')

# Result: Only 1 free dv parameter for ALL narrow lines
```

### Don't Convolve the Template

**IMPORTANT**: The template already includes instrumental broadening. Do NOT convolve it.

```python
# ✗ WRONG: Don't do this!
narrow = sagan.Line_template(...)
narrow_conv = sagan.convolve_lsf(narrow, ...)  # ← WRONG!

# ✓ CORRECT: Only convolve broad lines
broad = sagan.Line_MultiGauss(...)
broad_conv = sagan.convolve_lsf(broad, ...)
model = cont + broad_conv + narrow  # ← OK
```

---

## Troubleshooting

### "Template peak not at 0 km/s"

**Cause**: Template was generated incorrectly (e.g., by evaluating full doublet model)

**Solution**:
```python
# ✗ WRONG
flux_temp = sii_doublet(wave_eval)  # Contains both lines!

# ✓ CORRECT
flux_temp = sii_doublet_fit.gen_template(velc_temp, normalized=True)
```

### "Verification fit gives wrong [S II] 6731 amplitude"

**Cause**: Template contains doublet instead of single line

**Solution**: Ensure using `gen_template()` which extracts single-line profile

### "χ² much worse than original fit"

**Possible causes**:
1. Template was evaluated at wrong wavelengths
2. Wrong `dv` value used in verification fit
3. Template file corrupted

**Solution**: Regenerate template using `gen_template()` method

### "S/N too low for [S II]"

**Solution**: Use fixed-width Gaussian based on instrumental resolution

```python
# For SDSS with R ≈ 2000
lsf_sigma = ls_km / (2000 * 2.3548)  # ~64 km/s

velc_temp = np.linspace(-500, 500, 1000)
flux_temp = np.exp(-0.5 * (velc_temp / lsf_sigma)**2)
flux_temp = flux_temp / np.max(flux_temp)
```

---

## Complete Example

```python
#!/usr/bin/env python
"""
Extract narrow line template from [S II] doublet
Target: J094248.75-000240.3 (z = 0.0978)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
import sagan
from sagan import plot  # Import plot module
from sagan.utils import line_wave_dict, calculate_bic
from sagan.continuum import WindowedPowerLaw1D

# ========================================
# Load data
# ========================================
data = np.load('spectrum_data.npz')
wave = data['wave']
flux = data['flux']
ferr = data['ferr']
z = data['z']

# Define [S II] region
s2_region = (wave > 6700) & (wave < 6745)
wave_s2 = wave[s2_region]
flux_s2 = flux[s2_region]
ferr_s2 = ferr[s2_region]

# ========================================
# Build model (2 components)
# ========================================
cont_s2 = WindowedPowerLaw1D(
    amplitude=15.3, x_0=6720.0, alpha=9.8,
    x_min=6700, x_max=6745, name='Cont_SII'
)

sii_doublet = sagan.Line_MultiGauss_doublet(
    n_components=2,
    amp_c0=11.1, amp_c1=9.6,
    dv_c=140.0, sigma_c=124.0,
    amp_w0=0.37, dv_w0=-380.0, sigma_w0=74.0,
    wavec0=line_wave_dict['SII_6716'],
    wavec1=line_wave_dict['SII_6731'],
    name='SII_doublet'
)

model_init = cont_s2 + sii_doublet

# ========================================
# Fit
# ========================================
fitter = fitting.LevMarLSQFitter()
model_fit = fitter(model_init, wave_s2, flux_s2,
                   weights=1/ferr_s2**2, maxiter=10000)

print('[S II] fit completed')
print(f"  dv_c = {model_fit.dv_c_1.value:.2f} km/s")
print(f"  sigma_c = {model_fit.sigma_c_1.value:.2f} km/s")

# ========================================
# REQUIRED: Plot fitting result with plot.plot_fit()
# ========================================
ax = plot.plot_fit(wave_s2, flux_s2, model_fit, ferr_s2)
ax.set_title('[S II] Region - Template Generation Fit')
ax.axvline(line_wave_dict['SII_6716'], color='r', linestyle='--', alpha=0.3, label='[S II] 6716')
ax.axvline(line_wave_dict['SII_6731'], color='b', linestyle='--', alpha=0.3, label='[S II] 6731')
ax.legend()
plt.savefig('sii_fitting_result.png', dpi=150)
print("✓ Plot saved to: sii_fitting_result.png")
plt.close()

# ========================================
# Generate template
# ========================================
sii_doublet_fit = model_fit[1]  # Get doublet component
velc_temp = np.linspace(-800, 800, 2000)
flux_temp = sii_doublet_fit.gen_template(velc_temp, normalized=True)

print('\nTemplate generated:')
print(f'  Velocity range: {velc_temp[0]:.0f} to {velc_temp[-1]:.0f} km/s')
print(f'  Peak position: {velc_temp[np.argmax(flux_temp)]:.2f} km/s')

# ========================================
# REQUIRED: Plot template profile
# ========================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(velc_temp, flux_temp, 'k-', lw=2)
ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Center (v=0)')
ax.axhline(0.5, color='b', linestyle=':', alpha=0.5, label='Half max')
ax.axhline(1.0, color='g', linestyle=':', alpha=0.5, label='Peak')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Narrow Line Template Profile')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('narrow_template_profile.png', dpi=150)
print("✓ Template profile plot saved to: narrow_template_profile.png")
plt.close()

# ========================================
# Quality checks
# ========================================
# Check peak position
peak_idx = np.argmax(flux_temp)
peak_vel = velc_temp[peak_idx]
assert abs(peak_vel) < 10, f"Peak not at 0! Peak at {peak_vel:.2f} km/s"
print(f'\n✓ Peak centered at: {peak_vel:.2f} km/s')

# Measure FWHM
from scipy.interpolate import interp1d
f_interp = interp1d(velc_temp, flux_temp, kind='cubic',
                    bounds_error=False, fill_value=0)
velc_fine = np.linspace(velc_temp[0], velc_temp[-1], 10000)
flux_fine = f_interp(velc_fine)
crossings = np.where(np.diff(np.sign(flux_fine - 0.5)))[0]
if len(crossings) >= 2:
    fwhm = velc_fine[crossings[-1]] - velc_fine[crossings[0]]
    print(f"✓ Template FWHM: {fwhm:.1f} km/s")
    assert fwhm < 800, f"FWHM {fwhm:.1f} too large!"

# ========================================
# Save template
# ========================================
np.savetxt('narrow_template.txt',
           np.column_stack([velc_temp, flux_temp]),
           header='velocity_kms normalized_flux')

print('\n'+'='*60)
print('✓ Template generation complete!')
print('='*60)
print('Files created:')
print('  - sii_fitting_result.png (fit visualization)')
print('  - narrow_template_profile.png (template shape)')
print('  - narrow_template.txt (template data)')
print('='*60)
```

---

## References

- **Type 1 AGN Fitting Strategy**: `type1_agn.md`
- **Function Reference**: `../function_reference/` (split by module)
- **Key Points Summary**: `KEY_POINTS_TEMPLATE_GENERATION.md`
- **SAGAN Source**: `sagan/line_profile.py:432` (`gen_template()` implementation)
- **Plotting Guide**: `NARROW_LINE_TEMPLATE_PLOTTING_GUIDE.md` (Standard plotting formats)
- **Reusable Plotting Functions**: `sagan/plot.py` (`plot_narrow_line_diagnostic()`, `plot_narrow_line_template_validation()`)
- **Example Script**: `example/narrow_line_template_template.py` (Template script for users)
