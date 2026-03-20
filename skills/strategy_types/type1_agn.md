# Type 1 AGN Spectral Fitting Strategy

## Overview

This guide provides a detailed strategy for fitting Type 1 AGN and BAL QSO spectra using SAGAN. The approach is based on the methodology from Shangguan et al. (2026), developed for studying low-redshift broad absorption line quasars.

### Scientific Context (Shangguan et al. 2026)

SAGAN was developed for the study of low-redshift BAL QSOs with the following scientific goals:

1. **BAL Outflows**: Analysis of broad absorption lines to study quasar outflow kinematics
2. **Black Hole Masses**: Measurement from broad line widths (Hα, Hβ)
3. **Emission Line Decomposition**: Separate broad and narrow components
4. **Physical Parameters**: Optical depth, covering fraction, velocity shifts

### Characteristic Features of Type 1 AGN

**Emission Lines**:
- **Broad permitted lines**: Hα (6563 Å), Hβ (4861 Å), Hγ (4340 Å) with FWHM 1000-10000 km/s
- **Narrow forbidden lines**: [O III] 4959,5007, [S II] 6716,6731, [N II] 6548,6583 with FWHM 100-500 km/s
- **Semi-forbidden lines**: [O III] 4363, He II 4686
- **Other lines**: He I 5876, O I 6300,6364

**Absorption Features** (BAL QSOs only):
- Broad absorption troughs blueward of emission lines
- Typical velocity range: -1000 to -30000 km/s
- Multiple absorption components possible

**Continuum**:
- Power-law continuum: F_ν ∝ ν^α with α ≈ -0.5 to -1.5
- Iron emission (Fe II) blends, especially around Hβ
- Optional stellar continuum from host galaxy

## Table of Contents

1. [Two-Stage Fitting Strategy](#two-stage-fitting-strategy) ⭐ **READ THIS FIRST**
2. [CRITICAL: Weak Narrow Lines](#critical-weak-narrow-lines) ⭐ **NEW**
3. [Iterative Model Building](#iterative-model-building-start-simple-add-complexity)
4. [Data Preparation](#data-preparation)
5. [Stage 1: Create Narrow Line Template](#stage-1-create-narrow-line-template)
6. [Stage 2: Fit Broad Line Complexes](#stage-2-fit-broad-line-complexes)
7. [Component Selection Guidelines](#component-selection-guidelines)
8. [Parameter Tying Patterns](#parameter-tying-patterns)
9. [MCMC Fitting Strategy](#mcmc-fitting-strategy)
10. [Physical Measurements](#physical-measurements)
11. [Complete Example](#complete-example)

## Two-Stage Fitting Strategy ⭐ **READ THIS FIRST**

**Critical Principle**: Fit narrow lines **first**, generate a template, then use that template for all subsequent fitting. This dramatically reduces free parameters and prevents degeneracies between broad and narrow components.

### Why This Strategy Matters

When fitting Type 1 AGN spectra, the narrow forbidden lines (FWHM ~100-500 km/s) and broad permitted lines (FWHM ~1000-10000 km/s) are blended. If you fit both simultaneously with independent profiles, you introduce strong degeneracies:

- **Problem**: Broad line wings can masquerade as narrow lines
- **Problem**: Narrow lines can be over-subtracted, contaminating broad line measurements
- **Problem**: Too many free parameters lead to unconstrained fits

**Solution**: Use a **single, empirically-determined narrow line profile** for all narrow lines.

### The Two Stages

#### Stage 1: Create Narrow Line Template
**Goal**: Determine the intrinsic narrow line profile shape

1. **Choose a clean narrow line region**:
   - **Option 1 (Preferred)**: [S II] λλ6716,6731 doublet
     - Isolated from strong broad lines
     - No iron contamination
     - Doublet helps constrain profile shape
   - **Option 2**: [O III] λλ4959,5007 doublet
     - Use only if [S II] is unavailable or low S/N
     - May have blue wing that complicates template extraction
     - Can mask the blue wing region if needed

2. **Fit with Gaussian components**:
   - Use as many Gaussians as needed to capture the profile
   - Do NOT use `Line_template` yet — use individual `Line_Gaussian` or `Line_MultiGauss` components
   - Example: 2-3 Gaussians for [S II] doublet

3. **Save the template**:
   - Generate the best-fit model profile
   - Save to file (velocity + normalized flux)
   - This template will be used for **all** subsequent narrow line fits

#### Stage 2: Fit Broad Line Complexes
**Goal**: Decompose broad and narrow components using the template

1. **Apply the template to ALL narrow lines**:
   - Narrow Hα, Hβ, Hγ, [O III], [N II], [S II], [O I], He II, etc.
   - All use the **same** `template_velc` and `template_flux` from Stage 1
   - Only amplitude varies between lines

2. **Tie narrow line velocities**:
   - All narrow lines share the **same** `dv` parameter
   - Typically: tie all to `nHalpha.dv`
   - This reduces N narrow line velocity parameters to **1** free parameter

3. **Fit broad lines independently**:
   - Broad Hα, Hβ, Hγ with `Line_MultiGauss`

## CRITICAL: Weak Narrow Lines ⭐ **NEW**

### The Problem

**When [S II] or [O III] lines have low S/N (<20), creating an empirical template from those lines will FAIL.**

**Symptoms**:
- Fitted sigma hits upper bounds (e.g., 9000 km/s)
- Template looks noisy or unphysical
- Subsequent fitting produces poor results

**Real Example**:
```
Spectrum: DESI 39627747915469565
[S II] 6716 S/N ~6 (too weak!)
Fitted sigma: 9009 km/s → hitting bound
Result: Template unusable
```

### The Solution: Fixed-Width Gaussian Template

When narrow lines are too weak for empirical template extraction, use a **fixed-width Gaussian based on instrumental resolution**:

```python
# Calculate instrumental LSF width
lsf_sigma = ls_km / (resolving_power * 2.3548)  # ~64 km/s for R=2000

# Create fixed-width Gaussian template
velc_temp = np.linspace(-500, 500, 1000)  # km/s
flux_temp = np.exp(-0.5 * (velc_temp / lsf_sigma)**2)
flux_temp = flux_temp / np.max(flux_temp)

# Use this for ALL narrow lines
nha = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=estimated_amp,
    dv=0,
    wavec=line_wave_dict['Halpha'],
    name='nHalpha'
)
```

### Decision Flowchart

```
Check S/N of narrow line region
    │
    ├─ S/N > 20?
    │   │
    │   ├─ YES → Use empirical template (see Stage 1)
    │   │         Fit [S II] or [O III] with Gaussians
    │   │         Generate template from fit
    │   │
    │   └─ NO → Use fixed-width Gaussian
    │             Based on instrumental resolution
    │             sigma = c / (R * 2.3548)
    │
    └─ Proceed to Stage 2 with template
```

### How to Check S/N

```python
# In [S II] region
s2_region = (wave_rest > 6700) & (wave_rest < 6745)
flux_s2 = flux[s2_region]

# Continuum estimate
cont_mask = ((wave_rest > 6700) & (wave_rest < 6710)) | \
            ((wave_rest > 6735) & (wave_rest < 6745))
cont_level = np.median(flux_s2[cont_mask])
noise = np.std(flux_s2[cont_mask])

# S/N calculation
peak_flux = np.max(flux_s2)
sn_ratio = (peak_flux - cont_level) / noise

print(f"[S II] S/N: {sn_ratio:.1f}")

if sn_ratio < 20:
    print("WARNING: S/N too low for empirical template")
    print("         Use fixed-width Gaussian instead")
```

### When to Use Each Approach

| Scenario | Template Type | When to Use |
|----------|---------------|-------------|
| **High-quality narrow lines** | Empirical from [S II] or [O III] | S/N > 20, clear detection |
| **Weak narrow lines** | Fixed-width Gaussian | S/N < 20, or absent |
| **No narrow lines** | Fixed-width Gaussian | Type 1 AGN with only broad lines |

### Convolution Rules with Templates

**IMPORTANT**: The convolution rules differ for empirical vs fixed-width templates:

```python
# For EMPIRICAL templates (from Stage 1):
# Already includes instrumental broadening - DO NOT convolve
narrow = sagan.Line_template(template_velc=velc_temp, template_flux=flux_temp, ...)
broad_conv = sagan.convolve_lsf(broad, ...)  # Convolve only broad
model = cont + broad_conv + narrow  # Narrow NOT convolved

# For FIXED-WIDTH Gaussian templates:
# Based on instrumental width - still DO NOT convolve
# (the Gaussian already represents the instrumental profile)
```

**Key Point**: In both cases, **do NOT convolve the narrow line template**. The template already represents the line profile as it appears in the data (including instrumental broadening).
   - No degeneracy with narrow lines (profile shape is fixed)

### Parameter Reduction Example

For a typical Hα region with narrow Hα, [N II] doublet, [S II] doublet, [O I] doublet:

| Approach | Narrow Line Free Parameters |
|----------|----------------------------|
| **Without template** | ~7 lines × (amp + vel + width) = **21+ parameters** |
| **With template + tied dv** | ~7 amplitudes + **1 velocity** = **8 parameters** |

**This is a >60% reduction in free parameters**, leading to:
- More robust fits
- Better constrained uncertainties
- Faster MCMC convergence
- Physically meaningful results

### Visual Overview

```
STAGE 1: Extract Narrow Line Template
┌─────────────────────────────────────────────────────────┐
│  Fit [S II] or [O III] with multiple Gaussians        │
│  ─────────────────────────────────────────             │
│  → Save best-fit profile as template                   │
│  → Template contains: velc array + flux array         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
STAGE 2: Fit Broad Line Complexes
┌─────────────────────────────────────────────────────────┐
│  Hα region:                                             │
│  ├─ Broad Hα: Line_MultiGauss (3 components)          │
│  ├─ Narrow Hα: Line_template ────────┐                │
│  ├─ [N II] 6583: Line_template ───────┤                │
│  ├─ [N II] 6548: Line_template ───────┤                │
│  ├─ [S II] 6716: Line_template ───────┤ All use        │
│  ├─ [S II] 6731: Line_template ───────┤ SAME template  │
│  ├─ [O I] 6300: Line_template ────────┤ from Stage 1  │
│  └─ [O I] 6364: Line_template ────────┘                │
│                                                          │
│  Tying:                                                 │
│  └─ All narrow lines dv tied to nHalpha.dv            │
│     (1 velocity parameter for ALL narrow lines)        │
└─────────────────────────────────────────────────────────┘
```

### Common Mistakes to Avoid

❌ **Wrong**: Fit broad and narrow lines simultaneously without a template
❌ **Wrong**: Use different templates for different narrow lines
❌ **Wrong**: Let each narrow line have its own velocity parameter
❌ **Wrong**: Use `Line_Gaussian` for narrow lines in Stage 2
❌ **Wrong**: Start with a complex model with many components

✅ **Correct**: Extract template once from [S II] or [O III], reuse for everything
✅ **Correct**: Tie all narrow line velocities together
✅ **Correct**: Use `Line_template` for all narrow lines in Stage 2
✅ **Correct**: Start with a simple model, gradually add components

## Iterative Model Building: Start Simple, Add Complexity

**Critical Principle**: Always start with the simplest model that captures the basic features, then gradually increase complexity. This prevents the fit from getting stuck in local minima and ensures each component is necessary.

### Why Start Simple?

1. **Avoid local minima**: Complex models have many degeneracies
2. **Verify component necessity**: Each addition should improve the fit significantly
3. **Faster convergence**: Simple models fit faster and more reliably
4. **Better initial guesses**: Each stage provides good starting values for the next

### Recommended Progression

#### Level 1: Absolute Minimum
```python
# Start with just continuum + 1 broad component + 1 narrow component
model = cont + bha_simple + nha

# bha_simple: 1 component only
bha_simple = sagan.Line_MultiGauss(
    n_components=1,
    amp_c=50.0,
    dv_c=0,
    sigma_c=500,
    wavec=line_wave_dict['Halpha']
)
```

#### Level 2: Add Narrow Lines
```python
# Add forbidden lines ([N II], [S II])
# Still using simple broad component (1 Gaussian)
model = cont + bha_simple + nha + nn2 + ns2
```

#### Level 3: Add Broad Line Components
```python
# Now that narrow lines are constrained, add more broad components
bha_complex = sagan.Line_MultiGauss(
    n_components=2,  # Added second component
    amp_c=50.0,
    dv_c=0,
    sigma_c=500,
    amp_w0=0.2,
    dv_w0=100,
    sigma_w0=1500,
    wavec=line_wave_dict['Halpha']
)

model = cont + bha_complex + nha + nn2 + ns2
```

#### Level 4: Add Absorption (if BAL)
```python
# Add absorption trough
aha = sagan.Line_Absorption(logtau0=1.0, dv=-100, sigma=50, Cf=0.3,
                            wavec=line_wave_dict['Halpha'])

model = (cont + bha_complex) * aha + nha + nn2 + ns2
```

#### Level 5: Full Model
```python
# Add all remaining components
model = (cont + bha_full) * aha + nha + nn2 + ns2 + no1
```

### Step-by-Step Fitting Script

```python
# Define fitting progression
fitting_stages = [
    {
        'name': 'Level 1: Continuum + simple broad + narrow Hα',
        'components': ['Cont Ha', 'Broad Halpha', 'nHalpha'],
        'description': 'Basic model to get continuum and rough line widths'
    },
    {
        'name': 'Level 2: Add [N II], [S II]',
        'components': ['Cont Ha', 'Broad Halpha', 'nHalpha', 'NII_6583', 'NII_6548', 'SII_6716', 'SII_6731'],
        'description': 'Constrain narrow lines with template'
    },
    {
        'name': 'Level 3: Add second broad component',
        'components': ['Cont Ha', 'Broad Halpha', 'nHalpha', 'NII_6583', 'NII_6548', 'SII_6716', 'SII_6731'],
        'modify': {'Broad Halpha': {'n_components': 2}},
        'description': 'Add intermediate-width broad component'
    },
    {
        'name': 'Level 4: Add absorption',
        'components': ['Cont Ha', 'Broad Halpha', 'nHalpha', 'NII_6583', 'NII_6548', 'SII_6716', 'SII_6731', 'Abs. Halpha'],
        'description': 'Add BAL trough if needed'
    },
    {
        'name': 'Level 5: Full model with third broad component',
        'components': ['Cont Ha', 'Broad Halpha', 'nHalpha', 'NII_6583', 'NII_6548', 'SII_6716', 'SII_6731', 'Abs. Halpha', 'OI_6300', 'OI_6364'],
        'modify': {'Broad Halpha': {'n_components': 3}},
        'description': 'Complete model with all components'
    }
]

# Iterate through stages
for i, stage in enumerate(fitting_stages):
    print(f"\n{'='*70}")
    print(f"STAGE {i+1}: {stage['name']}")
    print(f"{'='*70}")
    print(f"Description: {stage['description']}")

    # Build model for this stage
    if i == 0:
        # Build from scratch
        model = build_model(stage['components'])
    else:
        # Use previous model as starting point
        model = previous_model
        if 'modify' in stage:
            # Modify components as needed
            for comp_name, mods in stage['modify'].items():
                # Rebuild component with new parameters
                ...

    # Fix unused parameters
    for name in model.submodel_names:
        if name not in stage['components']:
            for param in model[name].param_names:
                setattr(model[name], param, 'fixed')

    # Fit
    fitter = fitting.LevMarLSQFitter()
    model_fit = fitter(model, wave, flux, weights=weight, maxiter=10000)

    # Evaluate fit quality
    chi2 = np.sum(((flux - model_fit(wave)) / ferr)**2) / len(flux)
    print(f"χ²/ν = {chi2:.3f}")

    # Visual check
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(wave, flux, where='mid', color='k', alpha=0.5, label='Data')
    ax.plot(wave, model_fit(wave), color='C3', lw=2, label='Fit')
    ax.set_title(stage['name'])
    plt.legend()
    plt.show()

    # Ask user if this stage is good
    response = input("Continue to next stage? [y/n]: ")
    if response.lower() != 'y':
        print("Stopping at this stage.")
        break

    # Save for next iteration
    previous_model = model_fit

print(f"\nFinal model: {fitting_stages[i]['name']}")
```

### When to Add More Components

**Add another component if**:
- Residuals show systematic structure (not just noise)
- χ² decreases significantly (>10% improvement)
- The new component has a physically interpretable purpose

**Do NOT add if**:
- Fit is already good (χ² ≈ 1)
- New component amplitude is consistent with zero
- Residuals look like random noise

### Decision Flowchart

```
Start: Continuum + 1 broad Gaussian + 1 narrow template
    │
    ├─ Fit quality: χ² >> 1?
    │   │
    │   ├─ YES → Check residuals
    │   │   │
    │   │   ├─ Broad line wings poorly fit?
    │   │   │   → Add second broad component
    │   │   │
    │   │   ├─ Narrow lines present but not fit?
    │   │   │   → Add more narrow lines ([N II], [S II])
    │   │   │
    │   │   └─ Absorption trough visible?
    │   │       → Add absorption component
    │   │
    │   └─ NO → Fit is acceptable
    │       → Stop (or proceed to MCMC)
    │
    └─ Re-fit with new model
        └─ Repeat evaluation
```

### Practical Example: Hα Region

```python
# ===== STEP 1 =====
# Simplest model
model1 = cont + bha_1comp + nha
m1 = fitter(model1, wave, flux, weights=weight)
# Check: Is χ² reasonable? Are residuals random?
# If not, proceed...

# ===== STEP 2 =====
# Add forbidden lines
model2 = cont + bha_1comp + nha + nn2 + ns2
m2 = fitter(model2, wave, flux, weights=weight)
# Better fit? χ² improved significantly?

# ===== STEP 3 =====
# Add second broad component (if needed)
model3 = cont + bha_2comp + nha + nn2 + ns2
m3 = fitter(model3, wave, flux, weights=weight)
# Do we really need this third component?

# ===== STEP 4 =====
# Add absorption (if BAL QSO)
model4 = (cont + bha_2comp) * aha + nha + nn2 + ns2
m4 = fitter(model4, wave, flux, weights=weight)

# ===== STEP 5 =====
# Proceed to MCMC with the final model
```

### Key Takeaways

1. **Never start with the full model** - you'll get stuck in local minima
2. **Fit one component at a time** - verify each addition is necessary
3. **Use previous fit as initial guess** - warm start is better than cold start
4. **Check residuals at each step** - they tell you what's missing
5. **Know when to stop** - more components ≠ better fit

1. [Two-Stage Fitting Strategy](#two-stage-fitting-strategy) ⭐ **READ THIS FIRST**
2. [Data Preparation](#data-preparation)
3. [Stage 1: Create Narrow Line Template](#stage-1-create-narrow-line-template)
4. [Stage 2: Fit Broad Line Complexes](#stage-2-fit-broad-line-complexes)
5. [Component Selection Guidelines](#component-selection-guidelines)
6. [Parameter Tying Patterns](#parameter-tying-patterns)
7. [MCMC Fitting Strategy](#mcmc-fitting-strategy)
8. [Physical Measurements](#physical-measurements)
9. [Complete Example](#complete-example)

## Data Preparation

### 1. Initial Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import sagan
from sagan.utils import line_wave_dict, line_label_dict
from extinction import ccm89, remove

# Constants
ls_km = 2.99792e5  # Speed of light in km/s
resolving_power = 1800  # SDSS spectrograph resolution

# Load target information
from astropy.table import Table
tb = Table.read('target_info.ipac', format='ipac')
targname = tb['Name'][0]
zred = tb['zred'][0]
Av = tb['Av'][0]

# Load spectrum
spec = np.loadtxt('spectrum.txt')
wave_obs = spec[:, 0]
flux_obs = spec[:, 1]
ferr_obs = spec[:, 2]
```

### 2. Redshift Verification

Use narrow forbidden lines to verify or refine the redshift:

```python
# Get narrow line regions
wave_rest = wave_obs / (1 + zred)

# Check [O III] 5007 position
from scipy.signal import find_peaks
o3_region = (wave_rest > 5000) & (wave_rest < 5020)
peaks, _ = find_peaks(flux_obs[o3_region], height=np.max(flux_obs[o3_region])*0.5)

if len(peaks) > 0:
    peak_wave = wave_rest[o3_region][peaks[0]]
    dv_measured = (peak_wave - line_wave_dict['OIII_5007']) / line_wave_dict['OIII_5007'] * ls_km
    print(f"[O III] velocity shift: {dv_measured:.1f} km/s")
    # Small dv corrections are acceptable (< 100 km/s)
```

### 3. Define Fitting Windows

Based on the J0925+6409 example, use these wavelength windows:

```python
# Hα region: 6100-7000 Å (rest)
# Hβ + Hγ region: 4200-5400 Å (rest)
# He I region: 5400-6100 Å (rest)

line_windows = [(4200, 5400), (5400, 6100), (6100, 7000)]

# Create weight array
weight_lines = np.zeros_like(wave_rest)
for window in line_windows:
    weight_lines[(wave_rest >= window[0]) & (wave_rest <= window[1])] = 1

# Mask bad pixels
weight_lines[(wave_rest > 5398) & (wave_rest < 5403)] = 0  # Detector defect
weight_lines[(wave_rest > 6097) & (wave_rest < 6103)] = 0  # Detector defect
```

## Stage 1: Create Narrow Line Template

**Goal**: Extract the intrinsic narrow line profile shape from a clean, isolated line region.

### Choose Your Template Source

#### Option 1: [S II] λλ6716,6731 Doublet (Preferred)

**Advantages**:
- Isolated from strong broad lines
- No iron contamination
- Doublet provides strong constraints on profile shape
- Clean continuum regions on both sides

```python
# Define [S II] region
s2_region = (wave_rest > 6700) & (wave_rest < 6745)

# Fit continuum
from astropy.modeling import models, fitting
cont_s2 = models.Polynomial1D(degree=1)
fitter = fitting.LevMarLSQFitter()

# Continuum-only regions (avoid emission lines)
cont_mask = ((wave_rest > 6700) & (wave_rest < 6710)) | \
            ((wave_rest > 6735) & (wave_rest < 6745))

cont_fit = fitter(cont_s2, wave_rest[s2_region][cont_mask[s2_region]],
                  flux_obs[s2_region][cont_mask[s2_region]])

# Normalize spectrum
flux_norm = flux_obs[s2_region] / cont_fit(wave_rest[s2_region])

# Convert to velocity space (relative to [S II] 6716)
velc_sii = (wave_rest[s2_region] - line_wave_dict['SII_6716']) / \
           line_wave_dict['SII_6716'] * ls_km
flux_sii = flux_norm / np.max(flux_norm)
```

#### Option 2: [O III] λλ4959,5007 Doublet

**Use when**: [S II] region has low S/N, is contaminated, or is unavailable

**Caution**: [O III] often has a blue wing component. You may need to mask it.

```python
# Define [O III] region
o3_region = (wave_rest > 4940) & (wave_rest < 5020)

# Mask blue wing if present (typically -200 to -1000 km/s)
o3_mask = (velc_o3 > -300) & (velc_o3 < 200)  # Keep only core

# Continue with continuum normalization as above...
```

### Fit the Template with Gaussians

**BEST PRACTICE**: For doublets, use `Line_MultiGauss_doublet` to tie kinematics:

```python
# RECOMMENDED: Use Line_MultiGauss_doublet for [S II]
sii_doublet = sagan.Line_MultiGauss_doublet(
    n_components=1,
    amp_c0=8.0,     # [S II] 6716 amplitude
    amp_c1=6.0,     # [S II] 6731 amplitude
    dv_c=0,         # Shared velocity shift
    sigma_c=100,    # Shared width (km/s)
    wavec0=line_wave_dict['SII_6716'],
    wavec1=line_wave_dict['SII_6731'],
    name='SII_doublet'
)

# Add local continuum
cont_local = models.Polynomial1D(degree=1, name='cont_local')

# Combine
model_init = cont_local + sii_doublet

# Fit
fitter = fitting.LevMarLSQFitter()
model_fit = fitter(model_init, wave_rest[s2_region], flux_obs[s2_region],
                   weights=1/ferr_obs[s2_region]**2, maxiter=10000)

# Visual check
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.step(velc_sii, flux_sii, where='mid', color='k', label='Data')
ax.plot(velc_sii, model_fit(wave_rest[s2_region])/np.max(flux_sii),
        color='C3', lw=2, label='Fit')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Normalized Flux')
ax.legend()
plt.show()
```

**Why use Line_MultiGauss_doublet?**
- Both lines share the **same velocity shift** (`dv_c`)
- Both lines share the **same width** (`sigma_c`)
- Prevents degeneracies between the two components
- More robust than fitting separate Gaussians

**Alternative: Multiple components**
If the profile requires more than one component per line:

```python
sii_doublet = sagan.Line_MultiGauss_doublet(
    n_components=2,  # Core + wing
    amp_c0=8.0,     # Core: [S II] 6716
    amp_c1=6.0,     # Core: [S II] 6731
    dv_c=0,
    sigma_c=100,
    amp_w0=0.2,     # Wing relative amplitude
    dv_w0=50,       # Wing velocity offset
    sigma_w0=200,   # Wing width
    wavec0=line_wave_dict['SII_6716'],
    wavec1=line_wave_dict['SII_6731'],
    name='SII_doublet'
)
```

### Generate and Save the Template

**⚠️ CRITICAL: Template must ALWAYS be centered at dv=0**

The measured dv from the [S II] fit indicates **redshift error**, not the line profile shape. The template should only capture the **profile shape**.

```python
# Get fitted parameters
amp0 = model_sii_fit['SII_doublet'].amp_c0.value
amp1 = model_sii_fit['SII_doublet'].amp_c1.value
dv_measured = model_sii_fit['SII_doublet'].dv_c.value  # ← Store this!
sig = model_sii_fit['SII_doublet'].sigma_c.value

# IMPORTANT: dv_measured indicates redshift error
print(f"Measured dv = {dv_measured:.0f} km/s")
print(f"This indicates redshift error: Δz = {-dv_measured/ls_km:.6f}")
print(f"Corrected redshift: z = {z - dv_measured/ls_km:.6f}")

# Generate velocity array
velc_temp = np.linspace(-800, 800, 2000)  # km/s

# Generate template with dv=0 (NOT dv_measured!)
flux_6716 = amp0 * np.exp(-0.5 * ((velc_temp - 0) / sig)**2)  # ← Use dv=0!
flux_6731 = amp1 * np.exp(-0.5 * ((velc_temp - 0) / sig)**2)  # ← Use dv=0!

flux_temp = (flux_6716 + flux_6731)
flux_temp = flux_temp / np.max(flux_temp)

# Save template
np.savetxt('narrow_template.txt', np.column_stack([velc_temp, flux_temp]),
           header='velocity_kms normalized_flux')
```

**Why dv=0?**
- Template captures **line profile shape only**
- Any velocity shift is due to redshift inaccuracy
- Use `dv` parameter in `Line_template` to apply velocity shift during fitting
- If template itself is shifted, you can't distinguish redshift error from real kinematics

**See Also**: `typical_bugs.md` for more on parameter tying issues

### Generate and Save the Template

Once satisfied with the fit, extract the profile and save:

```python
# Generate velocity array for template
velc_temp = np.linspace(-1000, 1000, 2000)  # km/s

# Create template from the best-fit narrow lines only
# (excluding continuum)
flux_6716 = model_fit['SII_6716_init'](velc_temp, wavec=line_wave_dict['SII_6716'])
flux_6731 = model_fit['SII_6731_init'](velc_temp, wavec=line_wave_dict['SII_6731'])
flux_temp = (flux_6716 + flux_6731) / np.max(flux_6716 + flux_6731)

# Save to file
template_file = f'{targname}_narrow_template.txt'
np.savetxt(template_file, np.column_stack([velc_temp, flux_temp]),
           header='velocity_kms normalized_flux')

print(f"Template saved to: {template_file}")
print(f"Template range: {velc_temp[0]:.0f} to {velc_temp[-1]:.0f} km/s")
print(f"Template peak flux: {np.max(flux_temp):.3f}")
```

### Quality Checks

Before proceeding, verify:

1. **Profile shape looks smooth**: No wiggles or artifacts
2. **Continuum is properly subtracted**: Profile goes to zero at edges
3. **Doublet ratio is reasonable**: [S II] 6716/6731 should be 0.5-1.5
4. **Velocity width is realistic**: FWHM should be 100-500 km/s

```python
# Measure template FWHM
from scipy.interpolate import interp1d
f_interp = interp1d(velc_temp, flux_temp, kind='cubic',
                    bounds_error=False, fill_value=0)

# Find half-max
half_max = 0.5
velc_fine = np.linspace(velc_temp[0], velc_temp[-1], 10000)
flux_fine = f_interp(velc_fine)

crossings = np.where(np.diff(np.sign(flux_fine - half_max)))[0]
if len(crossings) >= 2:
    fwhm = velc_fine[crossings[-1]] - velc_fine[crossings[0]]
    print(f"Template FWHM: {fwhm:.1f} km/s")
    if fwhm < 50 or fwhm > 800:
        print("WARNING: FWHM outside expected range for narrow lines!")
```

## Stage 2: Fit Broad Line Complexes

**Goal**: Decompose broad and narrow components in Hα and Hβ regions using the template from Stage 1.

### ⚠️ CRITICAL: Include ALL Components From The Start

**Do NOT add components iteratively!**

Build the complete model with:
- Continuum
- **ALL** narrow lines (at once)
- Broad line (start with 1 component)

**Then** check residuals to decide if broad line needs more components.

**Why?**
- Narrow lines are well-constrained by template + fixed ratios
- Tying velocities reduces degeneracy
- **Only broad line complexity is uncertain**
- Adding narrow iteratively doesn't help and wastes time

```python
# ✅ CORRECT: Build complete model
continuum = Polynomial1D(...)
broad_ha = Line_MultiGauss(n_components=1, ...)
nha = Line_template(...)  # Narrow Hα
nii_6583 = Line_template(...)  # [N II]
nii_6548 = Line_template(...)  # [N II]
model = continuum + broad_ha + nha + nii_6583 + nii_6548

# Tie parameters (skip reference parameter!)
nii_6548.amplitude.tied = tie_template_amplitude('NII_6583', ratio=2.96)
for ln in ['NII_6583', 'NII_6548']:  # ← Skip 'nHalpha'!
    model[ln].dv.tied = tie_template_dv('nHalpha')

# Fit complete model
model_fit = fitter(model, wave, flux, weights=weight)

# Check residuals → decide if broad needs 2nd component
# NOT: check residuals → decide if need [N II] (already included!)
```

**❌ WRONG: Adding components iteratively**
```python
# Bad approach - adds narrow lines one by one
model = continuum + broad_ha
model = fit(model, ...)
model = model + nha  # ← Don't do this!
model = fit(model, ...)
model = model + nii_6583  # ← Don't do this!
```

**See Also**: `typical_bugs.md` - Bug #1: Tying a Parameter to Itself

### ⚠️ Important Note on dv Parameter

When using `Line_template` with the narrow line template:

- **Template is centered at v=0** (from Stage 1)
- **dv parameter is FREE** (allows velocity shift of all narrow lines)
- **Tie all narrow lines** to ONE dv parameter (reduces degeneracy)
- **Template shape stays fixed** (only dv varies, not the profile)

```python
# All narrow lines share ONE dv parameter
nha = sagan.Line_template(template_velc=velc_temp, template_flux=flux_temp,
                          amplitude=estimate, dv=0,  # ← Initial guess, FREE to vary!
                          wavec=line_wave_dict['Halpha'], name='nHalpha')

nii_6583 = sagan.Line_template(template_velc=velc_temp, template_flux=flux_temp,
                             amplitude=estimate, dv=0,
                             wavec=line_wave_dict['NII_6583'], name='NII_6583')

# Tie [N II] to narrow Hα (skip nHalpha in the loop!)
for ln in ['NII_6583', 'NII_6548']:
    model[ln].dv.tied = sagan.tie_template_dv('nHalpha')

# Result: ONE free dv parameter for all narrow lines
# Fitted value tells us the narrow line velocity shift
```

**Common mistake**: Don't worry if dv ≠ 0 after fitting! This is correct - it's the actual velocity shift of the narrow lines. The template shape stays centered at v=0.

### Stage 2A: Hα Region (6100-7000 Å)

**Components to fit**:
1. Continuum (Polynomial1D or WindowedPowerLaw1D)
2. **All narrow lines use the saved template** (Line_template)
3. Broad Hα (Line_MultiGauss with 2-3 components)
4. Hα absorption (Line_Absorption, if BAL)

**Continuum Choice**:
- **Small regions (<100 Å)**: Use `Polynomial1D(degree=1)`
  - More numerically stable
  - Sufficient for local continuum
  - Recommended for initial fitting
- **Large regions (>200 Å)**: Use `WindowedPowerLaw1D` or `PowerLaw1D`
  - Better for wide wavelength coverage
  - Physical power-law shape
  - Use for final physical modeling

#### Load the Template

```python
# Load the template created in Stage 1
template_data = np.loadtxt(template_file)
velc_temp = template_data[:, 0]
flux_temp = template_data[:, 1]

print(f"Loaded narrow line template from Stage 1")
print(f"Template shape: {velc_temp.shape}")
```

#### Build the Model

**CRITICAL**: All narrow lines use `Line_template` with the SAME `template_velc` and `template_flux`:

```python
# Continuum
cont_ha = sagan.WindowedPowerLaw1D(
    amplitude=9.0,
    x_0=6550.,
    alpha=1.5,
    x_min=6100,
    x_max=7000,
    name='Cont Ha'
)

# Broad Hα: 2-3 components (core + winds)
bha = sagan.Line_MultiGauss(
    n_components=3,
    amp_c=48.0,          # Core amplitude
    dv_c=-100,           # Core velocity shift (km/s)
    sigma_c=400,         # Core dispersion (km/s)
    amp_w0=0.15,         # Wind 0 relative amplitude
    dv_w0=20,            # Wind 0 velocity offset (km/s)
    sigma_w0=1400,       # Wind 0 dispersion (km/s)
    amp_w1=0.15,         # Wind 1 relative amplitude
    dv_w1=800,           # Wind 1 velocity offset (km/s)
    sigma_w1=200,        # Wind 1 dispersion (km/s)
    wavec=line_wave_dict['Halpha'],
    name='Broad Halpha'
)

# ===== ALL NARROW LINES USE THE SAME TEMPLATE =====
# Narrow Hα
nha = sagan.Line_template(
    template_velc=velc_temp,    # ← From Stage 1
    template_flux=flux_temp,    # ← From Stage 1
    amplitude=50.0,
    dv=10,
    wavec=line_wave_dict['Halpha'],
    name='nHalpha'
)

# [N II] doublet
nn2 = sagan.Line_template(
    template_velc=velc_temp,    # ← Same template
    template_flux=flux_temp,    # ← Same template
    amplitude=9.0,
    dv=10,
    wavec=line_wave_dict['NII_6583'],
    name='NII_6583'
) + sagan.Line_template(
    template_velc=velc_temp,    # ← Same template
    template_flux=flux_temp,    # ← Same template
    amplitude=3.0,  # Will be tied to NII_6583
    dv=10,
    wavec=line_wave_dict['NII_6548'],
    name='NII_6548'
)

# [S II] doublet
ns2 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=10.0,
    dv=10,
    wavec=line_wave_dict['SII_6716'],
    name='SII_6716'
) + sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=8.0,  # FREE parameter (not tied)
    dv=10,
    wavec=line_wave_dict['SII_6731'],
    name='SII_6731'
)

# [O I] doublet
no1 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=6.0,
    dv=10,
    wavec=line_wave_dict['OI_6300'],
    name='OI_6300'
) + sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=2.0,  # Will be tied to OI_6300
    dv=10,
    wavec=line_wave_dict['OI_6364'],
    name='OI_6364'
)

# Hα absorption (if BAL QSO)
aha = sagan.Line_Absorption(
    logtau0=2.0,
    dv=-160,
    sigma=40,
    Cf=0.4,
    wavec=line_wave_dict['Halpha'],
    name='Abs. Halpha'
)

# Combine: absorption multiplies broad + continuum
# Then add narrow lines (no absorption on narrow forbidden lines)
b_lines = (bha + cont_ha) * aha
b_lines = sagan.convolve_lsf(b_lines, wavec=bha.wavec, resolving_power=resolving_power)
n_lines = nha + nn2 + ns2 + no1
m_init_ha = b_lines + n_lines
```

#### Tie Parameters

**CRITICAL**: Tie all narrow line velocities to a single parameter:

```python
# 1. Tie doublet amplitude ratios (theoretical values)
m_init_ha['NII_6548'].amplitude.tied = sagan.tie_template_amplitude('NII_6583', ratio=2.96)
m_init_ha['OI_6364'].amplitude.tied = sagan.tie_template_amplitude('OI_6300', ratio=3.0)
# Note: [S II] ratio is physics-dependent, leave FREE

# 2. Tie ALL narrow line velocities to nHalpha
narrow_lines_ha = ['nHalpha', 'NII_6583', 'NII_6548', 'SII_6716', 'SII_6731', 'OI_6300', 'OI_6364']
for ln in narrow_lines_ha:
    m_init_ha[ln].dv.tied = sagan.tie_template_dv('nHalpha')

# Now we have ONLY 1 free velocity parameter for ALL narrow lines!
print(f"Narrow line velocity parameter: {m_init_ha['nHalpha'].dv.value:.1f} km/s")
```

#### LSQ Fit and MCMC

(See [MCMC Fitting Strategy](#mcmc-fitting-strategy) below)

### Stage 2B: Hβ + Hγ Region (4200-5400 Å)

**Components to fit**:
1. Continuum (WindowedPowerLaw1D)
2. **All narrow lines use the SAME template from Stage 1** (Line_template)
3. Broad Hβ, Hγ (Line_MultiGauss)
4. [O III] blue wing (Line_MultiGauss_doublet)
5. Fe II template (IronTemplate)
6. Hβ, Hγ absorption (Line_Absorption, if BAL)

**CRITICAL**: Use the exact same `velc_temp` and `flux_temp` from Stage 1:

```python
# Continuum
cont_hb = sagan.WindowedPowerLaw1D(
    amplitude=12.0,
    x_0=5100.,
    alpha=1.1,
    x_min=4200,
    x_max=5400,
    name='Cont Hb'
)

# Iron template (optional, depends on data quality)
iron = sagan.IronTemplate(
    amplitude=0.8,
    stddev=900/2.3548,
    z=0.003,
    name='Fe II'
)

# Broad Hβ: 2 components
bhb = sagan.Line_MultiGauss(
    n_components=2,
    amp_c=12.0,
    dv_c=-100,
    sigma_c=290,
    amp_w0=0.7,
    dv_w0=20,
    sigma_w0=820,
    wavec=line_wave_dict['Hbeta'],
    name='Broad Hbeta'
)

# Broad Hγ: 1 component
bhg = sagan.Line_MultiGauss(
    n_components=1,
    amp_c=7.0,
    dv_c=-100,
    sigma_c=500,
    wavec=line_wave_dict['Hgamma'],
    name='Broad Hgamma'
)

# ===== ALL NARROW LINES USE THE SAME TEMPLATE FROM STAGE 1 =====
# Narrow Hβ
nhb = sagan.Line_template(
    template_velc=velc_temp,    # ← Same template as Hα region
    template_flux=flux_temp,    # ← Same template as Hα region
    amplitude=6.0,
    dv=10,
    wavec=line_wave_dict['Hbeta'],
    name='nHbeta'
)

# [O III] core (narrow)
no3 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=44.0,
    dv=10,
    wavec=line_wave_dict['OIII_5007'],
    name='OIII_5007'
) + sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=15.0,
    dv=10,
    wavec=line_wave_dict['OIII_4959'],
    name='OIII_4959'
)

# [O III] blue wing: 2 components (broad, asymmetric)
no3_w = sagan.Line_MultiGauss_doublet(
    n_components=2,
    amp_c0=65.0,
    amp_c1=22.0,
    dv_c=-70,
    sigma_c=130,
    amp_w0=0.43,
    dv_w0=-80,
    sigma_w0=300,
    wavec0=line_wave_dict['OIII_5007'],
    wavec1=line_wave_dict['OIII_4959'],
    name='OIII_5007_w'
)

# He II 4686 (broad + narrow)
bhe2 = sagan.Line_MultiGauss(
    n_components=1,
    amp_c=2.0,
    dv_c=-300,
    sigma_c=1800,
    wavec=line_wave_dict['HeII_4686'],
    name='Broad HeII'
)

nhe2 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=5.0,
    dv=10,
    wavec=line_wave_dict['HeII_4686'],
    name='nHeII_4686'
)

# [O III] 4363 (narrow + wing)
no3_4360 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=5.0,
    dv=10,
    wavec=line_wave_dict['OIII_4363'],
    name='OIII_4363'
)

no3_4360_w = sagan.Line_MultiGauss(
    n_components=2,
    amp_c=6.0,
    dv_c=-70,
    sigma_c=130,
    amp_w0=0.43,
    dv_w0=-80,
    sigma_w0=300,
    wavec=line_wave_dict['OIII_4363'],
    name='OIII_4363_w'
)

# Narrow Hγ
nhg = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=5.0,
    dv=10,
    wavec=line_wave_dict['Hgamma'],
    name='nHgamma'
)

# Absorption (if BAL QSO)
ahb = sagan.Line_Absorption(
    logtau0=1.15,
    dv=-160,
    sigma=40,
    Cf=0.4,
    wavec=line_wave_dict['Hbeta'],
    name='Abs. Hbeta'
)

ahg = sagan.Line_Absorption(
    logtau0=0.7,
    dv=-160,
    sigma=40,
    Cf=0.4,
    wavec=line_wave_dict['Hgamma'],
    name='Abs. Hgamma'
)

# Combine
b_lines = (bhb + bhg + cont_hb) * ahb * ahg + bhe2
b_lines = sagan.convolve_lsf(b_lines, wavec=bhb.wavec, resolving_power=resolving_power)
m_init_hb = b_lines + iron + nhb + nhe2 + no3 + no3_w + nhg + no3_4360 + no3_4360_w
```

#### Tie Parameters

```python
# [O III] doublet ratios
m_init_hb['OIII_4959'].amplitude.tied = sagan.tie_template_amplitude('OIII_5007', ratio=2.98)
m_init_hb['OIII_4959_w'].amp_c0.tied = sagan.tie_MultiGauss_amp_c('OIII_5007_w', ratio=2.98)

# Tie ALL narrow line velocities to nHalpha (or nHbeta)
# This ensures consistency across the entire spectrum!
narrow_lines_hb = ['nHbeta', 'OIII_5007', 'OIII_4959', 'nHeII_4686',
                   'nHgamma', 'OIII_4363']
for ln in narrow_lines_hb:
    m_init_hb[ln].dv.tied = sagan.tie_template_dv('nHalpha')  # ← Tie to Hα region!

# [O III] wing kinematics
m_init_hb['OIII_4363_w'].dv_c.tied = sagan.tie_MultiGauss_dv_c('OIII_5007_w')
m_init_hb['OIII_4363_w'].sigma_c.tied = sagan.tie_MultiGauss_sigma_c('OIII_5007_w')
```

### Stage 2C: He I Region (5400-6100 Å) - Optional

**Components to fit**:
1. Continuum (WindowedPowerLaw1D)
2. Narrow Hα + [N II] + [S II] + [O I] (Line_template)
3. Broad Hα (Line_MultiGauss with 3 components)
4. Hα absorption (Line_Absorption, if BAL)

```python
# Continuum
cont_ha = sagan.WindowedPowerLaw1D(
    amplitude=9.0,
    x_0=6550.,
    alpha=1.5,
    x_min=6100,
    x_max=7000,
    name='Cont Ha'
)

# Broad Hα: 3 components (core + 2 winds)
bha = sagan.Line_MultiGauss(
    n_components=3,
    amp_c=48.0,          # Core amplitude
    dv_c=-100,           # Core velocity shift (km/s)
    sigma_c=400,         # Core dispersion (km/s)
    amp_w0=0.15,         # Wind 0 relative amplitude
    dv_w0=20,            # Wind 0 velocity offset (km/s)
    sigma_w0=1400,       # Wind 0 dispersion (km/s)
    amp_w1=0.15,         # Wind 1 relative amplitude
    dv_w1=800,           # Wind 1 velocity offset (km/s)
    sigma_w1=200,        # Wind 1 dispersion (km/s)
    wavec=line_wave_dict['Halpha'],
    name='Broad Halpha'
)

# Narrow Hα (using template)
nha = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=50.0,
    dv=10,               # Measured from narrow lines
    wavec=line_wave_dict['Halpha'],
    name='nHalpha'
)

# [N II] doublet
nn2 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=9.0,
    dv=10,
    wavec=line_wave_dict['NII_6583'],
    name='NII_6583'
) + sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=3.0,  # Will be tied to NII_6583
    dv=10,
    wavec=line_wave_dict['NII_6548'],
    name='NII_6548'
)

# [S II] doublet
ns2 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=10.0,
    dv=10,
    wavec=line_wave_dict['SII_6716'],
    name='SII_6716'
) + sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=8.0,  # FREE parameter (not tied)
    dv=10,
    wavec=line_wave_dict['SII_6731'],
    name='SII_6731'
)

# [O I] doublet
no1 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=6.0,
    dv=10,
    wavec=line_wave_dict['OI_6300'],
    name='OI_6300'
) + sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=2.0,  # Will be tied to OI_6300
    dv=10,
    wavec=line_wave_dict['OI_6364'],
    name='OI_6364'
)

# Hα absorption (if BAL)
aha = sagan.Line_Absorption(
    logtau0=2.0,        # Log optical depth
    dv=-160,            # Velocity shift (km/s)
    sigma=40,           # Dispersion (km/s)
    Cf=0.4,             # Covering fraction
    wavec=line_wave_dict['Halpha'],
    name='Abs. Halpha'
)

# Combine and convolve
b_lines = (bha + cont_ha) * aha
b_lines = sagan.convolve_lsf(b_lines, wavec=bha.wavec, resolving_power=resolving_power)
n_lines = nha + nn2 + ns2 + no1
m_fit_ha = b_lines + n_lines

# Tie narrow line parameters
m_fit_ha['NII_6548'].amplitude.tied = sagan.tie_template_amplitude('NII_6583', ratio=2.96)
m_fit_ha['OI_6364'].amplitude.tied = sagan.tie_template_amplitude('OI_6300', ratio=3.0)

# All narrow lines share same dv
for ln in ['NII_6583', 'NII_6548', 'SII_6716', 'SII_6731', 'OI_6300', 'OI_6364']:
    m_fit_ha[ln].dv.tied = sagan.tie_template_dv('nHalpha')
```

### Stage 2: Hβ + Hγ Region (4200-5400 Å)

**Components to fit**:
1. Continuum (WindowedPowerLaw1D)
2. Narrow Hβ, [O III] 4959,5007, [O III] 4363, He II 4686 (Line_template)
3. Broad Hβ, Hγ, He II 4686 (Line_MultiGauss)
4. [O III] blue wing (Line_MultiGauss_doublet)
5. Fe II template (IronTemplate)
6. Hβ, Hγ absorption (Line_Absorption, if BAL)

```python
# Continuum
cont_hb = sagan.WindowedPowerLaw1D(
    amplitude=12.0,
    x_0=5100.,
    alpha=1.1,
    x_min=4200,
    x_max=5400,
    name='Cont Hb'
)

# Iron template
iron = sagan.IronTemplate(
    amplitude=0.8,
    stddev=900/2.3548,  # Will be tied to broad Hα
    z=0.003,            # Small redshift offset
    name='Fe II'
)

# Broad Hβ: 2 components
bhb = sagan.Line_MultiGauss(
    n_components=2,
    amp_c=12.0,
    dv_c=-100,
    sigma_c=290,
    amp_w0=0.7,         # Wind component
    dv_w0=20,
    sigma_w0=820,
    wavec=line_wave_dict['Hbeta'],
    name='Broad Hbeta'
)

# Broad Hγ: 1 component (simpler than Hβ)
bhg = sagan.Line_MultiGauss(
    n_components=1,
    amp_c=7.0,
    dv_c=-100,
    sigma_c=500,
    wavec=line_wave_dict['Hgamma'],
    name='Broad Hgamma'
)

# Narrow Hβ
nhb = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=6.0,
    dv=10,
    wavec=line_wave_dict['Hbeta'],
    name='nHbeta'
)

# [O III] core (narrow)
no3 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=44.0,
    dv=10,
    wavec=line_wave_dict['OIII_5007'],
    name='OIII_5007'
) + sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=15.0,  # Will be tied to OIII_5007
    dv=10,
    wavec=line_wave_dict['OIII_4959'],
    name='OIII_4959'
)

# [O III] blue wing: 2 components
no3_w = sagan.Line_MultiGauss_doublet(
    n_components=2,
    amp_c0=65.0,        # 5007 amplitude
    amp_c1=22.0,        # 4959 amplitude
    dv_c=-70,           # Blue-shifted wing
    sigma_c=130,
    amp_w0=0.43,
    dv_w0=-80,          # Additional blue shift
    sigma_w0=300,
    wavec0=line_wave_dict['OIII_5007'],
    wavec1=line_wave_dict['OIII_4959'],
    name='OIII_5007_w'
)

# He II 4686 (broad)
bhe2 = sagan.Line_MultiGauss(
    n_components=1,
    amp_c=2.0,
    dv_c=-300,
    sigma_c=1800,
    wavec=line_wave_dict['HeII_4686'],
    name='Broad HeII'
)

nhe2 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=5.0,
    dv=10,
    wavec=line_wave_dict['HeII_4686'],
    name='nHeII_4686'
)

# [O III] 4363 (narrow + wing)
no3_4360 = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=5.0,
    dv=10,
    wavec=line_wave_dict['OIII_4363'],
    name='OIII_4363'
)

no3_4360_w = sagan.Line_MultiGauss(
    n_components=2,
    amp_c=6.0,
    dv_c=-70,           # Same as [O III] 5007 wing
    sigma_c=130,
    amp_w0=0.43,
    dv_w0=-80,
    sigma_w0=300,
    wavec=line_wave_dict['OIII_4363'],
    name='OIII_4363_w'
)

# Narrow Hγ
nhg = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=5.0,
    dv=10,
    wavec=line_wave_dict['Hgamma'],
    name='nHgamma'
)

# Hβ and Hγ absorption (if BAL)
ahb = sagan.Line_Absorption(
    logtau0=1.15,       # Will be tied to Hα
    dv=-160,
    sigma=40,
    Cf=0.4,
    wavec=line_wave_dict['Hbeta'],
    name='Abs. Hbeta'
)

ahg = sagan.Line_Absorption(
    logtau0=0.7,        # Will be tied to Hα
    dv=-160,
    sigma=40,
    Cf=0.4,
    wavec=line_wave_dict['Hgamma'],
    name='Abs. Hgamma'
)

# Combine and convolve
b_lines = (bhb + bhg + cont_hb) * ahb * ahg + bhe2
b_lines = sagan.convolve_lsf(b_lines, wavec=bhb.wavec, resolving_power=resolving_power)
m_init = b_lines + iron + nhb + nhe2 + no3 + no3_w + nhg + no3_4360 + no3_4360_w + m_fit_ha

# Tie parameters
# [O III] doublet ratios
m_init['OIII_4959'].amplitude.tied = sagan.tie_template_amplitude('OIII_5007', ratio=2.98)
m_init['OIII_4959_w'].amp_c0.tied = sagan.tie_MultiGauss_amp_c('OIII_5007_w', ratio=2.98)
# ... (see Parameter Tying Patterns section below)
```

### Stage 2C: He I Region (5400-6100 Å) - Optional

**Components to fit**:
1. Continuum (WindowedPowerLaw1D)
2. Narrow He I 5876 (Line_template - **same template from Stage 1**)
3. Broad He I 5876 (Line_MultiGauss)

```python
cont_he1 = sagan.WindowedPowerLaw1D(
    amplitude=15.0,
    x_0=5750.,
    alpha=1.0,
    x_min=5400,
    x_max=6100,
    name='Cont HeI'
)

bhe1 = sagan.Line_MultiGauss(
    n_components=1,
    amp_c=5.0,
    dv_c=-100,
    sigma_c=300,
    wavec=line_wave_dict['HeI_5876'],
    name='Broad HeI'
)

# Use the SAME template from Stage 1
nhe1 = sagan.Line_template(
    template_velc=velc_temp,    # ← Same template
    template_flux=flux_temp,    # ← Same template
    amplitude=5.0,
    dv=10,
    wavec=line_wave_dict['HeI_5876'],
    name='nHeI_5876'
)

m_fit_he1 = cont_he1 + bhe1 + nhe1

# Tie velocity to other narrow lines
m_fit_he1['nHeI_5876'].dv.tied = sagan.tie_template_dv('nHalpha')
```

## Component Selection Guidelines

### Continuum: WindowedPowerLaw1D

For AGN, use a power-law continuum:

```python
cont = sagan.WindowedPowerLaw1D(
    amplitude=1.0,       # Continuum level
    x_0=5000,            # Reference wavelength (Å)
    alpha=-1.0,          # Power-law index (F_ν ∝ ν^α)
    x_min=4500,          # Window start (Å)
    x_max=5500,          # Window end (Å)
    name='continuum'
)
```

**Typical values**:
- amplitude: 1-20 (normalized units)
- alpha: -0.5 to -1.5
- x_0: central wavelength of fitting window

### Narrow Lines: Line_template

Use template-based fitting for narrow forbidden lines:

```python
narrow_line = sagan.Line_template(
    template_velc=velc_temp,    # Velocity array (km/s)
    template_flux=flux_temp,    # Normalized flux array
    amplitude=10.0,             # Line amplitude
    dv=10,                      # Velocity shift (km/s)
    wavec=line_wave_dict['OIII_5007'],
    name='narrow_line'
)
```

**Which lines to include**:
- **Always**: Hα, Hβ, [O III] 4959,5007
- **Usually**: [N II] 6548,6583, [S II] 6716,6731
- **Sometimes**: [O I] 6300,6364, He II 4686, [O III] 4363

### Broad Lines: Line_MultiGauss

Multi-component Gaussians for broad permitted lines:

```python
# Simple case (Hγ): 1 component
bhg = sagan.Line_MultiGauss(
    n_components=1,
    amp_c=7.0,
    dv_c=-100,
    sigma_c=500,
    wavec=line_wave_dict['Hgamma'],
    name='Broad Hgamma'
)

# Complex case (Hα): 3 components
bha = sagan.Line_MultiGauss(
    n_components=3,
    amp_c=48.0,      # Core
    dv_c=-100,
    sigma_c=400,
    amp_w0=0.15,     # Wind 0
    dv_w0=20,
    sigma_w0=1400,
    amp_w1=0.15,     # Wind 1
    dv_w1=800,
    sigma_w1=200,
    wavec=line_wave_dict['Halpha'],
    name='Broad Halpha'
)
```

**Number of components**:
- Hα, Hβ: 2-3 components (core + winds)
- Hγ: 1-2 components
- He II 4686: 1 component

**Typical values**:
- sigma_c: 200-600 km/s (core)
- sigma_w0: 800-2000 km/s (intermediate wind)
- sigma_w1: 100-500 km/s (very broad wind)

### Absorption: Line_Absorption

For BAL troughs:

```python
absorption = sagan.Line_Absorption(
    logtau0=2.0,        # Log optical depth at line center
    dv=-160,            # Velocity shift (km/s)
    sigma=40,           # Velocity dispersion (km/s)
    Cf=0.4,             # Covering fraction (0-1)
    wavec=line_wave_dict['Halpha'],
    name='Abs. Halpha'
)
```

**Physical constraints**:
- logtau0: -2 to 2 (weak to strong absorption)
- dv: -1000 to -100 km/s (blueshifted absorption)
- sigma: 20-200 km/s
- Cf: 0.1-1.0 (partial to full covering)

### Iron Template: IronTemplate

For Fe II emission blends:

```python
iron = sagan.IronTemplate(
    amplitude=0.8,              # Template amplitude
    stddev=900/2.3548,          # Velocity dispersion (km/s)
    z=0.003,                    # Redshift offset (small)
    template_name='park2022'    # or 'boroson1992'
)
```

**Available templates**:
- `park2022`: Based on Mrk 493 (recommended)
- `boroson1992`: Based on I Zw 1

**Typical values**:
- amplitude: 0.1-2.0 (relative to continuum)
- stddev: 400-1500 km/s (tie to broad Hβ)
- z: -0.003 to 0.003

## Parameter Tying Patterns

Parameter tying is critical for reducing degeneracies and ensuring physical consistency.

### Narrow Line Velocity Tying

All narrow lines share the same velocity shift:

```python
# Tie all narrow lines to nHalpha
narrow_lines = ['nHbeta', 'OIII_5007', 'OIII_4959',
                'nHgamma', 'OIII_4363', 'nHeII_4686', 'nHeI_5876']

for ln in narrow_lines:
    model[ln].dv.tied = sagan.tie_template_dv('nHalpha')
```

### Doublet Amplitude Ratios

Forbidden line doublets have fixed theoretical ratios:

```python
# [O III] 5007/4959 = 2.98 (Storey & Zeippen 2000)
model['OIII_4959'].amplitude.tied = sagan.tie_template_amplitude('OIII_5007', ratio=2.98)

# [N II] 6583/6548 = 2.96 (Storey & Zeippen 2000)
model['NII_6548'].amplitude.tied = sagan.tie_template_amplitude('NII_6583', ratio=2.96)

# [O I] 6300/6364 = 3.0 (Storey & Zeippen 2000)
model['OI_6364'].amplitude.tied = sagan.tie_template_amplitude('OI_6300', ratio=3.0)

# [S II] 6716/6731 - NOT tied (physics varies)
# Leave amplitudes free
```

### Absorption Parameter Tying

For BAL troughs across multiple lines:

```python
# Velocity and dispersion tied across all Balmer lines
model['Abs. Hbeta'].dv.tied = sagan.tie_Absorption_dv('Abs. Halpha')
model['Abs. Hgamma'].dv.tied = sagan.tie_Absorption_dv('Abs. Halpha')

model['Abs. Hbeta'].sigma.tied = sagan.tie_Absorption_sigma('Abs. Halpha')
model['Abs. Hgamma'].sigma.tied = sagan.tie_Absorption_sigma('Abs. Halpha')

model['Abs. Hbeta'].Cf.tied = sagan.tie_Absorption_Cf('Abs. Halpha')
model['Abs. Hgamma'].Cf.tied = sagan.tie_Absorption_Cf('Abs. Halpha')

# Optical depth ratios (from theory)
# τ_Hβ / τ_Hα = 7.13 (Case B recombination)
model['Abs. Hbeta'].logtau0.tied = sagan.tie_Absorption_logtau0('Abs. Halpha', ratio=7.13)

# τ_Hγ / τ_Hα = 21.18
model['Abs. Hgamma'].logtau0.tied = sagan.tie_Absorption_logtau0('Abs. Halpha', ratio=21.18)
```

### Broad Line Velocity Tying

Broad lines share similar kinematics:

```python
# Core velocity tied across Balmer lines
model['Broad Hbeta'].dv_c.tied = sagan.tie_MultiGauss_dv_c('Broad Halpha')
model['Broad Hgamma'].dv_c.tied = sagan.tie_MultiGauss_dv_c('Broad Halpha')

# Wind components can vary
# Typically leave free or tie loosely
```

### Iron Template Width Tying

```python
# Option 1: Tie to broad Hα FWHM
from sagan.measure_method import line_emission_fwhm
fwhm_ha = line_emission_fwhm(model, ['Broad Halpha'], wave, line_wave_dict['Halpha'])
model['Fe II'].stddev = fwhm_ha / 2.3548
model['Fe II'].stddev.bounds = (fwhm_ha/2.3548/1.5, fwhm_ha/2.3548*1.5)

# Option 2: Leave free but constrain
model['Fe II'].stddev.bounds = (200, 2000)
```

### [O III] Blue Wing Tying

The blue wing components share parameters:

```python
# Tie [O III] 4959 wing to 5007 wing
model['OIII_4959_w'].amp_c0.tied = sagan.tie_MultiGauss_amp_c('OIII_5007_w', ratio=2.98)
model['OIII_4959_w'].dv_c.tied = sagan.tie_MultiGauss_dv_c('OIII_5007_w')
model['OIII_4959_w'].sigma_c.tied = sagan.tie_MultiGauss_sigma_c('OIII_5007_w')
model['OIII_4959_w'].amp_w0.tied = sagan.tie_MultiGauss_amp_w0('OIII_5007_w')
model['OIII_4959_w'].dv_w0.tied = sagan.tie_MultiGauss_dv_w0('OIII_5007_w')
model['OIII_4959_w'].sigma_w0.tied = sagan.tie_MultiGauss_sigma_w0('OIII_5007_w')

# Tie [O III] 4363 wing to 5007 wing
model['OIII_4363_w'].dv_c.tied = sagan.tie_MultiGauss_dv_c('OIII_5007_w')
model['OIII_4363_w'].sigma_c.tied = sagan.tie_MultiGauss_sigma_c('OIII_5007_w')
```

## MCMC Fitting Strategy

Once the models are built with the narrow line template, proceed with LSQ fitting followed by MCMC.

### Step 1: LSQ Fitting

First, use Levenberg-Marquardt minimization to get initial parameters:

```python
fitter = fitting.LevMarLSQFitter()
m_fit = fitter(m_init, wave_rest, flux_obs, weights=weight_lines, maxiter=10000)

# Visual check
fig, ax = plt.subplots(figsize=(15, 5))
ax.step(wave_rest, flux_obs, where='mid', color='k', label='Data')
ax.plot(wave_rest, m_fit(wave_rest), color='C3', lw=2, label='LSQ Fit')
ax.legend()
plt.show()
```

### Step 2: MCMC Fitting - Separate Runs

Use different MCMC runs for different component groups to improve convergence:

#### Run 1: Balmer Lines (Hα, Hβ, Hγ + absorption)

```python
# Define which parameters to fit
balmer_components = [
    'Broad Halpha', 'Broad Hbeta', 'Broad Hgamma',
    'Abs. Halpha', 'Abs. Hbeta', 'Abs. Hgamma',
    'Cont Ha', 'Cont Hb'
]

# Fix other parameters
for name in m_fit.submodel_names:
    if name not in balmer_components:
        for param in m_fit[name].param_names:
            setattr(m_fit[name], param, 'fixed')

# Initialize MCMC
mcmc_balmer = sagan.MCMC_Fit(
    m_fit,
    wave_rest,
    flux_obs,
    ferr_obs,
    nwalkers=200,
    nsteps=15000,
    nburn=10000,          # Will use step_initial
    step_initial=5000     # Initial burn-in
)

# Run fitting
mcmc_balmer.fit(progress=True)

# Check convergence
chain, tau = mcmc_balmer.check_convergence()
# Good: nsteps / tau_max > 50

# Save samples
mcmc_balmer.save_samples('samples_balmer.npz')
```

#### Run 2: Other Components ([O III], [N II], [S II], Fe II)

```python
# Reload best-fit model from Run 1
mcmc_balmer = ... # Load from file
m_fit_balmer, _, _ = mcmc_balmer.get_best_fit()

# Fix Balmer line parameters
for name in ['Broad Halpha', 'Broad Hbeta', 'Broad Hgamma',
             'Abs. Halpha', 'Abs. Hbeta', 'Abs. Hgamma']:
    for param in m_fit_balmer[name].param_names:
        setattr(m_fit_balmer[name], param, 'fixed')

# Free other parameters
# ... (free [O III], [N II], etc.)

# Initialize MCMC
mcmc_others = sagan.MCMC_Fit(
    m_fit_balmer,
    wave_rest,
    flux_obs,
    ferr_obs,
    nwalkers=200,
    nsteps=15000,
    step_initial=5000
)

# Run fitting
mcmc_others.fit(progress=True)
mcmc_others.check_convergence()
mcmc_others.save_samples('samples_others.npz')
```

### Step 3: Combine Results

```python
# Load both MCMC runs
mcmc_balmer.load_samples('samples_balmer.npz')
mcmc_others.load_samples('samples_others.npz')

# Get best-fit models
m_fit1, _, _ = mcmc_balmer.get_best_fit()
m_fit2, _, _ = mcmc_others.get_best_fit()

# Extract parameters
# See Physical Measurements section below
```

## Physical Measurements

### Black Hole Mass Estimation

Using the broad line width and luminosity:

```python
from sagan.measure_method import line_emission_fwhm

# Measure FWHM of broad Hα
fwhm_ha, w_l, w_r, w_peak = line_emission_fwhm(
    m_fit1,
    ['Broad Halpha'],
    wave_rest,
    line_wave_dict['Halpha']
)

print(f"Broad Hα FWHM: {fwhm_ha:.1f} km/s")

# Measure continuum luminosity
# L_5100 = λ * f_λ * 4π * d_L^2
# Use the continuum amplitude and redshift

# Black hole mass (Vestergaard & Osmer 2009)
# log(M_BH/M_sun) = a + b*log(L_5100/10^44) + c*log(FWHM_Hβ/10^3)
```

### Emission Line Fluxes

```python
# Integrate line flux
def integrate_line_flux(model, wave, line_names):
    """Integrate flux for specified lines."""
    total_flux = 0
    for name in line_names:
        line = model[name]
        # Get line wavelength range
        if hasattr(line, 'wavec'):
            wavec = line.wavec.value
            sigma = line.sigma.value if hasattr(line, 'sigma') else 200
            vel_range = np.linspace(-5*sigma, 5*sigma, 1000)
            wave_range = wavec * (1 + vel_range / ls_km)
            flux_range = line(wave_range)
            line_flux = np.trapz(flux_range, wave_range)
            total_flux += line_flux
    return total_flux

# Example: narrow Hα flux
flux_nha = integrate_line_flux(m_fit2, wave_rest, ['nHalpha'])
print(f"Narrow Hα flux: {flux_nha:.2e} erg/s/cm²")
```

### Absorption Strength

```python
# Get optical depth samples
logtau0_samples = mcmc_balmer.get_param_samples('Abs. Halpha', 'logtau0')

# Extract statistics
tau0_median = np.percentile(logtau0_samples, 50)
tau0_low = np.percentile(logtau0_samples, 0.3)  # 99.4% upper limit
tau0_high = np.percentile(logtau0_samples, 99.7)

print(f"Hα log(τ₀): {tau0_median:.2f} +{tau0_high-tau0_median:.2f} -{tau0_median-tau0_low:.2f}")
```

### Velocity Measurements

```python
# Narrow line velocity shift
dv_nha = m_fit2['nHalpha'].dv.value
print(f"Narrow Hα dv: {dv_nha:.1f} km/s")

# Broad line velocity shift
dv_bha = m_fit1['Broad Halpha'].dv_c.value
print(f"Broad Hα dv: {dv_bha:.1f} km/s")

# [O III] velocity shift
dv_o3 = m_fit2['OIII_5007'].dv.value
print(f"[O III] 5007 dv: {dv_o3:.1f} km/s")
```

## Complete Example

Condensed working example from J0925+6409, following the **two-stage strategy**:

```python
#!/usr/bin/env python
"""Type 1 AGN fitting example - J0925+6409
    Demonstrates the two-stage narrow line template strategy
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import sagan
from sagan.utils import line_wave_dict
from astropy.table import Table

ls_km = 2.99792e5
resolving_power = 1800

# ========================================
# STAGE 0: Load and Prepare Data
# ========================================
tb = Table.read('target_info.ipac', format='ipac')
targname = tb['Name'][0]
zred = tb['zred'][0]

spec = np.loadtxt('J0925+6409_nostar.txt')
wave_obs = spec[:, 0]
flux_obs = spec[:, 1]
ferr_obs = spec[:, 2]

# Convert to rest frame
wave = wave_obs / (1 + zred)
flux = flux_obs
ferr = ferr_obs

# ========================================
# STAGE 1: Create Narrow Line Template
# ========================================
# Fit [S II] doublet with multiple Gaussians to get the profile shape
s2_region = (wave > 6700) & (wave < 6745)

# Continuum
cont_s2 = models.Polynomial1D(degree=1)
cont_mask = ((wave > 6700) & (wave < 6710)) | ((wave > 6735) & (wave < 6745))
fitter = fitting.LevMarLSQFitter()
cont_fit = fitter(cont_s2, wave[s2_region][cont_mask[s2_region]],
                  flux[s2_region][cont_mask[s2_region]])

# Normalize
flux_norm = flux[s2_region] / cont_fit(wave[s2_region])

# Build model with Gaussians (NOT Line_template yet!)
sii_6716 = sagan.Line_Gaussian(amplitude=8.0, dv=0, sigma=100,
                               wavec=line_wave_dict['SII_6716'])
sii_6731 = sagan.Line_Gaussian(amplitude=6.0, dv=0, sigma=100,
                               wavec=line_wave_dict['SII_6731'])
model_sii = cont_s2 + sii_6716 + sii_6731

# Fit
model_sii_fit = fitter(model_sii, wave[s2_region], flux[s2_region],
                       weights=1/ferr[s2_region]**2, maxiter=10000)

# Generate template
velc_temp = np.linspace(-1000, 1000, 2000)
flux_6716 = model_sii_fit['SII_6716'](velc_temp, wavec=line_wave_dict['SII_6716'])
flux_6731 = model_sii_fit['SII_6731'](velc_temp, wavec=line_wave_dict['SII_6731'])
flux_temp = (flux_6716 + flux_6731) / np.max(flux_6716 + flux_6731)

# Save template
template_file = f'{targname}_narrow_template.txt'
np.savetxt(template_file, np.column_stack([velc_temp, flux_temp]),
           header='velocity_kms normalized_flux')
print(f"✓ Template saved to: {template_file}")

# ========================================
# STAGE 2: Fit Broad Line Complexes
# ========================================
# Now use the template for ALL narrow lines

# Hα region model
cont_ha = sagan.WindowedPowerLaw1D(amplitude=9.0, x_0=6550., alpha=1.5,
                                   x_min=6100, x_max=7000, name='Cont Ha')
bha = sagan.Line_MultiGauss(n_components=3, amp_c=48.0, dv_c=-100, sigma_c=400,
                            amp_w0=0.15, dv_w0=20, sigma_w0=1400,
                            amp_w1=0.15, dv_w1=800, sigma_w1=200,
                            wavec=line_wave_dict['Halpha'], name='Broad Halpha')
# Use the SAME template for all narrow lines!
nha = sagan.Line_template(template_velc=velc_temp, template_flux=flux_temp,
                          amplitude=50.0, dv=10, wavec=line_wave_dict['Halpha'],
                          name='nHalpha')
nii_6583 = sagan.Line_template(template_velc=velc_temp, template_flux=flux_temp,
                               amplitude=9.0, dv=10, wavec=line_wave_dict['NII_6583'],
                               name='NII_6583')
nii_6548 = sagan.Line_template(template_velc=velc_temp, template_flux=flux_temp,
                               amplitude=3.0, dv=10, wavec=line_wave_dict['NII_6548'],
                               name='NII_6548')
# ... add other narrow lines [S II], [O I], etc.

aha = sagan.Line_Absorption(logtau0=2.0, dv=-160, sigma=40, Cf=0.4,
                            wavec=line_wave_dict['Halpha'], name='Abs. Halpha')

# Combine
b_lines = (bha + cont_ha) * aha
b_lines = sagan.convolve_lsf(b_lines, wavec=bha.wavec, resolving_power=resolving_power)
n_lines = nha + nii_6583 + nii_6548
m_init_ha = b_lines + n_lines

# Tie parameters: all narrow lines share same velocity!
for ln in ['nHalpha', 'NII_6583', 'NII_6548']:
    m_init_ha[ln].dv.tied = sagan.tie_template_dv('nHalpha')
m_init_ha['NII_6548'].amplitude.tied = sagan.tie_template_amplitude('NII_6583', ratio=2.96)

# Define fitting window
weight = np.ones_like(wave)
weight[(wave > 5398) & (wave < 5403)] = 0
weight[(wave > 6097) & (wave < 6103)] = 0
line_windows = [(4200, 5400), (6100, 7000)]
for window in line_windows:
    weight[(wave >= window[0]) & (wave <= window[1])] = 1

# LSQ fit
fitter = fitting.LevMarLSQFitter()
m_fit = fitter(m_init_ha, wave, flux, weights=weight, maxiter=10000)

# MCMC fit
mcmc = sagan.MCMC_Fit(m_fit, wave, flux, ferr, nwalkers=200,
                      nsteps=15000, step_initial=5000)
mcmc.fit(progress=True)
mcmc.check_convergence()
mcmc.save_samples(f'{targname}_samples.npz')

# Results
m_fit_final, _, _ = mcmc.get_best_fit()
ax, axr = sagan.plot.plot_fit_new(wave, flux, m_fit_final, error=ferr)
ax.set_xlim(6400, 6700)
plt.show()

# Measurements
from sagan.measure_method import line_emission_fwhm
fwhm_ha, _, _, _ = line_emission_fwhm(m_fit_final, ['Broad Halpha'], wave,
                                       line_wave_dict['Halpha'])
print(f"Broad Hα FWHM: {fwhm_ha:.1f} km/s")
print(f"Narrow line dv: {m_fit_final['nHalpha'].dv.value:.1f} km/s")
```

**Key Points**:
1. **Stage 1** (lines 24-58): Extract narrow line profile from [S II] or [O III]
2. **Stage 2** (lines 61+): Use the template for ALL narrow lines
3. **Tying**: All narrow lines share 1 velocity parameter
4. **Result**: >60% reduction in free parameters, robust fit

## References

- **Shangguan et al. (2026)**: Low-redshift BAL QSO analysis methodology
- **Storey & Zeippen (2000)**: Atomic data for forbidden line ratios
- **Vestergaard & Osmer (2009)**: Black hole mass estimators
- **Kovačević et al. (2013)**: Iron template implementation

## Related Documentation

- **[Main Fitting Guide](../sagan_spectral_fitting.md)** - General workflow and data preparation
- **[Function Reference](../function_reference.md)** - Complete function documentation

---

**Version**: 1.0
**Last Updated**: 2025-03-20
