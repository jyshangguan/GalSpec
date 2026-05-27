# Python Tools for GalSpec Spectral Analysis

Reusable plotting and utility functions for AGN spectral analysis using GalSpec.

## Overview

This module provides convenient wrapper functions for visualizing continuum fitting results and continuum-subtracted spectra. These tools are designed to work seamlessly with the GalSpec (Spectral Analysis for Galaxies and AGN) package.

## Installation

The `python_tools` module is located in the GalSpec skills directory:

```
/Users/shangguan/Softwares/my_modules/GalSpec/skills/python_tools/
```

## Available Functions

### 1. `plot_continuum_fit_diagnostic()`

Visualize the fitted continuum model with all components, using GalSpec's `plot_fit_new()` function.

**What it shows:**
- Full spectrum (black step line)
- Continuum windows highlighted (gray weight line)
- Total continuum fit (red line)
- Individual components (AGN power-law, stellar, iron)
- Residuals (data - model)

**Usage:**
```python
from python_tools.plot_tools import plot_continuum_fit_diagnostic

plot_continuum_fit_diagnostic(
    wave_rest,           # Wavelength array (Å, rest frame)
    flux_rest,           # Flux array
    ferr_rest,           # Error array
    model_cont_fit,      # Fitted continuum model
    cont_mask,           # Boolean mask for continuum windows
    target_name='SDSS-J000111.15-100155.5',
    filename='continuum_fit_diagnostic.png'
)
```

**Parameters:**
- `wave, flux, error`: Spectrum data (same length)
- `model`: Fitted continuum model (CompoundModel from GalSpec)
- `cont_mask`: Boolean array (True = continuum window pixel)
- `target_name`: Optional target name for title
- `filename`: Output filename (if None, displays interactively)
- `xlabel, ylabel`: Axis labels (customizable)
- `figsize`: Figure size (default: 10×8 inches)
- `dpi`: Resolution (default: 150)

**Returns:**
- `ax`: Main plot axes
- `axr`: Residual plot axes

---

### 2. `plot_continuum_subtracted_spectrum()`

Plot the spectrum after continuum subtraction, highlighting continuum windows.

**What it shows:**
- Continuum-subtracted flux (line emission/absorption)
- Continuum windows highlighted (blue shaded regions)
- Zero reference line (red dashed)

**Usage:**
```python
from python_tools.plot_tools import plot_continuum_subtracted_spectrum

# Define continuum windows
windows = [
    (4200, 4300), (4430, 4560), (5060, 5400),
    (5600, 5700), (6180, 6230), (6800, 7000), (7500, 8000)
]

plot_continuum_subtracted_spectrum(
    wave_rest,
    flux_subtracted,    # flux - continuum
    ferr_rest,
    continuum_windows=windows,
    target_name='SDSS-J000111.15-100155.5',
    filename='continuum_subtracted_spectrum.png'
)
```

**Parameters:**
- `wave, flux_sub, error`: Spectrum data
- `continuum_windows`: List of (wmin, wmax) tuples defining windows
- `target_name`: Optional target name for title
- `filename`: Output filename (if None, displays interactively)
- `window_color`: Color for window shading (default: 'blue')
- `window_alpha`: Transparency for windows (default: 0.1)

**Returns:**
- `ax`: Plot axes

---

### 3. `verify_continuum_subtraction()`

Verify continuum subtraction quality by computing statistics in continuum windows.

**What it does:**
- Computes median and standard deviation in each continuum window
- Good subtraction should yield median ≈ 0 in all windows
- Prints statistics for quality assessment

**Usage:**
```python
from python_tools.plot_tools import verify_continuum_subtraction

results = verify_continuum_subtraction(
    wave_rest,
    flux_subtracted,
    continuum_windows=windows,
    verbose=True
)

# Check for problematic windows
for r in results:
    if abs(r['median']) > 1.0:
        print(f"Warning: Window {r['window']} has median = {r['median']:.2f}")
```

**Parameters:**
- `wave, flux_sub`: Spectrum data
- `continuum_windows`: List of (wmin, wmax) tuples
- `verbose`: If True, prints statistics (default: True)

**Returns:**
- `results`: List of dicts with statistics for each window

---

## Complete Workflow Example

Here's a complete example showing how to use these tools in your continuum fitting workflow:

```python
#!/usr/bin/env python
"""
Example: Continuum fitting with visualization using python_tools
"""
import numpy as np
from astropy.modeling import fitting
from astropy.io import fits
import sys

# Add GalSpec to path
sys.path.insert(0, '/path/to/GalSpec')
sys.path.insert(0, '/path/to/GalSpec/skills')

import galspec
from galspec.utils import ReadSpectrum
from galspec.continuum import WindowedPowerLaw1D
from python_tools.plot_tools import (
    plot_continuum_fit_diagnostic,
    plot_continuum_subtracted_spectrum,
    verify_continuum_subtraction
)

# ========================================
# 1. Load Spectrum
# ========================================
hdu = fits.open('spectrum.fits')
spec = ReadSpectrum(is_sdss=True, hdu=hdu)
wave_rest, flux_rest, ferr_rest = spec.unredden_res()

# ========================================
# 2. Define Continuum Windows
# ========================================
CONTINUUM_WINDOWS = [
    (4200, 4300), (4430, 4560), (5060, 5400),
    (5600, 5700), (6180, 6230), (6800, 7000), (7500, 8000)
]

# Create continuum mask
cont_mask = np.zeros(len(wave_rest), dtype=bool)
for wmin, wmax in CONTINUUM_WINDOWS:
    if wmin >= wave_rest.min() and wmax <= wave_rest.max():
        cont_mask |= (wave_rest > wmin) & (wave_rest < wmax)

wave_cont = wave_rest[cont_mask]
flux_cont = flux_rest[cont_mask]
ferr_cont = ferr_rest[cont_mask]

# ========================================
# 3. Build & Fit Continuum Model
# ========================================
wave_min = int(np.floor(wave_rest.min()))
wave_max = int(np.ceil(wave_rest.max()))

agn_cont = WindowedPowerLaw1D(
    amplitude=np.median(flux_cont),
    x_0=5500, alpha=-1.5,
    x_min=wave_min, x_max=wave_max,
    name='AGN_powerlaw'
)

stellar = galspec.StarSpectrum(
    amplitude=0.2, Star_type='G',
    velscale=200, delta_z=0, sigma=150,
    name='stellar'
)

iron = galspec.IronTemplate(
    amplitude=0.5, stddev=800/2.3548, z=0,
    template_name='park2022',
    name='iron'
)

model_cont = agn_cont + stellar + iron

fitter = fitting.LevMarLSQFitter()
model_cont_fit = fitter(model_cont, wave_cont, flux_cont,
                        weights=1/ferr_cont**2, maxiter=10000)

# ========================================
# 4. Visualize Continuum Fit
# ========================================
plot_continuum_fit_diagnostic(
    wave_rest, flux_rest, ferr_rest, model_cont_fit, cont_mask,
    target_name='SDSS-J000111.15-100155.5',
    filename='continuum_fit_diagnostic.png'
)
print("✓ Continuum fit diagnostic plot saved")

# ========================================
# 5. Generate Continuum-Subtracted Spectrum
# ========================================
continuum_all = model_cont_fit(wave_rest)
flux_subtracted = flux_rest - continuum_all

# ========================================
# 6. Verify Subtraction Quality
# ========================================
results = verify_continuum_subtraction(
    wave_rest, flux_subtracted, CONTINUUM_WINDOWS
)

# Check quality
all_good = True
for r in results:
    if abs(r['median']) > 1.0:
        print(f"WARNING: Window {r['window']} has large offset!")
        all_good = False

if all_good:
    print("✓ Continuum subtraction quality verified")

# ========================================
# 7. Visualize Continuum-Subtracted Spectrum
# ========================================
plot_continuum_subtracted_spectrum(
    wave_rest, flux_subtracted, ferr_rest,
    continuum_windows=CONTINUUM_WINDOWS,
    target_name='SDSS-J000111.15-100155.5',
    filename='continuum_subtracted_spectrum.png'
)
print("✓ Continuum-subtracted spectrum plot saved")
```

---

## Integration with Existing Scripts

To use these tools in your existing continuum fitting scripts, simply replace the plotting sections:

### Before (custom plotting code):
```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(wave_rest, flux_rest, 'k-')
# ... 20+ lines of custom plotting code ...
plt.savefig('continuum_fit_diagnostic.png')
```

### After (using python_tools):
```python
from python_tools.plot_tools import plot_continuum_fit_diagnostic

plot_continuum_fit_diagnostic(
    wave_rest, flux_rest, ferr_rest, model_cont_fit, cont_mask,
    target_name=TARGET_NAME,
    filename='continuum_fit_diagnostic.png'
)
```

**Benefits:**
- ✓ Standardized visualization across all projects
- ✓ Less code to maintain
- ✓ Consistent formatting and styling
- ✓ Built on GalSpec's `plot_fit_new()` function
- ✓ Easy to use with minimal parameters

---

## File Structure

```
python_tools/
├── __init__.py           # Package initialization
├── plot_tools.py         # Plotting functions
└── README.md            # This file
```

---

## Dependencies

These functions require:
- `numpy` - Array operations
- `matplotlib` - Plotting
- `astropy` - Model handling
- `galspec` - GalSpec package (for `plot_fit_new()`)

---

## Troubleshooting

### Import Error: "No module named 'galspec'"

**Solution:** Add GalSpec to your Python path before importing:
```python
import sys
sys.path.insert(0, '/path/to/GalSpec')
```

### Plot windows not highlighting correctly

**Check:** Ensure `cont_mask` is a boolean array with same length as `wave`:
```python
assert len(cont_mask) == len(wave)
assert cont_mask.dtype == bool
```

### Continuum windows show large offsets after subtraction

**Possible causes:**
1. Continuum model doesn't cover full wavelength range
2. Missing continuum components (stellar, iron)
3. Poor initial parameters

**Solution:** Review continuum fitting strategy in `../fitting_strategies/continuum_fitting.md`

---

## Related Documentation

- **Continuum Fitting Guide**: `../fitting_strategies/continuum_fitting.md`
- **Narrow Line Template**: `../fitting_strategies/narrow_line_template.md`
- **Type 1 AGN Strategy**: `../fitting_strategies/type1_agn.md`
- **Function Reference**: `../function_reference/`

---

## Version History

- **v1.0.0** (2024-03-24): Initial release with three plotting functions

---

## Contact

For questions or issues with these tools, please refer to the main GalSpec documentation or contact the GalSpec development team.
