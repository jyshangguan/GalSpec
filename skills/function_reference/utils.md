# Utility Functions (`sagan.utils`)

General utility functions for spectral analysis in SAGAN.

## Table of Contents

1. [Line Dictionaries](#line-dictionaries)
2. [Velocity Conversions](#velocity-conversions)
3. [calculate_bic](#calculate_bic)
4. [ReadSpectrum Class](#readspectrum-class)
5. [Spectral Resolution Degradation](#spectral-resolution-degradation)

---

## Line Dictionaries

Get rest wavelengths and LaTeX labels for common emission lines.

```python
from sagan.utils import line_wave_dict, line_label_dict

# Get rest wavelength (Å)
wave_ha = line_wave_dict['Halpha']      # 6562.819
wave_hb = line_wave_dict['Hbeta']       # 4861.369
wave_o3_5007 = line_wave_dict['OIII_5007']  # 5006.843

# Get plotting label
label_ha = line_label_dict['Halpha']    # r'H$\alpha$'
label_hb = line_label_dict['Hbeta']     # r'H$\beta$'
```

**Available Lines**:

**Hydrogen Balmer Series**:
```python
'Halpha': 6562.819
'Hbeta': 4861.333
'Hgamma': 4340.471
'Hdelta': 4101.742
```

**[O III] Doublet**:
```python
'OIII_4959': 4958.911
'OIII_5007': 5006.843
'OIII_4363': 4363.210
```

**[S II] Doublet**:
```python
'SII_6716': 6716.440
'SII_6731': 6730.810
```

**[N II] Doublet**:
```python
'NII_6548': 6548.050
'NII_6583': 6583.460
```

**[O I] Doublet**:
```python
'OI_6300': 6300.304
'OI_6364': 6363.776
```

**Other Lines**:
```python
'HeII_4686': 4685.710
'HeI_4471': 4471.479
'HeI_5876': 5875.624
'NaD_5890': 5891.583
'NaD_5896': 5897.558
```

---

## Velocity Conversions

Convert between wavelength and velocity.

```python
from sagan.utils import wave_to_velocity, velocity_to_wave

# Wavelength to velocity
vel = wave_to_velocity(wave_obs, wave_rest)  # km/s

# Velocity to wavelength
wave_obs = velocity_to_wave(vel, wave_rest)  # Å
```

**Parameters**:
- `wave_to_velocity(wave_obs, wave_rest)`: Converts observed wavelength to velocity in km/s
- `velocity_to_wave(vel, wave_rest)`: Converts velocity to observed wavelength

**Formula**: `v = c * (λ_obs - λ_rest) / λ_rest`

---

## calculate_bic

Calculate the Bayesian Information Criterion (BIC) for model comparison.

```python
from sagan.utils import calculate_bic

# After fitting a model
bic, chi2, n_params = calculate_bic(model_fit, wave, flux, error)

print(f"BIC: {bic:.1f}")
print(f"χ²: {chi2:.1f}")
print(f"Free parameters: {n_params}")
```

**BIC Formula**: `BIC = χ² + k × ln(n)`

where:
- `χ²` = sum(((flux - model(wave)) / error)²)
- `k` = number of free parameters (not fixed or tied)
- `n` = number of data points

**Model Comparison**:
```python
# Compare two models
bic1, _, _ = calculate_bic(model1_fit, wave, flux, error)
bic2, _, _ = calculate_bic(model2_fit, wave, flux, error)

delta_bic = bic2 - bic1

if delta_bic < -10:
    print("Strong evidence for model 2 (ΔBIC < -10)")
elif delta_bic > 10:
    print("Strong evidence for model 1 (ΔBIC > 10)")
else:
    print("Weak evidence (|ΔBIC| < 10), prefer simpler model")
```

**Interpretation**:
- ΔBIC < -10: Strong evidence for the new (more complex) model
- ΔBIC > 10: Strong evidence for the baseline (simpler) model
- |ΔBIC| < 10: Weak evidence, prefer the simpler model

**Example: Deciding Number of Broad Components**:
```python
# Fit 1-component model
model_1comp = fitter(model_1, wave, flux, weights=1/error**2)
bic1, chi2_1, n1 = calculate_bic(model_1comp, wave, flux, error)

# Fit 2-component model
model_2comp = fitter(model_2, wave, flux, weights=1/error**2)
bic2, chi2_2, n2 = calculate_bic(model_2comp, wave, flux, error)

print(f"1-component: BIC={bic1:.1f}, χ²/ν={chi2_1/len(wave):.3f}, n={n1}")
print(f"2-component: BIC={bic2:.1f}, χ²/ν={chi2_2/len(wave):.3f}, n={n2}")
print(f"ΔBIC = {bic2 - bic1:.1f}")

if bic2 < bic1 - 10:
    print("→ Use 2-component model")
else:
    print("→ Use 1-component model")
```

**Parameters**:
- `model` (astropy.modeling.Model): Fitted model (can be compound)
- `wave` (array): Wavelength values
- `flux` (array): Flux values
- `error` (array, optional): Error values. If None, χ² is calculated without weighting.

**Returns**:
- `bic` (float): BIC value
- `chi2` (float): Total χ²
- `n_params` (int): Number of free parameters

---

## ReadSpectrum Class

Read and preprocess astronomical spectra.

```python
readspec = sagan.ReadSpectrum(
    is_sdss=True,        # For SDSS spectra
    hdu=hdulist,         # SDSS HDU object
    z=0.0,              # Redshift (optional, from SDSS header)
    ra=None, dec=None   # Coordinates
)

# For custom spectra
readspec = sagan.ReadSpectrum(
    is_sdss=False,
    lam=wave, flux=flux, ferr=ferr,
    z=0.1, ra=ra, dec=dec
)

# Methods
flux_unred, err_unred = readspec.de_redden()  # MW extinction correction
lam_res, flux_res, err_res = readspec.rest_frame()  # Rest-frame conversion
lam_res, flux_res, err_res = readspec.unredden_res()  # Both corrections
```

**Example**:
```python
# Load SDSS spectrum
from astropy.io import fits
hdu = fits.open('spec.fits')
readspec = sagan.ReadSpectrum(is_sdss=True, hdu=hdu)

# Get MW-corrected rest-frame spectrum
wave, flux, ferr = readspec.unredden_res()
```

**Methods**:
- `de_redden()`: Correct for Milky Way extinction using SFD dust map
- `rest_frame()`: Convert observed frame to rest frame
- `unredden_res()`: Apply both extinction correction and rest-frame conversion

---

## Spectral Resolution Degradation

Degrade spectral resolution to match instruments.

```python
from sagan.utils import down_spectres

flux_lowres = down_spectres(
    wave, flux,
    R_org=2000,      # Original resolving power
    R_new=1000,      # New resolving power
    wave_new=None    # Optional new wavelength grid
)
```

**Parameters**:
- `wave`: Original wavelength array
- `flux`: Original flux array
- `R_org`: Original resolving power (R = λ/Δλ)
- `R_new`: Target resolving power
- `wave_new`: Optional new wavelength grid (if None, computed automatically)

**Usage**: Match template or model resolution to observed spectrum.

---

**Module**: `sagan.utils`
**Source File**: `sagan/utils.py`
