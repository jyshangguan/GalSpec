# Convolution Functions (`sagan.convolution`, `sagan.convolution_var`)

Instrumental broadening convolution functions in SAGAN.

## Table of Contents

1. [convolve_lsf](#convolve_lsf)
2. [convolve_lsf_var](#convolve_lsf_var)

---

## convolve_lsf

Convolve model with constant resolving power LSF.

```python
model_convolved = sagan.convolve_lsf(
    model,                  # CompoundModel to convolve
    wavec=6563,             # Reference wavelength (Å)
    resolving_power=1800,   # Spectral resolving power (R = λ/Δλ)
    conv_mode='auto'        # 'auto', 'manual', or 'none'
)
```

**Usage**: Apply instrumental broadening to model before fitting.

**Important**: Must be applied AFTER absorption multiplication: `(emission * absorption)` then `convolve_lsf()`

**Example**:
```python
# Build model with absorption
model = (broad_ha + cont_ha) * abs_ha + narrow_ha

# Apply LSF convolution
model_convolved = sagan.convolve_lsf(
    model,
    wavec=line_wave_dict['Halpha'],
    resolving_power=1800
)
```

---

## convolve_lsf_var

Convolve with wavelength-dependent resolving power (JWST NIRSpec).

```python
# First create resolution curve
res_curve = sagan.ResolutionCurve.from_file(
    'data/NIRSpec_prism_resolution.fits',
    wave_col='WAVELENGTH',
    res_col='RESOLUTION',
    interpolation='loglog'
)

model_convolved = sagan.convolve_lsf_var(
    model,
    wave_array,             # Wavelength array
    res_curve,              # ResolutionCurve object
    min_wave=4000,          # Minimum wavelength to convolve
    max_wave=5500           # Maximum wavelength to convolve
)
```

**Usage**: Instruments with variable resolving power (e.g., JWST NIRSpec prism).

---

**Modules**: `sagan.convolution`, `sagan.convolution_var`
**Source Files**: `sagan/convolution.py`, `sagan/convolution_var.py`
