# Measurement Methods (`sagan.measure_method`)

Physical parameter measurements from fitted models in SAGAN.

## Table of Contents

1. [line_emission_fwhm](#line_emission_fwhm)

---

## line_emission_fwhm

Calculate FWHM of emission lines from fitted models.

```python
from sagan.measure_method import line_emission_fwhm

fwhm, w_l, w_r, w_peak = line_emission_fwhm(
    model,                  # Fitted model
    ['Broad Halpha'],       # List of line component names
    wave,                   # Wavelength array
    line_wave_dict['Halpha'],  # Line center wavelength
    x0_limit=None,          # Left boundary (optional)
    x1_limit=None,          # Right boundary (optional)
    fwhm_disp=None          # Instrument dispersion to remove
)

print(f"Broad Hα FWHM: {fwhm:.1f} km/s")
```

**Parameters**:
- `model`: Fitted CompoundModel
- `component_names`: List of component names to measure
- `wave`: Wavelength array
- `wavec`: Line center wavelength in Å
- `x0_limit`: Optional left boundary for calculation
- `x1_limit`: Optional right boundary for calculation
- `fwhm_disp`: Instrumental FWHM to remove (in quadrature)

**Returns**:
- `fwhm`: Line FWHM in km/s
- `w_l`: Left boundary wavelength (half maximum)
- `w_r`: Right boundary wavelength (half maximum)
- `w_peak`: Peak wavelength

**Instrumental Correction**:
When `fwhm_disp` is provided, the instrumental FWHM is removed in quadrature:
```python
fwhm_intrinsic = sqrt(fwhm_observed² - fwhm_disp²)
```

**Example**:
```python
# Measure broad Hα with instrumental correction
fwhm_ha, wl, wr, wp = line_emission_fwhm(
    mcmc_ha.model_best,
    ['Broad Halpha'],
    wave_ha,
    line_wave_dict['Halpha'],
    fwhm_disp=180  # Instrumental FWHM at Hα
)

# Measure narrow Hα
fwhm_nha, *_ = line_emission_fwhm(
    mcmc_ha.model_best,
    ['nHalpha'],
    wave_ha,
    line_wave_dict['Halpha']
)

print(f"Broad Hα: {fwhm_ha:.0f} km/s")
print(f"Narrow Hα: {fwhm_nha:.0f} km/s")

# Calculate black hole mass (using broad Hα)
from sagan.utils import line_wave_dict

# Virial mass formula (Vestergaard & Peterson 2006)
import numpy as np
l_ha = 10**component.amplitude.value  # Luminosity in erg/s
mass = 1.4e6 * (l_ha / 1e42)**0.64 * (fwhm_ha / 1000)**2
print(f"M_BH = {mass:.2e} M☉")
```

---

**Module**: `sagan.measure_method`
**Source File**: `sagan/measure_method.py`
