# Line Profile Models (`sagan.line_profile`)

All line profile models for emission and absorption lines in SAGAN.

## Table of Contents

1. [Line_Gaussian](#line_gaussian)
2. [Line_GaussHermite](#line_gausshermite)
3. [Line_MultiGauss](#line_multigauss)
4. [Line_MultiGauss_doublet](#line_multigauss_doublet)
5. [Line_GaussHermite_doublet](#line_gausshermite_doublet)
6. [Line_template](#line_template)
7. [Line_Absorption](#line_absorption)
8. [Line_Absorption_doublet](#line_absorption_doublet)

---

## Line_Gaussian

Basic Gaussian emission/absorption line.

```python
sagan.Line_Gaussian(
    amplitude=1.0,          # Line peak amplitude
    dv=0,                   # Velocity shift (km/s), bounds: (-2000, 2000)
    sigma=200,              # Velocity dispersion (km/s), bounds: (20, 10000)
    wavec=5000,             # Central wavelength (Å) [fixed=True]
    name='line_name'
)
```

**Parameters**:
- `amplitude`: Peak amplitude of the line (bounds: (0, None))
- `dv`: Velocity offset from rest wavelength in km/s (bounds: (-2000, 2000))
- `sigma`: Velocity dispersion (Gaussian width) in km/s (bounds: (20, 10000))
- `wavec`: Central rest wavelength in Å (fixed by default)

**Usage**: Single Gaussian lines, suitable for narrow lines or simple broad lines.

**Equation**: `F = amplitude * exp(-0.5 * ((v - dv)/sigma)²)`

**Example**:
```python
from sagan.utils import line_wave_dict
line = sagan.Line_Gaussian(
    amplitude=10.0,
    dv=-50,
    sigma=300,
    wavec=line_wave_dict['Halpha'],
    name='Halpha'
)
```

---

## Line_GaussHermite

Fourth-order Gauss-Hermite expansion for asymmetric lines.

```python
sagan.Line_GaussHermite(
    amplitude=1.0,
    dv=0,
    sigma=200,
    h3=0,                   # Skewness parameter, bounds: (-0.4, 0.4)
    h4=0,                   # Kurtosis parameter, bounds: (-0.4, 0.4)
    wavec=5000,
    clip=True,              # Set negative flux to 0
    name='line_name'
)
```

**Parameters**:
- `amplitude`: Peak amplitude (bounds: (0, None))
- `dv`: Velocity shift in km/s (bounds: (-2000, 2000))
- `sigma`: Velocity dispersion in km/s (bounds: (20, 10000))
- `h3`: Skewness parameter (bounds: (-0.4, 0.4))
- `h4`: Kurtosis parameter (bounds: (-0.4, 0.4))
- `wavec`: Central wavelength (fixed)
- `clip`: If True, replace negative values with 0

**Usage**: Non-Gaussian line profiles, asymmetric emission lines.

**Equation**: `F = G * (1 + h3*H3 + h4*H4)`

where G is the Gaussian and H3, H4 are Hermite polynomials.

**Example**:
```python
line = sagan.Line_GaussHermite(
    amplitude=10.0,
    dv=-50,
    sigma=500,
    h3=0.1,      # Positive skew
    h4=-0.05,    # Negative kurtosis
    wavec=line_wave_dict['Hbeta'],
    name='Hbeta_asymmetric'
)
```

---

## Line_MultiGauss

Multiple Gaussian components for complex line profiles.

```python
sagan.Line_MultiGauss(
    n_components=3,         # Number of Gaussian components
    amp_c=1.0,              # Core amplitude
    dv_c=0,                 # Core velocity shift (km/s)
    sigma_c=400,            # Core velocity dispersion (km/s)
    wavec=6563,             # Central wavelength
    par_w={                 # Wind/broad components
        'amp_w0': 0.3,      # Relative amplitude of wind 0
        'dv_w0': 500,       # Velocity offset of wind 0 (km/s)
        'sigma_w0': 1500,   # Dispersion of wind 0 (km/s)
        # Add amp_w1, dv_w1, sigma_w1 for more components
    },
    name='broad_line'
)
```

**Parameters**:
- `n_components`: Number of Gaussian components (int ≥ 1)
- `amp_c`: Amplitude of core component (bounds: (0, None))
- `dv_c`: Velocity shift of core (km/s) (bounds: (-2000, 2000))
- `sigma_c`: Dispersion of core (km/s) (bounds: (20, 10000))
- `wavec`: Central wavelength (fixed)
- `par_w`: Dictionary of wind component parameters
  - `amp_w{i}`: Relative amplitude of wind i (bounds: (0, 1))
  - `dv_w{i}`: Velocity offset of wind i (km/s) (bounds: (-8000, 8000))
  - `sigma_w{i}`: Dispersion of wind i (km/s) (bounds: (0, 10000))

**Usage**: Complex broad line profiles with multiple kinematic components (core + winds).

**Example**:
```python
# 3-component broad Hα
bha = sagan.Line_MultiGauss(
    n_components=3,
    amp_c=48.0,      # Core
    dv_c=-100,
    sigma_c=400,
    amp_w0=0.15,     # Intermediate wind
    dv_w0=20,
    sigma_w0=1400,
    amp_w1=0.15,     # Very broad wind
    dv_w1=800,
    sigma_w1=200,
    wavec=line_wave_dict['Halpha'],
    name='Broad Halpha'
)

# Access subcomponents
for comp in bha.subcomponents:
    print(f"{comp.name}: amp={comp.amplitude.value:.2f}, "
          f"sigma={comp.sigma.value:.1f}")
```

---

## Line_MultiGauss_doublet

Doublet with multi-Gaussian profile (e.g., [O III] 4959,5007).

```python
sagan.Line_MultiGauss_doublet(
    n_components=2,
    amp_c0=1.0,             # Amplitude of line 1
    amp_c1=1.0,             # Amplitude of line 2
    dv_c=0,                 # Common velocity shift
    sigma_c=200,            # Common dispersion
    wavec0=5007,            # Line 1 wavelength
    wavec1=4959,            # Line 2 wavelength
    par_w={},               # Wind components (shared)
    name='doublet'
)
```

**Parameters**: Similar to Line_MultiGauss but with two central wavelengths.

**Usage**: Doublet lines with shared kinematics.

**Example**:
```python
# [O III] blue wing
o3_wing = sagan.Line_MultiGauss_doublet(
    n_components=2,
    amp_c0=65.0,      # 5007
    amp_c1=22.0,      # 4959 (will be tied to 5007)
    dv_c=-70,         # Blue-shifted
    sigma_c=130,
    amp_w0=0.43,
    dv_w0=-80,
    sigma_w0=300,
    wavec0=line_wave_dict['OIII_5007'],
    wavec1=line_wave_dict['OIII_4959'],
    name='OIII_wing'
)
```

---

## Line_GaussHermite_doublet

Doublet with Gauss-Hermite profile.

```python
sagan.Line_GaussHermite_doublet(
    amp_c0=1.0,             # Amplitude of line 1
    amp_c1=1.0,             # Amplitude of line 2
    dv_c=0,                 # Common velocity shift
    sigma_c=200,            # Common dispersion
    h3=0,                   # Skewness
    h4=0,                   # Kurtosis
    wavec0=5007,            # Line 1 wavelength
    wavec1=4959,            # Line 2 wavelength
    clip=True,
    name='doublet_gh'
)
```

**Usage**: Asymmetric doublet profiles.

---

## Line_template

Template-based line profile from observed or theoretical spectra.

```python
sagan.Line_template(
    template_velc=velc_array,    # Velocity array (km/s)
    template_flux=flux_array,    # Flux array (normalized)
    amplitude=1.0,
    dv=0,                        # Velocity shift (km/s)
    wavec=4861,                  # Central wavelength (Å)
    name='narrow_line'
)
```

**Parameters**:
- `template_velc`: 1D velocity array in km/s
- `template_flux`: 1D flux array (normalized)
- `amplitude`: Line amplitude (bounds: (0, None))
- `dv`: Velocity shift (bounds: (-2000, 2000))
- `wavec`: Central wavelength (fixed)

**Usage**: Narrow lines using high-S/N template spectra.

**Example**:
```python
# Load template from [S II] region
velc_temp, flux_temp = np.loadtxt('narrow_template.txt').T

# Use for narrow Hα
nha = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=50.0,
    dv=10,
    wavec=line_wave_dict['Halpha'],
    name='nHalpha'
)
```

---

## Line_Absorption

Gaussian absorption line profile.

```python
sagan.Line_Absorption(
    logtau0=0.0,            # Log optical depth at line center
    dv=0,                   # Velocity shift (km/s)
    sigma=200,              # Velocity dispersion (km/s)
    Cf=1.0,                 # Covering fraction (0-1)
    wavec=6563,             # Central wavelength (Å)
    name='absorption'
)
```

**Parameters**:
- `logtau0`: Log₁₀ of optical depth at line center (bounds: (-2, 2))
- `dv`: Velocity shift (bounds: (-10000, 10000))
- `sigma`: Velocity dispersion (bounds: (20, 10000))
- `Cf`: Covering fraction (bounds: (0, 1))
- `wavec`: Central wavelength (fixed)

**Usage**: BAL troughs, intrinsic/absorption lines.

**Equation**: `F = 1 - Cf + Cf * exp(-τ₀ * exp(-0.5 * ((v - dv)/σ)²))`

where τ₀ = 10^logtau0

**Example**:
```python
# BAL trough in Hα
aha = sagan.Line_Absorption(
    logtau0=2.0,      # Strong absorption
    dv=-160,          # Blue-shifted
    sigma=40,
    Cf=0.4,           # Partial covering
    wavec=line_wave_dict['Halpha'],
    name='Abs. Halpha'
)
```

---

## Line_Absorption_doublet

Doublet absorption lines (e.g., CIV 1548,1551).

```python
sagan.Line_Absorption_doublet(
    logtau0=0.0,            # Log τ for line 1
    logtau1=0.0,            # Log τ for line 2
    dv=0,                   # Common velocity shift
    sigma=200,              # Common dispersion
    Cf=1.0,                 # Common covering fraction
    wavec0=1548,            # Line 1 wavelength
    wavec1=1550,            # Line 2 wavelength
    name='doublet_abs'
)
```

**Usage**: Absorption doublets with shared kinematics.

**Equation**: `F = (1 - Cf + Cf*exp(-τ₀)) * (1 - Cf + Cf*exp(-τ₁))`

---

**Module**: `sagan.line_profile`
**Source File**: `sagan/line_profile.py`
