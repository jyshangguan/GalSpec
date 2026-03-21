# Continuum Models (`sagan.continuum`)

Continuum models for AGN and stellar emission in SAGAN.

## Table of Contents

1. [WindowedPowerLaw1D](#windowedpowerlaw1d)
2. [BlackBody](#blackbody)
3. [BalmerPseudoContinuum](#balmerpseudocontinuum)
4. [extinction_ccm89](#extinction_ccm89)

---

## WindowedPowerLaw1D

Power law continuum active only within [x_min, x_max].

```python
sagan.WindowedPowerLaw1D(
    amplitude=1.0,
    x_0=5000,               # Reference wavelength (Å)
    alpha=-1.0,             # Power law index (F_ν ∝ ν^α)
    x_min=4500,             # Window start (Å)
    x_max=5500,             # Window end (Å)
    name='continuum'
)
```

**Parameters**:
- `amplitude`: Amplitude at x_0
- `x_0`: Reference wavelength (fixed)
- `alpha`: Power law index (negative for typical AGN)
- `x_min`: Window start (fixed)
- `x_max`: Window end (fixed)

**Usage**: AGN power-law continuum.

**Equation**: `F = amplitude * (x/x_0)^(-alpha)` for x_min ≤ x < x_max, else 0

**Example**:
```python
cont = sagan.WindowedPowerLaw1D(
    amplitude=10.0,
    x_0=5100,
    alpha=-1.2,
    x_min=4200,
    x_max=5400,
    name='Hb_continuum'
)
```

---

## BlackBody

Black body radiation model.

```python
sagan.BlackBody(
    temperature=5000,       # Temperature (K)
    scale=1.0,              # Amplitude scale
    name='blackbody'
)
```

**Parameters**:
- `temperature`: Black body temperature in K
- `scale`: Amplitude scale factor

**Usage**: Stellar continuum approximation.

---

## BalmerPseudoContinuum

Balmer jump and high-order Balmer lines (Kovačević et al. 2013).

```python
sagan.BalmerPseudoContinuum(
    i_ref=1.0,              # Hβ reference intensity
    sigma=1000,             # Doppler width (km/s)
    dv=0,                   # Velocity shift (km/s)
    te=15000,               # Electron temperature (K) [fixed]
    tau_be=1.0,             # Optical depth at Balmer edge [fixed]
    lambda_be=3646,         # Balmer edge wavelength (Å) [fixed]
    name='balmer'
)
```

**Usage**: Accurate modeling of Balmer continuum and high-order lines for stellar populations.

---

## extinction_ccm89

Extinction model of Cardelli et al. (1989).

```python
sagan.extinction_ccm89(
    a_v=0,                  # A_V extinction in magnitudes
    r_v=3.1,                # R_V = A_V / E(B-V)
    name='extinction'
)
```

**Usage**: Milky Way extinction correction.

---

**Module**: `sagan.continuum`
**Source File**: `sagan/continuum.py`
