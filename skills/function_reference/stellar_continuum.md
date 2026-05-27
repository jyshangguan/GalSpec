# Stellar Continuum (`galspec.stellar_continuum`)

Stellar spectrum template models in GalSpec.

## Table of Contents

1. [StarSpectrum](#starspectrum)
2. [Multi_StarSpectrum](#multi_starspectrum)

---

## StarSpectrum

Single stellar spectrum template.

```python
galspec.StarSpectrum(
    amplitude=1.0,
    star_type='G',          # 'A', 'F', 'G', 'K', or 'M'
    velscale=200,           # Velocity scale for rebinning (km/s)
    delta_z=0,              # Redshift offset (km/s)
    sigma=200,              # Velocity dispersion (km/s)
    name='stellar'
)
```

**Available Stellar Templates**:
- A0V: HD 97633
- F: HD 89254
- G: HD 140027
- K: HD 49520
- M: HD 44478

**Example**:
```python
stellar = galspec.StarSpectrum(
    amplitude=0.3,
    star_type='G',
    velscale=200,
    delta_z=0,
    sigma=150,              # Stellar velocity dispersion
    name='stellar'
)
```

---

## Multi_StarSpectrum

Multiple stellar components (e.g., bulge + disk).

```python
galspec.Multi_StarSpectrum(
    n_components=2,
    star_types=['G', 'K'],   # Stellar types for each component
    amplitudes=[1.0, 0.5],   # Relative amplitudes
    velscale=200,
    delta_z=0,
    sigma=200,
    name='multi_stellar'
)
```

---

**Module**: `galspec.stellar_continuum`
**Source File**: `galspec/stellar_continuum.py`
