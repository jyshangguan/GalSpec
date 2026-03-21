# Iron Templates (`sagan.iron_template`)

Iron emission template models for AGN spectra in SAGAN.

## Table of Contents

1. [IronTemplate](#irontemplate)
2. [IronTemplate_new](#irontemplate_new)
3. [IronTemplate_tied](#irontemplate_tied)

---

## IronTemplate

Iron emission template from AGN.

```python
sagan.IronTemplate(
    amplitude=1.0,
    stddev=900/2.3548,      # Velocity dispersion (km/s)
    z=0,                    # Redshift
    template_name='park2022'  # 'park2022' or 'boroson1992'
)
```

**Available Templates**:
- `park2022`: Based on Mrk 493 (Park et al. 2022)
- `boroson1992`: Based on I Zw 1 (Boroson & Green 1992)

**Intrinsic Widths**:
- park2022: 800/2.3548 km/s
- boroson1992: 900/2.3548 km/s

**Example**:
```python
iron = sagan.IronTemplate(
    amplitude=0.8,
    stddev=900/2.3548,
    z=0.003,
    template_name='park2022'
)
```

---

## IronTemplate_new

Enhanced iron template with velocity shift parameter.

```python
sagan.IronTemplate_new(
    amplitude=1.0,
    sigma=900/2.3548,       # Velocity dispersion (km/s)
    z=0,                    # Redshift [fixed parameter]
    dv=0,                   # Velocity shift (km/s), bounds: (-2000, 2000)
    template_name='park2022'
)
```

**Advantages**: Can fit iron template velocity offset independently of systemic redshift.

**Example**:
```python
iron_new = sagan.IronTemplate_new(
    amplitude=0.8,
    sigma=900/2.3548,
    z=0,
    dv=-100,                # Free velocity shift
    template_name='park2022'
)
```

---

## IronTemplate_tied

Iron template with tied convolution kernel.

```python
sagan.IronTemplate_tied(
    amplitude=1.0,
    stddev=910/2.3548,
    z=0,
    template_name='park2022',
    kernel=None             # Optional: Line_MultiGauss object
)
```

---

**Module**: `sagan.iron_template`
**Source File**: `sagan/iron_template.py`
