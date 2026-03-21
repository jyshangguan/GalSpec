# Parameter Tying Functions

Link parameters across different components to reduce degrees of freedom in SAGAN.

## Overview

Parameter tying allows you to link parameters across different spectral components, enforcing physical relationships and reducing the number of free parameters in your fit.

## Table of Contents

1. [Absorption Tying](#absorption-tying)
2. [MultiGauss Tying](#multigauss-tying)
3. [Template Tying](#template-tying)
4. [Stellar Continuum Tying](#stellar-continuum-tying)

---

## Absorption Tying

Tie absorption parameters across Hα and Hβ (or other lines).

```python
# Tie absorption parameters across Hα and Hβ
mcmc_hb['Abs. Hbeta'].dv.tied = sagan.tie_Absorption_dv('Abs. Halpha')
mcmc_hb['Abs. Hbeta'].sigma.tied = sagan.tie_Absorption_sigma('Abs. Halpha')
mcmc_hb['Abs. Hbeta'].Cf.tied = sagan.tie_Absorption_Cf('Abs. Halpha')

# Tie τ₀ with theoretical ratio (Hβ/Hα = 7.13 for Case B)
mcmc_hb['Abs. Hbeta'].logtau0.tied = sagan.tie_Absorption_logtau0('Abs. Halpha', ratio=7.13)
```

**Physical Basis**: In BAL outflows, absorption troughs in different lines should have the same kinematics (dv, sigma) and covering fraction (Cf). The optical depths scale with theoretical line ratios.

**Available Functions**:
- `sagan.tie_Absorption_dv(reference_name)`
- `sagan.tie_Absorption_sigma(reference_name)`
- `sagan.tie_Absorption_Cf(reference_name)`
- `sagan.tie_Absorption_logtau0(reference_name, ratio)`

---

## MultiGauss Tying

Tie MultiGauss broad line parameters across different lines.

```python
# Tie MultiGauss parameters
m_hb['Broad Hbeta'].dv_c.tied = sagan.tie_MultiGauss_dv_c('Broad Halpha')
m_hb['Broad Hbeta'].sigma_c.tied = sagan.tie_MultiGauss_sigma_c('Broad Halpha')
m_hb['Broad Hbeta'].amp_w0.tied = sagan.tie_MultiGauss_amp_w0('Broad Halpha')
m_hb['Broad Hbeta'].dv_w0.tied = sagan.tie_MultiGauss_dv_w0('Broad Halpha')
m_hb['Broad Hbeta'].sigma_w0.tied = sagan.tie_MultiGauss_sigma_w0('Broad Halpha')
```

**Physical Basis**: Broad line regions in AGNs should have similar kinematics across different hydrogen lines. Core and wind components track the same gas dynamics.

**Available Functions**:
- `sagan.tie_MultiGauss_dv_c(reference_name)`
- `sagan.tie_MultiGauss_sigma_c(reference_name)`
- `sagan.tie_MultiGauss_amp_w0(reference_name)`
- `sagan.tie_MultiGauss_dv_w0(reference_name)`
- `sagan.tie_MultiGauss_sigma_w0(reference_name)`
- `sagan.tie_MultiGauss_amp_w1(reference_name)`
- `sagan.tie_MultiGauss_dv_w1(reference_name)`
- `sagan.tie_MultiGauss_sigma_w1(reference_name)`

---

## Template Tying

Tie template-based narrow line parameters.

```python
# Tie template amplitude and dv
m_hb['nHbeta'].amplitude.tied = sagan.tie_template_amplitude('nHalpha', ratio=2.86)
m_hb['nHbeta'].dv.tied = sagan.tie_template_dv('nHalpha')
```

**Physical Basis**: Narrow lines come from the same low-density gas, so they should have:
- Same velocity shift (dv)
- Amplitude ratios given by theoretical recombination values

**Theoretical Ratios** (Case B, T=10⁴ K):
- Hβ/Hα = 2.86
- Hγ/Hα = 0.108
- Hδ/Hα = 0.048

**Available Functions**:
- `sagan.tie_template_amplitude(reference_name, ratio)`
- `sagan.tie_template_dv(reference_name)`

---

## Stellar Continuum Tying

Tie stellar population parameters across multiple components.

```python
# Tie stellar parameters between multiple stellar populations
m_stellar2['stellar_g'].delta_z.tied = sagan.tie_StarSpectrum_deltaz('stellar_k')
m_stellar2['stellar_g'].sigma.tied = sagan.tie_StarSpectrum_sigma('stellar_k')
```

**Physical Basis**: Different stellar populations in a galaxy typically share the same systemic velocity and velocity dispersion.

**Available Functions**:
- `sagan.tie_StarSpectrum_deltaz(reference_name)`
- `sagan.tie_StarSpectrum_sigma(reference_name)`

---

## Usage Notes

1. **Reference Names**: When tying parameters, use the exact `name` of the reference component as specified in the model.

2. **Apply Before Fitting**: Tying must be done *before* calling the fit method.

3. **Checking Ties**: After tying, you can verify with:
   ```python
   print(component.parameter.tied)
   ```

4. **Untying**: To remove a tie:
   ```python
   component.parameter.tied = None
   ```

---

**Related Functions**: Available in various `sagan` modules
