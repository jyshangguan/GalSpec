# SAGAN Documentation Updates Summary

**Date**: 2025-03-20
**Session**: BIC Model Selection & Continuum Guidance

---

## Files Updated

### 1. ✅ `skills/strategy_types/type1_agn.md`

#### Added Section: "Model Selection with BIC"
- **Location**: After "Iterative Model Building", before "Data Preparation"
- **Content**:
  - BIC formula and interpretation
  - When to use BIC for model comparison
  - Common use cases (number of broad components, [O III] wing, Fe II template)
  - **Key message**: Use BIC, not residual thresholds!
  - Complete code examples for BIC calculation and interpretation

#### Updated Section: "Define Fitting Windows"
- **Key changes**:
  - ⚠️ **CRITICAL**: Always plot first!
  - Emphasized that ranges are EXAMPLES, not fixed rules
  - Added guidance for very broad lines (FWHM > 5000 km/s)
  - Decision process for adjusting ranges based on plotted data
  - Why visual inspection matters
  - Removed fixed wavelength ranges - now emphasizes adjusting based on data

#### Updated Section: "Continuum: WindowedPowerLaw1D"
- **Key changes**:
  - **Power-law is FIRST choice for AGN** (not "one option")
  - Added parameter estimation code
  - Clarified when power-law fails and how to use polynomial fallback
  - Added physical explanation (AGN continuum is power-law)

#### Updated Table of Contents
- Added "Model Selection with BIC" to TOC

---

### 2. ✅ `skills/strategy_types/practical_tips.md`

#### Added Section: "Model Selection with BIC"
- **Location**: Section 3.5 (after "Start Simple, Add Complexity")
- **Content**:
  - Why BIC matters
  - Complete code examples
  - BIC interpretation table
  - **Key message**: Use BIC, not residuals!
  - Common use cases with code

#### Added Section: "Continuum Modeling for AGN"
- **Location**: Section 6.5 (after "When Narrow Lines Are Weak")
- **Content**:
  - WindowedPowerLaw1D as FIRST choice
  - Why power-law (physical AGN continuum)
  - Why window it (prevent numerical issues)
  - Parameter estimation from data
  - When power-law fails (polynomial fallback)

---

## Key Messages Emphasized

### 1. BIC for Model Selection (NEW!)
- **Use BIC, not residual thresholds**
- ΔBIC < -10: Strong evidence for complex model
- ΔBIC > 10: Strong evidence for simple model
- |ΔBIC| < 10: Weak evidence, prefer simple
- Works for: number of components, [O III] wing, Fe II inclusion

### 2. Power-Law Continuum (CORRECTED!)
- **Always use WindowedPowerLaw1D FIRST for AGN**
- Physical model (F_ν ∝ ν^α)
- Window to prevent extrapolation issues
- Polynomial only as fallback (numerical errors, very small range)

### 3. Wavelength Ranges (CLARIFIED!)
- **Ranges are EXAMPLES, not rules**
- Always plot first!
- Adjust based on:
  - Continuum visibility on both sides
  - Broad line width (very broad → wider range)
  - Contaminating lines (exclude them)
- For FWHM > 5000 km/s: extend range

---

## Code Examples Added

### BIC Calculation
```python
from sagan.utils import calculate_bic

bic, chi2, n_params = calculate_bic(model_fit, wave, flux, error)

# Compare models
bic1 = calculate_bic(model_1comp, wave, flux, error)[0]
bic2 = calculate_bic(model_2comp, wave, flux, error)[0]

if bic2 < bic1 - 10:
    print("→ Use 2-component")
```

### Power-Law Continuum
```python
from sagan.continuum import WindowedPowerLaw1D

cont = WindowedPowerLaw1D(
    amplitude=cont_level,
    x_0=line_wave_dict['Hbeta'],
    alpha=-1.0,
    x_min=wave_min,  # Prevent extrapolation
    x_max=wave_max,
    name='continuum'
)
```

### Wavelength Range Selection
```python
# Plot first!
fig, ax = plt.subplots()
ax.plot(wave_rest, flux)
ax.axvline(6563, color='r', linestyle='--', label='Hα')

# Adjust based on plot
ha_region = (wave_rest > 6450) & (wave_rest < 6700)  # MODIFY AS NEEDED
```

---

## Impact

### Before Updates
- Users might use arbitrary residual thresholds
- Continuum choice unclear (polynomial vs power-law)
- Fixed wavelength ranges regardless of data
- No statistical model comparison

### After Updates
- ✅ Statistical model comparison with BIC
- ✅ Clear guidance: power-law first for AGN
- ✅ Emphasized visual inspection and range adjustment
- ✅ Complete code examples for all concepts

---

## Related Files

### ✅ `sagan/utils.py`
- Added `calculate_bic()` function
- Returns: (bic, chi2, n_params)
- Handles compound models correctly
- Added to `__all__` for proper exporting

### ✅ `skills/function_reference.md`
- Added complete documentation for `calculate_bic()`
- Usage examples
- Model comparison guidelines
- BIC interpretation table

---

## Future Work

Potential additions:
1. Example notebook demonstrating BIC-based model selection
2. More examples in `type1_agn.md` showing BIC in practice
3. Cross-references in main guide to BIC section

---

**Status**: ✅ Documentation updated
**All critical lessons from Hα and Hβ fitting documented**
