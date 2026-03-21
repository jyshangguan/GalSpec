# Practical Tips for SAGAN Fitting

## 1. Always Check Your Data Quality First

Before starting the fit, check:
- **S/N of narrow line regions**: If S/N < 20, empirical template will fail
- **Presence of cosmic rays or bad pixels**: Mask them out
- **Continuum shape**: Affects initial parameter guesses

```python
# Check S/N before creating template
cont_level = np.median(flux[cont_mask])
noise = np.std(flux[cont_mask])
peak = np.max(flux)
sn_ratio = (peak - cont_level) / noise

if sn_ratio < 20:
    print("WARNING: S/N too low for empirical template")
    print("         Use fixed-width Gaussian instead")
```

## 2. Estimate Initial Parameters from Data

**Never use fixed initial guesses** - they may be completely wrong for your spectrum.

```python
# GOOD: Estimate from data
cont_mask = ((wave > 6450) & (wave < 6480)) | ((wave > 6670) & (wave < 6700))
cont_level = np.median(flux[cont_mask])
peak_flux = np.max(flux)
line_amplitude = peak_flux - cont_level

# Estimate sigma from line width
half_max = cont_level + line_amplitude * 0.5
# ... find FWHM and convert to sigma

# BAD: Fixed guesses that may be wrong
init_amp = 100.0
init_sigma = 500  # Might not match your data
```

## 3. Start Simple, Add Complexity

Follow the iterative approach - don't start with a complex model.

| Step | What to Add | χ² Goal |
|------|-------------|---------|
| 1 | Continuum only | Establish baseline |
| 2 | + Broad line | Major improvement |
| 3 | + Narrow lines | Moderate improvement |
| 4 | + Additional components | Only if needed |

**Stop when**:
- χ² doesn't improve by >5%
- Residuals look random
- New component amplitude is consistent with zero

## 3.5 Model Selection with BIC ⭐ **IMPORTANT**

**The Problem**: Adding more components always improves χ², but is it statistically justified?

**The Solution**: Use BIC (Bayesian Information Criterion) for objective model comparison.

### Why BIC Matters

BIC balances:
- **Goodness of fit** (χ²)
- **Model complexity** (number of parameters)
- **Sample size** (ln n penalty)

```python
from sagan.utils import calculate_bic

# Fit 1-component model
model_1comp = fitter(model_1, wave, flux, weights=1/error**2)
bic1, chi2_1, n1 = calculate_bic(model_1comp, wave, flux, error)

# Fit 2-component model
model_2comp = fitter(model_2, wave, flux, weights=1/error**2)
bic2, chi2_2, n2 = calculate_bic(model_2comp, wave, flux, error)

# Compare
delta_bic = bic2 - bic1

print(f"1-comp: BIC={bic1:.1f}, χ²/ν={chi2_1/len(wave):.3f}, n={n1}")
print(f"2-comp: BIC={bic2:.1f}, χ²/ν={chi2_2/len(wave):.3f}, n={n2}")
print(f"ΔBIC = {delta_bic:.1f}")

if delta_bic < -10:
    print("→ Use 2-component (statistically justified)")
elif delta_bic > 10:
    print("→ Use 1-component (simpler preferred)")
else:
    print("→ Weak evidence, prefer simpler 1-component")
```

### BIC Interpretation

| ΔBIC (complex - simple) | Interpretation |
|------------------------|----------------|
| **ΔBIC < -10** | Strong evidence for complex model |
| **ΔBIC > 10** | Strong evidence for simple model |
| **\|ΔBIC\| < 10** | Weak evidence, prefer simple |

Lower BIC = better model (accounts for complexity).

### Use BIC, Not Residuals!

❌ **WRONG**:
```python
# Don't use arbitrary thresholds!
if np.max(np.abs(residuals)) > 5:
    add_second_component()
```

✅ **CORRECT**:
```python
# Use statistical criterion
bic1 = calculate_bic(model_1comp, wave, flux, error)[0]
bic2 = calculate_bic(model_2comp, wave, flux, error)[0]

if bic2 < bic1 - 10:
    use_2component_model()
```

### Common Use Cases

**1. Number of Broad Components**
```python
# Test 1, 2, 3 components
for n_comp in [1, 2, 3]:
    model = fit_broad_halpha(wave, flux, n_components=n_comp)
    bic[n_comp] = calculate_bic(model, wave, flux, error)[0]

best_n = np.argmin(bic)
print(f"Optimal: {best_n + 1} components")
```

**2. Include [O III] Wing?**
```python
bic_no_wing = calculate_bic(model_no_wing, wave, flux, error)[0]
bic_with_wing = calculate_bic(model_with_wing, wave, flux, error)[0]

if bic_with_wing < bic_no_wing - 10:
    print("→ Include [O III] wing")
```

**3. Include Fe II Template?**
```python
bic_no_fe = calculate_bic(model_without_fe, wave, flux, error)[0]
bic_with_fe = calculate_bic(model_with_fe, wave, flux, error)[0]

if bic_with_fe < bic_no_fe - 10:
    print("→ Include Fe II template")
```

## 4. LSF Convolution Rules

**What to convolve**:
- ✅ Broad lines: `Line_Gaussian`, `Line_MultiGauss`
- ✅ Iron templates: `IronTemplate`
- ✅ Stellar continua: `StarSpectrum`

**What NOT to convolve**:
- ❌ Empirical narrow line templates (`Line_template` with `template_velc`)
  - Already includes instrumental broadening
  - Convolving again would over-broaden

**Example**:
```python
# CORRECT
broad_conv = sagan.convolve_lsf(broad_line, wavec=..., resolving_power=...)
model = cont + broad_conv + narrow_template  # Template NOT convolved

# WRONG
narrow_conv = sagan.convolve_lsf(narrow_template, ...)  # Don't do this!
model = cont + broad_conv + narrow_conv
```

## 5. Use Appropriate Line Fitters

| Line Type | Recommended Class | Why |
|-----------|-------------------|-----|
| **Doublets** ([S II], [N II], [O III]) | `Line_MultiGauss_doublet` | Ties kinematics, prevents degeneracies |
| **Broad single lines** | `Line_MultiGauss` | Multiple components possible |
| **Narrow forbidden lines** | `Line_template` | Uses empirical profile |
| **Absorption troughs** | `Line_Absorption` | Physical model for BAL |

**Doublet example**:
```python
# Use Line_MultiGauss_doublet for [S II]
sii_doublet = sagan.Line_MultiGauss_doublet(
    n_components=1,
    amp_c0=10.0,    # [S II] 6716 amplitude
    amp_c1=3.0,     # [S II] 6731 amplitude
    dv_c=0,         # Shared velocity shift
    sigma_c=100,    # Shared width
    wavec0=line_wave_dict['SII_6716'],
    wavec1=line_wave_dict['SII_6731'],
    name='SII_doublet'
)
```

## 6. When Narrow Lines Are Weak

**Problem**: If [S II] or [O III] have S/N < 20, the empirical template approach fails.

**Solution**: Use a fixed-width Gaussian based on instrumental resolution.

```python
# Calculate instrumental LSF width
lsf_sigma = ls_km / (resolving_power * 2.3548)

# Create fixed-width Gaussian template
velc_temp = np.linspace(-500, 500, 1000)  # km/s
flux_temp = np.exp(-0.5 * (velc_temp / lsf_sigma)**2)
flux_temp = flux_temp / np.max(flux_temp)

# Use this template for all narrow lines
nha = sagan.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=...,
    dv=...,
    wavec=line_wave_dict['Halpha'],
    name='nHalpha'
)
```

**When to use fixed-width vs empirical template**:
- **Empirical template**: S/N > 20, clear narrow line detection
- **Fixed-width Gaussian**: S/N < 20, or narrow lines absent

## 6.5 Continuum Modeling for AGN

**First Choice: WindowedPowerLaw1D**

For AGN spectra, always try a power-law continuum first:

```python
from sagan.continuum import WindowedPowerLaw1D

cont = WindowedPowerLaw1D(
    amplitude=cont_level,    # From data
    x_0=line_wave_dict['Hbeta'],  # Reference wavelength
    alpha=-1.0,              # Power-law index (F_ν ∝ ν^α)
    x_min=wave_min,          # Window start - prevents extrapolation
    x_max=wave_max,          # Window end - prevents extrapolation
    name='continuum'
)
```

**Why Power-Law?**
- AGN continuum is physical: F_ν ∝ ν^α
- α ≈ -0.5 to -1.5 for typical AGN
- Power-law from accretion disk emission

**Why Window It?**
- Prevents numerical issues at λ → 0
- Constrains model to fitting range
- Avoids extrapolation artifacts

**Parameter Estimation**:
```python
# Continuum level from line-free regions
cont_regions = ((wave > 4500) & (wave < 4700)) | \
               ((wave > 5100) & (wave < 5250))
cont_level = np.median(flux[cont_regions])

# Power-law index from two continuum regions
cont1 = np.median(flux[(wave > 4500) & (wave < 4600)])
cont2 = np.median(flux[(wave > 5300) & (wave < 5400)])
alpha_init = -np.log(cont2/cont1) / np.log(5350/4550)

print(f"Continuum: {cont_level:.2f}, α={alpha_init:.2f}")
```

**When Power-Law Fails**:

If you get numerical errors (NonFiniteValueError):

```python
# Fallback: Polynomial1D
from astropy.modeling import models
cont = models.Polynomial1D(degree=1, c0=cont_level, c1=0)
```

This is rare with proper windowing. Only use polynomial if:
- Power-law produces numerical errors
- Very small range (< 50 Å)
- Non-AGN object (e.g., star-forming galaxy)

## 7. Monitor χ² and Residuals

**χ² progression**:
- Good: χ² decreases significantly (>5%) at each step
- Bad: χ² stays the same or increases

**Residual analysis**:
```python
resid = (flux - model(wave)) / ferr

# Check for structure
resid_std = np.std(resid)
resid_max = np.max(np.abs(resid))

if resid_max > 5 * resid_std:
    print("Structured residuals - missing component")
elif resid_max < 3 * resid_std:
    print("Good fit - residuals look random")
```

## 8. Common Pitfalls to Avoid

❌ **Don't**: Start with the full complex model
✅ **Do**: Build up iteratively

❌ **Don't**: Use fixed initial guesses for all spectra
✅ **Do**: Estimate from data

❌ **Don't**: Convolving empirical templates
✅ **Do**: Only convolve broad lines and high-res templates

❌ **Don't**: Fit doublet components separately
✅ **Do**: Use `Line_MultiGauss_doublet`

❌ **Don't**: Add components without checking if they're needed
✅ **Do**: Monitor χ² improvement and residuals

❌ **Don't**: Let sigma hit bounds (e.g., 9000 km/s)
✅ **Do**: Check parameter values are physical

## 9. Debugging Checklist

If your fit looks wrong:

1. **Check initial parameters**: Are they reasonable?
2. **Check template quality**: Is S/N high enough?
3. **Check convolution**: Are you convolving the right things?
4. **Check parameter values**: Is any parameter hitting a bound?
5. **Check residuals**: Do they show structure?
6. **Check χ²**: Is it improving with each step?

## 10. Final Model Validation

Before accepting the fit:

1. **Visual inspection**: Does the model match the data?
2. **Parameter values**: Are all parameters physical?
   - Broad sigma: 500-5000 km/s
   - Narrow sigma: 50-500 km/s
   - Velocity shifts: -500 to +500 km/s typically
3. **χ²/ν**: Should be close to 1-5 for good data
4. **Residuals**: Should be random, no structure
5. **Component amplitudes**: All should be non-zero (if included)

---

**Remember**: The best fit is the simplest model that adequately describes the data.
