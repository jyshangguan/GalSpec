---
name: sagan-spectral-fitting
description: This skill should be used when the user asks to "fit astronomical spectra", "spectral fitting", "fit AGN spectrum", "fit emission lines", "fit absorption lines", "fit Type 1 AGN", "fit BAL QSO", mentions "SAGAN package", "spectral analysis", "line profile fitting", "MCMC fitting", "Bayesian inference", or discusses fitting astronomical spectra with complex line profiles, iron templates, BAL troughs, or emission/absorption line decomposition. Also use when user needs to model continuum, broad lines, narrow lines, absorption components, or measure physical parameters from spectra.

**CRITICAL KEYWORD TRIGGERS - Must read specified files BEFORE creating a plan:**
- "doublet", "[S II]", "S II", "[N II]", "N II", "[O III]", "O III", "narrow line template", "LSF", "instrumental profile" → MUST read: `fitting_strategies/narrow_line_template.md`
- "Type 1 AGN", "broad Hα", "broad Hβ", "broad line", "Balmer" → MUST read: `fitting_strategies/type1_agn.md`
- "BAL", "absorption trough", "broad absorption line" → MUST read: `fitting_strategies/type1_agn.md` (BAL section)
version: 1.0.0
---

# SAGAN Spectral Fitting

**Description:** SAGAN (Spectral Analysis for Galaxies and AGN) is a comprehensive Python package for fitting astronomical spectra, particularly designed for AGN and galaxy spectroscopy with Bayesian inference tools.

**Author:** Shangguan et al.
**Version:** 1.0.0
**Package Location:** `/Users/shangguan/Softwares/my_modules/SAGAN/`

---

## When to Use This Skill

Use this skill when you need to:
- Fit astronomical spectra with complex line profiles
- Decompose spectra into continuum, broad lines, narrow lines, and absorption components
- Perform Bayesian parameter estimation with MCMC or nested sampling
- Fit Type 1 AGN or BAL QSO spectra with iron templates
- Model broad emission lines with multiple components
- Fit absorption troughs in BAL QSOs
- Measure black hole masses from broad line widths
- Apply instrumental LSF convolution to models
- Use line template fitting for narrow forbidden lines

**Supported Object Types:**
- Type 1 AGN (broad-line AGN)
- BAL QSOs (broad absorption line quasars)
- Type 2 AGN (narrow-line AGN)
- Galaxies with stellar continua
- Any object with emission/absorption lines

---

## ⚠️ Task-Specific Documentation Routing

**BEFORE creating a plan, read the specified file first:**

| Task / Keywords | MUST Read This File |
|-----------------|---------------------|
| **Fit doublet lines** ([S II], [N II], [O III]) | `fitting_strategies/narrow_line_template.md` |
| **Derive narrow line template** from doublets | `fitting_strategies/narrow_line_template.md` |
| **Characterize instrumental LSF** or line profile | `fitting_strategies/narrow_line_template.md` |
| **Fit Type 1 AGN** with broad Balmer lines | `fitting_strategies/type1_agn.md` |
| **Fit BAL QSO** with absorption troughs | `fitting_strategies/type1_agn.md` |
| **Broad Hα / Hβ fitting** | `fitting_strategies/type1_agn.md` |
| Simple single emission line | `function_reference/line_profile_models.md` |
| Parameter tying between components | `function_reference/parameter_tying.md` |

### Critical Keywords

**If user mentions ANY of these, STOP and read `narrow_line_template.md` first:**
- "doublet", "[S II]", "S II doublet", "[N II] doublet", "[O III] doublet"
- "narrow line template", "derive template", "extract template"
- "LSF", "line spread function", "instrumental profile", "instrumental broadening"
- "6716", "6731", "6716/6731" (referring to [S II] wavelengths)

**If user mentions ANY of these, STOP and read `type1_agn.md` first:**
- "Type 1 AGN", "Type 1", "broad line AGN"
- "broad Hα", "broad Hbeta", "broad Balmer", "broad permitted lines"
- "BAL", "BAL QSO", "absorption trough", "broad absorption line"

---

## Quick Reference

### Most Common Usage

```python
import numpy as np
import sagan
from sagan.utils import line_wave_dict
from astropy.modeling import models, fitting

# Load and prepare spectrum
wave, flux, ferr = load_your_spectrum()

# Define model
cont = sagan.WindowedPowerLaw1D(amplitude=1.0, x_0=5000, alpha=-1.0,
                                x_min=4500, x_max=5500)
line = sagan.Line_Gaussian(amplitude=1.0, dv=0, sigma=500,
                           wavec=line_wave_dict['Halpha'])
model = cont + line

# LSQ fit
fitter = fitting.LevMarLSQFitter()
model_lsq = fitter(model, wave, flux, weights=1/ferr**2)

# MCMC fit
mcmc = sagan.MCMC_Fit(model_lsq, wave, flux, ferr, nwalkers=50, nsteps=1000)
samples, model_fit, param_names = mcmc.fit()

# Plot results
ax, axr = sagan.plot.plot_fit_new(wave, flux, model_fit, error=ferr)
```

### Key Components

| Component | Class | Usage | Documentation |
|-----------|-------|-------|---------------|
| **Doublet lines** | `Line_MultiGauss_doublet` | **Fitting [S II], [N II], [O III] doublets** | `fitting_strategies/narrow_line_template.md` ⭐ |
| Power-law continuum | `WindowedPowerLaw1D` | AGN continuum | `function_reference/continuum_models.md` |
| Narrow lines | `Line_template` | Forbidden lines using extracted profile | `function_reference/line_profile_models.md` |
| Broad lines | `Line_MultiGauss` | Permitted lines (Hα, Hβ) | `function_reference/line_profile_models.md` |
| Asymmetric lines | `Line_GaussHermite` | Skewed profiles | `function_reference/line_profile_models.md` |
| Absorption | `Line_Absorption` | BAL troughs | `function_reference/line_profile_models.md` |
| Iron template | `IronTemplate` | Fe II emission | `function_reference/iron_templates.md` |
| Stellar continuum | `StarSpectrum` | Host galaxy | `function_reference/stellar_continuum.md` |

**⭐ CRITICAL:** For fitting [S II], [N II], or [O III] doublets to derive the instrumental LSF/narrow line template, **ALWAYS use `Line_MultiGauss_doublet`**. See `fitting_strategies/narrow_line_template.md` for the complete workflow.

---

## Documentation Structure

The SAGAN documentation is organized into three main files:

### 1. Main Guide (`skills/sagan_spectral_fitting.md`)
- Installation instructions
- Data preparation (MW extinction, redshift determination)
- General fitting strategy
- Step-by-step fitting workflow
- Choosing between LSQ, MCMC, and Dynesty
- Best practices and troubleshooting

### 2. Type 1 AGN Strategy (`skills/fitting_strategies/type1_agn.md`)
- Specific model building sequence for Type 1 AGN
- Wavelength windows (Hα: 6100-7000 Å, Hβ: 4200-5400 Å)
- Component selection guidelines
- Parameter tying patterns (doublet ratios, absorption ties)
- Complete working example from J0925+6409
- Physical measurements (black hole masses, line fluxes)

### 3. Function Reference (`skills/function_reference/`)
- Complete documentation of all SAGAN functions
- Organized by module (line_profile, continuum, iron_template, etc.)
- Parameter descriptions and bounds
- Usage examples for each function
- Split into separate files by module for easier navigation

---

## Common Workflows

### Workflow 1: Simple Emission Line Fitting
```python
# Fit Hα with Gaussian
line = sagan.Line_Gaussian(amplitude=10.0, dv=0, sigma=300,
                           wavec=line_wave_dict['Halpha'])
model = line + cont
mcmc = sagan.MCMC_Fit(model, wave, flux, ferr)
```

### Workflow 2: Type 1 AGN with Broad + Narrow Components
```python
# Broad Hα (3 components) + narrow Hα (template)
bha = sagan.Line_MultiGauss(n_components=3, amp_c=48.0, dv_c=-100, sigma_c=400)
nha = sagan.Line_template(template_velc=velc_temp, template_flux=flux_temp)
aha = sagan.Line_Absorption(logtau0=2.0, dv=-160, sigma=40, Cf=0.4)
model = (bha + cont) * aha + nha
model = sagan.convolve_lsf(model, wavec=6563, resolving_power=1800)
```

### Workflow 3: BAL QSO with Multiple Absorption Components
```python
# Tie absorption parameters across Hα, Hβ, Hγ
model['Abs. Hbeta'].dv.tied = sagan.tie_Absorption_dv('Abs. Halpha')
model['Abs. Hbeta'].logtau0.tied = sagan.tie_Absorption_logtau0('Abs. Halpha', ratio=7.13)
```

### Workflow 4: Iron Template Fitting
```python
# Add iron template to Hβ region
iron = sagan.IronTemplate(amplitude=0.8, stddev=900/2.3548, z=0.003)
model = iron + bhb + nhb + no3
```

---

## Key Features

- **Bayesian Inference**: MCMC (emcee) and nested sampling (dynesty)
- **Complex Line Profiles**: Multi-component Gaussians, Gauss-Hermite, templates
- **Iron Templates**: Park 2022 and Boroson & Green 1992 templates
- **Absorption Modeling**: Covering fraction, optical depth, velocity shifts
- **Instrument Convolution**: Constant and wavelength-dependent LSF
- **Parameter Tying**: Reduce degeneracies by linking parameters
- **Diagnostics**: Corner plots, chain plots, convergence checking

---

## File Locations

**Package:**
```
/Users/shangguan/Softwares/my_modules/SAGAN/
```

**Documentation:**
```
skills/sagan_spectral_fitting.md          # Main guide
skills/fitting_strategies/type1_agn.md    # Type 1 AGN strategy
skills/function_reference/                # Function reference (split by module)
```

**Key Modules:**
```
sagan/line_profile.py    # Line profile models
sagan/continuum.py        # Continuum models
sagan/iron_template.py    # Iron templates
sagan/stellar_continuum.py # Stellar populations
sagan/convolution.py      # LSF convolution
sagan/mcmc_fit.py        # MCMC fitting
sagan/dynesty_fit.py     # Nested sampling
sagan/plot.py            # Plotting functions
sagan/utils.py           # Utility functions
```

---

## Installation

```bash
cd /Users/shangguan/Softwares/my_modules/SAGAN
pip install -e .
```

**Required packages:**
```bash
pip install numpy scipy matplotlib astropy
pip install emcee corner    # For MCMC
pip install dynesty          # For nested sampling
pip install multiprocess     # For parallel processing
pip install extinction sfdmap  # For MW extinction correction
```

---

## Examples

Example notebooks are available in the `example/` directory:
- `wrk_sagan_bl_J0925+6409.ipynb` - Type 1 AGN fitting with BAL troughs
- `wrk_sagan_bl_J1025+1402.ipynb` - Another Type 1 AGN example
- `abs_fitting_example.ipynb` - Absorption line fitting

---

## Tips for Best Results

1. **CRITICAL: Read routing documentation first** - Before creating any plan, check the "Task-Specific Documentation Routing" table above. For doublet fitting or narrow line template derivation, you MUST read `fitting_strategies/narrow_line_template.md` first.
2. **For doublet lines ([S II], [N II], [O III])**: Use `Line_MultiGauss_doublet` class, NOT separate Gaussian components. See `narrow_line_template.md` for workflow.
3. **Start with LSQ fitting** to get initial parameters before MCMC
4. **Use appropriate component types**: Line_template for narrow lines, Line_MultiGauss for broad lines, Line_MultiGauss_doublet for doublets
5. **Apply LSF convolution** after absorption multiplication
6. **Check convergence**: nsteps / tau_max should be > 50
7. **Tie parameters** to reduce degeneracies (narrow line dv, doublet ratios)
8. **Use separate MCMC runs** for different component groups (Balmer vs others)
9. **Verify extinction correction** before fitting
10. **Normalize spectrum** to local continuum for line fitting

---

## Parameter Tying Examples

```python
# Narrow lines share velocity
for ln in ['NII_6583', 'SII_6716', 'OIII_5007']:
    model[ln].dv.tied = sagan.tie_template_dv('nHalpha')

# [O III] doublet ratio
model['OIII_4959'].amplitude.tied = sagan.tie_template_amplitude('OIII_5007', ratio=2.98)

# Absorption tied across Balmer lines
model['Abs. Hbeta'].dv.tied = sagan.tie_Absorption_dv('Abs. Halpha')
model['Abs. Hbeta'].logtau0.tied = sagan.tie_Absorption_logtau0('Abs. Halpha', ratio=7.13)

# Broad line kinematics
model['Broad Hbeta'].dv_c.tied = sagan.tie_MultiGauss_dv_c('Broad Halpha')
```

---

## Notes

- SAGAN uses velocity (km/s) instead of wavelength for line widths
- All wavelengths are in rest frame (divide by 1+z before fitting)
- The package follows astropy.modeling conventions
- MCMC requires ≥ 2 × number of parameters walkers
- Convergence is checked via autocorrelation time
- For publication-quality results, use MCMC after LSQ initialization

---

## References

- Shangguan et al. (2026): Low-redshift BAL QSO analysis methodology
- Kovačević et al. (2013): Balmer pseudo-continuum
- Park et al. (2022): Mrk 493 iron template
- Boroson & Green (1992): I Zw 1 iron template
