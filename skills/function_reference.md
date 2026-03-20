# SAGAN Function Reference

Complete reference documentation for all SAGAN functions, classes, and modules.

## Table of Contents

1. [Line Profile Models](#line-profile-models-saganline_profile)
2. [Continuum Models](#continuum-models-sagancontinuum)
3. [Iron Templates](#iron-templates-saganiron_template)
4. [Stellar Continuum](#stellar-continuum-saganstellar_continuum)
5. [Convolution Functions](#convolution-functions-saganconvolution)
6. [Fitting Classes](#fitting-classes)
7. [Parameter Tying Functions](#parameter-tying-functions)
8. [Plotting Functions](#plotting-functions-saganplot)
9. [Utility Functions](#utility-functions-saganutils)

---

## Line Profile Models (`sagan.line_profile`)

### Line_Gaussian

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

### Line_GaussHermite

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

### Line_MultiGauss

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

### Line_MultiGauss_doublet

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

### Line_GaussHermite_doublet

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

### Line_template

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

### Line_Absorption

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

### Line_Absorption_doublet

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

## Continuum Models (`sagan.continuum`)

### WindowedPowerLaw1D

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

### BlackBody

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

### BalmerPseudoContinuum

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

### extinction_ccm89

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

## Iron Templates (`sagan.iron_template`)

### IronTemplate

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

### IronTemplate_new

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

### IronTemplate_tied

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

## Stellar Continuum (`sagan.stellar_continuum`)

### StarSpectrum

Single stellar spectrum template.

```python
sagan.StarSpectrum(
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
stellar = sagan.StarSpectrum(
    amplitude=0.3,
    star_type='G',
    velscale=200,
    delta_z=0,
    sigma=150,              # Stellar velocity dispersion
    name='stellar'
)
```

### Multi_StarSpectrum

Multiple stellar components (e.g., bulge + disk).

```python
sagan.Multi_StarSpectrum(
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

## Convolution Functions (`sagan.convolution`)

### convolve_lsf

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

### convolve_lsf_var

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

## Fitting Classes

### MCMC_Fit (`sagan.mcmc_fit`)

Bayesian fitting using emcee MCMC sampler.

```python
mcmc = sagan.MCMC_Fit(
    model,                  # CompoundModel
    wave_use,               # Wavelength array (Å)
    flux_use,               # Flux array (normalized)
    ferr,                   # Flux error array
    nwalkers=100,           # Number of walkers (≥ 2*ndim)
    nsteps=6000,            # Total steps per walker
    nburn=5000,             # Burn-in steps to discard
    step_initial=2000,      # Initial steps for convergence (optional)
    initial_frac=1e-4       # Random initialization fraction
)
```

**Methods**:
- `fit(progress=True)`: Run MCMC sampling
- `fit_ncores(ncores=4)`: Run with multi-core parallelization
- `check_convergence()`: Check autocorrelation time
- `get_best_fit(discard=0)`: Get best-fit model and parameters
- `get_param_samples(model_name, param_name, discard=0)`: Get samples for specific parameter
- `plot_corner(thin=1, **kwargs)`: Plot corner plot
- `plot_chain(thin=1)`: Plot MCMC chains
- `save_samples(filename, thin=1)`: Save samples to file
- `load_samples(filename)`: Load samples from file

**Example**:
```python
# Initialize
mcmc = sagan.MCMC_Fit(
    model,
    wave,
    flux,
    ferr,
    nwalkers=100,
    nsteps=6000,
    nburn=5000
)

# Run fitting
samples, model_fit, param_names = mcmc.fit(progress=True)

# Check convergence
chain, tau = mcmc.check_convergence()
print(f"nsteps/tau_max = {chain.shape[0] / tau.max():.1f}")

# Get best fit
model_best, par_names, theta_best = mcmc.get_best_fit()

# Corner plot
fig = mcmc.plot_corner(thin=100)
plt.show()
```

### Dynesty_Fit (`sagan.dynesty_fit`)

Nested sampling fitting using dynesty.

```python
dynesty_fit = sagan.Dynesty_Fit(
    model,
    wave_use,
    flux_use,
    ferr,
    bounds_dict=None,       # Custom bounds: {param_name: (lower, upper)}
    default_bounds=None,    # Use built-in defaults
    sample_method='rwalk',  # 'unif', 'rwalk', 'rslice', 'hslice'
    nlive=500,              # Number of live points
    bound='multi',          # Bounding method
    rstate=None             # Random state for reproducibility
)
```

**Methods**:
- `fit()`: Run nested sampling
- `get_best_fit()`: Get best-fit model
- `get_param_samples(model_name, param_name)`: Get parameter samples

**Example**:
```python
# Define bounds
bounds_dict = {
    'Halpha.amplitude': (0.1, 100),
    'Halpha.sigma': (100, 2000),
}

# Initialize
dynesty_fit = sagan.Dynesty_Fit(
    model,
    wave,
    flux,
    ferr,
    bounds_dict=bounds_dict,
    sample_method='rwalk',
    nlive=500
)

# Run fitting
results = dynesty_fit.fit()

# Print evidence
print(f"Log evidence: {results.logz:.2f} ± {results.logzerr:.2f}")

# Get best fit
model_best, par_names, theta_best = dynesty_fit.get_best_fit()
```

---

## Parameter Tying Functions

Link parameters across different components to reduce degrees of freedom.

### Absorption Tying

```python
# Tie absorption parameters across Hα and Hβ
mcmc_hb['Abs. Hbeta'].dv.tied = sagan.tie_Absorption_dv('Abs. Halpha')
mcmc_hb['Abs. Hbeta'].sigma.tied = sagan.tie_Absorption_sigma('Abs. Halpha')
mcmc_hb['Abs. Hbeta'].Cf.tied = sagan.tie_Absorption_Cf('Abs. Halpha')

# Tie τ₀ with theoretical ratio (Hβ/Hα = 7.13 for Case B)
mcmc_hb['Abs. Hbeta'].logtau0.tied = sagan.tie_Absorption_logtau0('Abs. Halpha', ratio=7.13)
```

### MultiGauss Tying

```python
# Tie MultiGauss parameters
m_hb['Broad Hbeta'].dv_c.tied = sagan.tie_MultiGauss_dv_c('Broad Halpha')
m_hb['Broad Hbeta'].sigma_c.tied = sagan.tie_MultiGauss_sigma_c('Broad Halpha')
m_hb['Broad Hbeta'].amp_w0.tied = sagan.tie_MultiGauss_amp_w0('Broad Halpha')
m_hb['Broad Hbeta'].dv_w0.tied = sagan.tie_MultiGauss_dv_w0('Broad Halpha')
m_hb['Broad Hbeta'].sigma_w0.tied = sagan.tie_MultiGauss_sigma_w0('Broad Halpha')
```

### Template Tying

```python
# Tie template amplitude and dv
m_hb['nHbeta'].amplitude.tied = sagan.tie_template_amplitude('nHalpha', ratio=2.86)
m_hb['nHbeta'].dv.tied = sagan.tie_template_dv('nHalpha')
```

### Stellar Continuum Tying

```python
# Tie stellar parameters
m_stellar2['stellar_g'].delta_z.tied = sagan.tie_StarSpectrum_deltaz('stellar_k')
m_stellar2['stellar_g'].sigma.tied = sagan.tie_StarSpectrum_sigma('stellar_k')
```

---

## Plotting Functions (`sagan.plot`)

### plot_fit_new

Plot fitted spectrum with components and residuals.

```python
ax, axr = sagan.plot.plot_fit_new(
    wave,                   # Wavelength array (Å)
    flux,                   # Observed flux array
    model_fit,              # Fitted CompoundModel
    weight=None,            # Optional weights
    error=flux_err,         # Flux error array
    xlim=(6400, 6700),      # Optional x-axis limits
    components_to_plot=[    # Optional: specific components to plot
        'Broad Halpha',
        'nHalpha',
        'Abs. Halpha'
    ]
)
```

**Customization**:
```python
ax.set_ylabel(r'$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=16)
axr.set_xlabel('Rest Wavelength (Å)', fontsize=16)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fit_result.png', dpi=300)
```

### plot_fit

Alternative plotting function with more options.

```python
ax, axr = sagan.plot.plot_fit(
    wave, flux, model,
    weight=None,
    error=None,
    ax=None,                # Optional axes
    axr=None,               # Optional residual axes
    xlim=None,
    ylim0=None,             # Main panel y-limits
    ylim1=None,             # Residual panel y-limits
    xlabel=None,
    ylabel=None,
    legend_kwargs=None,
    plot_weight=True,
    ignore_list=None,       # Components to ignore
    legend_map=None,        # Custom legend mapping
    mask_list=None          # Wavelength ranges to mask
)
```

---

## Utility Functions (`sagan.utils`)

### Line Dictionaries

```python
from sagan.utils import line_wave_dict, line_label_dict

# Get rest wavelength (Å)
wave_ha = line_wave_dict['Halpha']      # 6562.819
wave_hb = line_wave_dict['Hbeta']       # 4861.369
wave_o3_5007 = line_wave_dict['OIII_5007']  # 5006.843

# Get plotting label
label_ha = line_label_dict['Halpha']    # r'H$\alpha$'
label_hb = line_label_dict['Hbeta']     # r'H$\beta$'
```

**Available Lines**:
```python
# Hydrogen
'Halpha': 6562.819, 'Hbeta': 4861.333, 'Hgamma': 4340.471, 'Hdelta': 4101.742

# [O III]
'OIII_4959': 4958.911, 'OIII_5007': 5006.843, 'OIII_4363': 4363.210

# [S II]
'SII_6716': 6716.440, 'SII_6731': 6730.810

# [N II]
'NII_6548': 6548.050, 'NII_6583': 6583.460

# [O I]
'OI_6300': 6300.304, 'OI_6364': 6363.776

# Others
'HeII_4686': 4685.710, 'HeI_4471': 4471.479, 'HeI_5876': 5875.624
'NaD_5890': 5891.583, 'NaD_5896': 5897.558
```

### Velocity Conversions

```python
from sagan.utils import wave_to_velocity, velocity_to_wave

# Wavelength to velocity
vel = wave_to_velocity(wave_obs, wave_rest)  # km/s

# Velocity to wavelength
wave_obs = velocity_to_wave(vel, wave_rest)  # Å
```

### ReadSpectrum Class

```python
readspec = sagan.ReadSpectrum(
    is_sdss=True,        # For SDSS spectra
    hdu=hdulist,         # SDSS HDU object
    z=0.0,              # Redshift (optional, from SDSS header)
    ra=None, dec=None   # Coordinates
)

# For custom spectra
readspec = sagan.ReadSpectrum(
    is_sdss=False,
    lam=wave, flux=flux, ferr=ferr,
    z=0.1, ra=ra, dec=dec
)

# Methods
flux_unred, err_unred = readspec.de_redden()  # MW extinction correction
lam_res, flux_res, err_res = readspec.rest_frame()  # Rest-frame conversion
lam_res, flux_res, err_res = readspec.unredden_res()  # Both corrections
```

**Example**:
```python
# Load SDSS spectrum
from astropy.io import fits
hdu = fits.open('spec.fits')
readspec = sagan.ReadSpectrum(is_sdss=True, hdu=hdu)

# Get MW-corrected rest-frame spectrum
wave, flux, ferr = readspec.unredden_res()
```

### Spectral Resolution Degradation

```python
from sagan.utils import down_spectres

flux_lowres = down_spectres(
    wave, flux,
    R_org=2000,      # Original resolving power
    R_new=1000,      # New resolving power
    wave_new=None    # Optional new wavelength grid
)
```

---

## Measurement Methods (`sagan.measure_method`)

### line_emission_fwhm

Calculate FWHM of emission lines.

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

---

## I/O Functions (`sagan.mcmc_fit_io`, `sagan.dynesty_fit_io`)

### Save/Load MCMC Results

```python
import sagan

# Save
sagan.save_mcmc(mcmc_object, 'model.pkl')

# Load
mcmc_loaded = sagan.load_mcmc('model.pkl')
```

---

## Related Documentation

- **[Spectral Fitting Guide](sagan_spectral_fitting.md)** - General workflow and data preparation
- **[Type 1 AGN Strategy](strategy_types/type1_agn.md)** - Specific strategies for Type 1 AGN fitting

---

**Version**: 1.0
**Last Updated**: 2025-03-20
