# Fitting Classes (`galspec.mcmc_fit`, `galspec.dynesty_fit`)

Bayesian fitting classes for spectral modeling in GalSpec.

## Table of Contents

1. [MCMC_Fit](#mcmc_fit)
2. [Dynesty_Fit](#dynesty_fit)

---

## MCMC_Fit

Bayesian fitting using emcee MCMC sampler.

```python
mcmc = galspec.MCMC_Fit(
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
mcmc = galspec.MCMC_Fit(
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

---

## Dynesty_Fit

Nested sampling fitting using dynesty.

```python
dynesty_fit = galspec.Dynesty_Fit(
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
dynesty_fit = galspec.Dynesty_Fit(
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

**Modules**: `galspec.mcmc_fit`, `galspec.dynesty_fit`
**Source Files**: `galspec/mcmc_fit.py`, `galspec/dynesty_fit.py`
