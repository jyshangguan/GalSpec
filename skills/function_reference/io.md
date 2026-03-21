# I/O Functions (`sagan.mcmc_fit_io`, `sagan.dynesty_fit_io`)

Save and load fitting results in SAGAN.

## Table of Contents

1. [MCMC I/O Functions](#mcmc-io-functions)
2. [Dynesty I/O Functions](#dynesty-io-functions)

---

## MCMC I/O Functions

Save and load MCMC fitting results.

```python
import sagan

# Save MCMC results
sagan.save_mcmc(mcmc_object, 'model.pkl')

# Load MCMC results
mcmc_loaded = sagan.load_mcmc('model.pkl')
```

### save_mcmc

Save complete MCMC fitting object to file.

```python
sagan.save_mcmc(
    mcmc_object,      # MCMC_Fit object
    filename          # Output filename (typically .pkl)
)
```

**Parameters**:
- `mcmc_object`: Fitted `MCMC_Fit` object with samples
- `filename`: Output filename (pickle format)

**What Gets Saved**:
- Model configuration
- MCMC sampler chains
- Parameter names and values
- Input data (wave, flux, error)
- Fitting configuration (nwalkers, nsteps, nburn)

### load_mcmc

Load previously saved MCMC results.

```python
mcmc_loaded = sagan.load_mcmc(filename)
```

**Parameters**:
- `filename`: Input filename (pickle format)

**Returns**:
- `MCMC_Fit` object with all methods available:
  - `get_best_fit()`
  - `get_param_samples()`
  - `plot_corner()`
  - `plot_chain()`

**Example**:
```python
# After fitting
mcmc = sagan.MCMC_Fit(model, wave, flux, ferr, ...)
samples, model_fit, param_names = mcmc.fit(progress=True)

# Save results
sagan.save_mcmc(mcmc, 'agn_halpha_mcmc.pkl')

# Later, load and analyze
mcmc_loaded = sagan.load_mcmc('agn_halpha_mcmc.pkl')
model_best, par_names, theta_best = mcmc_loaded.get_best_fit()

# Continue analysis
fig = mcmc_loaded.plot_corner(thin=100)
plt.savefig('corner_plot.png', dpi=300)
```

---

## Dynesty I/O Functions

Save and load Dynesty nested sampling results.

```python
import sagan

# Save Dynesty results
sagan.save_dynesty(dynesty_object, 'model_dynesty.pkl')

# Load Dynesty results
dynesty_loaded = sagan.load_dynesty('model_dynesty.pkl')
```

### save_dynesty

Save complete Dynesty fitting object to file.

```python
sagan.save_dynesty(
    dynesty_object,   # Dynesty_Fit object
    filename          # Output filename (typically .pkl)
)
```

**Parameters**:
- `dynesty_object`: Fitted `Dynesty_Fit` object with results
- `filename`: Output filename (pickle format)

**What Gets Saved**:
- Model configuration
- Nested sampling results
- Log evidence and error
- Parameter samples
- Input data

### load_dynesty

Load previously saved Dynesty results.

```python
dynesty_loaded = sagan.load_dynesty(filename)
```

**Parameters**:
- `filename`: Input filename (pickle format)

**Returns**:
- `Dynesty_Fit` object with all methods available:
  - `get_best_fit()`
  - `get_param_samples()`

**Example**:
```python
# After fitting
dynesty_fit = sagan.Dynesty_Fit(model, wave, flux, ferr, ...)
results = dynesty_fit.fit()

print(f"Log Z: {results.logz:.2f} ± {results.logzerr:.2f}")

# Save results
sagan.save_dynesty(dynesty_fit, 'agn_halpha_dynesty.pkl')

# Later, load and analyze
dynesty_loaded = sagan.load_dynesty('agn_halpha_dynesty.pkl')
model_best, par_names, theta_best = dynesty_loaded.get_best_fit()

# Get parameter constraints
halpha_amp_samples = dynesty_loaded.get_param_samples('Broad Halpha', 'amplitude')
print(f"Hα amplitude: {np.median(halpha_amp_samples):.2f} ± {np.std(halpha_amp_samples):.2f}")
```

---

## Best Practices

1. **Use Descriptive Filenames**: Include object name, spectral region, and date
   ```python
   sagan.save_mcmc(mcmc, 'J1234+5678_Halpha_20250320.pkl')
   ```

2. **Organize Results**: Keep results in a dedicated directory
   ```python
   import os
   results_dir = 'fitting_results'
   os.makedirs(results_dir, exist_ok=True)
   sagan.save_mcmc(mcmc, f'{results_dir}/model.pkl')
   ```

3. **Document Fits**: Keep a log file alongside saved results
   ```python
   with open(f'{results_dir}/fit_log.txt', 'w') as f:
       f.write(f"Object: {object_name}\n")
       f.write(f"Date: {datetime.now()}\n")
       f.write(f"Model: {model_description}\n")
       f.write(f"BIC: {bic:.1f}\n")
   ```

4. **Version Control**: Track major changes in SAGAN version
   ```python
   import sagan
   print(f"SAGAN version: {sagan.__version__}")  # Check before saving
   ```

---

## File Size Considerations

MCMC chains can be large, especially with many walkers and steps:

```python
# Estimate file size
nwalkers = 100
nsteps = 6000
n_params = 20

# Sample size in bytes (float64)
size_bytes = nwalkers * nsteps * n_params * 8
size_mb = size_bytes / (1024**2)

print(f"Estimated file size: {size_mb:.1f} MB")
```

To reduce file size, you can save only the best-fit model and parameter samples:

```python
# Save only essential results
import pickle

results = {
    'model_best': mcmc.get_best_fit()[0],
    'param_names': mcmc.param_names,
    'theta_best': mcmc.get_best_fit()[2],
    'samples': mcmc.sampler.get_chain(discard=5000, thin=10)
}

with open('results_essential.pkl', 'wb') as f:
    pickle.dump(results, f)
```

---

**Modules**: `sagan.mcmc_fit_io`, `sagan.dynesty_fit_io`
**Source Files**: `sagan/mcmc_fit_io.py`, `sagan/dynesty_fit_io.py`
