# GalSpec

Galaxy Spectrum Fitting

GalSpec is a Python package for fitting astronomical spectra, specifically designed for AGN and galaxy spectra with complex emission and absorption line features.

<p align="center">
  <img src="docs/source/_static/logo.jpg" alt="GalSpec Logo" width="300"/>
</p>

## Key Features

- **Astropy Integration**: All models inherit from Astropy's modeling framework
- **Multiple Fitting Methods**: LSQ, MCMC (emcee), and nested sampling (dynesty)
- **Comprehensive Line Database**: Built-in UV/optical emission lines
- **Template Support**: Fe II and stellar population templates
- **Flexible Model Composition**: Easily combine continuum, line, and template components
- **GPU Acceleration (optional)**: JAX implementations of the same models, with
  a JNesty sampler -- 50-105x faster batched likelihood evaluations and 4-22x
  faster complete nested-sampling fits on an RTX 4090 (the larger the CPU model,
  the larger the gain; see docs/source/jax_gpu.rst)

## Documentation

Full documentation is available at: **https://jyshangguan.github.io/GalSpec/**

The documentation includes:
- Installation instructions
- Quick start tutorial with real SDSS data
- API reference for all modules
- Examples for continuum fitting, emission lines, absorption lines, and advanced techniques

## Installation

Install from source:

```bash
git clone https://github.com/jyshangguan/GalSpec.git
cd GalSpec
pip install -e .
```

## Claude Code Integration (Optional)

GalSpec includes a Claude Code skill that allows you to use GalSpec directly within Claude Code for interactive spectral fitting assistance.

To install the skill:

```bash
# From the GalSpec repository root
./install_skill.sh
```

This will:
- Create a symlink from `~/.claude/skills/galspec` to the `skills/` directory
- Enable GalSpec commands in Claude Code

To use the skill in Claude Code:
```
/skills
Use GalSpec to fit the spectrum
```

To uninstall:
```bash
rm ~/.claude/skills/galspec
```

## Quick Example

```python
import numpy as np
from astropy.modeling.models import Linear1D
from galspec import Line_Gaussian
from astropy.modeling.fitting import LevMarLSQFitter

# Load your spectrum
wave = np.linspace(6400, 6700, 1000)
flux = np.random.randn(len(wave)) * 0.1  # Your data here

# Define model: continuum + Hα line
continuum = Linear1D(slope=0.0, intercept=1.0)
halpha = Line_Gaussian(amplitude=5.0, dv=0, sigma=200, wavec=6563)
model = continuum + halpha

# Fit
fitter = LevMarLSQFitter()
fitted_model = fitter(model, wave, flux)

print(f"Hα amplitude: {fitted_model[1].amplitude.value:.3f}")
```

## GPU Acceleration (JAX)

The `galspec.jax_models` subpackage re-implements the models as pure JAX
functions, so a fit can run on a GPU. It is optional -- `import galspec` does not
require JAX.

```python
import numpy as np
import galspec
from galspec import jax_models as jxm
from galspec.jnesty_fit import JNesty_Fit

# same constructors, same fixed/tied/bounds/name semantics as the astropy models
continuum = jxm.PowerLaw1D(amplitude=0.5, x_0=5000, alpha=1.0,
                           fixed={"x_0": True}, name="Continuum")
line = jxm.Line_MultiGauss_doublet(n_components=2, amp_c0=1.7, amp_c1=0.57,
                                   dv_c=-43, sigma_c=250, wavec0=5006.84,
                                   wavec1=4958.91, name="OIII",
                                   amp_w0=0.35, dv_w0=-280, sigma_w0=340)
line.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("OIII", 2.98)
model = continuum + line

fit = JNesty_Fit(model, wave, flux, error, nlive=200, device="auto")
fit.fit(dlogz=0.5)
best_model, names, theta = fit.get_best_fit()
print(fit.get_evidence())          # (logZ, logZ_err)
```

Install the dependencies with `pip install -e ".[gpu]"` (or `".[jax]"` for a
CPU-only JAX) and install [JNesty](https://github.com/jyshangguan/jnesty) from
its repository. See `docs/source/jax_gpu.rst` for the model list, parity
guarantees, benchmarks and the recipe for adding a new model, and
`develop/dev_jax/` for the benchmark and demo scripts.

The JAX models are verified against the CPU models by
`tests/test_jax_parity.py`; fluxes agree to ~1e-16 relative (float64), so the
GPU and CPU fits are guaranteed to be solving the same problem.

## Example Notebooks

The `example/` directory contains detailed notebooks demonstrating:
- Continuum fitting with stellar templates
- Emission line fitting for AGN
- Absorption line and BAL fitting
- Physical parameter estimation

## Requirements

- Python ≥ 3.9
- numpy, scipy, matplotlib, astropy, pandas
- emcee, dynesty (for Bayesian fitting)
- jax + jnesty (optional, for GPU-accelerated fitting)

**Note**: When using `ReadSpectrum` class for extinction correction, SFD dust map data (4 FITS files, ~167MB total) will be automatically downloaded from [kbarbary/sfddata](https://github.com/kbarbary/sfddata) on first use.

## License

MIT License - see LICENSE file for details

## Links

- **Documentation**: https://jyshangguan.github.io/GalSpec/
- **GitHub Repository**: https://github.com/jyshangguan/GalSpec
- **Issue Tracker**: https://github.com/jyshangguan/GalSpec/issues
