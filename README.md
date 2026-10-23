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

A complete worked example lives in `example/fit_iras07546_jax.py` (script,
about 10 min on one GPU) and `example/fit_iras07546_jax.ipynb` (the same fit broken
into annotated cells). It fits **4200-6800 A** of IRAS 07546+3928 (rest frame,
56 956 pixels, H-gamma to [S II]) in one go: a power-law continuum plus an Fe II
template, H-alpha/H-beta/H-gamma each modelled **twice** (a narrow `Line_Gaussian`
and a broad `Line_MultiGauss` with a core and a wind), He I 5876 and He II 4686
with one broad Gaussian each, [O III] with a broad wing, and [N II], [S II], [O I]
and [O III] 4363 -- 30 free parameters describing seventeen models, with every
narrow line tied to the [O III] kinematics, the broad components tied across the
three Balmer lines, and no least-squares step anywhere. Nothing blueward of H-gamma or redward of
[S II] is modelled, so no window has to be masked; the Na D region, which used to
be masked, is now covered by fitting He I 5876.

The doublet ratios are tied at their atomic values except [S II], whose 6716/6731
ratio is a density diagnostic and is therefore measured. The Fe II template is
fitted in its `IronTemplate_tied` form: the iron is broadened by the profile of
the broad H-alpha component, tied component by component -- the core to the broad
core, the wind to the wind -- so its only free parameter is its amplitude. It
carries no weights of its own (a `Line_MultiGauss` is already a weighted sum, and
the convolution renormalises it). The two width ranges of the broad line do not
overlap, because the components are exchangeable and the posterior would otherwise
be bimodal; `develop/dev_jax/io_test/iron_tied_io_test.html` measures that and
shows a mock recovered with it. (`IronTemplate_new` -- one velocity dispersion
and a velocity shift -- remains available, and the two agree exactly for a
one-component kernel.) The error bar on every pixel is the measured pixel-to-pixel
scatter plus **5% of the flux** in quadrature, which stands for the photon noise
of the source.

The model is not convolved with the LSF (this spectrum grossly oversamples the
instrument); the fit is drawn with `galspec.plot.plot_fit`, so every component and
the sigma-scaled residuals appear in one figure, the marginals are drawn with
`galspec.plot.plot_posterior_hist` (axis from the posterior, not the prior), and
the whole run is written to a FITS file that
`galspec.jnesty_fit_io.load_jnesty_fit_from_fits` turns back into a working fit.
Parameter labels come from `galspec.param_labels`, so plots print
`BH$\alpha$:$\sigma_{w0}$` instead of `Halpha.sigma_w0`.

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
