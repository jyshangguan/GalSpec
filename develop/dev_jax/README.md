# JAX / GPU development scripts

Scripts for the optional JAX backend (`galspec.jax_models`) and the JNesty
fitter (`galspec.jnesty_fit`). Everything here is additive: the astropy/CPU
models and `galspec.Dynesty_Fit` are untouched.

| file | purpose |
| --- | --- |
| `common.py` | the CPU *and* JAX versions of the example models (IRAS 07546+3928 [O III] window, full spectrum, stellar continuum) plus the data loaders; used by the tests and the benchmarks |
| `bench_likelihood.py` | cost of one likelihood evaluation: astropy/CPU vs JAX `jit` vs JAX `vmap` batches |
| `bench_device_scaling.py` | Device and precision scaling (CPU vs GPU, float64 vs float32). |
| `bench_jnesty.py` | complete nested-sampling fits: dynesty serial / dynesty parallel / JNesty CPU / JNesty GPU, with the same data, priors, `nlive` and `dlogz` |
| `demo_ir07546_jnesty.py` | user-facing demo: fit the [O III] complex with JNesty on the GPU |
| `sdss_agn.py` | a realistic SDSS AGN model (H-alpha + H-beta + H-gamma, continuum, iron, all strong forbidden lines, 23 free parameters, LSF convolution) in both flavours |
| `run_sdss_fit.py` | run one configuration of that fit (`--config dynesty-serial\|dynesty-parallel\|jnesty-cpu\|jnesty-gpu`) and save JSON + NPZ |
| `make_report.py`, `report_pages.py` | build the multi-page HTML report from the saved runs |
| `iron_cpu_vs_jax.py` | does the JAX `IronTemplate_tied` reproduce the CPU one at the same parameters? Both are compared against an exact velocity-space convolution; writes `io_test/cpu_vs_jax.html` |
| `iron_tied_io_test.py` | input/output test of `IronTemplate_tied` + `Line_MultiGauss`: generate a mock (iron convolved with a known broad H-beta profile, plus H-beta), fit it back with the JAX models, compare with the input; writes `io_test/iron_tied_io_test.html` |
| `results/` | benchmark output (JSON + figures; `results/report/` holds the HTML) |
| `io_test/` | output of the input/output test: the HTML report, the figures as PNGs, and `io_test_results.json` |

## Running

From the repository root:

```bash
# throughput benchmark (GPU if available, export JAX_PLATFORMS=cpu to force CPU)
python develop/dev_jax/bench_likelihood.py

# end-to-end nested sampling, the demo problem from develop/dev_dynesty
python develop/dev_jax/bench_jnesty.py

# the same for the 39-parameter full-spectrum model, a binned spectrum to keep
# the CPU reference affordable
python develop/dev_jax/bench_jnesty.py --problem full --bin-pixels 6000 \
    --nlive 100 --dlogz 1e-4 --maxiter 4000 --queue-size 64 --tag full

# the demo
python develop/dev_jax/demo_ir07546_jnesty.py

# the realistic SDSS AGN fit: H-gamma, H-beta and H-alpha at once, 23 parameters
python develop/dev_jax/run_sdss_fit.py --config jnesty-gpu      --nlive 400 --dlogz 0.01
python develop/dev_jax/run_sdss_fit.py --config dynesty-serial  --nlive 400 --dlogz 0.01
python develop/dev_jax/run_sdss_fit.py --config dynesty-parallel --processes 16 \
    --nlive 400 --dlogz 0.01

# HTML report from whatever runs finished.  report.html holds all four sections
# in one file (figures inlined, anchor navigation), so it renders anywhere;
# index.html / model.html / results.html / glossary.html are the same content
# as separate self-contained pages.
python develop/dev_jax/make_report.py      # -> results/report/report.html

# the input/output test of the tied iron + multi-Gaussian kernel (about 6 min on
# one GPU): checks the parameter names, the normalisation, the exactness of the
# convolution, and whether a fit reproduces a known input
python develop/dev_jax/iron_cpu_vs_jax.py          # -> io_test/cpu_vs_jax.html

python develop/dev_jax/iron_tied_io_test.py --noise
# -> develop/dev_jax/io_test/iron_tied_io_test.html
```

**Two reports, two questions.**
`io_test/cpu_vs_jax.html` (from `iron_cpu_vs_jax.py`, about a minute) asks whether
the JAX `IronTemplate_tied` reproduces the CPU one at the same parameters, and
measures both against an exact velocity-space convolution. Short answer: they differ
by 8-10% of the iron peak, and against an independent velocity-space convolution the
JAX model is exact to 0.02% over the whole template while the CPU one is 7-23% off and
moves when its deconvolution, kernel window or intrinsic width are changed.
`io_test/iron_tied_io_test.html` (from `iron_tied_io_test.py`, about six minutes) is
the longer input/output study: a mock fitted back, the unit checks, and the
core-wind degeneracy.

**Read `io_test/cpu_vs_jax.html` before trusting either tied iron.** It is the
one place where the model is checked against a known answer: the mock is generated
by the *CPU* classes, the fit uses the JAX ones, and the test separates what the
code must get exactly right (the convolution, the ties, the normalisation:
machine precision) from what is a convention shared with the CPU model (the
template's linear wavelength sampling: 4% of the iron peak). It also measures the
one real trap in the model -- component 0 and component 1 of a `Line_MultiGauss`
are *exchangeable*, so the posterior is bimodal unless their width priors are
disjoint, which is why `example/fit_iras07546_jax.py` splits them at 3000 km/s.

## The realistic comparison (numbers as measured)

One SDSS spectrum of IRAS 07546+3928 (59 146 pixels, 4200-6900 A), 21 free
parameters, H-gamma + H-beta + H-alpha fitted together with the continuum, the
iron template and every strong forbidden line, convolved with the SDSS LSF.
Same priors, same `nlive=400`, same `dlogz=0.01` everywhere.

| | wall clock | sampler calls/s |
| dynesty, 1 core | 8.9 h (measured) | 86 |
| dynesty, 16 cores | 83.7 min | 513 |
| JNesty, GPU | 6.3 min | 7 680 |

* 13.3x faster than 16 CPU cores, 84.7x faster than one core (8.9 h measured).
* All 21 posterior medians agree to <=0.05 of their error bar (median 0.02);
  reduced chi2 is 1.978 for both.
* log Z differs by 20.2 nats between the samplers while each claims +/-0.6;
  doubling `nlive` moves JNesty by 0.3, so both have converged and the gap is
  between integrators, not GPU precision: the same dynesty code on 1 core vs
  16 cores differs by 13.6 nats (22x its quoted error). Prefer parameters over
  evidence ratios when mixing samplers.

`sdss_agn.py` can be run on its own: it prints the prior of every parameter and
checks that the astropy and JAX models agree (they do, to 1.5e-15 of the peak
flux). The priors are derived from the data's own flux range rather than from a
least-squares fit, so the fit can be run on a new spectrum with no user
intervention.

`bench_jnesty.py` runs each configuration in its own subprocess: dynesty
parallelises with `os.fork`, which is unsafe in a process that has already
initialised JAX, and separate processes also give every configuration a cold
cache. The JNesty numbers report the *sampling* time separately from the
one-off setup (CUDA context + compilation), and absorb JNesty's own
first-call compilation with a short discarded fit; pass `--no-warm-first` to
time the very first run instead.

## Numerics

The JAX models reproduce the CPU fluxes to float64 rounding (~1e-16 relative),
which `tests/test_jax_parity.py` asserts for every model and for the example
compound models. Differences between the samplers in `results/` are therefore
stochastic (posterior draws), not systematic.