# GPU / JAX fitting (`galspec.jax_models`, `galspec.jnesty_fit`)

The astropy models in GalSpec each have a JAX twin under `galspec.jax_models`
(imported as `jxm`) and the twin can be sampled with a JAX nested sampler,
`galspec.jnesty_fit.JNesty_Fit`.  On a decent GPU this is 10-100x faster than
`Dynesty_Fit` on the astropy models, because the whole model and the whole
likelihood are fused into one device kernel.

Use it whenever the user asks for JAX, GPU, speed, or "many components / many
parameters" — a 30-parameter global fit takes minutes instead of hours.

## Table of Contents

1. [Is the GPU actually there](#1-is-the-gpu-actually-there)
2. [Building the model](#2-building-the-model)
3. [Sampling with JNesty_Fit](#3-sampling-with-jnestly_fit)
4. [The batch_size / queue_size trap](#4-the-batchsize--queuesize-trap)
5. [Benchmarking before a long run](#5-benchmarking-before-a-long-run)
6. [Ties and priors](#6-ties-and-priors)
6b. [Gating the component count](#6b-gate-the-component-count-instead-of-fitting-a-fixed-one)
7. [Reading the posterior](#7-reading-the-posterior)
8. [Checkpointing and reloading](#8-checkpointing-and-reloading)
9. [Making a figure from a JAX fit](#9-making-a-figure-from-a-jax-fit)
10. [Checklist](#10-checklist)

## 1. Is the GPU actually there

```python
import jax, jnesty
print(jax.__version__, jnesty.__version__, jax.devices())
# [CudaDevice(id=0)]          <-- good
# [CpuDevice(id=0)]           <-- JAX did not see the driver
```

`JNesty_Fit(..., device="auto")` prints which device it picked and how much
memory it took.  If JAX reports a CPU device while `nvidia-smi` works, the
process is missing `/dev/nvidia*` (a container or a `bwrap --dev /dev` sandbox
hides them and `jax.devices()` silently falls back to the CPU, which is 40x
slower for this workload — always check).

## 2. Building the model

`jxm` mirrors `galspec`: same class names, same constructor keywords.

```python
import galspec
from galspec import jax_models as jxm
from galspec.utils import line_wave_dict as LW

cont = jxm.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=0.5, name="Continuum")
line = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, dv_c=0.0, sigma_c=1500.0,
                           amp_w0=0.1, dv_w0=-1000.0, sigma_w0=6000.0,
                           wavec=LW["Halpha"], name="b_halpha")
model = cont + line
```

Everything that works on the astropy side works here too: `name=`,
`model["Continuum"]`, `getattr(model[sub], par).fixed = True`, and the
`galspec.tie_*` objects (`tie_template_amplitude`, `tie_MultiGauss_dv_c`, ...).
`Line_template` takes the same `velc` / `template_flux` arrays.

Two useful attributes — note that `free_param_names` does **not** exist on the
astropy compound models, only here:

```python
model.free_param_names        # ['Continuum.amplitude', 'Continuum.alpha', ...]
model.qualified_param_names   # same order, names qualified by submodel
model.param_names             # flat names, e.g. 'amplitude_0'
model.param_dict()            # every parameter, ties already resolved
model.flux(wave, theta)       # theta = free parameters, in free_param_names order
model.flux_batch(wave, thetas)  # (n_samples, ndim) -> (n_samples, npix), vmapped
```

`flux_batch` is the reason to use JAX: it evaluates thousands of spectra in one
call.

## 3. Sampling with JNesty_Fit

```python
from galspec.jnesty_fit import JNesty_Fit

fit = JNesty_Fit(
    model, wave, flux, ferr,
    nlive=400,
    bound="multi",
    device="auto",
    queue_size=64,             # see section 4 -- this is the speed knob
)
fit.fit(progress=True, dlogz=0.05)
print(fit.log_evidence, fit.log_evidence_err)
```

Priors are set on the model, not passed to the fitter (section 3b).  A free
parameter with no prior is uniform over its bounds, and the defaults are wide, so
check that every parameter you care about has one:

```python
missing = [q for q in model.free_param_names if model.get_prior(q) is None]
assert not missing, missing
```

* `nlive` of 300-500 is right for 30-50 parameters; 100 is a test only — it
  converges to a *wrong* mode sometimes and the error bars come out 10x too
  small, which is easy to miss.
* `dlogz=0.05` is a sensible stopping rule.  For model comparison by evidence,
  quote `log_evidence_err` and remember that two runs differing by < 1 in log Z
  are not distinguished.
* Log-uniform priors for amplitude-like parameters avoid the posterior piling
  against zero:

  ```python
  from galspec import priors
  model.set_prior("Halpha.amplitude", priors.LogUniform(1e-4, 10.0))
  ```

## 3b. Priors

A prior is written on the parameter, next to `value`, `fixed` and `bounds` -- the
slot astropy reserves for user fitters:

```python
from galspec import priors

model.sigma_c.prior = priors.Uniform(30.0, 800.0)
model.amp_c.prior = priors.LogUniform(1e-3, 3.0)         # flat per decade
model.amp_w0.prior = priors.Uniform(0.0, 1.0)

# one parameter's prior may name another: H-beta's width near H-alpha's, and
# never more than five tie widths away
model.set_prior("b_hbeta.sigma_c",
                priors.Normal(mean="b_halpha.sigma_c", relative=0.20, nsigma=5))
```

* `priors.Normal(mean=..., sigma=... | relative=..., scale_from=..., nsigma=...)`:
  the centre is a parameter name or a number; `relative` makes the width a
  fraction of the centre; `scale_from` takes the width from a third parameter;
  `nsigma` is a *hard* limit, so a posterior pressed against it is a bound rather
  than a measurement.
* `priors.SpikeAndSlab(p_off=..., lower=..., upper=...)` switches a component off:
  `p_off` of the prior volume is at exactly zero.
* The fitter works out which parameter to draw first; nothing has to be declared.
  A cycle raises rather than hanging.
* Priors are stored with the fit (type plus arguments) and rebuilt on load, so a
  saved file can be resumed.  A prior passed as a bare Python function (the older
  `custom_priors` spelling) cannot be, and the loader says so.

## 4. The batch_size / queue_size trap

**`batch_size` is silently ignored when `bound="multi"`.**  This is worth
knowing before concluding that the GPU is slow.

Inside `jnesty.NestedSampler.__init__`:

```python
if queue_size is None:
    queue_size = 8 if bound == "multi" else 0
if queue_size > 1:
    batch_size = 1            # <-- whatever you passed is overwritten
```

So with the default `bound="multi"` the sampler runs 8 concurrent walks, i.e.
it sends 8-likelihood batches to the GPU, no matter what `batch_size` you
passed through `JNesty_Fit`.  Measured on a 31-parameter fit of a high-S/N
X-shooter spectrum:

| setting | iterations | wall clock |
|---|---|---|
| default (`queue_size=8`) | 17972 | 5.20 min |
| `batch_size=256` | 17972 | 5.22 min (ignored) |
| `queue_size=64` | 16935 | 2.84 min |

**The knob is `queue_size`**, not `batch_size`.  Larger is better up to the
point where the bound update or the move-step count starts to dominate; 32-128
is a good range.  Memory grows with `queue_size * nlive` only weakly, but
`nlive=400` plus `queue_size=64` already takes ~19 GB, so drop `queue_size`
first if you hit an out-of-memory error.

Also note `rwalk_K = max(25, ndim + 20)`: each walk takes that many steps, so
the likelihood count per iteration is `rwalk_K` divided over the queue.  A run
that looks slow may simply be burning `rwalk_K` steps per accepted live point.

## 5. Benchmarking before a long run

Do not launch a multi-hour run on a guess.  Time the likelihood itself:

```python
import time
theta = np.array([...])                    # one point inside the prior
model.flux(wave, theta).block_until_ready()          # triggers jit
t0 = time.perf_counter()
for _ in range(20):
    jax.block_until_ready(model.flux(wave, theta))
single = (time.perf_counter() - t0) / 20

thetas = np.random.uniform(size=(256, model.ndim))
jax.block_until_ready(model.flux_batch(wave, thetas))
t0 = time.perf_counter()
for _ in range(10):
    jax.block_until_ready(model.flux_batch(wave, thetas))
batched = (time.perf_counter() - t0) / 10 / 256
print(f"single {single*1e3:.2f} ms, batched {batched*1e3:.3f} ms/likelihood")
```

Typical result: 30 ms single, 0.4 ms batched on an RTX 4090 with 14k pixels.
If the single-call number is 30 ms, the run *must* be batched — this is exactly
what `queue_size` buys you.

## 6. Ties and priors

Ties behave as on the astropy side (`galspec.tie_*`), with two practical notes:

* A tied parameter is **not** in `free_param_names` and must have **no** prior of
  its own (`set_prior` raises for a tied or fixed parameter).  Iterate over
  `model.free_param_names` rather than a hand-written list, or a `--tie` flag will
  break the run.
* Wind/core ratios (`amp_w0` relative to `amp_c`) are best sampled on uniform
  [0, 1] and multiplied by the core amplitude *inside the model call*.  Without
  that ordering the core and the wind can trade flux and the posterior is
  bimodal.

## 6b. Gate the component count instead of fitting a fixed one

`priors.SpikeAndSlab` turns "how many components?" into a posterior instead of a
sequence of fits.  One run, one evidence, and the *count* falls out of it:

```python
P_OFF = (0.2, 0.5, 0.8, 0.95)          # prior P(wing i absent); higher order, less likely
MIN_FRACTION = 0.03                     # "on" = above 3 % of its own core peak

model = ...Line_MultiGauss(n_components=1 + len(P_OFF), ...)
for i in range(len(P_OFF)):
    model.set_prior(f"b_halpha.amp_w{i}",
                    priors.SpikeAndSlab(p_off=P_OFF[i], lower=0.0, upper=1.0))
```

Reading it back.  The amplitude of wing *i* is either exactly 0 or on its slab,
so:

```
P(on) = mean(amp_w_i > MIN_FRACTION)
BF    = (P(on) / P(off)) / toggle_prior_odds(p_off_i)      # prior odds divided out
count = number of wings with P(on) > 0.5                   # components = count + 1
```

and `P(N)` is the histogram of `sum(amp_w_i > MIN_FRACTION)` over the samples,
which is label-invariant: it counts components without claiming that "wing 2" is
a particular structure.

Three things that bite:

* **The floor is part of the answer, not a hidden prior.** Say what counts as
  "on" and report it next to the probability.  A component at 2 % of the core can
  be bought from the noise by an unneeded Gaussian.
* **Return a bound when `P(on)` saturates.** With `n` equal-weight samples,
  `P(on)=1` means `p_off < 1/n`, so the Bayes factor is `> (n-1)/odds` — an
  honest lower bound — not infinity.
* **The log Z of a gated fit is not comparable with a plain-prior fit.** A
  spike-and-slab is a mixture of sub-models; its normalisation is not the same
  integral.  Use the gate run for the *count*, then refit the plain model at that
  count for an evidence you can put beside other runs.

Budget: 12 gates on a 66-parameter model with `nlive=1200` took 56 min on a 4090
at 2.6 ms per likelihood.  Raise `max_ellipsoids` (to ~60, from the default 20):
a dozen point masses make the posterior strongly multimodal.

Independent gates on several lines at once do *not* give a common decomposition —
see `typical_bugs.md` bug #10 before writing centred priors across them.

The same gate works on a *single* extra component, where the question is
"whether" rather than "how many".  Add a second Gaussian to a forbidden line and
gate its amplitude against even prior odds:

```python
model["OIIIwing"].amp_w0.prior = priors.SpikeAndSlab(0.5, lower=0.0, upper=1.0)
P(on) = mean(amp_w0 > 0.03)      # 3 % of the first component's peak
BF    = (P(on) / (1 - P(on))) / 1.0
```

Read the result off the *shape* of the amplitude posterior as well as its mass: a
sharp peak at 0.14 with no samples at zero (P(on) = 1.000) is a detection; a broad
pile near the floor is not.  And keep the older component's boxes untouched so the
richer model contains the poorer one — see `typical_bugs.md` bug #11.

Before building any of this, bin the residual of the simpler model in velocity
and check that the feature is where the physics says it is.  For a doublet, scale
one line by the ratio and overlay: a real profile feature appears in both lines in
that ratio, while noise, a continuum error and unmodelled blends do not.  That
test costs a minute and decides whether the extra component is worth an hour of
sampling.

## 7. Reading the posterior

```python
lo, med, hi = fit.get_quantiles((0.16, 0.5, 0.84))
samples = fit.samples_equal_weight          # (nsamp, ndim), equally weighted
names = fit.full_param_names                # ALL parameters, ties included
print(fit.ndim, fit.n_iterations, fit.log_evidence, fit.log_evidence_err)
```

`full_param_names` lists every parameter of the model; the tied ones are
resolved.  To get the value of a tied parameter for every sample you must do
the tie arithmetic yourself:

```python
ix = {q: i for i, q in enumerate(fit.full_param_names)}
amp_w0 = samples[:, ix["b_hbeta.amp_w0"]]
```

For plotting only, `fit.get_best_fit()` gives
`(model, names, theta)` and `model.param_dict()` resolves every tie.

## 8. Checkpointing and reloading

```python
from galspec.jnesty_fit_io import save_jnesty_fit_to_fits, load_jnesty_fit_from_fits
save_jnesty_fit_to_fits(fit, "fit.fits")
fit2 = load_jnesty_fit_from_fits("fit.fits")
```

The FITS file carries the model tree, the samples, the bounds and the
evidence, so every post-processing step (plots, derived quantities, tables) can
run later without the sampler.

**The prior transform is not stored** (callables cannot go into FITS).  The
loader raises a `UserWarning` listing the custom priors it could not restore.
Consequence: you can analyse a reloaded fit, but you must not resume sampling
or call `fit.fit()` on it.  Reload with the same `wave` grid you fitted on —
pass `wave=` explicitly if the stored grid was rebinned.

## 9. Making a figure from a JAX fit

Pass the JAX model straight to `plot_fit_jax`:

```python
model_jax, _, theta = fit.get_best_fit()        # also sets the current values
ax, axr = galspec.plot.plot_fit_jax(wave, flux, model_jax, error=error,
                                    residual_in_sigma=True,
                                    xlabel="Rest wavelength  [A]")
```

Same signature and same figure as `plot_fit`, plus two things it cannot do for a
JAX model:

- it refreshes tied parameters first (`model.sync_tied_values()`), because a JAX
  leaf keeps its own storage and a tied one holds its construction default until
  something writes the tie back;
- it breaks `Line_MultiGauss` / `Line_MultiGauss_doublet` into their Gaussians as
  dashed curves, via the model's `subcomponents` property (core, then one per wind;
  a doublet gets one of each under each line).

```python
model_jax["b_halpha"].n_components      # 2
parts = model_jax["b_halpha"].subcomponents   # [Line_Gaussian(core), Line_Gaussian(wind0)]
assert np.allclose(sum(p(wave) for p in parts), model_jax["b_halpha"](wave))
```

Do **not** pass a JAX model to `plot_fit`: its `isinstance(m, Line_MultiGauss)`
test is against the *astropy* class, so the dashed subcomponents silently vanish
(everything else duck-types, which is what makes the bug quiet).

A `JNesty_Fit` works too — `plot_fit_jax(wave, flux, fit, ...)` calls
`get_best_fit()` itself.

If you need an astropy twin anyway (to hand the model to code that only accepts
astropy models), build it and check the parity rather than assuming it:

```python
dev = np.max(np.abs(model_cpu(wave) - model_jax.flux(wave, theta))) / np.max(np.abs(model_jax.flux(wave, theta)))
assert dev < 1e-6, dev
```

## 10. Checklist

- [ ] `jax.devices()` shows the GPU
- [ ] every `free_param_names` entry has a bound or a custom prior
- [ ] `queue_size` set explicitly (not `batch_size`)
- [ ] likelihood benchmarked single vs batched
- [ ] `nlive >= 300` for a production run, `dlogz <= 0.1`
- [ ] FITS checkpoint written immediately after `fit.fit()`
- [ ] CPU/JAX parity checked before any figure
