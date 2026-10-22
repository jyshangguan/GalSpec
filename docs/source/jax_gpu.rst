JAX / GPU acceleration
======================

GalSpec's original spectral models are `astropy`_ Fittable1DModel classes fitted
with numpy. That is convenient but slow: every likelihood evaluation writes the
free parameters onto the model through ``setattr`` and then re-evaluates a Python
expression tree, and every evaluation happens on a single CPU core.

``galspec.jax_models`` re-implements the same models as pure `JAX`_ functions, so
that

* the model evaluation is a single ``jax.jit``-compiled computation,
* it can be ``vmap``-ed over batches of parameter vectors (JNesty feeds the GPU
  this way, which is where the largest gain comes from),
* and it runs on a GPU when one is available.

The models are numerically **identical** to the CPU versions: with float64
enabled, fluxes agree to ~1e-16 relative, i.e. to floating-point rounding
(``tests/test_jax_parity.py`` asserts this for every model and for the
example compound models).

What is available
-----------------

.. list-table::
   :header-rows: 1

   * - CPU (astropy) model
     - JAX equivalent
   * - ``astropy.modeling.models.PowerLaw1D``
     - ``galspec.jax_models.PowerLaw1D``
   * - ``astropy.modeling.models.BrokenPowerLaw1D``
     - ``galspec.jax_models.BrokenPowerLaw1D``
   * - ``astropy.modeling.models.Const1D``, ``Linear1D``, ``Polynomial1D``
     - ``Const1D``, ``Linear1D``, ``Polynomial1D``
   * - ``astropy.modeling.models.Gaussian1D``
     - ``Gaussian1D``
   * - ``galspec.WindowedPowerLaw1D``, ``WindowedLinear``
     - ``WindowedPowerLaw1D``, ``WindowedLinear``
   * - ``galspec.Line_Gaussian``
     - ``Line_Gaussian``
   * - ``galspec.Line_GaussHermite``, ``Line_GaussHermite_doublet``
     - ``Line_GaussHermite``, ``Line_GaussHermite_doublet``
   * - ``galspec.Line_MultiGauss``, ``Line_MultiGauss_doublet``
     - ``Line_MultiGauss``, ``Line_MultiGauss_doublet``
   * - ``galspec.Line_Absorption``, ``Line_Absorption_doublet``
     - ``Line_Absorption``, ``Line_Absorption_doublet``
   * - ``galspec.Line_template``
     - ``Line_template``
   * - ``galspec.Line_Exponential``
     - ``Line_Exponential``
   * - ``galspec.IronTemplate``
     - ``IronTemplate``
   * - ``galspec.StarSpectrum``, ``galspec.Multi_StarSpectrum``
     - ``StarSpectrum``, ``Multi_StarSpectrum``
   * - ``galspec.convolve_lsf``, ``galspec.convolve_lsf_var``
     - ``convolve_lsf``, ``convolve_lsf_var``

The ``galspec.tie_*`` classes are reused unchanged: they only read
``model[name].parameter`` and do arithmetic, which works the same way on a JAX
model.

API
---

The JAX models deliberately mirror the astropy ones: the same keyword arguments,
the same ``fixed``/``tied``/``bounds``/``name`` semantics, the same parameter
names (``amplitude_0``, ``amp_c0_1``, ... where the suffix is the submodel
index), and the same ``+``/``*`` composition. Only the evaluation is different:
a JAX model offers ``flux(x, theta)`` and ``flux_batch(x, thetas)`` for the
``jit``/``vmap`` path, while ``model(x)`` evaluates with the *current* parameter
values just like astropy.

.. code-block:: python

   import numpy as np
   from galspec import jax_models as jxm
   from galspec.jnesty_fit import JNesty_Fit

   continuum = jxm.PowerLaw1D(amplitude=0.5, x_0=5000, alpha=1.0,
                              fixed={"x_0": True}, name="Continuum")
   line = jxm.Line_MultiGauss_doublet(n_components=2, amp_c0=1.7, amp_c1=0.57,
                                      dv_c=-43, sigma_c=250,
                                      wavec0=5006.84, wavec1=4958.91,
                                      name="OIII", amp_w0=0.35, dv_w0=-280,
                                      sigma_w0=340)
   line.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("OIII", 2.98)
   model = continuum + line

   fit = JNesty_Fit(model, wave, flux, error, nlive=200, device="auto")
   fit.fit(dlogz=0.5)

   best_model, names, theta = fit.get_best_fit()
   print(fit.get_evidence())
   print(fit.prior_dominated())   # parameters pressed against a prior edge
   fit.plot_corner()

Priors are built with exactly the same rules as :class:`galspec.Dynesty_Fit`
(``bounds_dict`` first, then the parameter's own finite bounds, then a
per-parameter-type default with log-uniform amplitudes), so the same
``bounds_dict`` gives the same prior -- and therefore a comparable evidence --
in both fitters.

Precision
---------

``galspec.jax_models`` enables JAX's 64-bit mode on import, because that is what
makes the fluxes agree with the CPU models to rounding. float32 is available and
considerably faster on consumer GPUs, at ~4e-5 relative accuracy (measured for
the [O III] model):

.. code-block:: bash

   GALSPEC_JAX_X64=0 python my_fit.py     # disable float64 (float32 everywhere)

Adding a new model
------------------

Subclass :class:`galspec.jax_models.JaxModel`, declare the parameters and write
``evaluate`` as a pure ``jnp`` function of ``(x, params)``:

.. code-block:: python

   import jax.numpy as jnp
   from galspec.jax_models.core import JaxModel, ParamSpec

   class TwoSlope(JaxModel):
       PARAMS = (
           ParamSpec("amplitude", 1.0),
           ParamSpec("x_break", 5000.0),
           ParamSpec("alpha_1", 0.0),
           ParamSpec("alpha_2", 1.0),
       )

       def evaluate(self, x, params):
           ratio = x / params["x_break"]
           slope = jnp.where(x < params["x_break"], params["alpha_1"],
                             params["alpha_2"])
           return params["amplitude"] * ratio ** (-slope)

Everything else -- ``fixed``, ``tied``, ``bounds``, ``name``, compound ``+`` and
``*``, ``JNesty_Fit`` -- is inherited. Note that anything computed in
``evaluate`` must have a shape that is independent of the parameter *values*;
if a static size is needed (e.g. a convolution kernel radius), take it from a
constructor argument and mask with ``jnp.where``. ``prepare(x)`` is the hook for
x-dependent static data, and ``sigma_pix_max`` is the usual pattern for widths
(see ``IronTemplate`` or ``Line_Exponential``).

Benchmarks
----------

Everything below was measured on one machine (RTX 4090, 19 GB; 192-core host)
with the scripts in ``develop/dev_jax``. Raw numbers are stored in
``develop/dev_jax/results/``.

Cost of one likelihood evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``develop/dev_jax/bench_likelihood.py``. This is what nested sampling spends its
time on. "batched" means ``vmap`` over a block of parameter vectors, which is how
the GPU is kept busy.

.. list-table::
   :header-rows: 1

   * - model
     - astropy on CPU
     - JAX ``jit``, one point
     - JAX ``vmap``, batches
   * - [O III] window, 35 056 pixels, 8 free parameters
     - 0.697 ms
     - 0.030 ms (23x)
     - 0.013 ms (53x)
   * - full spectrum, 51 918 pixels, 39 free parameters
     - 7.663 ms
     - 0.192 ms (40x)
     - 0.073 ms (105x)

The log-likelihoods agree with the CPU fitter to ``|dlogL| = 0`` at the model's
own parameters and to 2e-16 relative over random prior draws, so the speed is
not bought with accuracy.

Complete fits
~~~~~~~~~~~~~

``develop/dev_jax/bench_jnesty.py``. Same data, priors, ``nlive`` and ``dlogz``
for every sampler; each configuration runs in its own process, and the JNesty
entries exclude the one-off compilation.

.. list-table::
   :header-rows: 1

   * - problem
     - dynesty, 1 CPU
     - dynesty, 8 CPUs
     - JNesty, CPU
     - JNesty, GPU
   * - [O III], 35 056 px, 8 par, converged (nlive 80, dlogz 1)
     - 57.7 s
     - 36.3 s (1.6x)
     - 13.9 s (4.1x)
     - 14.0 s (4.1x)
   * - AGN: power law + Fe II + [O III], 35 056 px, 10 par, converged
     - 111.7 s
     - --
     - 18.2 s (6.1x)
     - 17.7 s (6.3x)
   * - full spectrum, 51 918 px, 39 par, 4000 iterations
     - 1006 s
     - --
     - 45.2 s (22x)
     - 45.4 s (22x)

(speedups are relative to the single-process dynesty run. The full-spectrum row
used an equal iteration budget instead of convergence: neither sampler had
converged after 4000 iterations of a 39-dimensional problem, so only the timings
of that row are meaningful -- the evidences are not.)

For the converged problems all samplers agree: the maximum-likelihood fits have
the same chi-square, the best-fit parameters agree to within one posterior sigma,
and the evidences span ~3 with a median reported error of ~1. In other words the
JAX path reproduces the CPU result, as ``tests/test_jax_parity.py`` does at the
model level.

What the speedup depends on
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The three converged rows above differ only in how expensive the *astropy* model
is per evaluation, and that is what sets the speedup. Measured, on the same
machine:

.. list-table::
   :header-rows: 1

   * - CPU model cost per call
     - JNesty wall time to convergence
     - speedup vs dynesty
   * - 0.70 ms ([O III], 8 parameters)
     - 14.0 s
     - 4.1x
   * - 0.95 ms (AGN, 10 parameters)
     - 17.7 s
     - 6.3x
   * - 4.6 ms (full spectrum, 39 parameters)
     - 45.4 s
     - 22x

The JAX model costs the same either way -- roughly 0.013 ms per evaluation when
batched -- and what JNesty adds on top is close to a constant, ~0.15 ms of
per-call overhead. Hence

    speedup  =  (astropy cost per call) / 0.15 ms,

so a ten-fold gain needs the astropy model to cost more than ~1.5 ms per
evaluation. A two- or three-component fit to a few thousand pixels is cheaper
than that and lands at 4-6x; a full optical spectrum with a dozen components is
far above it and lands at 20x or more.

There is also a hard ceiling on small problems: JNesty needs ~2 ms per iteration
even with a trivial likelihood (measured), so a problem that converges in 3 000
iterations cannot finish faster than ~6 s no matter how fast the model becomes --
8.7x for the [O III] fit above. Pushing past that means giving JNesty's loop more
work per kernel launch, which is where the remaining headroom sits (see below).

Where the GPU does -- and does not -- help
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``develop/dev_jax/bench_device_scaling.py`` separates the two effects (device and
precision) for the [O III] model:

.. list-table::
   :header-rows: 1

   * - spectrum
     - JAX-CPU float64
     - JAX-GPU float64
     - JAX-CPU float32
     - JAX-GPU float32
   * - 4 382 pixels
     - 16.9 ms / iteration
     - 17.2 ms
     - 16.2 ms
     - 16.4 ms
   * - 35 056 pixels
     - 11.1 ms / iteration
     - 11.2 ms
     - 7.5 ms
     - 7.6 ms
   * - 350 560 pixels
     - 51.8 ms / iteration
     - 44.2 ms
     - 22.8 ms
     - 13.2 ms

At the spectrum sizes this package is used for, the GPU and the JAX-CPU backend
take the same time: a nested-sampling iteration is bound by the sampler's own
bookkeeping and by per-call dispatch, not by the arithmetic. The GPU only pulls
ahead when the model is expensive enough to dominate that floor (350 k pixels:
1.2x in float64, 1.7x in float32).

The arithmetic itself is much faster than the sampler currently uses: the batched
evaluation above runs at 15.6 us/point (float64) and 1.5 us/point (float32) for
the 35 k-pixel model, i.e. 45-460x faster than the astropy model's 0.70 ms. The
headroom is in the sampler: JNesty's random walk evaluates the likelihood for one
candidate at a time, so the batched throughput shown above is only partly
realised end-to-end. Feeding larger batches per kernel launch is the lever with
the largest remaining payoff for GPU runs.

Float32 is worth knowing about: on this GeForce the 64-bit pipeline runs at a
small fraction of the 32-bit rate, so float32 is ~3-10x faster per evaluation
and ~2-4x faster end-to-end, at ~4e-5 relative accuracy instead of ~1e-16.

Drawing twenty components
~~~~~~~~~~~~~~~~~~~~~~~~

``galspec.plot.plot_fit`` draws every component, the total model and the residual
panel from one call.  Two conventions in it exist because a twenty-component fit
broke the defaults:

* **Colours.**  matplotlib's ten-colour cycle is not enough for twenty components:
  two unrelated lines end up in the same colour, and the worst case is a line
  taking the same red as the total model.  ``plot_fit`` therefore cycles a
  24-colour palette (``galspec.plot._COMPONENT_COLORS``) with the red reserved for
  the total model.
* **Residuals.**  ``residual_in_sigma=True`` (with ``error=``) divides the residual
  by the flux uncertainty and labels the panel ``Res./sigma``, with dotted guides
  at +/-1.  In raw flux units a single bright pixel sets the scale of the whole
  panel, so a well-fitted zoom looks flat and the stated rms cannot be checked by
  eye.  Pass ``ylim1`` per panel for the same reason.

The legend is placed by the caller: with twenty entries, ``bbox_to_anchor`` outside
the axes is the only way to keep it from covering the data.

Saving a fit, and what lives on the GPU
---------------------------------------

``galspec.jnesty_fit_io`` writes a completed fit to FITS and reads it back::

    from galspec.jnesty_fit_io import save_jnesty_fit_to_fits, load_jnesty_fit_from_fits

    save_jnesty_fit_to_fits(fit, "iras07546_jnsty.fits")     # thin=2 to shrink it
    loaded = load_jnesty_fit_from_fits("iras07546_jnsty.fits")
    loaded.get_quantiles((0.16, 0.5, 0.84))

The layout is the same one ``galspec.dynesty_fit_io`` uses -- ``WAVE_USE``,
``FLUX_USE``/``FERR``, ``SAMPLES``, ``LOGWT``, ``LOGL``, ``LOGVOL``,
``SAMPLES_EQ``, ``PARAM_NAMES``, ``PARAM_BOUNDS``, ``MODEL_PARAM`` and
``MODEL_STR`` -- so a JNesty file can be read with the same habits and the same
scripts. Three extensions exist only here: ``SAMPLES_U`` (the unit-cube
coordinates), ``DLZ_TRAJ``/``SCALE_TR`` (the remaining-evidence estimate and the
random-walk step scale against iteration), and the ``X64``/``JAXVER``/``JNESTYV``/
``DEVICE``/``NITER``/``RUNTIME`` keywords needed to describe a run. The model is
serialised by :mod:`galspec.jax_models.serialization`, which knows about the
constructor arguments the astropy walker never sees (``n_components``,
``degree``, ``template_name``, a stellar type) and about convolution wrappers.

A loaded fit evaluates the model, the best fit, the quantiles and the evidence
exactly as the original did -- the reloaded model reproduces the flux to 0.0
difference in double precision, ties included. What it cannot reproduce is the
*prior transform*: ``custom_priors`` are Python callables, and a closure defined
in ``__main__`` cannot be written to a file. The file stores the resolved bounds
and the names of the parameters that had a custom prior; loading such a file warns
about it. That restriction is shared with the dynesty, nautilus and MCMC loaders.

**Memory.** JNesty's GPU footprint for a fit like this is ~1 GB and does *not*
grow with the number of iterations: the dead-point history is one buffer allocated
once at ``max_iterations`` size (``ndim * max_iterations * 8`` bytes -- 61 MB for
38 parameters and 200 000 iterations, 2.4 GB for 8 million), and the sampler runs
inside a single ``lax.while_loop`` with a fixed-shape state. ``memory_frac`` only
caps the proposal batch, so raising it does not reserve memory, and with
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (the default in the example environment)
the driver keeps the high-water mark for the life of the process without ever
shrinking it -- ``nvidia-smi`` therefore shows the peak, which can look like slow
growth if another job is sharing the card. Two things are worth knowing when the
card is shared:

* set ``max_iterations`` to a realistic budget (a value ten times larger than
  needed reserves a buffer ten times larger);
* call :meth:`galspec.jnesty_fit.JNesty_Fit.release_device_buffers` when a run is
  over. It drops the sampler (and JAX's compiled kernels) while keeping the
  results, so plots and summaries still work. Saving to FITS first makes that
  lossless.

Streaming the dead points to disk *during* a run is possible -- ``run_nested``
accepts ``init_live_x``/``init_live_logL``/``init_logZ_val``/``init_logX``/
``init_iter_offset``, which is how JNesty continues a run -- but it does not lower
the peak footprint, because the peak is the fixed history buffer. Its real value
is checkpoint/resume after an interruption, and it needs a joint-evidence
recombination on the host (``jnesty.dynamic.compute_integrals``) because each
segment's ``logZ`` is computed from that segment alone.

Caveats
-------

* JAX's first call in a process pays for compilation (about 1-3 s for these
  models, 5-20 s including CUDA initialisation for a 39-parameter model).
  Amortise it over a fit; :meth:`galspec.jnesty_fit.JNesty_Fit.warmup` does that
  explicitly.
* The CUDA backend cannot be selected after JAX has initialised; set
  ``JAX_PLATFORMS=cpu`` (or pass ``device="cpu"``) if you want to force the CPU.
* ``float64`` on consumer GPUs (GeForce) runs at a small fraction of the
  ``float32`` rate. The numbers above already include that penalty; on
  data-centre GPUs (A100/H100) ``float64`` is much less penalised.
* JNesty has no ``seed`` argument (it uses a fixed internal key), so repeated
  runs are reproducible but cannot be re-seeded.
* JNesty does not report the number of likelihood calls; ``JNesty_Fit.ncall``
  follows JNesty's own convention (``iterations * rwalk_K + nlive``) and is an
  estimate.

.. _astropy: https://www.astropy.org/
.. _JAX: https://jax.readthedocs.io/

.. _jax-gpu-sdss:

A complete AGN fit: CPU and GPU, same answer
--------------------------------------------

The benchmarks above use windows of a spectrum. The case that actually matters is
a full, realistic fit, so ``develop/dev_jax/run_sdss_fit.py`` runs one and writes
everything needed to compare: the SDSS spectrum of IRAS 07546+3928 (z = 0.0953,
dereddened, rest frame, 4200-6900 A, 59 146 pixels at 0.046 A sampling), a model
that covers H-gamma, H-beta and H-alpha together with all the strong forbidden
lines, and 23 free parameters.

The model
~~~~~~~~~

``develop/dev_jax/sdss_agn.py`` builds the *same* model twice, once from the
astropy classes and once from the JAX classes, from a single shared list of
components:

* a power-law continuum;
* the Park et al. (2022) Fe II template (the iron pseudo-continuum);
* H-alpha, H-beta, H-gamma: a narrow and a broad Gaussian each;
* [O III] 4959/5007 with a narrow core and a broad wing;
* [N II] 6548/6583, [S II] 6716/6731, [O I] 6300/6364, He II 4686, [O III] 4363.

The iron template's broadening and the line wavelengths are fixed (the former is
degenerate with the continuum: freed, it runs to whatever upper limit it is given
while chi2/dof is unchanged).

Twenty-one free parameters is what remains after the standard ties (all narrow
lines share the kinematics of [O III]; all broad Balmer lines share the kinematics
of H-alpha) and the fixed doublet ratios.  For the timing comparison the
[S II] 6716/6731 ratio is tied too; physically it should not be (it is an
electron-density diagnostic, not a fixed number), and the worked example in
``example/fit_iras07546_jax.py`` leaves it free, adds a second broad Gaussian
per Balmer line and four high-order Balmer lines (H-epsilon and H8-H11), which is
why that fit has 38 free parameters over twenty components. The benchmark model
is also convolved with
the SDSS line-spread function (Gaussian, R = 2000 at 5500 A): at 0.046 A sampling
the instrument, not the gas, sets the width of the narrow cores, so convolving
makes the comparison the harder, more realistic one.  (The worked example does
*not* convolve: for that spectrum the instrument contributes ~150 km/s next to
~700 km/s of gas, and its own reference fit in
``example/wrk_sagan_iras07546.ipynb`` likewise ignores the LSF.)

The two implementations agree to 1.5e-15 of the peak flux at the best fit, and
the log-likelihoods agree to 3e-5 out of ~1e5 (relative 3e-10) over 25 random
prior draws. This is what makes the timing comparison meaningful: the GPU is not
buying speed with accuracy.

Priors without a preliminary fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The priors are derived from the data's own flux range, not from a least-squares
fit. Every emission-line amplitude is log-uniform between 1e-3 and twice the
brightest pixel; narrow-line widths span 30-1000 km/s (FWHM 70-2400 km/s); broad
Balmer widths span 500-20 000 km/s; the continuum slope may range from -5 to +5.
Nothing is tuned to a known answer, which is what makes the exercise meaningful:
a nested sampler with wide priors has to find the solution itself.

The fit
~~~~~~~

.. list-table::
   :header-rows: 1

   * - quantity
     - value
   * - pixels / free parameters
     - 59 146 / 21
   * - reduced chi2
     - 1.98
   * - residual rms
     - 1.40 sigma
   * - narrow-line FWHM (shared by H-alpha .. [O I])
     - 824 km/s
   * - broad H-alpha FWHM
     - 4 690 km/s
   * - log evidence
     - 114 264 +/- 0.6

The residual scatter is 1.4 sigma, so the model describes a real spectrum at the
level expected from a 1% photon noise budget plus a 3% systematic allowance for
flux calibration, the iron template and the imperfection of a single power law.
The strongest lines ([O III], H-beta) retain a few-sigma misfit at their very
cores: two Gaussian components cannot reproduce the longer wings of a real line
profile. That limitation is identical for the two code paths and does not affect
the comparison.

Timing
~~~~~~

Same model, same data, same ``nlive`` (400) and same stopping rule
(``dlogz = 0.01``, i.e. less than 1% of the evidence may remain unaccounted for).
Each configuration runs in its own process; compilation is excluded from the
JNesty numbers.

.. list-table::
   :header-rows: 1

   * - configuration
     - wall clock
     - likelihood calls
     - log Z
   * - dynesty, 1 CPU core
     - 8.9 h (32104 s)
     - 2.76 x 10^6
     - 114271.0 +/- 0.6
   * - dynesty, 16 CPU cores
     - 83.7 min (5 025 s)
     - 2.58 x 10^6
     - 114 284.7 +/- 0.6
   * - JNesty, GPU
     - 6.3 min (379 s)
     - 2.91 x 10^6
     - 114 264.5 +/- 0.6

All three rows are measured wall-clock times for the same fit with the same
stable stopping rule.  The GPU is 84.7x faster than one core and 13.3x faster
than sixteen; sixteen cores are only 6.4x faster than one, because dynesty's
parallel mode funnels proposals through a parent process.  One model evaluation
costs 11.6 ms on this machine (32 104 s / 2.76 x 10^6 calls, and cross-checked
by a counter wrapped around the likelihood), which is the quantity that sets
every one of these numbers.

Three things are worth taking from the table.

* **The parameters are identical.** Over all 21 fitted quantities the two
  samplers' posterior medians differ by at most 0.05 of a combined standard
  error (median 0.02), and reduced chi2 is 1.978 for both.
* **Parallel CPU scaling is poor here.** Sixteen cores buy only 6.4x over
  one core (5 025 s against 32 104 s, both measured), because dynesty's parallel mode funnels every proposal through a single
  parent process: on this problem the workers sit at ~36% CPU. The GPU does not
  have that bottleneck, which is why the gain is larger than the per-call ratio.
* **The evidence agrees less well than the parameters.** log Z differs by 20.2
  between the two codes while each reports +/-0.6. Doubling ``nlive`` (400 to
  800) moves JNesty's answer by 0.3, so its integration has converged. The
  decisive control is the single-core run: the *same* dynesty code, on the same
  problem, differing only in core count, returns 114 271.0 against 114 284.7 --
  13.6 nats apart, 22x its quoted error. Sampler-to-sampler scatter of this size
  is therefore normal and is not GPU precision (the models agree to 3e-5 in
  log L). Report evidence ratios across runs made with the same sampler, or
  quote both.

A readable HTML version of this comparison, written for readers who are new to
spectral fitting, is produced by ``develop/dev_jax/make_report.py``.  It writes
``develop/dev_jax/results/report/report.html``: a single self-contained file
(every figure inlined, navigation by in-page anchors) that renders anywhere,
including in a sandboxed preview pane.  The same script also writes the four
sections as separate pages (``index.html``, ``model.html``, ``results.html``,
``glossary.html``) for reading straight from a checkout.
