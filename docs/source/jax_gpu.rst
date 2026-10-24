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

The iron template, and letting the lines broaden it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``galspec.jax_models.IronTemplate`` speaks the same language as the CPU
``galspec.IronTemplate_new``: ``amplitude``, ``sigma`` (velocity dispersion in
km/s), ``dv`` (velocity shift) and a ``z`` that is **fixed by default**.  This is
the parameterisation to fit with, because ``sigma`` and ``dv`` are the numbers a
paper quotes and the redshift is known:

.. code-block:: python

   iron = jxm.IronTemplate_new(amplitude=0.2, sigma=2500.0, dv=0.0, name="FeII")
   iron.z.fixed = True                  # already the default
   # sigma and dv are free: the Fe II optical multiplet is a measurement

``IronTemplate_tied`` is the stricter model: the template is broadened not by one
Gaussian but by a **multi-Gaussian profile that is fitted at the same time** --
normally the profile of the broad line, tied component by component.  Its kernel
parameters carry the *same names* as the ``Line_MultiGauss`` they come from, and
``n_components`` counts the kernel's components, so ``2`` is a core plus one wind:

.. code-block:: python

   # the broad component of H-alpha, on its own: component 0 is the broad core
   b_ha = jxm.Line_MultiGauss(n_components=2, amp_c=0.65, dv_c=0.0, sigma_c=1500.0,
                              amp_w0=0.10, dv_w0=0.0, sigma_w0=6000.0,
                              wavec=6563.0, name="b_halpha")

   iron = jxm.IronTemplate_tied(n_components=2, amplitude=0.2, name="FeII")
   iron.dv_c.tied     = tie.tie_MultiGauss_dv_c("b_halpha")      # the broad core
   iron.sigma_c.tied  = tie.tie_MultiGauss_sigma_c("b_halpha")
   iron.amp_w0.tied   = tie.tie_MultiGauss_amp_w0("b_halpha")    # the wind
   iron.dv_w0.tied    = tie.tie_MultiGauss_dv_w0("b_halpha")
   iron.sigma_w0.tied = tie.tie_MultiGauss_sigma_w0("b_halpha")
   # free iron parameters left: amplitude, and z (fixed by default)

There are **no weight parameters**, and no ``amp_c``: a ``Line_MultiGauss`` already
*is* a weighted sum of Gaussians, whose weights are its own amplitudes
(``amp_c`` for component 0, ``amp_c * amp_w0`` for the wind).  The iron only needs
the *shape* of that sum, and the convolution renormalises it, so the core
amplitude cancels identically and the wind ratio -- the line's own ``amp_w0``,
tied -- is the only weight left.  Adding a separate weight would introduce a flat
direction that the data cannot close.

With one component it is *identical* to ``IronTemplate`` (verified to machine
precision in ``tests/test_model_features.py``), so the generalisation cannot
quietly change the simpler model's meaning.  Shifts are relative to the core
(``dv_c + dv_w0``), exactly as in the line model, so the iron inherits the line's
centroid.

**Bound the wind ratio, and the core-wind ambiguity disappears.**  The two
components of a ``Line_MultiGauss`` are *exchangeable*: the map

.. math::

   A_c \to A_c A_w,\quad \sigma_c \to \sigma_w,\quad \Delta v_c \to \Delta v_c + \Delta v_w,
   \quad A_w \to 1/A_w,\quad \Delta v_w \to -\Delta v_w,\quad \sigma_w \to \sigma_c

leaves the spectrum unchanged to 3e-16 (a unit check in
``develop/dev_jax/iron_tied_io_test.py``), and the tied iron follows it.  So a fit
with ``A_w`` allowed above 1 has two equally good answers and the words "core" and
"wind" mean nothing.  Since a wind is by definition no brighter than its own core,
put ``A_w`` in ``[0, 1]``: the swap needs ``A_w' = 1/A_w > 1`` and is excluded,
while the widths keep their full data-driven range.  Measured on the worked
example's posterior (67 620 samples): zero samples on the mirror branch, and the
widths come out the same as with artificially disjoint width priors -- 1409 and
7247 km/s here, 1409 and 7255 km/s there -- with the same error bars, so the bound
costs no information.  ``develop/dev_jax/io_test/cpu_vs_jax.html`` is the separate
question of whether this model agrees with the CPU one.

The implementation subtracts the template's own width in quadrature, component by
component, instead of deconvolving it: exact for Gaussian kernels, and it needs no
host-side kernel array, so the model stays differentiable and ``jit``-able
through every kernel parameter.

The CPU ``galspec.IronTemplate_tied`` now takes the same route, component for
component: every kernel component is smoothed once with
``sqrt(sigma**2 - sigma_intr**2)`` on the template grid, projected onto the data grid
at its own shift, and added with the line's own relative weight.  The two backends
therefore agree to **1e-15** of the iron peak -- the same function, not two
approximations to it -- and both are within **0.01-0.02%** of an independent
velocity-space convolution (measured in ``develop/dev_jax/io_test/cpu_vs_jax.html``;
the residue is the 126 km/s sampling of the template, not the algorithm).  Before
this revision the CPU class FFT-*deconvolved* the template by a 200 km/s intrinsic
Gaussian, filtered the result with a Wiener filter and convolved with a kernel
sampled on a fixed +/-10 000 km/s window; it was 8.5% (mock) and 10.0% (fit) of the
iron peak away from the exact answer, and 8-10% from this model.  That algorithm is
kept, unmodified and clearly labelled, as ``legacy_iron()`` in the comparison script
so the old numbers stay reproducible.

**Cost.**  The convolution runs on the template grid (2 389 channels for
``park2022``, 801 of them the tabulated range and the rest the zero padding), not on
the 57 000-pixel spectrum, and only the sum is interpolated onto the data.  Measured
on this machine, a two-component kernel costs **+0.40 ms per evaluation** -- 56% of a
full-spectrum model, which is 0.32 ms without it.  A fit needs 1e5-1e6 evaluations,
so call it **40 s to 7 minutes** on top of the sampling.  It is not free -- the kernel costs one smoothing and one
interpolation per component -- but it does not change the scale of the problem.
Two things keep it there: the smoothing happens on the template rather than on the
spectrum, and the kernel radius is static ``jnp`` data sized from the widest width
the model can be asked for (the prior bound inside a fit, the widest constructor
value outside one), so nothing re-traces when a sample moves.  The zero padding
triples the template, which triples the smoothing; it is the price of not reflecting
the template's edge flux into the fit, and it is paid once per evaluation.

**The template grid is even in the logarithm of wavelength.**  As tabulated, both
templates are even in *wavelength* (``park2022``: exactly 2 A over 4000-5600 A), so
one channel would be 150 km/s at the blue end and 107 km/s at the red end and a
Gaussian of a given ``sigma`` would be a different number of channels at each end --
29% narrower where the template is red.  Both backends therefore resample at load
time, through :func:`galspec.iron_template.load_iron_template` (which calls
:func:`galspec.stellar_continuum.log_rebin`), onto a grid with constant
``dln(lambda)``: 126 km/s per channel over 4000-5600 A.  Every model built on the
template -- the CPU and JAX ``IronTemplate``, ``IronTemplate_new`` and
``IronTemplate_tied`` -- broadens it in velocity space by construction, and the CPU
and JAX versions still agree to 1e-16 because they share the grid.

The tabulated range is then extended with zeros by 100 000 km/s on each side, on the
same grid, for a total of 2 389 channels.  Without that the smoothing pads by
mirroring, and the template's red end (still carrying about 4% of its peak at
5600 A) is mirrored back into the spectrum as a constant floor: the model stayed at
5% of the iron peak all the way to 6800 A even though there is no iron there, and it
mis-set the iron wing at 5450-5600 A.  With the zeros the model fades out where the
template runs out.  Against an independent convolution on a uniform velocity grid the
JAX tied iron is then **0.02% of the iron peak over the whole range**, at every kernel
width tested (0.01-0.47%, the 0.47% at the narrowest kernel, where the template's own
sampling is what limits it).  ``tests/test_model_features.py`` pins the grid, the
padding and the velocity-space broadening.

LaTeX labels for parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fitters report ``amp_c0_7`` and ``Continuum.alpha``; plots should print
:math:`A_{c0}` and :math:`\alpha`.  ``galspec.param_labels`` splits every
parameter name into a base and a subscript, so one table covers the whole
library, and prefixes the owning model:

.. code-block:: python

   model["Halpha"].label = r"BH$\alpha$"      # display name, may contain maths
   fit = JNesty_Fit(model, wave, flux, error)
   print(fit.param_labels[:3])
   # ['Continuum:$A$', 'Continuum:$\alpha$', 'BH$\alpha$:$A_{c}$']

Both fitters expose ``param_labels`` (aligned with ``full_param_names``), so
``fit.plot_corner()`` and ``galspec.plot.plot_posterior_hist`` use them without
further work.  The model's ``name`` is untouched: it is still the key you index
with (``model["Halpha"]``).

A model class that needs a different symbol declares it in a ``PARAM_SYMBOLS``
class attribute, which overrides the shared table for that model only::

    class Line_Exponential(Line_Gaussian):
        PARAM_SYMBOLS = {"sigma": r"\varsigma", "amplitude": r"\mathcal{A}"}

    >>> parameter_labels(Line_Exponential(amplitude=1.0, sigma=300.0))
    ['$\mathcal{A}$', '$\Delta v$', '$\varsigma$', '$\lambda$']

One-dimensional posteriors
~~~~~~~~~~~~~~~~~~~~~~~~~~

A corner plot of thirty parameters is unreadable.  ``plot_posterior_hist`` draws
one histogram per requested parameter, with the median, the 16-84 percentile band
and the value printed on top:

.. code-block:: python

   galspec.plot.plot_posterior_hist(
       fit, params=["Halpha.sigma_w0", "OIII.sigma_c", "HeI.amp_w0"],
       ncols=3, filename="marginals.png")

It accepts a fit object or a plain ``(nsamples, ndim)`` array, and labels can be
overridden with ``labels=``.

The x range of each panel comes from the posterior (``view_quantiles``, 0.1% to
99.9% by default, plus ``pad`` of that span), **not** from the prior.  That is not
cosmetic: the prior on a broad width spans 300-8000 km/s while the posterior spans
a few km/s, so an axis stretched to the prior would draw a well-measured parameter
as a spike one pixel wide -- the shape, which is the reason for the panel, would be
invisible.  When a prior bound falls outside the plotted range, its value is
written in the panel corner, so the zoom hides nothing.  A parameter that really
has filled its prior is still obvious, because its histogram is flat rather than
peaked.

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
``example/fit_iras07546_jax.py`` leaves it free, models each Balmer line as two
separate models (a narrow ``Line_Gaussian`` tied to [O III] and a broad
``Line_MultiGauss`` with a core and a wind), fits He I 5876 the same way, and ties
the iron template's kernel to the broad H-alpha component by component, which is
why that fit covers 4200-6800 A with 30 free parameters over seventeen models. The
benchmark model is also convolved with
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

How many components do the broad lines need?
--------------------------------------------

``example/fit_iras07546_jax.py`` gives each broad Balmer line a core and one wind.
That number was a modelling choice made before the fit; the fit then measured the
widths of the components it was given.  The question this section answers is the
one before it: **does the data need the second component at all, or a third?**

The approach implemented here is to let the sampler decide.  A line is given
``k_max`` components, and the amplitude of every component after the first carries
a *toggle* prior: with some probability the amplitude is **exactly zero**, and that
component contributes nothing.  The number of components then becomes a posterior
-- one run reports how many Gaussians the profile needs, and how sure the data are.

The same construction is available for any parameter, through
:mod:`galspec.priors`.

The model side: any number of components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing had to change in the models.  ``Line_MultiGauss(n_components=k)`` is a core
(``amp_c``, ``dv_c``, ``sigma_c``) followed by ``k - 1`` wings, each with its own
amplitude **ratio** to the core (``amp_w0``, ``amp_w1``, ...) and its own shift and
width; ``IronTemplate_tied(n_components=k)`` takes the same kernel.  Both backends
loop over ``n_components`` rather than over hard-coded indices, and
``tests/test_jax_parity.py`` compares them for k up to 5 (flux agrees to 1e-16,
i.e. to rounding, as for every other model).

The tie helpers are generated for ``w0`` .. ``w4``
(``tie_MultiGauss_amp_w2``, ``tie_MultiGauss_sigma_w3``, ...), so a component can
be shared between lines as before: in the example the broad H-beta and H-gamma
wings take H-alpha's shift and width component by component (they keep their own
amplitudes -- each Balmer line has its own decrement), and the Fe II kernel follows
H-alpha's profile.

Extra components are cheap to evaluate and expensive to *sample*: on a 2401-pixel
mock the batched forward model costs 57.2 ms per 400 parameter vectors at k = 2 and
84.7 ms at k = 5, while the sampler's own cost grows with the dimension (each extra
component adds 3 parameters per line).

A prior that can switch a component off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import galspec

   priors = {
       # a spike at exactly zero (probability p_off), else a linear slab on (0, 1]
       "b_halpha.amp_w0": galspec.uniform_toggle(p_off=0.2),
       "b_hbeta.amp_w0": galspec.uniform_toggle(p_off=0.2),
       "b_hgamma.amp_w0": galspec.uniform_toggle(p_off=0.2),
   }
   fit = JNesty_Fit(model, wave, flux, error, bounds_dict=bounds,
                    custom_priors=priors, bound="multi", nlive=1000)

   p = galspec.toggle_prior_odds(0.2)       # 0.2/0.8 = 0.25, for the Bayes factor

The prior is applied as a *transform of the unit cube*: a uniform draw ``u`` with
``u < p_off`` maps to ``0.0`` and everything else maps to the slab.  Two
consequences matter.

* The zero is **exact**, not small.  That is what makes it a spike-and-slab prior
  in the statistical sense (George & McCulloch 1993; Kuo & Mallick 1998): a point
  mass at zero, which has *finite* prior volume -- ``p_off`` of the cube -- as
  opposed to a density on the reals, which has none.  A nested sampler can find a
  region with finite volume; it could never find a point of measure zero.
* Nothing couples the parameters.  ``JNesty_Fit`` applies each entry of
  ``custom_priors`` to one cube coordinate, independently, so the transform cannot
  enforce a hierarchical constraint among components.  All it does is allow a
  component's amplitude to be zero.

``galspec.priors`` also has ``log_uniform_toggle``: the same spike, but the slab is
uniform *in the log*, which is the right shape when the gated quantity spans
decades by itself -- an absolute amplitude in flux units, say, rather than a ratio
to a core.  ``component_toggle`` is the general form both wrappers call (``slab=
"linear"`` or ``"log"``), and a plain ``log_uniform`` is exported too, because the
same file is the natural home for a fit's whole prior dictionary.

For the wing *ratios* the linear slab is the honest choice: a ratio bounded by 1
has no decade structure to be uniform in, and it is the same uniform prior the
published fit already gives its wind.  On the mocks, a log slab actually did worse
-- it concentrates density at the 1% amplitudes where an extra Gaussian hides in
the noise (measured below).

The Bayes factor of a component follows from the prior odds it was given:

.. math::

   \mathrm{BF} = \frac{P(\text{on})/P(\text{off})}{p_{\rm on}/p_{\rm off}}
               = \frac{P(\text{on})/P(\text{off})}{(1 - p_{\rm off})/p_{\rm off}}.

Dividing the prior odds out matters here: with a ladder of ``p_off``, the higher
components are *a priori* unlikely, and would otherwise look rejected for that
reason alone.

...and an amplitude floor
~~~~~~~~~~~~~~~~~~~~~~~~~

A switched-off component is unambiguous (``amp_w = 0`` exactly).  A component that
is barely on is not: at 1% noise an unneeded Gaussian can sit at 1% of the core
peak and buy a little chi-square from the noise, with a Bayes factor of ~400 in
favour.  Measured on the mock in ``develop/dev_jax/broad_line_components.py``
(2401 pixels, 1% noise, a line whose true profile has two components): a
log-uniform slab produces exactly that false positive, at amplitudes of
0.012-0.015 and BF = 391, and one sample of the run failed to converge.  The
linear slab is better behaved, but the conclusion is not a property of the slab
shape.

So the *count* is defined with a floor: a component counts as needed when its peak
exceeds a fixed fraction of the core peak (3% in the worked example).  The floor is
part of the definition of "needed", not a hidden extra prior -- the fit still puts
posterior mass between zero and the floor, and the study reports the count at
several floors (1%, 3%, 5%, 10%) so that a stable answer can be recognised.

The recipe that works
~~~~~~~~~~~~~~~~~~~~~

Applied to IRAS 07546+3928 (``example/fit_iras07546_jax_toggle.py``):

* **only the three Balmer lines** get four components; He I 5876 and He II 4686
  keep one broad Gaussian, and the Fe II kernel follows broad H-alpha, k = 4;
  40 free parameters in total;
* **one width prior for everything**: the core and all three wings span
  300-8000 km/s.  A wing is *not* required to be broader than the core -- a
  narrower component is a legitimate second structure, and the sampler is free to
  find one, but the ladder of ``p_off`` already discourages higher components
  without also constraining their widths;
* **a ladder of prior probabilities**: ``p_off = 0.2, 0.5, 0.7`` for wings 0, 1, 2.
  The likelihood *cannot* tell two wings apart -- exchanging them leaves the
  spectrum bit-identical -- so the prior is the only place an asymmetry can live;
* **decide by the mode of the floored count**, then refit at that ``k`` with plain
  priors if an evidence is wanted.

That run is set up (``--broad-wings 3`` on the example gives the same model without
the toggles) and its output lands in ``example/output/iras07546_jax_toggle_*``.

The answer for IRAS 07546+3928
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The run asked for at most four broad components and returned **four**: in every
one of the 228 281 equally weighted samples all three wings are on, and at the 3%
floor ``P(N = 3) = 1.000``.  The three Bayes factors are unbounded above (no sample
in the run switched any of the wings off), so this is not a marginal call.  Sorted
by width, broad H-alpha comes out as:

.. list-table::
   :header-rows: 1

   * - component
     - peak / core peak
     - FWHM [km/s]
     - dv [km/s]
     - share of the broad flux
   * - core
     - 1.000
     - 2 066
     - +294
     - 46%
   * - wing 1
     - 0.085
     - 2 484
     - -919
     - 4.7%
   * - wing 2
     - 0.283
     - 5 050
     - +4
     - 31.9%
   * - wing 3
     - 0.041
     - 18 830
     - -2 617
     - 17.3%

The three extra components are not cosmetic.  Comparing the same data with the
published two-component model, chi2/dof falls from 0.321 to 0.195
(``dchi2 ~ 7200`` for ten extra parameters); and inside this one run the best
three-component sample is worse than the best four-component one by ``dchi2 = 62``
for the three parameters the fourth component costs.  The published two-component
model put all of that excess into one wind at ``sigma ~ 6930`` km/s; here the same
excess is three structures -- a near-core shoulder at 2 484 km/s, a 5 050 km/s
component, and a pedestal broader than the adopted ceiling -- and forcing it back
into one Gaussian is what costs the ``dchi2 ~ 7200``.

Two caveats belong with that number, and both are visible in the same output.

* **The pedestal sits on the width ceiling.**  ``b_halpha.sigma_w`` reaches
  7 996 km/s against the 8 000 km/s upper bound and ``prior_dominated()`` flags it,
  so the width of that component -- and possibly the number of components that
  describe the base of the line -- is limited by the adopted range rather than by
  the data.  Widen ``SIGMA_RANGE`` and the decomposition may merge the pedestal
  into the 5 050 km/s component, or ask for one more.
* **The count is floor-sensitive in the expected way.**  The pedestal peaks at
  4.1% of the core, just above the 3% floor: at a 5% floor the count is 2 and at
  10% it is 1.  The pedestal's Bayes factor is unbounded, so it is not noise -- a
  floor of 5% would simply be a different, coarser definition of "a component".

The run's ``prior_dominated()`` list needs reading with care too: it flags a
parameter when its median is within 1% of the *prior width* of an edge, and that
criterion also catches amplitudes that are small compared with their (deliberately
wide) prior box -- the faint He I, He II and forbidden lines appear there although
their posteriors are perfectly well constrained (the [S II] doublet comes back at
0.054 and 0.071, a ratio of 0.76).  The genuine warning in that list is the flagged
*width*; the amplitude entries are the diagnostic's tolerance, not the data.

This is a single object, and the point of the recipe is that the answer is measured
rather than assumed: on the mocks the same code stops at two wings where two are
enough, and here it does not stop until four components are in.

Reading the answer
~~~~~~~~~~~~~~~~~~

Two numbers need care.

* **Labels are not identified.**  Two components of similar width can swap index
  from sample to sample, and the sampler is not asked which one is "the" wind.
  Only the *count* and the *width-sorted* set of components are meaningful: on
  the two-wing mock the two real wings come back at their true widths under
  whatever index the sampler happens to give them, and the same physical component
  is wing 0 in one sample and wing 2 in the next.  The study script therefore
  reports the count, the per-index probabilities, *and* a width-sorted table; the
  width-sorted one is the table to quote.
* **The evidence of a gated fit is not an evidence for a k-component model.**  A
  toggled model is a *mixture* over subsets of components, so its log Z is not
  comparable with a plain-prior fit at any one ``k``.  Quote the component count
  from the gated run, and -- if you want an evidence ratio -- refit at that count
  with plain priors and compare those runs.

Nested sampling is what makes this affordable.  The spike owns ``p_off`` of that
parameter's prior volume, so reaching either state costs a *bounded* amount of
compression -- about ``ln(1/p_off)`` e-folds, i.e. 1.6, 0.7 and 0.4 for the ladder
above -- and the sampler reports the volume it found on each side.  A point mass
with no volume would instead be a state the sampler never visits, however long it
runs (which is also why the amplitude inside the spike is exactly zero rather than
"very small").

Calibration on mocks
~~~~~~~~~~~~~~~~~~~~

``develop/dev_jax/broad_line_components.py`` asks the question of data whose answer
is known: four mocks (2401 pixels around H-beta, 1% noise) built from the example's
own truth, with 0, 1, 2 and 3 wings, each fitted with three configurations.  It
writes ``develop/dev_jax/io_test/component_selection.html`` (figures, per-component
posteriors and the Bayes factors) and a JSON file with the same numbers.

The column is the mode of the floored count -- the number of wings reported -- with
its posterior probability in brackets:

.. list-table::
   :header-rows: 1

   * - mock
     - generated wings
     - linear slab, nlive 400
     - linear slab, nlive 1000
     - log slab, nlive 1000
   * - 0 wings
     - 0
     - 0 (1.00)
     - 0 (1.00)
     - 0 (0.90)
   * - 1 wing
     - 1
     - 1 (0.99)
     - 1 (1.00)
     - 1 (0.63)
   * - 2 similar wings
     - 2
     - 2 (1.00)
     - 2 (1.00)
     - 2 (1.00)
   * - 3 overlapping wings
     - 3
     - 2 (1.00)
     - 2 (1.00)
     - 2 (0.83)

Three things are worth reading off this table.

* **The count is the minimum the data need, not the count that generated them.**
  For 0, 1 and 2 wings it is exactly the generating number.  For the three-wing
  mock it is 2 -- and that is the right answer to the question as asked: a
  three-component model (core plus two wings) fits that mock at chi2/dof = 1.013,
  and adding the third wing buys ``dchi2 = 1.1`` for three extra parameters
  (p = 0.78).  Three overlapping wings that differ only in width are not separately
  required by 1%-noise data, so two is the minimum sufficient count.  A mock whose
  components *are* separable cannot be made by piling on overlapping Gaussians --
  the family is too flexible -- which is itself the main result for real data: the
  number of components saturates at what the profile shape and the noise support.
* **The linear slab is the recommended one.**  The log-uniform slab puts a second
  wing above the 3% floor in 0.37 of the posterior on the one-wing mock (0.52
  before the floor), against 0.00 for the linear slab, by concentrating density at
  the small amplitudes where an extra Gaussian can hide in the noise.  The floor is
  what stops that wing from being reported; the slab shape is what lets it exist.
* **Use 1000 live points when the count is the answer.**  In the run tabulated
  above, 400 and 1000 agree on all four mocks; but a repeat of the one-wing mock at
  400 live points put 0.91 of the posterior on a spurious second wing, which is
  exactly the failure mode the floor and the linear slab are there to suppress.  At
  1000 live points the answer was 1 (0.99) in both runs.  400 is marginal for a
  40-parameter model with gated amplitudes; the cost is a factor of a few in wall
  time, and the count is the number the run exists to produce.

What it does not do
~~~~~~~~~~~~~~~~~~~

* It does not make an extra Gaussian expensive.  The prior says a component is
  unlikely *a priori*; the likelihood is still free to prefer one, and with 1%
  noise it sometimes does.  The floor is what turns a continuum of small
  amplitudes into a decision.
* It does not identify which component is which, and it does not order them.
* It does not fit a different number of components per line: one count is shared by
  all the Balmer lines, which keeps the model self-consistent (they are the same
  gas) and the answer readable.  Per-line counts would need one gate per line and a
  rule for combining them.

Where the idea comes from
~~~~~~~~~~~~~~~~~~~~~~~~~

The statistics is standard: a spike-and-slab prior for variable selection goes back
to George & McCulloch (1993, *JASA* 88, 881) and was given its Gibbs-sampling
treatment by Kuo & Mallick (1998, *Biometrika* 85, 65).  The new part here is not
the statistics but its use on emission-line decomposition, and a reader should know
what the astronomy literature actually contains.  For line profiles the two
established routes are

* **fit separately and compare evidences** (what most AGN papers do: 1, 2 or 3
  Gaussian components, ranked by log Z or by an information criterion), and
* **trans-dimensional sampling**, which makes ``k`` itself a parameter: Green's
  reversible-jump MCMC (1995) or, in a nested-sampling framework, the
  birth/death moves of diffusive nested sampling (Brewer 2014,
  arXiv:1411.3921), which samples models of different dimension with a uniform
  prior on the number of components.

I did not find a paper that applies a spike-and-slab toggle prior to the
decomposition of emission lines, which is why the worked example also carries the
mock calibration above.  The trans-dimensional route is the more general one and
needs no floor: it samples ``k`` directly and gives a posterior on ``k`` that is a
model comparison.  It was not used here because a changing parameter count defeats
the static shapes that ``jit``/``vmap`` batching needs -- JNesty's throughput on
the GPU comes from evaluating a fixed-size parameter vector -- and because the
toggle needs no change to the sampler at all.  If a component count is the
*scientific* question rather than a diagnostic, diffusive nested sampling is the
better tool.
