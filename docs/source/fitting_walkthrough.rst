A realistic fit, start to finish
================================

One AGN spectrum, one model, one sampler, four steps. This page is the whole fit as
the script runs it -- ``example/fit_iras07546_jax.py``, of which every code block below
is a verbatim slice -- together with the products that run writes (the figures are
copies of the files in ``example/output/``). The same story is in
``example/fit_iras07546_jax.ipynb`` if you would rather execute it cell by cell.

IRAS 07546+3928 at z = 0.0953: 4200-6800 A in the rest frame, 57 000 pixels at 0.046 A
sampling, H-gamma through H-alpha with the strong forbidden lines, fitted in one go
with 30 free parameters. Full settings on one GPU: 400 live points, ``dlogz = 0.01``.

.. contents:: On this page
   :local:
   :depth: 1

1. Read the spectrum
--------------------

The data come from a FITS file, are dereddened, shifted to the rest frame and cropped;
the error array is re-derived from the scatter of the spectrum itself rather than
trusted from the header. That last step matters: a wrong error scale makes every
posterior too narrow or too wide by the same factor.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% load
   :end-before: # %% helpers

2. Build the model, and give every parameter a prior
----------------------------------------------------

The model is a compound astropy model: a power-law continuum, the iron template, and
one submodel per line. The step itself is two calls; the functions behind them are
``build_model`` (133 lines of component construction) and ``set_priors`` (76 lines of
prior recipe, written out on :doc:`priors`).

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-2-model
   :end-before: # %% step-3-fitter
   :dedent: 4

``set_priors`` attaches one prior per free parameter, in four groups: continuum and
iron, every line peak (log-uniform), narrow kinematics (one shift and one width shared
by every narrow line, measured from [O III]), and broad kinematics (one width box of
300-8000 km/s for the core *and* the wind, with each Balmer line keeping its own
amplitude). The recipe is written out on :doc:`priors`, which also explains what each
shape does and how to make one parameter's prior depend on another's.

A fit with an unset prior is a silent bug: the parameter then samples its whole default
box. The check above turns that into an error before the sampler starts.

3. Run the sampler
------------------

Nested sampling on the GPU. ``nlive`` live points are kept, and the run stops when less
than ``dlogz`` of the evidence may remain unseen.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-3-fitter
   :end-before: # %% step-4-fit
   :dedent: 4

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-4-fit
   :end-before: # %% step-5-results
   :dedent: 4

The run prints its own summary::

          log Z = 130889.60 +- 0.62   (66956 iterations, 10.2 min)
       chi2/dof = 0.307, residual rms 0.55 sigma, worst pixel 3.9 sigma

4. Present the results
----------------------

The best fit, the residual, and the table a paper would quote -- medians with the 16th
and 84th percentiles. Widths and integrated fluxes are *derived* from those numbers, so
they are computed here rather than sampled.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-5-results
   :end-before: # %% step-6-figure
   :dedent: 4

.. code-block:: text

   parameter                  median         -         +  unit
   ----------------------------------------------------------------------------
   Continuum.amplitude       0.54103    0.0003   0.00029  1e-14 erg/s/cm2/A
   Continuum.alpha            1.0778     0.005    0.0051
   FeII.amplitude            0.25047    0.0015    0.0015  1e-14 erg/s/cm2/A
   n_halpha.amplitude         2.3372     0.019     0.018  1e-14 erg/s/cm2/A
   b_halpha.amp_c              1.523     0.011     0.012  1e-14 erg/s/cm2/A
   b_halpha.dv_c              187.24       2.4       2.4  km/s
   b_halpha.sigma_c           1301.5       6.6       6.7  km/s
   b_halpha.amp_w0           0.10704    0.0018    0.0017  ratio to core
   b_halpha.dv_w0            -519.32        28        26  km/s
   b_halpha.sigma_w0          5143.4        67        74  km/s
   n_hbeta.amplitude         0.58991    0.0082    0.0083  1e-14 erg/s/cm2/A
   b_hbeta.amp_c             0.62641    0.0039    0.0038  1e-14 erg/s/cm2/A
   b_hbeta.amp_w0            0.15966    0.0034    0.0034  ratio to core
   n_hgamma.amplitude        0.17306    0.0066    0.0065  1e-14 erg/s/cm2/A
   b_hgamma.amp_c            0.29017    0.0036    0.0037  1e-14 erg/s/cm2/A
   b_hgamma.amp_w0          0.099886    0.0062    0.0066  ratio to core
   OIII.amp_c0                1.9275    0.0095    0.0096  1e-14 erg/s/cm2/A
   OIII.dv_c                  70.717      0.95      0.96  km/s
   OIII.sigma_c               298.88       1.2       1.2  km/s
   OIII.amp_w0              0.036375    0.0017    0.0017  ratio to core
   OIII.sigma_w0              1343.9        47        47  km/s
   NII.amp_c0                0.75139     0.014     0.014  1e-14 erg/s/cm2/A
   SII.amp_c0               0.022533    0.0019    0.0019  1e-14 erg/s/cm2/A
   SII.amp_c1               0.053145    0.0019    0.0019  1e-14 erg/s/cm2/A
   OI.amp_c0                0.062106    0.0021    0.0022  1e-14 erg/s/cm2/A
   HeII.amplitude           0.033405    0.0041    0.0041  1e-14 erg/s/cm2/A
   b_heii.amp_c             0.086096    0.0023    0.0023  1e-14 erg/s/cm2/A
   OIII4363.amplitude       0.045406    0.0045    0.0047  1e-14 erg/s/cm2/A
   n_hei.amplitude           0.04289    0.0024    0.0026  1e-14 erg/s/cm2/A
   b_hei.amp_c              0.040666    0.0011    0.0012  1e-14 erg/s/cm2/A

The figure: data, model, and residuals in units of the error, with the 1-sigma guides
drawn so that a good fit is visibly flat and inside them.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-6-figure
   :end-before: # %% step-7-zooms
   :dedent: 4

.. figure:: _static/example_figures/iras07546_jax_fit.png
   :width: 100%

   The full 4200-6800 A window: data, total model, and every component underneath it.
   The lower panel is the residual in units of the error, with the +-1 sigma guides.

Three windows: 870 A, 380 A and 250 A wide, against the 2600 A in the figure above.
On that page a 700 km/s narrow core is two pixels; here it is a profile, and the wings
of the broad lines are visible at all.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-7-zooms
   :end-before: # %% step-8-posteriors
   :dedent: 4

.. figure:: _static/example_figures/iras07546_jax_fit_Hgamma_Hbeta.png
   :width: 95%

   H-gamma + H-beta with the [O III] doublet.

.. figure:: _static/example_figures/iras07546_jax_fit_Halpha_NII_SII.png
   :width: 95%

   H-alpha with [N II] and [S II]. The narrow lines sit on broad wings that the model
   does not try to fit with the same component.

.. figure:: _static/example_figures/iras07546_jax_fit_HeI_NaD.png
   :width: 95%

   He I 5876 and Na D, the faintest features in the model.

The posterior itself: one histogram per parameter, and the joint distribution of the
parameters that a reader is most likely to want a covariance for.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-8-posteriors
   :end-before: # %% step-9-save
   :dedent: 4

.. figure:: _static/example_figures/iras07546_jax_histograms.png
   :width: 100%

   Marginal posteriors, with the prior range marked on each panel.

.. figure:: _static/example_figures/iras07546_jax_corner.png
   :width: 75%

   The joint posterior of the parameters the fit is most sensitive to.

5. Save it, and read it back
----------------------------

The samples as plain arrays, and the whole fit as one FITS file: model, priors, samples,
evidence, and the GPU-side arrays needed to resume. Reloading gives a
:class:`galspec.JNesty_Fit` whose quantiles, best fit and plots work unchanged.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-9-save
   :end-before: # %% entry
   :dedent: 4

.. code-block:: python

   from galspec.jnesty_fit_io import load_jnesty_fit_from_fits

   fit = load_jnesty_fit_from_fits("example/output/iras07546_jax_fit.fits")
   print(fit.log_evidence)                     # the same number, from the file

What is in the file -- and the one thing that cannot be stored -- is on
:doc:`jax_gpu`; the priors that went into it are on :doc:`priors`.

Where to go next
----------------

* ``example/fit_iras07546_jax_tied.py``: the same fit with the Balmer kinematics
  released and centred on H-alpha, which turns "the lines share one profile" into a
  measurement (:doc:`priors` has the construction and the result).
* ``example/fit_iras07546_jax_toggle.py``: how many broad components the profile needs,
  decided by the sampler rather than assumed.
* :doc:`jax_gpu`: the JAX backend, the batching, and the CPU-versus-GPU comparison.