A realistic fit, start to finish
================================

One AGN spectrum, one model, one sampler, and the step before it: measuring the narrow
lines' profile. This page is the whole fit as the scripts run it. Every code block
below is a verbatim slice of a file in the repository, and the figures are copies of
the files those scripts write, so nothing here is hand-assembled:

* ``example/fit_iras07546_o3_template.py`` -- step 2, measures the narrow-line profile
  on the [O III] doublet and writes it to a file (the code calls this *stage 1*: it is
  the first of the two fits);
* ``example/iras07546_model.py`` -- the model the fit uses, one piece at a time;
* ``example/fit_iras07546_jax.py`` -- steps 4 to 7, the whole-spectrum fit.

``example/fit_iras07546_jax.ipynb`` runs the last of those cell by cell.

IRAS 07546+3928 at z = 0.0953: 4200-6800 A in the rest frame, 57 000 pixels at 0.046 A
sampling, H-gamma through H-alpha with the strong forbidden lines. The fit has 27 free
parameters, runs with 400 live points and ``dlogz = 0.01``, and takes about twelve
minutes on one GPU.

Twelve of those 27 are narrow-line parameters -- eleven peak amplitudes and one shared
velocity shift; the narrow *profile* is not fitted at all. It is one measured object,
``example/data/iras07546_narrow_template.txt``, extracted from the [O III] doublet in
step 2 and shared by all ten narrow lines. That is what makes the parameter count 27
instead of 30, and step 2 is where the shape comes from, so it comes first.

.. contents:: On this page
   :local:
   :depth: 2

1. Read the spectrum
--------------------

The data come from a FITS file, are dereddened, shifted to the rest frame and cropped;
the error array is re-derived from the scatter of the spectrum itself rather than
trusted from the header. That last step matters: a wrong error scale makes every
posterior too narrow or too wide by the same factor.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-after: # %% load
   :end-before: # %% helpers

2. Measure the narrow-line profile on [O III]
---------------------------------------------

A narrow emission line in an AGN is not a Gaussian. It is the virial and turbulent
kinematics of a whole region, and its profile has wings that a Gaussian cannot bend
into. Fitting a Gaussian per line therefore does two bad things at once: it spends a
free width per line on a shape the data constrain much better once, and the widths it
returns depend on which lines are bundled together.

The alternative is to measure the shape once, on the line with the most signal per
pixel, and give it to the rest with an amplitude each. [O III] 5007 is that line here:
narrow, strong, and sitting in a window where the continuum is smooth and close by.

Two decisions make the measurement a measurement rather than a choice:

* **A local fit, not a global one.** The window is 160 A, 3 505 pixels. The broad
  H-beta wing runs underneath it and would otherwise be free to reshape the narrow
  profile -- which is exactly the degeneracy that a global fit cannot avoid.
* **The iron stays in.** [O III] 5007 sits where the Fe II pseudo-continuum is still
  strong -- the published fit gives it an amplitude of 0.25, and the local fit below
  starts there. Leave it out and the window fit has nowhere to put that flux but a very
  broad wing on [O III] itself, which then goes into the template and is added under
  every narrow line.

The window's model is three objects: a degree-3 polynomial on an abscissa mapped onto
[-1, 1], the iron template, and a three-component ``Line_MultiGauss_doublet`` -- a core
plus two winds. Three components is not assumed: it is what the evidence chose between
two and three (the scan that did that is ``develop/dev_template/stage1_o3_template.py``,
a local script that is not part of this repository; what is in this repository is the
run that produces the template). The wing width is capped at 1 500 km/s, because a
broader component is indistinguishable from the continuum over this window and will
absorb it if allowed to.

.. literalinclude:: ../../example/fit_iras07546_o3_template.py
   :language: python
   :start-after: # %% model
   :end-before: # %% fit

Two details in there are worth naming, because they are the same tricks the whole
spectrum model uses later. The 5007/4959 ratio is atomic, so it is *tied* and not
fitted -- ``o3.amp_c1.tied = tie_MultiGauss_doublet_ratio("OIII", 2.98)``. And the
priors are scaled from the line-free level of the window rather than from a previous
fit, so the sampler starts from the data and not from somebody's answer.

The fit itself is four lines:

.. literalinclude:: ../../example/fit_iras07546_o3_template.py
   :language: python
   :start-after: # %% fit
   :end-before: # %% template

It prints::

   window 4920-5080 A: 3505 pixels, line-free level 0.653

   -- fit the window --
     ln Z = 7450.85 +- 0.42   (36790 iterations, 0.8 min)

The fit is good to the eye and to the residuals, but the part that matters is what
comes next -- the profile. ``gen_template`` turns the fitted doublet into a single-line
shape on a velocity grid: the core at zero velocity, the wings at their own ``dv_w``,
normalised to unit peak. The line's own ``dv_c`` is *not* part of it, because that
belongs to each line that uses the template, which is how one profile can serve H-alpha
and [O III] at the same time.

.. literalinclude:: ../../example/fit_iras07546_o3_template.py
   :language: python
   :start-after: # %% template
   :end-before: # %% verify

The grid is 8 001 points from -4 000 to +4 000 km/s, at 1 km/s -- three times finer
than the 2.8 km/s a pixel spans here, so the template does not add resolution the data
do not have. The quality control is the check at the end: the profile has to have
decayed to below 0.1% of its peak at both edges of the grid, or every narrow line would
get a truncated wing and a hard edge.

.. figure:: _static/example_figures/iras07546_o3_window_fit.png
   :width: 100%

   The [O III] window fit: data, total model, the degree-3 continuum, the iron, and the
   three dashed curves that are the doublet's core and two winds. Residuals in units of
   the error in the lower panel. Drawn by ``plot_fit_jax``, which takes the dashed
   curves from the submodels themselves.

.. figure:: _static/example_figures/iras07546_narrow_template_profile.png
   :width: 72%

   The profile that comes out: the fitted [O III] shape in velocity space, normalised to
   unit peak, with the half-maximum marked. FWHM 588 km/s, peak 9 km/s blueward of the
   systemic velocity. The shaded 16-84% envelope of the posterior is +-1% of the peak --
   thinner than the line, and thinner than the plot's own line width.

That last sentence is the honest part of this step. The decomposition into a core and
two wings is *degenerate*: the three components trade amplitude against width at almost
constant likelihood, so the parameters of the decomposition are not individually
determined -- the posterior envelope of the profile is +-1% of the peak. What is
determined is the shape they all share, and that is the only thing the template uses.
``profile_envelope`` in the script measures that envelope by resampling the posterior
and printing it next to the profile, so "the shape is safe to freeze" is a number and
not an assumption.

The script then writes the profile to ``example/data/iras07546_narrow_template.txt``
with its provenance on top, so that a reader who finds only the file still knows where
it came from::

   # narrow-line template of IRAS 07546+3928
   # measured on the [O III] 4959/5007 doublet, normalised to unit peak
   # generated by example/fit_iras07546_o3_template.py
   # window 4920-5080 A, degree-3 local continuum, 3-component doublet, ln Z = 7450.85
   # FWHM 587.9 km/s  (sigma_equivalent 249.6 km/s), peak at -9.0 km/s
   # the frozen profile fits the window at chi2/dof = 0.086
   # col 1: velocity [km/s]   col 2: normalised flux

The ``chi2/dof`` line is a self-consistency check, not a victory: the same window is
refitted with the *frozen* profile in place of the three Gaussians, and if the profile
were a poor description of the data this number would be orders of magnitude above the
0.086 the free fit reaches. The frozen profile is a smoother description, so it is not
expected to be identical -- only close.

3. Build the model, one piece at a time
---------------------------------------

The whole-spectrum model is a compound astropy model: a power-law continuum, the iron
template, and one submodel per line. It is built by four independent functions -- one
per physical piece -- that are then summed and tied, which is the only reason the
pieces can be drawn and checked separately, as the four figures below do.

The script's step is one call, after the spectrum and the profile are read:

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-2-model
   :end-before: # %% step-3-fitter
   :dedent: 4

Everything it calls lives in ``example/iras07546_model.py``, so this example,
``fit_iras07546_jax_tied.py`` and ``fit_iras07546_jax_toggle.py`` fit the same thing
instead of three copies of it.

3.1. The continuum
~~~~~~~~~~~~~~~~~~

One power law over 2 600 A: an amplitude and a slope. It is anchored at 5500 A, in the
middle of the range, so that amplitude and slope are nearly independent -- move the
anchor to the blue end and the two parameters become a line's intercept and gradient,
correlated at the 0.99 level.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def build_continuum():
   :end-before: def build_broad_lines(

.. figure:: _static/example_figures/iras07546_jax_model_continuum.png
   :width: 78%

   The continuum alone, drawn from the fitted parameters, against the data. ``alpha``
   comes out at 1.07 in F-lambda, which is a slope of -0.93 in F-nu -- redder than a
   bare quasar's -0.5, the starburst showing through.

3.2. The broad lines
~~~~~~~~~~~~~~~~~~~~

Three Balmer lines and two helium lines. Each Balmer line is a core plus a wind
(``Line_MultiGauss``), and He I 5876 and He II 4686 are weak enough to be one Gaussian
each.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def build_broad_lines(
   :end-before: def build_narrow_lines(

What is *not* in this function is as important as what is: no ties yet. The pieces are
built loose and tied afterwards (sections 3.5-3.7), because a tie is a callable resolved by
name at evaluation time and does not need the compound model to exist.

.. figure:: _static/example_figures/iras07546_jax_model_broad.png
   :width: 100%

   Left: the five broad components and their sum, from the fitted parameters. Right: the
   same five lines aligned on their peaks and normalised. The shift and the width are
   shared, so the curves nearly coincide; what separates them is the wind amplitude,
   which is free per line -- the Balmer decrement of the wind, and a measurement.

3.3. The narrow lines
~~~~~~~~~~~~~~~~~~~~~

Ten lines, all on the measured profile: one amplitude each, the atomic ratio for the
second peak of the forbidden doublets, and no width anywhere, because the profile
carries it. [S II] is the exception that keeps both amplitudes free, because
6716/6731 is a density diagnostic.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def build_narrow_lines(
   :end-before: def build_iron_template(

.. figure:: _static/example_figures/iras07546_jax_model_narrow.png
   :width: 100%

   Left: where the ten narrow lines sit, each with its amplitude from the fit. Right: the
   same ten, aligned on their peaks and normalised -- one curve, not ten, which is the
   whole point of the template. The offset curves are the doublets keeping their atomic
   separation: [N II] at -1 594 km/s, [S II] at -670 km/s.

3.4. The iron template
~~~~~~~~~~~~~~~~~~~~~~

The Fe II pseudo-continuum is not a line: it is a forest of blends between 4 200 and
5 600 A that no individual transition can be fitted. It is used as a template --
``IronTemplate_tied`` -- convolved with a broadening kernel and scaled by one number.
It is not a small correction: in the fit below it reaches 0.18 at 4 556 A against a
continuum of 0.66, a quarter of the total, and that structure has to go somewhere. A
model without it gives the flux to whatever else is nearby -- the continuum, or a wing
on the nearest broad line.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def build_iron_template(
   :end-before: def tie_narrow_lines(

.. figure:: _static/example_figures/iras07546_jax_model_iron.png
   :width: 78%

   The Fe II pseudo-continuum under H-beta, [O III] and H-gamma, from the fitted
   amplitude. The wiggles are the template's own structure; what the fit decides is the
   broadening (section 3.7) and the amplitude.

3.5. Tying the narrow lines: one shift, one ratio per doublet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The narrow lines share a velocity: eleven peaks become one measurement of the
narrow-line-region systemic velocity, and no line can drift on its own. The second peak
of a forbidden doublet is not fitted at all, because the ratio is atomic.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def tie_narrow_lines(
   :end-before: def tie_broad_lines(

A tie is a callable, not a value: ``tie_MultiGauss_dv_c("OIII")`` looks the reference up
by name when the model is evaluated, so ties can be assigned before the compound model
exists and they follow their reference through every sample. That is also why the
pieces above are built loose and tied afterwards: the model does not have to be
assembled in one expression.

Tying every narrow shift is a statement rather than a measurement: a line that really
sits at a different velocity is absorbed into its amplitude and into the residual, and
the model cannot report it. These lines do come from the same gas, so it is a good bet
-- but it is the kind of assumption worth writing down rather than leaving implicit in
the code. The right-hand panel of the narrow-line figure above is what it buys: ten
curves that are one curve.

3.6. Tying the broad lines: one shape, three amplitudes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Broad H-beta and H-gamma follow H-alpha component by component: the core's shift and
width, and every wind's shift and width. What is *not* tied is the amplitude, of the
core or of the wind -- the Balmer decrement is a measurement, and it is one of the
things this fit is for.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def tie_broad_lines(
   :end-before: def tie_iron_template(

This is the "one broad-line region" assumption, and it is the one
``example/fit_iras07546_jax_tied.py`` tests: that script releases the eight parameters
removed here, puts a Gaussian on the difference instead of a tie, and reports how far
each line moved in units of that width (:doc:`priors` has the construction and the
result -- H-gamma's wind is the one that disagrees).

3.7. Tying the iron template: its kernel *is* broad H-alpha
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The iron is a template convolved with a broadening kernel. Every parameter of that
kernel -- core shift and width, wind shift and width -- is tied to the broad H-alpha
component, which is the strongest statement in the model: the iron lines come from the
same gas as the Balmer line, so they cannot have a velocity structure the Balmer line
does not have.

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def tie_iron_template(
   :end-before: def build_model(

Why tie rather than fit the kernel? Because a free kernel width lets the iron absorb Fe
II flux into a fake wind, and earlier fits of this object did exactly that -- the
narrow lines were fitted well and the broad profile was quietly wrong. With the kernel
tied, the iron costs exactly one free parameter, its amplitude.

3.8. The composition, and a prior for every parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The four pieces are summed in a fixed order -- narrow and broad alternating, so that a
reader finds [O III] next to H-beta rather than in a block at the end -- and then tied:

.. literalinclude:: ../../example/iras07546_model.py
   :language: python
   :start-at: def build_model(
   :end-before: # %% priors

Seventeen components, 27 free parameters: two for the continuum, one for the iron,
twelve for the narrow lines (eleven peaks and one shared shift), twelve for the broad
lines.

The priors are attached by ``set_priors``, in four groups: continuum and iron, every
line peak (log-uniform), narrow kinematics (the one shared shift -- there is no narrow
width to give a prior to, because the profile carries it), and broad kinematics (one
width box of 300-8 000 km/s for the core *and* the wind, with each Balmer line keeping
its own amplitude). The recipe is written out on :doc:`priors`, which also explains what
each shape does and how to make one parameter's prior depend on another's.

A fit with an unset prior is a silent bug: the parameter then samples its whole default
box. ``set_priors`` ends by walking every free parameter and raising if one has no
prior, which turns that bug into an error before the sampler starts.

4. Run the sampler
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

      template: iras07546_narrow_template.txt, 8001 points, -4000 to +4000 km/s, FWHM 588 km/s

      model: 17 components, no LSF convolution, 27 free parameters
             narrow lines are the template: 10 of the 17 components, one shift, no widths

          log Z = 130677.27 +- 0.58   (60678 iterations, 11.7 min)
       chi2/dof = 0.315, residual rms 0.56 sigma, worst pixel 3.9 sigma

5. Present the results
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
   Continuum.amplitude       0.54259   0.00026   0.00027  1e-14 erg/s/cm2/A
   Continuum.alpha            1.0747    0.0044    0.0044
   FeII.amplitude            0.24733    0.0015    0.0015  1e-14 erg/s/cm2/A
   n_halpha.amplitude         2.4281     0.025     0.024  1e-14 erg/s/cm2/A
   b_halpha.amp_c             1.5429     0.013     0.013  1e-14 erg/s/cm2/A
   b_halpha.dv_c              209.44       2.6       2.5  km/s
   b_halpha.sigma_c           1194.4       5.5       5.9  km/s
   b_halpha.amp_w0           0.13541    0.0019    0.0019  ratio to core
   b_halpha.dv_w0            -420.46        18        18  km/s
   b_halpha.sigma_w0          4316.4        41        42  km/s
   n_hbeta.amplitude         0.61144      0.01    0.0099  1e-14 erg/s/cm2/A
   b_hbeta.amp_c             0.62481    0.0045    0.0044  1e-14 erg/s/cm2/A
   b_hbeta.amp_w0            0.19568    0.0039    0.0039  ratio to core
   n_hgamma.amplitude        0.16966    0.0078    0.0078  1e-14 erg/s/cm2/A
   b_hgamma.amp_c             0.2956     0.004    0.0042  1e-14 erg/s/cm2/A
   b_hgamma.amp_w0            0.1264    0.0068    0.0071  ratio to core
   OIII.amp_c0                2.2814    0.0077    0.0076  1e-14 erg/s/cm2/A
   OIII.dv_c                  95.596      0.97         1  km/s
   NII.amp_c0                0.65068     0.016     0.016  1e-14 erg/s/cm2/A
   SII.amp_c0               0.033291    0.0022    0.0022  1e-14 erg/s/cm2/A
   SII.amp_c1               0.062543    0.0021    0.0022  1e-14 erg/s/cm2/A
   OI.amp_c0                0.086604     0.002     0.002  1e-14 erg/s/cm2/A
   HeII.amplitude           0.031606     0.005    0.0046  1e-14 erg/s/cm2/A
   b_heii.amp_c                0.098    0.0023    0.0024  1e-14 erg/s/cm2/A
   OIII4363.amplitude       0.047033     0.005    0.0049  1e-14 erg/s/cm2/A
   n_hei.amplitude          0.045207    0.0031     0.003  1e-14 erg/s/cm2/A
   b_hei.amp_c              0.040059    0.0013    0.0014  1e-14 erg/s/cm2/A

The table is three rows shorter than the Gaussian model's and has no ``OIII.sigma_c``
row: the narrow lines contribute one peak each (``amp_c0``, or both peaks for [S II],
whose ratio is a density diagnostic) and one shared shift. The [S II] ratio that comes
out is 0.0625/0.0333 = 1.88, which is high for a narrow-line region and is the number
to check against an electron-density calculation.

The figure: data, model, and residuals in units of the error, with the 1-sigma guides
drawn so that a good fit is visibly flat and inside them. The dashed curves are the
sub-components of the multi-Gaussian lines -- broad H-alpha's core and its wind, one
curve each -- which ``plot_fit_jax`` takes from ``model[...].subcomponents``.

.. literalinclude:: ../../example/fit_iras07546_jax.py
   :language: python
   :start-after: # %% step-6-figure
   :end-before: # %% step-6b-model-pieces
   :dedent: 4

.. figure:: _static/example_figures/iras07546_jax_fit.png
   :width: 100%

   The full 4200-6800 A window: data, total model, and every component underneath it,
   with the broad lines' cores and winds drawn as dashed curves in the colour of the
   line they belong to. The lower panel is the residual in units of the error, with the
   +-1 sigma guides.

The four pieces of the model are also drawn on their own, from the fitted parameters, by
one call in the script (``plot_model_pieces``, in the same file). Those are the four
figures in section 3, and they are the reason the model is written as four functions:
each piece can be drawn, compared with the data, and corrected before the next one is
added.

Three windows: 870 A, 380 A and 250 A wide, against the 2 600 A in the figure above. On
that page a 700 km/s narrow core is two pixels; here it is a profile, and the wings of
the broad lines are visible at all.

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
   does not try to fit with the same component, and the two dashed purple curves are
   broad H-alpha's core and wind, drawn separately.

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

6. Save it, and read it back
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

7. What the numbers say
-----------------------

The fit prints its own comparison, which is the point of writing the model this way:

.. code-block:: text

     model                       free         ln Z  chi2/dof
     --------------------------------------------------------
     published                     30    130889.60     0.307
    *template                      27    130677.27     0.315
                                        Delta = -212.33

The published model is the same spectrum, the same continuum, the same broad lines and
the same priors, with a Gaussian per narrow line and one width shared between them. The
only difference is the narrow-line profile, so the 212 nats are the price of that
assumption -- not of the code: the templated model fits [O III] itself better, and a
second, independently written implementation of the same model agrees to a hundredth of
a nat. Re-running this one lands a nat or two away, inside its 0.58 error bar -- see
:doc:`priors` on why the fixed key does not buy bitwise identity -- and re-extracting the
template moves it by more than that again, so the comparison is between runs made on the
same template file.

The reason is measurable rather than mysterious. The same local treatment applied to
narrow H-alpha instead of [O III] gives a width of about 300 km/s, against an equivalent
249.6 km/s for the [O III] profile -- a ratio of 0.84. The narrow Balmer lines in this
object are broader than [O III], so a profile measured on [O III] and stamped under
H-alpha, the strongest narrow line in the spectrum, under-fits it. The residual is not
flat across that line: it runs at -0.6 sigma over the core and +2.3 sigma on the red
shoulder (6570-6580 A), which is the signature of a shape very slightly too narrow and
too peaked for this line -- and it is what the H-alpha zoom above shows. A Gaussian
model with one free common width can split the difference; a frozen profile cannot.

That ratio was measured with the portability check in
``develop/dev_template/stage1_o3_template.py``, the local script that also scanned the
component count; it is not part of this repository, so treat the number as provenance
rather than as something the page reproduces. That is a result about IRAS 07546+3928 and
a lesson about the method, and it is worth knowing before reaching for a template:
**measure the profile on the line you trust, then check that the other lines share it**.
For the forbidden lines alone, which do share their kinematics, the extraction is sound
and the model is the right one; for a Balmer-dominated narrow-line region, anchor the
template on a Balmer line instead.

The [S II] ratio that comes out of the fit is 0.0625/0.0333 = 1.88, measured rather than
tied (it is a density diagnostic), and the [N II] and [O I] ratios are at their atomic
values because nothing physical varies them.

Where to go next
----------------

* ``example/fit_iras07546_o3_template.py``: step 2 on its own -- the window fit, the
  profile, and the file the whole-spectrum fit reads.
* ``example/iras07546_model.py``: the model itself -- the seventeen components, the
  ties and the priors, shared by all three fitting scripts so that each one runs on its
  own.
* ``example/fit_iras07546_jax_tied.py``: the same fit with the Balmer kinematics
  released and centred on H-alpha, which turns "the lines share one profile" into a
  measurement (:doc:`priors` has the construction and the result).
* ``example/fit_iras07546_jax_toggle.py``: how many broad components the profile needs,
  decided by the sampler rather than assumed -- four on this object, with every sample
  agreeing (:doc:`priors` has the numbers).
* :doc:`jax_gpu`: the JAX backend, the batching, and the CPU-versus-GPU comparison.