Priors
======

A prior says two things about a parameter: where it may be, and how likely each value
in that range is. In GalSpec it is set on the parameter itself -- the attribute astropy
already reserves for the user's fitter -- so the prior sits next to the thing it
describes instead of in a dictionary passed to the sampler at call time::

   model.sigma_w0.prior = priors.Uniform(300.0, 8000.0)

This page covers the shapes that exist, what the fitter does with them, and three
worked examples taken from ``example/``: the published recipe, a prior centred on
another parameter, and a prior that can switch a component off entirely. For the fit
those priors belong to, see :doc:`fitting_walkthrough`.

.. contents:: On this page
   :local:
   :depth: 1

The four shapes
---------------

.. code-block:: python

   from galspec import priors

   model.amplitude.prior = priors.LogUniform(1e-3, 30.0)   # flat per decade
   model.sigma_c.prior   = priors.Uniform(300.0, 8000.0)   # flat in the value
   model.amp_w0.prior    = priors.Uniform(0.0, 1.0)        # a ratio: bounded
   model.amp_w1.prior    = priors.SpikeAndSlab(p_off=0.5)  # may be exactly zero

   # a prior that names another parameter: broad H-beta's wind shift is centred on
   # broad H-alpha's, with 5% of H-alpha's wind width as its sigma and five of those
   # widths as a hard limit
   model.set_prior("b_hbeta.dv_w0",
                   priors.Normal(mean="b_halpha.dv_w0", sigma=0.05,
                                 scale_from="b_halpha.sigma_w0", nsigma=5))

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - shape
     - use it when
     - in the published recipe
   * - ``Uniform(lower, upper)``
     - the range is already the physical one
     - every width, shift and ratio; the continuum power-law index
   * - ``LogUniform(lower, upper)``
     - the parameter spans decades
     - every line peak, and the iron amplitude: from invisible to three times the
       brightest pixel
   * - ``Normal(...)``
     - the value is expected near a number, or near **another parameter**
     - the soft tie (below)
   * - ``SpikeAndSlab(p_off)``
     - the parameter may be *absent*, not merely small
     - the wing gate in the component-count run (below)

A parameter whose prior is never mentioned is uniform over its default bounds, and the
examples refuse to start in that state -- a stray parameter then samples its whole box
and the run looks fine while it happens.

A centred prior, in detail
--------------------------

``priors.Normal`` returns a :class:`galspec.CenteredPrior`; ``mean`` is a number or the
name of another parameter. The width of the Gaussian can be written three ways:

* **absolute** (default): ``sigma`` is in the parameter's own units;
* ``relative=True``: ``sigma`` is a fraction of the *centre's* value, so the prior
  widens with the profile it is centred on;
* ``scale_from="name"``: ``sigma`` is a fraction of *that* parameter's value -- how a
  shift is constrained in units of a line width.

``log=True`` makes it multiplicative (a log-normal), the right shape for an amplitude;
it cannot be combined with ``relative``, because in log space ``sigma`` is already a
fraction.

``nsigma`` is a **hard** limit, not a suggestion: beyond it the prior is exactly zero.
The default is seven, where 1.3e-12 of the Gaussian's mass is left and float64 can no
longer tell a cumulative probability from 1. The prior is truncated to the parameter's
own box *intersected* with that range, so when the box is narrower, the box wins. A
posterior that piles up against a hard limit is then a bound on the parameter, not a
measurement of it -- which is why the tied example reports how much of the posterior
sits there.

What the fitter does with them
------------------------------

**A prior is a transform of the unit cube.** The sampler draws ``u`` in [0, 1] and the
prior maps it one-to-one onto the parameter's range, carrying the Jacobian inside the
transform. Because the map is a bijection and not a sampling density, the number the
run reports is the evidence of *the stated prior* -- nested sampling's prior volume
stays meaningful.

**Dependencies are resolved, not assumed.** A prior that names another parameter must be
drawn after it. :func:`galspec.resolve_prior_dependencies` resolves names (flat, then
qualified, then a unique bare match -- an ambiguous one is an error rather than a guess),
rejects cycles, and returns an order in which independent parameters keep their original
position, so a model without centred priors transforms exactly as it did before. Both
fitters walk that order, and ``tests/test_priors_tied.py`` asserts they agree to 1e-9.

**Log-scaled parameters are truncated in log space.** An amplitude's box is stored in
log10 units, so the truncation range is exponentiated before it reaches the prior; the
fitters do that, or a tie silently lands in the wrong place.

The published recipe
--------------------

``set_priors`` in ``example/fit_iras07546_jax.py``, condensed:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - what
     - range
     - why that shape
   * - line peaks
     - 1e-3 to 3x the brightest pixel, **log-uniform**
     - four decades, from invisible to implausibly strong
   * - wind / core ratio
     - 0-1, uniform
     - no wind is brighter than its own core
   * - narrow shift and width
     - +-1000 and 30-800 km/s, uniform
     - shared by every narrow line, measured from [O III]: unresolved up to the widest
       known narrow-line region
   * - [O III] wing width
     - 300-4000 km/s
     - a wing on the narrow core, not the broad-line wind
   * - broad shifts
     - core +-2 500, wind +-6 000 km/s, uniform
     - a core within a few widths of rest, a wind free to be offset by more
   * - broad core *and* wind
     - 300-8000 km/s, uniform
     - one box for both: a wing is not assumed broader than its core, and a 25 000 km/s
       ceiling only puts prior where the data are not
   * - continuum level
     - 0.2-4x the line-free windows, log-uniform
     - cannot be negative, cannot swamp the lines
   * - Fe II amplitude
     - 1e-3 to 3x the continuum level
     - the only free iron parameter

Worked example: a prior centred on another parameter
----------------------------------------------------

The three Balmer lines come from the same gas, so the published fit *ties* broad
H-beta's and H-gamma's shifts and widths to broad H-alpha's. A tie is an absolute
statement: the tied parameter cannot disagree, so the fit cannot report whether the tie
was right, and a real difference would be absorbed silently by the amplitudes.

The alternative is not to free all eight parameters -- the *difference* between two
widths is then dominated by whatever the data cannot constrain. What is wanted is "near
H-alpha's value, and not farther than this":

.. code-block:: python

   # a width: 20% of H-alpha's width, at most five widths away
   model.set_prior("b_hbeta.sigma_c",
                   priors.Normal(mean="b_halpha.sigma_c", relative=0.20, nsigma=5))
   # a shift: 5% of that component's H-alpha *width* (not of the shift, which can be
   # near zero), used as the standard deviation
   model.set_prior("b_hbeta.dv_c",
                   priors.Normal(mean="b_halpha.dv_c", sigma=0.05,
                                 scale_from="b_halpha.sigma_c", nsigma=5))

That is the whole construction, and ``example/fit_iras07546_jax_tied.py`` is the published
fit with it applied: 38 free parameters instead of 30, with ``WIDTH_FRACTION = 0.20``,
``SHIFT_FRACTION = 0.05``, ``MAX_SIGMA = 5.0`` and ``CAP_TOLERANCE`` (when a pull counts
as sitting on the limit) as named constants at the top of the file.

The number to read is the **pull** -- the difference from H-alpha's value in units of
the tie width. A pull of zero means the data agree with the tie; a pull of one means the
difference is as large as the prior allowed for; a pull of five or more means the data
asked for more than the prior permits, and the fitted value is a compromise rather than
the likelihood's own answer.

The evidence moves too, and its sign has to be read carefully. For a Gaussian of width
:math:`\sigma_{\rm tie}` against a posterior of width :math:`\sigma_{\rm post}`, each
released parameter costs

.. math::

   \Delta \ln Z \;\approx\; \ln \frac{\sigma_{\rm tie}\sqrt{2\pi}}{\sigma_{\rm post}} ,

which for the eight parameters is about 29 nats *if the tie is right* (the script
computes it from each parameter's own posterior width). A drop of that size is the
expected cost of a freedom the data did not need; a rise means the likelihood gained
more than the prior volume cost, i.e. the tie was wrong.

That arithmetic is the whole argument for measuring instead of tying, so it needs a
control: nested sampling's evidence has its own error, and two runs can differ for
reasons that have nothing to do with the model. The control is to rerun one of them
with more live points and see whether the number moves.

.. figure:: _static/example_figures/iras07546_jax_tied_ties.png
   :width: 80%

   Each released parameter against H-alpha, in tie widths. The shaded band is where the
   prior allows the difference; the red points sit against the hard limit.

On IRAS 07546+3928 the answer is split, and that is the point of doing it. The eight
released parameters (H-beta and H-gamma, core and wind, width and shift) separate into
two groups:

.. list-table::
   :header-rows: 1
   :widths: 26 14 14 14 16 16

   * - parameter
     - H-alpha
     - line
     - difference
     - fractional
     - pull
   * - ``b_hbeta.sigma_c``
     - 1 256.6
     - 1 156.7
     - -100.1
     - -8.0%
     - -0.40
   * - ``b_hbeta.dv_c``
     - 215.6
     - 56.0
     - -159.7
     - -74.0%
     - -2.54
   * - ``b_hbeta.sigma_w0``
     - 4 016.9
     - 7 991.8
     - +3 971.7
     - +98.9%
     - **+4.94**
   * - ``b_hbeta.dv_w0``
     - -397.7
     - -724.7
     - -325.6
     - -81.9%
     - -1.62
   * - ``b_hgamma.sigma_c``
     - 1 256.6
     - 1 093.7
     - -162.6
     - -12.9%
     - -0.65
   * - ``b_hgamma.dv_c``
     - 215.6
     - 204.5
     - -11.7
     - -5.4%
     - -0.19
   * - ``b_hgamma.sigma_w0``
     - 4 016.9
     - 7 966.7
     - +3 944.9
     - +98.2%
     - **+4.91**
   * - ``b_hgamma.dv_w0``
     - -397.7
     - -1 321.4
     - -923.7
     - -232.2%
     - **-4.60**

The four **core** parameters agree with H-alpha's -- pulls inside 2.5 tie widths, so
for the narrow cores the hard tie was right and could have stayed. The four **wing**
parameters do not, and none of them can be quoted as a number:

* both wing widths sit against the 8 000 km/s ceiling of the width prior -- every
  sample of ``b_hbeta.sigma_w0`` is above 7 895, 92% of ``b_hgamma.sigma_w0`` above
  7 900 -- so what those two rows say is "at least 8 000 km/s";
* H-gamma's wind shift is at -4.60 of its five-width window, with 5% of its posterior
  against the limit.

What survives is directional: the wind in H-beta and H-gamma is far from H-alpha's,
broader than the width prior allows for, and in H-gamma shifted as well.

The ceiling is not the tie, and it is worth being exact about which one binds. At
H-alpha's 4 017 km/s the five-width tolerance reaches 10 300 km/s, but the width prior
ends at 8 000, so the box stops the two wings before the tie window does: raising
``MAX_SIGMA`` alone would change nothing, and freeing them means widening the width
prior in ``set_priors`` first. Which is the honest form of the answer -- the adopted
300-8000 km/s range, chosen for the core, is what the wind runs into.

The fit itself: 38 free parameters, ``log Z = 132 153.01 +- 0.45`` in 36.6 min,
``chi2/dof = 0.261`` against the hard tie's 0.307, so nothing is worse. The evidence
is **+1263 nats** above the hard tie's 130 889.60, where the eight parameters' own
prior volume is worth only +29. The difference is the better fit and not the
arithmetic: dropping chi2/dof from 0.307 to 0.261 over 56 956 degrees of freedom is
``0.5 x 2 620 = +1 310`` nats of likelihood, and 1 310 - 29 = +1 281 against the
measured +1 263. The two agree to the accuracy of a chi2/dof printed to three digits
(0.001 of it is 57 nats), which is enough to say where the evidence went: into the
likelihood, not into prior volume.

A gain that size is worth checking against the sampler before it is quoted, and the
control is cheap: the published fit rerun at 1000 live points instead of 400 returns
130 887.64 +- 0.39 against 130 889.60 +- 0.62, a shift of 2 nats, with the same
posteriors. Repeating it at 400 points moves it by up to 5 nats (three runs gave
130 894.66, 130 893.37 and 130 889.60), so the honest error on a single evidence is
about 5 nats and not the 0.6 the run reports -- still small next to this difference.
The hard tie is rejected at 1263 +- 6 nats: a likelihood gain from a model that was
genuinely misfitting, not a difference in prior volume.

One number in that list is a warning about all of them. Releasing the two wing widths
moves H-alpha's *own* wind width from 5 143 +- 70 to 4 017 +- 14 km/s: 1 126 km/s,
sixteen times the plain fit's error bar and eighty times the tied one's. It is not the
sampler -- the plain fit's wind width is stable in ``nlive`` (5 143 +- 70 at 400
points, 5 145 +- 70 at 1000), and the tied fit's posterior is not multimodal. It is
the model. So those error bars are *conditional*: they measure the width of a narrow
ridge, not the uncertainty of a quantity. The wind's shape is set jointly with the
balance between core, continuum and wings, and changing how the wings are described
slides the ridge. That is the honest limit of every number in this section, and it is
the reason the tie is worth writing down rather than leaving implicit.

Two caveats belong with it:

* **The tie width is a physical statement, and it is the choice that decides the
  answer.** 20% of the width says the two profiles move like the same gas to within a
  fifth; a galaxy whose H-beta region really is twice as broad as its H-alpha region
  would show pulls of ~5 and would be reported as such, not smoothed away.
* **Component *i* is tied to component *i*.** With one wind per line the pairing is
  unambiguous; with more it is a choice, and the example pairs them in order.

Worked example: switching a component off
-----------------------------------------

How many broad components does a profile need? The alternative to fitting ``k`` and
comparing evidences is to let the sampler decide: give the line ``k`` components and let
every component after the first carry a *toggle* prior -- with some probability its
amplitude is **exactly zero**.

Nothing special is needed in the models: ``Line_MultiGauss(n_components=k)`` is a core
plus ``k - 1`` wings, each with its own amplitude ratio, shift and width.

.. code-block:: python

   # a spike at exactly zero (probability 0.2), else a linear slab on (0, 1]
   for name in ("b_halpha", "b_hbeta", "b_hgamma"):
       model.set_prior(f"{name}.amp_w0", priors.SpikeAndSlab(p_off=0.2, lower=0.0,
                                                            upper=1.0))

The zero is **exact**, not small. That is what makes it work: a point mass has *finite*
prior volume -- ``p_off`` of the cube -- while a density on the reals has none. A nested
sampler can find a region of finite volume; it could never find a point of measure zero.
The Bayes factor of the component follows from the prior odds it was given:

.. math::

   \mathrm{BF} = \frac{P(\text{on})/P(\text{off})}{(1 - p_{\rm off})/p_{\rm off}} .

Dividing the prior odds out matters: with a ladder of ``p_off`` the higher components are
*a priori* unlikely and would otherwise look rejected for that reason alone.

A component that is switched off is unambiguous. One that is *barely* on is not: at 1%
noise an unneeded Gaussian can sit at 1% of the core peak and buy a little chi-square
from the noise, with a Bayes factor of ~400 in favour. So the *count* is defined with a
floor -- 3% of the core peak in the example -- and the fit still puts posterior mass
between zero and the floor. The floor is part of the definition of "needed", not a hidden
prior.

The recipe, as ``example/fit_iras07546_jax_toggle.py`` applies it:

* only the three Balmer lines get four components; He I and He II keep one broad
  Gaussian, and the Fe II kernel follows broad H-alpha. 40 free parameters;
* one width prior for everything, 300-8000 km/s, the same box as the published fit --
  a wing is not required to be broader than the core;
* a ladder ``p_off = 0.2, 0.5, 0.7``. The likelihood cannot tell two wings apart
  (exchanging them leaves the spectrum bit-identical), so the prior is the only place an
  asymmetry can live;
* decide by the *mode* of the floored count, then refit at that ``k`` with plain priors
  if an evidence is wanted.

.. figure:: _static/example_figures/iras07546_jax_toggle_components.png
   :width: 85%

   The four components the run reports for broad H-alpha, sorted by width.

.. figure:: _static/example_figures/iras07546_jax_toggle_gates.png
   :width: 95%

   Where the count comes from: left, the posterior on how many components the profile
   needs (0.84 for four); right, the probability that each of the three wings is on.

The answer on IRAS 07546+3928: the run asked for at most four broad components and
returned **four** -- a core plus three wings. In 97% of the 236 466 equally weighted
samples all three wings are on, and in 84% all three stay above the 3% floor, so the
posterior on the *wing* count is ``P(3 wings) = 0.84``. Two of the three Bayes factors
are unbounded above and the third is 76, so the count is not a marginal call. Sorted by
width, broad H-alpha comes out as:

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

Two caveats come with that answer. The pedestal (wing 3) sits on the width ceiling --
its sigma reaches 7 996 km/s against the 8 000 km/s bound, so how wide it is, and
possibly how many components describe the base of the line, is limited by the adopted
range rather than by the data. And the count is floor-sensitive in the expected way: the
pedestal peaks at 4.1% of the core, just above the 3% floor, so at a 5% floor the count is
two and at 10% it is one -- not because the pedestal is noise (its Bayes factor is
unbounded) but because a coarser floor is a different definition of "a component".

Calibration on mocks with known answers (``develop/dev_jax/broad_line_components.py``,
four mocks of 1, 2, 3 and 4 components at 1% noise) says three things. The count is the
*minimum the data need*, not the number that generated them: it is exact for the 1-, 2-
and 3-component mocks, and 3 for the 4-component one, because three overlapping wings that
differ only in width are not separately required by 1%-noise data. The linear slab is the
right one -- a log slab puts a spurious wing above the floor in a third of the posterior
on the 2-component mock by concentrating density where an extra Gaussian hides in the
noise. And use 1000 live points when the count is the answer: 400 gets that same mock
wrong, reproducibly.

What it does not do:

* it does not make an extra Gaussian expensive -- the prior says a component is unlikely
  *a priori*, and with 1% noise the likelihood sometimes still prefers one. The floor is
  what turns a continuum of small amplitudes into a decision;
* it does not identify which component is which, or order them;
* its evidence is not an evidence for a ``k``-component model: a toggled model is a
  mixture over subsets, so quote the count from the gated run and refit at that count for
  an evidence ratio.

The statistics is standard -- a spike-and-slab prior for variable selection goes back to
George & McCulloch (1993) and Kuo & Mallick (1998). Trans-dimensional sampling (Green
1995; diffusive nested sampling, Brewer 2014) is the more general route and needs no
floor, but it changes the parameter count from sample to sample, which defeats the
fixed-size vectors that ``jit``/``vmap`` batching needs. The variant here needs no change
to the sampler at all.

Saving priors with a fit
------------------------

A saved fit carries its priors: each is stored as its type plus its constructor
arguments, together with the name of any parameter it is centred on, so the file is
enough to rebuild the prior transform and resume the sampler. The one exception is a
prior supplied as a bare Python function -- that has no data form, so the loader warns
and keeps only the resolved bounds.