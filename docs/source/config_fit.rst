.. _config_fit:

Fitting From a Configuration File
=================================

``galspec.pipeline`` fits one spectrum, or a whole sample with one model, from a
TOML file instead of a Python script::

    ./example/fit_iras07546_config.sh            # the published IRAS 07546 fit
    ./example/fit_iras07546_config.sh --dry-run  # what would be fitted, no sampling

The pipeline is a *consumer* of the models: it builds the same
:class:`~galspec.jax_models.JaxCompoundModel` a script would assemble, and runs
the same :class:`~galspec.jnesty_fit.JNesty_Fit`. It adds no model code and
changes none, so a fit produced from a file and a fit produced by hand of the
same model and data are the same fit. ``galspec/__init__.py`` does not import
it; nothing changes for code that does not use it.

The running example in this chapter is
``example/fit_iras07546_config.toml``: the seventeen-component, 27-parameter
model of ``example/fit_iras07546_jax.py``, written as a file and checked against
the script parameter by script parameter
(``develop/dev_config/test_iras07546_config.py``). Read it next to this page.

A file can be much shorter than that one. Nested sampling draws from the prior, so
neither the starting values nor the priors have to be written down -- the IRAS file
below writes no starting value at all, only what the model *is* (line centres, a fixed
redshift, the ties) -- and the narrow-line profile can travel inside the spectrum's own
FITS file. A complete configuration for one emission line is::

    [data]
    file = "spectrum.fits"
    narrow_template_key = "NARROW"      # optional; a Gaussian if omitted

    [model]
    [[model.component]]
    type = "Line_template"
    wavec = "@Halpha"

That file has two free parameters, the profile's amplitude and its velocity, and
a continuum component would be the natural next line. ``--dry-run`` prints the
prior every omitted parameter ended up with, and the same list goes into
``run_report.json``.

Starting from a template
------------------------

The file format has a lot of optional keys, so the command will write you a
commented starter::

    $ python -m galspec.pipeline --template config.toml
    wrote config.toml
      edit [data] file, then
        python -m galspec.pipeline config.toml --dry-run    # what would be fitted
        python -m galspec.pipeline config.toml              # the fit

``--template`` alone writes ``config.toml`` in the current directory; give it a
path to write elsewhere, ``-`` to print to standard output, and ``--force`` to
overwrite a file that is already there (without it an existing file is an error,
so the command cannot eat a configuration you have written). It exits without
touching any data.

The starter is a *working* configuration rather than a sheet of placeholders: in a
checkout it points at the two-line mock spectrum in ``example/data``, so the first
``--dry-run`` reports one source and eight free parameters before you edit
anything, and the fit it describes takes about forty seconds on a CPU. From there
the edits are: ``[data] file`` (or ``glob``/``catalog`` for a sample), the
components in ``[model]``, and the priors. Every section it writes is explained
below.

Why a file
----------

A sample is fitted with *the same* model, the same priors and the same sampler
settings for every source, and only the data and a few flux scales differ.
Writing that down once -- instead of copy-pasting a script per object -- is what
makes a sample fit reproducible: the file is hashed into the output, and the
:ref:`run report <pipeline-output>` records every setting that produced each
result.

Requirements on the data
------------------------

#. **One FITS file per source.** The pipeline reads one file at a time; a
   "sample" is a list of files, not a list of rows inside one file.

#. **Three arrays: wavelength, flux and error.** All three are mandatory. A
   missing error array is an error, not a guess: an uncertainty invented from
   ``std(flux)`` is not a measurement, so the pipeline refuses to make one.

   The arrays are found either as columns of a binary table or as image HDUs
   whose ``EXTNAME`` matches the key::

       [data]
       wave_key = "WAVELENGTH"     # default; matched case-insensitively
       flux_key = "FLUX"
       error_key = "ERROR"

   A column name is matched first; an image HDU is looked up by extension name.
   Use ``wave_hdu``/``flux_hdu``/``error_hdu`` when a file has more than one
   candidate (they take an HDU index or an extension name).

#. **Wavelength is converted to Angstrom, flux is not converted.** The unit is
   taken from ``TUNITn`` (a table column) or ``BUNIT``/``CUNIT1`` (an image), or
   from ``[data] wave_unit``/``flux_unit``. A wavelength in ``nm`` or ``um`` is
   converted; an axis that is not a length (velocity, frequency, pixels) is
   rejected, because the models are evaluated on a wavelength grid. The flux
   unit is *recorded* in the output header and the report and otherwise left
   alone: no ``f_lambda``/``f_nu`` conversion is attempted.

#. **The header supplies the two corrections.** A spectrum is fitted in the rest
   frame, with the Galaxy's reddening removed, and both numbers can be read from
   the file rather than written into the configuration. One keyword each:

   .. list-table::
      :header-rows: 1
      :widths: 18 26 56

      * - quantity
        - header keyword (``[data]``)
        - what the pipeline does
      * - redshift
        - ``z_key``, default ``"ZRED"``
        - divides the wavelength by ``1 + z``; the model's line centres are rest
          wavelengths (``"@Halpha"`` or ``6562.819``)
      * - Galactic reddening
        - ``e_bv_key``, default ``"EBV"``
        - de-reddens the flux *and* the error with ``extinction.ccm89`` at
          ``extinction_rv``, on the observed grid before the wavelength is
          shifted. The keyword's name says how to read it: a colour excess is
          taken as is, an ``A_V`` is divided by ``R_V`` first, so
          ``e_bv_key = "A_V"`` is a complete statement.

   A single keyword, not a list of spellings, because a file either has the
   keyword or it does not: a list turns "this file has no redshift in it" into a
   quiet fallback onto whichever of five keys happened to be there for an unrelated
   reason. If the file stores it under another name, name it once
   (``z_key = "MYZ"``). ``extinction_rv`` is a configuration value only -- no header
   is trusted for it, since a survey that stores ``A_V`` leans on ``R_V`` and a wrong
   ``R_V`` would propagate into every flux without leaving a trace. It defaults to
   ``3.1``, the Milky Way value.

   The precedence is: the catalog's ``z`` column, the header, ``z_default``.
   A value that is *not* in the file is assumed to be zero **and the run warns**:
   the report records ``z_source`` and ``e_bv_source`` per source, so a
   correction that was assumed is visible next to one that was applied. Writing
   the key in the configuration (``z_default = 0.0``, or ``extinction = false``
   for a synthetic spectrum) counts as stating it, and the run is then silent.

   .. note::
      The pipeline does not mask sky lines and does not synthesize errors: if
      your spectra need that, prepare a FITS file with the three arrays -- that
      file *is* the input, and it keeps the processing visible and repeatable.
      See `Preparing a file`_ for the IRAS spectrum, which needs both.

Preparing a file
----------------

Two files in ``example/data/`` show why the three arrays are a requirement rather
than a convenience, and what "prepare it" means in practice.  A template is the
one extra thing a file may carry, and it is deliberately optional: a spectrum
without one is fitted with Gaussian lines.

``spec-0651-52141-0434.fits`` is a BOSS coadd: it has ``flux``, ``loglam`` and
``ivar``, so there is a wavelength grid and a noise estimate, but no error array
and no length unit. One short script turns it into pipeline input -- a table with
``WAVELENGTH = 10**loglam``, ``FLUX = flux`` and ``ERROR = 1/sqrt(ivar)``, with
the masked pixels dropped -- and the choices it makes (which pixels to keep, what
to do with zero ``ivar``) stay in the file instead of being buried in a fitting
script.

``ir07546sed.fits`` is a single image with no error and an observed-frame
logarithmic axis; the published fit (:mod:`example.iras07546_model`) dereddens it
with ``extinction.ccm89``, divides by ``1 + z`` and derives the error from a
running median. Only the *corrections* can move into the pipeline, because the
header can carry them; the error cannot, because it is not a number in the file.
``develop/dev_config/convert_iras07546.py`` writes the input therefore::

    python develop/dev_config/convert_iras07546.py     # -> example/data/iras07546_pipeline.fits

It stores the observed-frame ``WAVELENGTH``, ``FLUX`` and ``ERROR`` columns, puts
``REDSHIFT`` and ``EBV`` (and ``EBVRV``) in the header, adds the measured
narrow-line profile as a fourth extension (``NARROW``: velocity in km/s,
normalised flux -- what ``[data] narrow_template_key`` reads, so the profile
travels with the spectrum instead of beside it), and then *checks itself*:
it applies the pipeline's own de-reddening and de-shifting to what it wrote and
asserts that the result is ``iras07546_model.load_spectrum()`` to ``1e-12``
relative. Wavelengths round trip exactly (0.0), flux and error at ``1e-16``. That
check is what makes the configuration a reproduction of the script rather than an
approximation of it.  ``python develop/dev_config/convert_iras07546.py
--check-only`` runs the check without writing.

A configuration is then a file of declarations, and the section-by-section
walkthrough below reads ``example/fit_iras07546_config.toml`` in order.

The run section
---------------

.. code-block:: toml

    [run]
    output_dir = "out/iras07546"
    workers = 1

``[run]`` says where the outputs go and how many processes do the work:

.. list-table::
   :header-rows: 1
   :widths: 20 14 66

   * - key
     - default
     - meaning
   * - ``output_dir``
     - ``"out"``
     - resolved relative to the configuration file, not the working directory.
   * - ``workers``
     - ``1``
     - ``1`` fits in this process; ``>1`` spawns that many processes, each
       fitting its own sources.
   * - ``force``
     - ``false``
     - refit even when a source's outputs exist.
   * - ``progress``
     - unset
     - the sampler's progress bar: unset means "on a terminal, with
       ``workers = 1``"; ``true`` forces it, ``false`` never draws it.

The data section
----------------

.. code-block:: toml

    [data]
    file = "data/iras07546_pipeline.fits"
    narrow_template_key = "NARROW"     # the measured line profile, inside the file

    z_key = "REDSHIFT"                 # one keyword; the default is "ZRED"
    e_bv_key = "EBV"                   # ... and "EBV"
    extinction = true
    extinction_rv = 3.1                # R_V of the law; the default, config-only

    wave_min = 4200.0
    wave_max = 6800.0
    continuum_percentile = 15.0
    min_pixels = 50000

.. _pipeline-data:

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - key
     - default
     - meaning
   * - ``file``, ``files``, ``glob``, ``catalog``
     - --
     - exactly one of them: one spectrum, a list of spectra, a shell pattern
       (relative to the config file, or ``**`` for recursion), or a CSV/TSV with
       one row per source.
   * - ``catalog_id_column``
     - ``id``
     - catalog column with the source identifier (falls back to the file stem).
   * - ``catalog_file_column``
     - ``file``
     - catalog column with the spectrum path, relative to the catalog.
   * - ``catalog_z_column``
     - --
     - catalog column with the redshift, if it has one.
   * - ``wave_key``, ``flux_key``, ``error_key``
     - ``WAVELENGTH``, ``FLUX``, ``ERROR``
     - array names, matched case-insensitively.
   * - ``wave_hdu``, ``flux_hdu``, ``error_hdu``
     - --
     - restrict a lookup to one HDU (index or extension name).
   * - ``wave_unit``, ``flux_unit``
     - from the header
     - override the header's unit.
   * - ``narrow_template_key``
     - --
     - name (or index) of an HDU holding the narrow-line profile as a two-column
       table: velocity in km/s, then normalised flux.  Each spectrum then carries
       its own profile, which is what a sample usually needs -- the lines of
       different galaxies are not the same shape.  A component's
       ``template_file`` overrides it; components whose model takes no profile
       ignore it, and every other array lookup is unaffected.  The columns are
       read by position, so their names do not matter; a velocity column with a
       unit is converted to km/s, and one without is taken to be km/s already.
   * - ``z_key``
     - ``"ZRED"``
     - the header keyword holding the object's redshift.  One keyword, not a list:
       name the one your files use (``"MYZ"``); a file without it falls back to
       ``z_default`` and says so.
   * - ``z_default``
     - ``0.0``
     - redshift when neither the catalog nor the header says.  Declaring it
       explicitly is what silences the "assumed" warning.
   * - ``e_bv_key``
     - ``"EBV"``
     - the header keyword holding the Galactic reddening.  A colour-excess name
       (``EBV``, ``E(B-V)``) is read as is; an extinction name (``A_V``) is divided
       by ``extinction_rv``.
   * - ``extinction``
     - ``true``
     - de-redden with that keyword; ``false`` declares that this spectrum needs no
       correction (and suppresses the warning).
   * - ``extinction_rv``
     - ``3.1``
     - ``R_V`` of the reddening law.  Configuration only -- no header is consulted
       for it, because a wrong ``R_V`` would scale every flux silently.
   * - ``wave_min``, ``wave_max``
     - --
     - rest-frame range to fit.
   * - ``mask``
     - --
     - list of ``[low, high]`` rest-frame windows to drop, e.g. residuals of a
       telluric band.
   * - ``min_pixels``
     - ``50``
     - refuse a source with fewer pixels than this after the cuts.  The IRAS
       window keeps 56956; the value in the file fails loudly if a wrong window
       is used by accident.
   * - ``continuum_percentile``
     - ``50``
     - percentile behind the ``continuum_level`` expression token.  The example
       script uses ``np.percentile(flux, 15.0)``, so the file says ``15``.

A catalog looks like::

    id,file,z
    J0925,spectra/J0925.fits,0.0312
    J0931,spectra/J0931.fits,0.0455

The fit section
---------------

.. code-block:: toml

    [fit]
    nlive = 400
    dlogz = 0.01
    max_iterations = 200000
    share_compilation = true
    device = "auto"

``[fit]`` passes sampler settings to :class:`~galspec.jnesty_fit.JNesty_Fit`; the
values above are the ones ``example/fit_iras07546_jax.py`` uses
(``NLIVE = 400``, ``DLOGZ = 0.01``).

.. list-table::
   :header-rows: 1
   :widths: 20 14 66

   * - key
     - default
     - meaning
   * - ``nlive``
     - ``500``
     - live points. The main cost knob; 500 to 1000 is usual for a dozen
       parameters, 400 is what the IRAS model was published with.
   * - ``dlogz``
     - ``0.01``
     - stop when the remaining evidence is below this (nats).
   * - ``max_iterations``
     - ``200000``
     - safety cap.
   * - ``bound``
     - ``"multi"``
     - ``none``/``single``/``multi`` bounding.
   * - ``device``
     - ``"auto"``
     - ``auto``/``cpu``/``gpu``.
   * - ``queue_size``, ``batch_size``, ``rwalk_K``, ``rwalk_step_scale``
     - automatic
     - forwarded to JNesty; leave unset unless you are tuning.
   * - ``max_ellipsoids``, ``memory_frac``, ``min_eff``, ``min_ncall``
     - ``20``, ``0.9``, ``10``, --
     - forwarded to JNesty.
   * - ``share_compilation``
     - ``true``
     - compile the likelihood once for the whole sample (see
       `Compiling once for a sample`_).
   * - ``compilation_cache_dir``
     - --
     - a directory for JAX's on-disk compilation cache.

The output section
------------------

.. code-block:: toml

    [output]
    fit_plot = true
    corner = true
    corner_params = [
        "Continuum.amplitude", "Continuum.alpha", "FeII.amplitude",
        "b_halpha.amp_c", "b_halpha.amp_w0", "b_halpha.dv_c", "b_halpha.sigma_c",
        # ... five more
    ]

    histogram = true
    hist_params = ["Continuum.alpha", "FeII.amplitude", "b_halpha.amp_c", # ...]

``[output]`` controls the files: ``thin`` (default ``1``) saves every n-th
posterior sample; ``fit_plot`` (default ``true``); ``corner`` and ``histogram``
(default ``false``).

``corner_params`` and ``hist_params`` select the parameters to draw, and both
spellings are accepted: the qualified name (``b_halpha.amp_c``), which does not
depend on the summation order, or the flat one (``amp_c_3``), which is what the
posterior file stores. A name that is not a free parameter of the model is an
error listing the free ones. A 27-parameter corner plot is a 27x27 square nobody
reads, which is why the IRAS file names thirteen.

The figures are the library's, not the pipeline's own drawings: the fit plot is
:func:`galspec.plot.plot_fit_jax` -- data, total model, every component and the
residuals in units of the error, the same figure ``example/fit_iras07546_jax.py``
writes -- the histograms are :func:`galspec.plot.plot_posterior_hist`, and the
corner is :func:`galspec.plot.plot_corner`. So a pipeline figure and the figure a
hand-written script makes of the same fit agree, and the qualified LaTeX labels and
prior annotations are the ones the rest of GalSpec uses. One consequence worth
knowing: a corner plot needs a posterior that moved, so a run stopped before the
sampler explored skips that one figure with a warning instead of failing the fit.

The model section
-----------------

``[model]`` holds ``require_priors`` (default ``false``) and one
``[[model.component]]`` per additive term, **in the order they are summed**:

.. code-block:: toml

    [model]
    # require_priors = true        # every prior must be written out to fit

    [[model.component]]
    type = "PowerLaw1D"            # any kind galspec.jax_models knows
    name = "Continuum"             # used by ties, priors and plots
    amplitude = 0.55               # optional: the model has a default too
    x_0 = 5500.0                   # not optional: it is where the power law sits
    alpha = 1.0
    fixed = { x_0 = true }

    [model.component.priors.amplitude]
    kind = "loguniform"            # optional: the default is loguniform over
    lower = "0.2*continuum_level"  # 1e-3 .. 3*max_flux, which is wider
    upper = "4*continuum_level"

The order matters twice: it is the order of the sum, and the flat parameter names
carry the index (``amp_c_3`` is the fourth component's ``amp_c``). Reordering the
file therefore renames parameters; the qualified spelling is the stable one, which
is why ties, priors and plots use it.

Keys of a component are one of:

* ``type`` and ``name``;
* a **parameter value** -- anything the model names as a parameter
  (``amplitude``, ``dv``, ``sigma``, ``amp_c0``, ...), which may be a number or
  an expression.  Optional: an omitted value keeps the model class's default,
  which nested sampling ignores anyway.  Three values are *not* optional in
  practice -- ``wavec``/``wavec0``/``wavec1`` and ``x_0`` -- because their model
  default (5000 A, 1 A) does not describe the line or the pivot you mean: a line
  component has to name its centre, with a number or with ``"@Halpha"``, or pin it
  with ``[model.component.bounds.wavec]``;
* a **constructor argument** (``n_components``, ``par_w``, ``degree``,
  ``template_name``, ``Star_types``, ``R``, ``template_file``) -- these are
  passed to the model's constructor and must not collide with a parameter name.
  A key ending in ``_file`` is a path: it is resolved **relative to the
  configuration file** (an absolute path is used as written) and never parsed as
  an expression;
* the tables ``fixed``, ``bounds``, ``params``, ``ties`` and ``priors``.

Anything else is rejected with its TOML path, so a typo cannot become an unused
keyword argument that silently does nothing.

Expressions
~~~~~~~~~~~

Any *number* in ``[model]`` and ``[data]`` may be an expression over the
statistics of the source being fitted, which is what lets one file serve a sample
whose flux scale differs from spectrum to spectrum:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - token
     - value for the current source
   * - ``peak_flux``
     - ``max(|flux|)``, after despiking (see below)
   * - ``max_flux``, ``min_flux``
     - ``max(flux)``, ``min(flux)``, after despiking
   * - ``median_flux``, ``continuum_level``
     - the median, and the ``continuum_percentile`` percentile, of the raw pixels
   * - ``rms``
     - standard deviation of the flux
   * - ``n_pixels``
     - pixels being fitted
   * - ``z``
     - the redshift used

The language is numbers, those names, ``+ - * / **`` and parentheses.
Everything else -- a call, an attribute, a subscript -- is rejected when the
file is read, before any spectrum is touched. ``"3*max_flux"`` sets a starting
value; ``[model.component.bounds] amplitude = ["0.01*peak_flux",
"10*peak_flux"]`` sets the box in the same units, which is what makes a
magnitude-limited sample fit with one file.

How the statistics are measured
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tokens are computed once per source, from the pixels that are actually
fitted -- after the rest-frame window, the masks and the de-reddening, but in the
flux unit the file carries, so a token inherits that unit and no conversion is
attempted.

``continuum_level`` and ``median_flux`` come straight from the flux array:
``np.percentile(flux, continuum_percentile)`` and ``np.median(flux)``.  A
percentile is what a continuum level should be -- it is a statement about *where
the spectrum sits*, not about its brightest feature -- and it is already immune to
a few bad pixels, so nothing is smoothed.  ``continuum_percentile`` therefore has
to be chosen for the spectrum: a spectrum with 60% of its window filled by
emission lines wants something like 15 or 20, one that is mostly continuum wants
the default 50.

``max_flux``, ``min_flux`` and ``peak_flux`` set the *scale* of every amplitude
default and are measured on a despiked copy of the flux, because a single hot
pixel would otherwise decide them:

#. a running **median** of five pixels, which outvotes any outlier up to two
   pixels wide -- a cosmic ray, a bad column, a sky residual left in one channel;
#. a **sigma clip** of the residual against that running median: a pixel that
   stands more than three residual spreads away *and* agrees with neither
   neighbour is replaced by the median.  The neighbour test is what keeps a real
   line, whose flanks do not look like that;
#. the extremes of the result.

A real line is wide compared with five pixels at any sensible dispersion, so its
peak survives: the despiked maximum of the clean IRAS spectrum differs from
``max(flux)`` by 4e-4 in relative terms -- while a one-pixel spike of any height does
not.  A *narrow* line loses the tip of its profile -- a one-pixel line keeps 80%
of its peak, a two-pixel line 93%, five pixels or more 98% -- which is the right
trade for a number whose job is to set a prior scale.  ``rms`` keeps the raw array, since it
is a noise estimate and despiking it would remove exactly what it measures.

The IRAS file uses ``max_flux`` rather than ``peak_flux`` on purpose: the script's
``set_priors()`` uses ``np.max(flux)``, which is not ``max(|flux|)`` when the
noise dips below zero, and a reproduction should not depend on that difference.

Arithmetic between constants is allowed and is used for the atomic ratios, so a
starting value is the ratio itself rather than a rounded copy::

    amp_c0 = 1.8
    amp_c1 = "1.8/2.98"            # the [O III] 5007/4959 ratio

Line names such as ``"@Halpha"`` resolve to the rest wavelength from
:data:`galspec.utils.line_wave_dict` (``Halpha``, ``Hbeta``, ``OIII_5007``,
``NII_6583``, ``SII_6716``, ``HeI_5876``, ...); other values are written in
Angstrom.

Fixed, bounds, params
~~~~~~~~~~~~~~~~~~~~~

To hold a parameter at its value::

    [model.component]
    type = "Line_Gaussian"
    name = "Halpha"
    fixed = { wavec = true }       # Line_Gaussian already fixes it by default

To change a box::

    [model.component.bounds]
    amplitude = ["0.01*peak_flux", "5*peak_flux"]
    sigma = [50.0, 1200.0]

Each parameter can also be written as its own table, which is convenient when
it needs several settings::

    [model.component.params.sigma]
    value = 300.0
    bounds = [50.0, 1200.0]
    prior = { kind = "uniform", lower = 50.0, upper = 1200.0 }

Ties
~~~~

A tie makes one parameter follow another. The plain form is a reference::

    [model.component.ties]
    dv = "Halpha.dv"                 # this line moves with Halpha
    sigma = { ref = "Halpha.sigma", scale = 1.5 }   # 1.5 * Halpha.sigma

The named kinds are the GalSpec conventions, written as
``reference * scale + offset`` so that they survive serialisation as pure data:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - ``kind``
     - reference parameter
     - value
   * - ``ref``, ``follow``, ``copy``
     - ``param`` (required)
     - ``reference``
   * - ``doublet_ratio``
     - ``amp_c0``
     - ``amp_c0 / ratio`` (e.g. 2.98 for [O III])
   * - ``multigauss_amp_c``
     - ``amp_c``
     - ``amp_c / ratio``
   * - ``template_amplitude``
     - ``amplitude``
     - ``amplitude / ratio``
   * - ``logtau0_ratio``
     - ``logtau0``
     - ``logtau0 - log10(ratio)``
   * - ``template_dv``, ``absorption_dv``, ``absorption_sigma``,
       ``absorption_cf``, ``starspectrum_sigma``, ``starspectrum_deltaz``
     - that parameter
     - ``reference``

The IRAS model is held together by 29 such ties, and they are the whole physical
content of the model beyond the line centres and the fixed values: one narrow-line
velocity, one broad-line kinematics, and three atomic ratios. Written as they appear in the
file, for the four kinds of reader -- the profile comes from the data file
(``[data] narrow_template_key``), so no component repeats a path:

.. code-block:: toml

    # long-wavelength narrow lines share the [O III] velocity
    [[model.component]]
    type = "Line_template_doublet"
    name = "NII"
    amp_c0 = 1.0
    amp_c1 = "1.0/2.96"
    dv_c = 0.0
    wavec0 = "@NII_6583"
    wavec1 = "@NII_6548"

    [model.component.ties]
    amp_c1 = { kind = "doublet_ratio", ref = "NII", ratio = 2.96 }
    dv_c = "OIII.dv_c"                       # ... and [O III] owns the velocity

A component that needs a *different* profile from the rest of the spectrum writes
``template_file = "profiles/nii.txt"`` (two columns, velocity then flux, resolved
relative to the configuration) and overrides ``narrow_template_key`` for itself.

    # broad H-beta keeps its own peak and its own wind amplitude, and copies the
    # shape of broad H-alpha, which is the only line whose kinematics are free
    [[model.component]]
    type = "Line_MultiGauss"
    name = "b_hbeta"
    n_components = 2
    amp_c = 0.22
    dv_c = 0.0
    sigma_c = 1500.0
    wavec = "@Hbeta"
    amp_w0 = 0.20
    dv_w0 = 0.0
    sigma_w0 = 6000.0

    [model.component.ties]
    dv_c = "b_halpha.dv_c"
    sigma_c = "b_halpha.sigma_c"
    dv_w0 = "b_halpha.dv_w0"
    sigma_w0 = "b_halpha.sigma_w0"

The iron pseudo-continuum is tied the same way (five kernel parameters follow
broad H-alpha), so the iron costs the sampler exactly one number, its amplitude.

Ties are validated when the model is built: a reference to a component or
parameter that does not exist is an error naming the alternatives. Tying a
parameter that is fixed is allowed (the tie then simply never matters); putting a
prior on a tied parameter is rejected, because the prior would never be used.

Priors
~~~~~~

A parameter with no prior is uniform over its bounds. Every prior takes the
arguments of its class in :mod:`galspec.priors`, plus expressions where a number
is expected::

    [model.component.priors.amp_c0]
    kind = "loguniform"            # uniform, tophat, loguniform, spikeandslab,
    lower = "0.01*peak_flux"       # normal, clippednormal
    upper = "5*peak_flux"

    [model.component.priors.dv]
    kind = "normal"
    mean = 0.0
    sigma = 150.0

    [model.component.priors.sigma_w0]
    kind = "normal"
    mean = "Halpha.sigma_c"        # centred on another parameter
    sigma = 0.4
    relative = true
    nsigma = 3

Default priors
~~~~~~~~~~~~~~

Nested sampling draws from the prior, so a configuration does not have to say
where to start: an omitted parameter value falls back to the model class's own
default (``amplitude = 1``, ``sigma = 200`` km/s, ...), which never biases the
answer -- it only changes the point the sampler prints while it warms up.

A parameter the file says nothing about also gets a **default prior**, chosen
from its *name* and resolved against the source's own statistics:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - parameter
     - default prior
     - why
   * - ``amplitude``, ``amp_c``, ``amp_c0``, ``amp_c1``, ...
     - ``loguniform(1e-3*continuum_level, 3*max_flux)``
     - a line is between a thousandth of the continuum and three times the
       brightest pixel, and "twice as faint" and "twice as bright" are equally
       probable.  Both ends are *scaled* by the spectrum, so one file serves a
       sample whose flux spans decades; an absolute ``1e-3`` would be a different
       physical limit for every source.  If the resolved interval is empty, or its
       lower end is not positive (a continuum level of zero or below), it falls
       back to ``1e-3*max_flux`` and ``--dry-run`` says so.
   * - ``amp_w0``, ``amp_w1``, ...
     - ``uniform(0, 1)``
     - a wind amplitude is a fraction of the line.
   * - ``dv``, ``dv_c``
     - ``uniform(-3000, 3000)`` km/s
     - systematics and a few hundred km/s of gravitational slop.
   * - ``dv_w0``, ``dv_w1``, ...
     - ``uniform(-6000, 6000)`` km/s
     - a wind is wider and can be more offset.
   * - ``sigma``, ``sigma_c``, ``stddev``
     - ``uniform(10, 8000)`` km/s
     - any width a spectrograph can resolve, from a stellar line in a
       high-resolution spectrum to a broad-line-region profile.  A *broad* line
       wants a floor near 300 km/s, which is a statement about the object rather
       than about units, so write it out -- the IRAS file does.
   * - ``sigma_w0``, ``sigma_w1``, ...
     - ``uniform(10, 8000)`` km/s
     - the wing, i.e. an outflow.
   * - ``alpha``
     - ``uniform(-5, 5)``
     - a power-law index, both signs.
   * - ``h3``, ``h4``
     - ``uniform(-0.5, 0.5)``
     - Hermite moments.
   * - ``logtau``
     - ``uniform(-3, 3)``
     - optical depth of an absorption model.
   * - ``cf``
     - ``uniform(0, 1)``
     - covering fraction.
   * - ``z``, ``redshift``
     - ``uniform(-0.1, 0.1)``
     - a redshift correction.
   * - ``velscale``
     - ``uniform(10, 1000)`` km/s
     - a broadening kernel.

A prior or a bound written in the file always wins, so the defaults are what is in
force *only* where the file is silent. ``--dry-run`` lists every parameter that
took one, with the numbers it resolved to, and the same list is in
``run_report.json`` under ``model.default_priors``: nothing about the prior is
invisible.

Two kinds of parameter are deliberately *not* defaulted, because the right range
depends on the data in a way a name cannot express -- a linear continuum's
``slope`` and ``intercept``, and a *free* ``wavec``: a wrong box there changes the
evidence silently instead of failing. Those still need

.. code-block:: toml

    [model.component.bounds.slope]
    ...  # = [low, high]

``require_priors = true`` demands a prior *written in the file* on every free
parameter -- defaults do not count -- so it is the way to make a configuration
state its priors even though it no longer has to. Two checks run before the
sampler starts, so a mistake costs a second instead of a night:

* every free parameter needs a finite sampling range -- both bounds, or a prior
  with finite limits (a centred prior such as ``normal`` additionally needs the
  bounds, because they truncate it);
* no free parameter means nothing to fit, which is also an error.

The IRAS file leaves **twenty-one of its twenty-seven** priors to the defaults:
every line amplitude (``loguniform(1e-3*continuum_level, 3*max_flux)``, the
reference script's ``loguniform(1e-3, 3*max_flux)`` with a floor that scales
instead of one absolute number), ``alpha``, the three wind amplitudes and one wind
velocity. Six are written out, and each is worth knowing about when writing a new
file:

* the **broad** lines' widths: 300-8000 km/s, while the default floor is 10 km/s,
  which a *narrow* line needs. A floor of 300 is a statement about the object, not
  about units, so it cannot be a default;
* the continuum's ``0.2-4*continuum_level`` and the iron template's
  ``3*continuum_level`` ceiling -- a pseudo-continuum is scaled by the continuum,
  not by the brightest emission line;
* the two narrow-line velocity windows, broad H-alpha at 2500 km/s and [O III] at
  1000 km/s, which the systemic reference needs.

The scaled amplitude floor is not cosmetic for this object: 1e-3 of a continuum
level of 0.4991 is 5e-4, half the script's absolute floor, so a *log-uniform* prior
with the same ceiling is wider by ``ln(12.99/5e-4) / ln(12.99/1e-3) = 1.074`` in its
log-range, and a log-uniform density scales as ``1/ln(b/a)``. Each of the sixteen
amplitudes therefore lowers the evidence by ``ln(1.074) = 0.071``, and the total is
about ``-1.1``: the published ``130677.27`` belongs to the script's prior, and a file
that uses the scaled floor would report about ``130676.1``.

Only the *number* moves. The two priors differ by a constant factor wherever both are
non-zero, so the posterior is the same up to the renormalisation the added faint tail
implies -- and that tail (amplitudes between 5e-4 and 1e-3, where no line of this
spectrum lives) can only bring the shift back toward zero. The offset is also worth
putting next to the ±0.58 sampling error: it is a deliberate difference in the
*family of models*, not a defect in the fit. A short run cannot measure it either --
a nested-sampling run truncated at a few hundred iterations is dominated by its
un-converged initial state, and the numbers it reports differ by far more than one
nat -- which is why the value above is the exact prior-volume ratio rather than a
measured difference. The mapping is one line of the script per table of the file, which is the
property to aim for when writing a new configuration:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - in ``fit_iras07546_jax.py``
     - in ``fit_iras07546_config.toml``
   * - ``build_continuum()``
     - the ``Continuum`` component
   * - ``build_iron_template()`` + ``tie_iron_template()``
     - the ``FeII`` component and its five ties
   * - ``build_narrow_lines()`` + ``tie_narrow_lines()``
     - eight ``Line_template``/``Line_template_doublet`` components, their
       ``dv`` ties and three ratio ties
   * - ``build_broad_lines()`` + ``tie_broad_lines()``
     - five ``Line_MultiGauss`` components and their ties
   * - ``set_priors(model, flux)``
     - the six ``[model.component.priors.*]`` tables, plus the pipeline's default
       prior for the other twenty-one parameters
   * - ``JNesty_Fit(..., nlive=NLIVE, dlogz=DLOGZ, ...)``
     - ``[fit]``

Output
------

.. _pipeline-output:

For a source ``<id>``::

    <output_dir>/<id>/<id>_fit.fits          # JNesty_Fit, with provenance keywords
    <output_dir>/<id>/<id>_posterior.npz     # samples, log-likelihood, data
    <output_dir>/<id>/<id>_fit.png           # data, best model, residuals (optional)
    <output_dir>/<id>/<id>_corner.png        # optional
    <output_dir>/<id>/<id>_histogram.png     # optional
    <output_dir>/run_report.json             # what was fitted, with what, how

``<id>_fit.fits`` is the file
:func:`galspec.jnesty_fit_io.save_jnesty_fit_to_fits` writes, plus keywords for
the source identifier, the input file, the redshift and where it came from, the
number of pixels, the config file name and the first 16 characters of its SHA-256
(the same keys are in the report), and the extinction that was removed
(``EBV``, ``EBVRV``) and where it came from (``EXTINCT``).

``<id>_posterior.npz`` is readable with ``numpy.load(..., allow_pickle=False)``:
``samples_equal_weight``, ``samples``, ``logl``, ``logwt``, ``param_names``
(flat names, as the model spells them), ``log_evidence``,
``log_evidence_error``, ``n_iterations``, ``redshift`` and the ``wave``/``flux``/
``error`` arrays that were fitted.

``run_report.json`` records the settings, library versions, and per source: the
status, evidence, iterations, timing (compile and sampling separately), a
``parameters`` table with the median, the 16th and 84th percentile and the
maximum-likelihood value of every free parameter, which parameters are pressed
against a prior limit (``prior_dominated``), which parameters took a default prior
(``model.default_priors``, with the numbers it resolved to for this source), the
flux tokens that were used, the redshift and extinction provenance, any warning
raised while reading, and the outputs. A source that fails is recorded with its error and the run continues; the
report is written even when every source fails.

Resuming and parallelism
~~~~~~~~~~~~~~~~~~~~~~~~

A source whose output files all exist is skipped unless ``force = true`` (or
``--force``), so an interrupted run is continued by running the same command
again. ``workers > 1`` fits sources in spawned processes; each process writes its
own sources, and the report is assembled from their results. Failures are
isolated per source in both modes. Set ``GALSPEC_PIPELINE_TRACE=1`` to keep the
traceback in the report.

Compiling once for a sample
---------------------------

Every JAX program has to be compiled before it runs, and compiling the
likelihood and the prior of a full model takes a fraction of a second to a second
(0.6 s for the 39-parameter example in ``develop/dev_jax/bench_shared_compilation.py``).
With ``share_compilation = true`` -- the pipeline's default -- the spectrum
travels to the compiled likelihood as an *argument*, so the compiled program does
not contain the data and is reused for the next source. In a four-worker run of
``develop/dev_config/mock_sample.toml`` the report shows 0.26-0.38 s of compiling
for the first source each worker fitted and 0.01 s for the second.

JNesty's own loops accept the same context, so its kernels are reused as well,
provided the model structure (not the data) stays the same. The report records
``compile_seconds`` and ``sample_seconds`` separately, so the saving -- or its
absence -- is visible per source rather than argued about.

The one thing that defeats the reuse is a model that carries per-source numbers
of its own: a narrow-line profile read out of each spectrum
(``[data] narrow_template_key``) is a *constant* inside the traced program, not an
argument, so two sources with different profiles trace two different programs.
The cost is one recompile of the whole likelihood kernel -- there is no "template
part" to compile separately, because the profile is arithmetic inside the same
expression as the model sum and the residual. Measured on the 27-parameter IRAS
model (56956 pixels, CPU): **0.5-0.7 s** for a new profile, **0.07 s** when the
profile repeats (the trace plus a registry hit). The prior transform is a separate
program with no model in it, so it is reused; JNesty's own sampling kernels never
touch the model and are not recompiled at all.

Avoiding even that would mean passing the profile as a runtime array argument,
which needs the model to *accept* one -- a change to :mod:`galspec.jax_models`,
and the pipeline deliberately changes no model. Half a second per source against
minutes of sampling is the trade the file takes; the report records it per source
as ``compile_seconds``.

The registry hashes the program *including its constants*, which matters here: JAX
prints a large array constant in abbreviated form, so two models that differ only
in their profile trace to the same text, and a key made from the text alone would
hand the second source the first source's kernel -- a fit with the wrong line
shape, and nothing downstream would show it. ``tests/test_shared_compilation.py``
checks that a model differing only in a constant compiles again and gives its own
likelihood.

This is a compilation saving, not a speed-up of the sampling: the compiled
arithmetic is the same, and per-iteration cost is unchanged within run-to-run
noise. The numbers do not change either -- the same computation is traced, and
``tests/test_shared_compilation.py`` checks that the likelihood, the prior
transform and a short run agree with the per-source path.
``compilation_cache_dir`` additionally keeps JAX's on-disk cache, which is what
helps across processes (``workers > 1``) and across runs.

Command line
------------

::

    python -m galspec.pipeline CONFIG.toml [options]

::

    --template [PATH]  write a starter configuration (PATH defaults to config.toml;
                       "-" prints it) and exit; --force overwrites
    --dry-run          read every spectrum and report what would be fitted
    --list-sources     print the source ids and files the config selects
    --workers N        fit in N processes
    --force            refit even when the outputs exist
    --output DIR       write here instead of [run] output_dir
    --device CPU|GPU   override [fit] device

    --quiet            only failures and the final summary

A fit prints nothing between the ``fitting`` line and its result, which for a
27-parameter model on a CPU means hours: nested sampling converges by itself and
has nothing to report in between. In a terminal the sampler draws a progress bar, so
silence there means something else; when the output is redirected the bar is
suppressed (it is written with carriage returns, which is noise in a log) and the run
says so before it starts. ``[run] progress`` forces either behaviour, and ``--quiet``
turns both the log and the bar off.

``--dry-run`` is the one to run first on a new sample: it reads every file,
resolves every expression against that source's flux, builds the model, and
reports the pixels, the rest-frame range, the redshift and its source, the
tokens and the number of free parameters -- without sampling. For the IRAS file::

    $ ./example/fit_iras07546_config.sh --dry-run
    [1/1] iras07546_pipeline: 56956 pixels, 4200.0-6800.0 A rest, z=0.0953 (header:REDSHIFT), peak_flux=4.332, 27 free parameters

``example/fit_iras07546_config.sh`` is a two-line wrapper around the command; it
passes its arguments through, so ``--device cpu`` or ``--output`` work with it
too. The same thing from Python:

.. code-block:: python

    from galspec.pipeline import load_config, run_pipeline

    config = load_config("sample.toml")
    report = run_pipeline(config)            # or workers=8, force=True
    print(report["counts"], report["sources"][0]["log_evidence"])

The IRAS reproduction
---------------------

``example/fit_iras07546_config.toml`` is the model of
``example/fit_iras07546_jax.py`` -- seventeen additive components, 27 free
parameters, 56956 pixels -- and it reproduces the published run::

    ./example/fit_iras07546_config.sh                # ~12 min on a GPU
    ./example/fit_iras07546_config.sh --device cpu   # hours

The claim is checkable and is checked by
``develop/dev_config/test_iras07546_config.py``, which runs in a few minutes:

* the spectrum equals ``iras07546_model.load_spectrum()`` to ``1.3e-16``
  relative -- the best possible, since both are the same float64 numbers;
* the model is the same: same 17 components in the same order, same 27 free
  parameters in the same order, same starting values, bounds, priors (kind and
  numbers) and the same 29 ties;
* ``prior_transform`` and the log-likelihood agree point by point, so the
  sampling problem is identical;
* a capped fit (50 live points, 300 iterations) drives the sampler end to end and
  writes every output.

Published numbers to compare against
(``example/iras07546_model.REFERENCE``)::

    Gaussian narrow lines          30 free   log Z = 130889.60 +/- 0.62   (10.2 min)
    measured narrow profile        27 free   log Z = 130677.27 +/- 0.58   (11.7 min)   <- this file

Repeat runs of one configuration have given 130656.22, 130656.34 and 130658.06
-- about two nats of scatter, from the last bits of a GPU reduction reaching the
sampler's ordering -- and re-extracting the narrow-line profile moved the evidence
by about twenty. Quote an evidence from this family to a nat or two; the
differences the model is judged on are two hundred. What the file reproduces
exactly is the *problem* -- the data, the model, the priors -- not the
floating-point trajectory of the sampler.

``develop/dev_config/index.html`` (written by
``develop/dev_config/make_report.py``) is the illustrated version of this
walkthrough: it draws the spectrum the converter writes, the model at its
starting values split into continuum, iron, narrow and broad, the prior boxes,
and the parity figures.

A small example to copy from
----------------------------

``develop/dev_config/mock_two_line.toml`` fits the two-line mock spectrum that
ships with GalSpec -- a linear continuum plus two Gaussians, one tied velocity,
token priors::

    python -m galspec.pipeline develop/dev_config/mock_two_line.toml --dry-run
    python -m galspec.pipeline develop/dev_config/mock_two_line.toml

The fit recovers the line centres to a fraction of a pixel and the two very
different widths of the mock (about 1300 and 160 km/s), with residuals consistent
with the 1% noise. ``develop/dev_config/mock_sample.toml`` is the same model on
six spectra of different brightness:
``python develop/dev_config/make_mock_sample.py`` writes them, then
``--workers 4`` fits them in about 45 s and the report shows both widths
recovered in every source while the amplitudes follow each source's
``peak_flux``.

These two live in ``develop/`` because they are tests of the notation, not
science; ``example/`` holds the two examples a reader should start from
(``fit_iras07546_jax.py`` in Python and ``fit_iras07546_config.toml`` in TOML).

Limitations
-----------

* One FITS file per source; no row-slicing of a multispec file.
* No error synthesis, no sky-line masking and no flux-unit conversion: prepare
  the file you want fitted. The two corrections the pipeline *does* apply, the
  redshift and the Galactic extinction, are read from the header and recorded.
* The model is a sum of components; a product (e.g. a convolution wrapper) is not
  expressible in the file yet.
* A prior cannot be nested inside another prior (``spikeandslab`` uses its
  default slab).
* ``dynamic.py``'s ``DynamicNestedSampler`` is not covered by the shared
  compilation path; ``JNesty_Fit`` uses ``NestedSampler``.