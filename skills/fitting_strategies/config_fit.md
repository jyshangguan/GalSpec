# The configuration-file fitting mode

**Purpose**: fit one spectrum, or a whole sample, **without writing Python**. A TOML
file declares the data, the model and the sampler; `python -m galspec.pipeline` does
the rest and writes per-source outputs plus a JSON/HTML report.

**When to use it**: the model is known and the job is repetition --

- the same model for many spectra (`glob`, `files`, a catalog);
- a reproducible fit that a paper can quote as a file rather than a script;
- a batch to run overnight/weekend and inspect afterwards.

**When not to use it**: the model is still being designed. Build it with the Python
API first (`skills/function_reference/*`, `galspec.jax_models`), because a
configuration cannot invent a component the package does not have.

The reference file is `example/fit_iras07546_config.toml`: the 17-component,
27-parameter model of `example/fit_iras07546_jax.py`, checked against that script
parameter by parameter. The full walkthrough is `docs/source/config_fit.rst`.

---

## The two commands

```bash
python -m galspec.pipeline fit.toml --dry-run   # read + validate + list, no sampling
python -m galspec.pipeline fit.toml             # the fit
```

`--dry-run` is not decoration: it is the fastest way to catch a wrong path, a wrong
column name, a missing prior or an empty sampling box, and it prints the resolved
tokens (`peak_flux`, `continuum_level`, ...), the free parameters, and the ones that
took a default prior. Run it before every long fit.

---

## Minimal file

```toml
[data]
file = "spectrum.fits"      # or files = [...] / glob = "spectra/*.fits" / catalog = "s.csv"

[model]
[[model.component]]
type = "Line_MultiGauss"
n_components = 2
wavec = "@Halpha"
```

`[fit]` is optional (`nlive = 500`, `dlogz = 0.01`, `device = "auto"`,
`share_compilation = true`), and so is `[output]` (`figures = "pdf"`,
`fit_plot = true`, `corner = false`, `histogram = false`, `thin = 1`). That is a complete, runnable configuration: every free
parameter takes the default prior of its kind, and **no starting value is needed** --
nested sampling draws its live points from the prior, so a value in the file would
only say where a search would have begun.

---

## What the file declares, and what it must not repeat

| section | declares |
|---|---|
| `[data]` | where the spectra are, which arrays hold wavelength/flux/error, the rest-frame window, masks, the redshift keyword, the reddening keyword |
| `[model]` | one `[[model.component]]` per additive term, in sum order; `[model.component.ties]`, `.bounds`, `.priors` per component |
| `[fit]` | live points, stopping criterion, device, compilation sharing |
| `[output]` | thinning, fit plot, corner plot, histogram |
| `[run]` | output directory, worker processes, progress |

Rules that save time:

- **Ties replace values and priors.** A tied parameter is computed from its reference,
  so writing a value for it is harmless but meaningless.
- **`wavec` is the line's identity** and stays in the file (`wavec = "@Halpha"`);
  the pipeline refuses to guess a centre, because the default (5000 Å) is never the
  line you meant.
- **Both ends of every amplitude prior scale with the spectrum**
  (`loguniform(1e-3*continuum_level, 3*max_flux)`), so one file fits a sample whose
  flux spans decades. Override only when the object, not the unit, demands it
  (e.g. a 300–8000 km/s broad-line width, a continuum-scaled iron ceiling).
- **`require_priors = true`** means "every prior written in this file". Use it when
  you want the check; leave it out when the defaults are what you want.

---

## The two header keywords

Each correction is one keyword, matched exactly and case-insensitively:

```toml
[data]
z_key     = "ZRED"      # default; the redshift the file carries
e_bv_key  = "EBV"       # default; Galactic reddening (A_V is divided by R_V)
extinction = true       # set false to fit as observed (no dereddening)
# extinction_rv = 3.1   # R_V of the law: configuration only, no header fallback
```

- Name the keyword your files actually use (`z_key = "MYZ"`) instead of relying on a
  list of spellings -- the list mode is gone on purpose, because it turns "this file
  has no redshift" into a quiet fallback on a key that happened to be present.
- Precedence for the redshift: **catalog row → header keyword → `z_default`**. A value
  assumed rather than read makes the run warn, and the report records `z_source` /
  `e_bv_source` per source, so an assumption is visible next to a measurement.
- Declaring `z_default` (or `extinction = false`) counts as stating it: the run is
  then silent.
- `extinction_rv` has no header fallback by design: a survey storing `A_V` leans on
  `R_V`, and a wrong `R_V` would scale every flux without a trace.

---

## Outputs, per source

```
<output_dir>/<id>/<id>_fit.fits    # the model, the data and the whole posterior
<output_dir>/<id>/<id>_fit.pdf     # a text summary page, then the figures
<output_dir>/run_report.json       # the run: tokens, priors, evidence, timings
```

Two files, and deliberately no others: no `.npz` archive (every array it would hold
is in the FITS) and, by default, no loose `.png`s. `[output] figures = "png"` writes
the same pages as PNGs instead -- for slides -- and `"both"` writes both sets; the
default `"pdf"` is the one to archive, because it is vector and re-readable at any
zoom.

`<id>_fit.fits` is the result. Read it with astropy -- nothing GalSpec-specific is
needed:

```python
from astropy.io import fits
hdul = fits.open("out/<id>/<id>_fit.fits")
hdr = hdul[0].header                    # LOGZ, LOGZERR, NITER, NCALL, INFORM, ...
samples = hdul["SAMPLES"].data          # (niter, ndim), columns in PARAM_NAMES order
weights = hdul["LOGWT"].data            # ln weight; LOGL and LOGVOL beside it
equal = hdul["SAMPLES_EQ"].data         # the equal-weight resample already made
names = hdul["PARAM_NAMES"].data["PARAM_NAME"]
```

That is everything the dynesty-style diagnostics need (samples, weights, volumes,
`samples_u`, the `LOGZ`/`DLZ`/`SCALE` trajectories), so a failed-looking run can be
diagnosed from the file alone. `SAMPLES_EQ` is what the quoted percentiles come from;
`THETA_BEST` is the maximum-likelihood sample used by the fit page.

`<id>_fit.pdf` is one document: a summary page (evidence, diagnostics, provenance,
and every free parameter's median with its 16th and 84th percentile, `*` marking a
default prior and `!` a posterior pressed against a limit), then the pages `[output]`
asks for (`fit_plot`, `corner`, `histogram`). Vector, so it stays sharp at any zoom.
A corner plot is skipped with a warning when the run stopped before exploring -- the
rest of the document is still written.

`run_report.json` is the machine-readable record of the *run*: per source the evidence
and its error, the resolved tokens, the prior each parameter ended up with (and
whether it came from the file), and `compile_seconds` / `sample_seconds`. Quote
evidence from `log_evidence`, and remember that a different prior *family* is a
different evidence -- a changed prior detail moves log Z without moving the posteriors.

---

## Samples

```toml
[data]
glob = "spectra/*.fits"     # or files = [...] / catalog = "sample.csv" (+ catalog_z_column)
[fit]
share_compilation = true    # one compiled likelihood kernel for the whole sample
workers = 4                 # >1 fits the sources in that many processes
```

- The model must be identical across the sample; only the data (and optional
  per-source numbers such as `catalog_z_column`) may change. That is what makes the
  kernel shareable, and it is the whole speed argument for using this mode.
- A **per-source narrow-line profile** (`[data] narrow_template_key`, a table HDU in
  each file) puts that profile *inside* the traced program, so a sample whose profiles
  differ recompiles once per source (~0.5 s on a CPU, against minutes of sampling).
  That is the one thing that defeats `share_compilation`, and it is a fair trade.
- Already-fitted sources are skipped unless `[run] force = true`, so re-running a long
  sample after adding files continues rather than restarts.

---

## Pitfalls

- **A wrong redshift is silent.** Check `z_source` in the dry run: `header:ZRED` is a
  measurement, `z_default` is an assumption, and the run warns only for the latter.
- **`peak_flux` / `max_flux` ignore noise spikes** (median + sigma clip) but a
  *sub-pixel* line loses part of its peak (a one-pixel line keeps ~80%). Fine for a
  prior scale, not a flux measurement -- measure fluxes from the fitted model.
- **Corner plots stop being readable above ~10 parameters**; use
  `[output] corner_params` to name a handful.
- **The evidence is not comparable across prior changes.** Changing a prior detail
  changes log Z by the ratio of prior volumes (a factor 1.074 in `ln(b/a)` is 0.07 nat
  per parameter) while leaving the posteriors alone.
- **Nothing is invented.** A missing error array is an error, not a `std(flux)`
  estimate; a missing redshift warns; a missing extinction keyword warns.