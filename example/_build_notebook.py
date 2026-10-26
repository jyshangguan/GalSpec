"""Generate ``example/fit_iras07546_jax.ipynb`` from ``example/fit_iras07546_jax.py``.

The notebook is a *projection* of the script: every code cell is a slice of the
script, cut at its ``# %% <name>`` markers, so the two cannot drift apart.  This
file adds only what belongs in a notebook and not in a batch script -- the prose,
and five cells that exist to look at something (the spectrum, the prior spread, the
width-and-flux arithmetic, reading the fit back, releasing the GPU).

    python example/_build_notebook.py

Regenerating the notebook discards its outputs; an executed copy that is already
there is moved aside first (it is the result of a sampler run).
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "fit_iras07546_jax.py"
OUT = HERE / "fit_iras07546_jax.ipynb"


def script_cells(path: Path = SCRIPT) -> dict[str, str]:
    """``{name: source}`` for every ``# %% name`` block in the script.

    A block runs from its marker to the next one, so a marker inside ``main()``
    yields the body of one step, dedented: the notebook runs it as top-level code,
    which is exactly what a reader of the script sees on the page.
    """
    text = path.read_text()
    parts = re.split(r"(?m)^[ \t]*# %% ?(\S*)[ \t]*$", text)
    cells = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        cells[name] = textwrap.dedent(body).strip("\n")
    return cells


# --------------------------------------------------------------------------------------
# Cells that exist only in the notebook
# --------------------------------------------------------------------------------------
PLOT_DATA = '''\
# Before fitting anything: the spectrum, the error the loader built for it, and the
# line-free level the priors are scaled from.
step = slice(None, None, 5)                       # 57 000 pixels is more than a screen
error_percent = 100.0 * error / np.abs(flux)

fig, axes = plt.subplots(2, 1, figsize=(13.0, 6.0), sharex=True,
                         gridspec_kw={"height_ratios": [3.0, 1.0]},
                         constrained_layout=True)
axes[0].plot(wave[step], flux[step], color="0.35", lw=0.6, label="spectrum")
axes[0].plot(wave[step], error[step], color="#b5364a", lw=0.8, label="1-sigma error")
axes[0].axhline(np.percentile(flux, 15.0), color="#2b6cb0", lw=0.8, ls="--",
                label="line-free level (15th percentile)")
for centre, text in ((LINE_WAVE["Hgamma"], r"H$\\gamma$"),
                     (LINE_WAVE["Hbeta"], r"H$\\beta$"),
                     (LINE_WAVE["HeI_5876"], "He I"),
                     (LINE_WAVE["Halpha"], r"H$\\alpha$")):
    axes[0].axvline(centre, color="#2b6cb0", lw=0.7, ls=":", zorder=0)
    axes[0].annotate(text, (centre, 0.97), xycoords=("data", "axes fraction"),
                     xytext=(2, 0), textcoords="offset points", fontsize=11,
                     color="#2b6cb0", va="top")
axes[0].set_ylabel("flux density  [1e-14 erg/s/cm$^2$/A]")
axes[0].set_title("IRAS 07546+3928: 4200-6800 A, fitted in one go")
# the legend goes above the axes, so it cannot sit on the line labels
axes[0].legend(fontsize=9, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.005))
axes[1].plot(wave[step], error_percent[step], color="#b5364a", lw=0.8)
axes[1].set_ylabel("error [%]")
axes[1].set_xlabel("rest wavelength  [A]")
axes[1].set_ylim(0, 1.25 * np.percentile(error_percent, 99.0))
'''

PRIOR_SPREAD = '''\
# Draw parameter sets from the joint prior, evaluate the model for each, and look at
# the spread.  If the data sit inside the band, the prior contains the answer -- which
# is why no least-squares seed is needed.  The band is wide on purpose.
model.prepare(wave)
rng = np.random.default_rng(0)
theta_prior = np.array([fit.prior_transform(row) for row in rng.random((120, fit.ndim))])
draws = np.asarray(model.flux_batch(wave, theta_prior))

fig, ax = plt.subplots(figsize=(13.0, 3.8))
step = slice(None, None, 11)
ax.fill_between(wave[step], np.percentile(draws, 1, axis=0)[step],
                np.percentile(draws, 99, axis=0)[step], color="#2b6cb0",
                alpha=0.18, label="1-99% of 120 draws from the priors")
ax.plot(wave[step], flux[step], color="0.35", lw=0.6, label="data")
for centre in (LINE_WAVE["Hgamma"], LINE_WAVE["Hbeta"],
               LINE_WAVE["HeI_5876"], LINE_WAVE["Halpha"]):
    ax.axvline(centre, color="#b5364a", lw=0.6, ls=":")
ax.set_yscale("symlog", linthresh=1.0)
ax.set_xlabel("rest wavelength  [A]")
ax.set_ylabel("flux density  [1e-14 erg/s/cm$^2$/A]")
ax.set_title("The priors alone already bracket the spectrum")
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
'''

DERIVED = '''\
# Widths and integrated fluxes are what a paper quotes, and they are combinations of
# parameters, so they are computed per posterior sample rather than from the medians:
# that is how their error bars inherit the correlations in the fit.
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))                  # sigma -> FWHM
C_KM_S = 299792.458

samples = np.asarray(fit.samples_equal_weight, dtype=float)
column = {name: index for index, name in enumerate(fit.full_param_names)}


def flux_of(amplitude_name, sigma_name, wavelength):
    """Integrated flux of one Gaussian, in 1e-17 erg/s/cm2, per sample.

    ``A sigma sqrt(2 pi)`` is the integral when the width is in wavelength, so the
    velocity width is converted first: one km/s is ``wavelength / c`` Angstrom.
    """
    peak = samples[:, column[amplitude_name]]
    width_angstrom = samples[:, column[sigma_name]] * wavelength / C_KM_S
    return peak * width_angstrom * np.sqrt(2.0 * np.pi) * 1e3


def summarise(label, width_name, flux):
    """Median and 16-84 band of a width (as FWHM) and of a flux."""
    width = samples[:, column[width_name]]
    fwhm = np.median(width) * FWHM
    fwhm_lo, fwhm_hi = np.percentile(width, [16, 84]) * FWHM
    median = np.median(flux)
    flux_lo, flux_hi = np.percentile(flux, [16, 84])
    print(f"  {label:22s} {fwhm:8.0f} +{fwhm_hi - fwhm:5.0f} -{fwhm - fwhm_lo:4.0f}"
          f"   {median:9.1f} +{flux_hi - median:6.1f} -{median - flux_lo:6.1f}")


# the wind is a ratio to the core, so its flux is the core's flux times the ratio of
# its width to the core's (the amplitudes already differ by that ratio)
core = flux_of("b_halpha.amp_c", "b_halpha.sigma_c", LINE_WAVE["Halpha"])
wind = (samples[:, column["b_halpha.amp_w0"]]
        * samples[:, column["b_halpha.sigma_w0"]] / samples[:, column["b_halpha.sigma_c"]]
        * core)

print(f"  {'component':22s} {'FWHM':>17s}   {'flux':>18s}")
print(f"  {'':22s} {'[km/s]':>17s}   {'[1e-17 erg/s/cm2]':>18s}")
print("  " + "-" * 63)
summarise("[O III] 5007 core", "OIII.sigma_c",
          flux_of("OIII.amp_c0", "OIII.sigma_c", LINE_WAVE["OIII_5007"]))
summarise("narrow H-beta", "OIII.sigma_c",
          flux_of("n_hbeta.amplitude", "OIII.sigma_c", LINE_WAVE["Hbeta"]))
summarise("broad H-alpha core", "b_halpha.sigma_c", core)
summarise("broad H-alpha wind", "b_halpha.sigma_w0", wind)
print("\\n  (the narrow widths are the observed ones: no LSF was removed)")
'''

RELOAD = '''\
# Reading it back gives a JNesty_Fit with the same posterior, evidence, bounds, priors
# and model -- and no sampler.  This is the file worth keeping.
loaded = galspec.jnesty_fit_io.load_jnesty_fit_from_fits(fits_path)

print(f"log Z stored    {fit.log_evidence:12.2f} +/- {fit.log_evidence_err:.2f}")
print(f"log Z reloaded  {loaded.log_evidence:12.2f} +/- {loaded.log_evidence_err:.2f}")
print("same parameters:", list(loaded.param_names) == list(fit.param_names))
print("same posterior: ", np.allclose(loaded.samples_equal_weight, fit.samples_equal_weight))
print("priors restored:", all(loaded.model.get_prior(q) is not None
                             for q in free_qualified_names(loaded.model)))
'''

FREE = '''\
# The save step above called fit.release_device_buffers(): the sampler's buffers on the
# card (sized by max_iterations) are gone, while the results -- samples, evidence,
# quantiles -- survive, because they live in host memory.
print("sampler released; the quantiles are still available:", fit.get_quantiles().shape)
print("and so is the model:", len(model.submodel_names), "components,",
      len(model.free_param_names), "free parameters")
'''

#: The notebook runs in a kernel, so it needs the interactive backend; the script picks
#: Agg for itself (see the ``get_ipython`` guard in its imports cell).
MAGIC = "%matplotlib inline\n"

# --------------------------------------------------------------------------------------
# The notebook: prose, and named slices of the script
# --------------------------------------------------------------------------------------
cells = [
    ("md", """# Fitting IRAS 07546+3928 with JNesty on the GPU

This notebook fits one real spectrum with the GPU-native side of GalSpec: JAX models
(`galspec.jax_models`) sampled by **JNesty** (`galspec.jnesty_fit`). It is the
notebook form of `fit_iras07546_jax.py`, and every code cell below is a slice of that
script -- the cells are cut from it automatically, so the two cannot drift apart. The
script is what to use in a batch pipeline; the notebook is for looking at what happens.

## What is fitted

**4200-6800 A** of the quasar **IRAS 07546+3928** (z = 0.0953) in the rest frame --
56 956 pixels, from H-gamma to the [S II] doublet. All of it is fitted at once, so the
continuum and the lines are constrained by the same posterior:

| wavelength | what | model |
| --- | --- | --- |
| 4340 A | H-gamma | a narrow Gaussian + a broad MultiGauss (core + wind) |
| 4861 A | H-beta | a narrow Gaussian + a broad MultiGauss (core + wind) |
| 6563 A | H-alpha | a narrow Gaussian + a broad MultiGauss (core + wind) |
| 5876 A | He I | a narrow Gaussian + one broad Gaussian |
| 4959 / 5007 A | [O III] | narrow core + one broad wing |
| 6548 / 6584 A | [N II] | narrow, doublet ratio 2.96 (atomic physics) |
| 6716 / 6731 A | [S II] | narrow, **both peaks free** (the ratio is a density diagnostic) |
| 6300 / 6364 A | [O I] | narrow, doublet ratio 3.00 |
| 4686 A | He II | a narrow Gaussian + one broad Gaussian |
| 4363 A | [O III] | single narrow Gaussian |
| all | continuum | power law + Fe II template |

Nothing outside that window is fitted: no H-delta or higher Balmer lines, no [Ne III]
3869 or [O II] 3727, no [Ar III] 7136 or [O II] 7325. Stopping at H-gamma on the blue
side and at [S II] on the red side means **no masks are needed** -- the Ca II H+K
absorption of the host galaxy is below 4200 A, and the atmospheric O2 A band is beyond
6800 A. The Na D window, which used to be masked, now stays in the fit: He I 5876 sits
inside it and is modelled instead.

## How it is fitted

* **No least-squares step.** Nothing is optimised before nested sampling starts, so the
  priors cannot inherit a preliminary answer. JNesty walks the prior directly.
* **No instrumental smoothing.** At 0.046 A per pixel the spectrum grossly oversamples
  the spectrograph: R = 2000 means ~150 km/s of instrument, while the narrow lines here
  are ~700 km/s. So the model is *not* convolved, exactly as in
  `wrk_sagan_iras07546.ipynb`; the fitted narrow width is the observed one.
* **Wide priors, taken from the data.** Line peaks from invisible to 3x the brightest
  pixel (log-uniform); the broad core 300-8000 km/s and the wind 800-25000 km/s; the
  wind's amplitude anywhere between 0 and 1 of the core. Section 5 shows the prior
  spread against the data.
* **Physical ties.** Every narrow line shares one velocity shift and one width
  (measured from [O III]); the broad core and the wind are each shared across the three
  Balmer lines and He I; and the Fe II template is broadened by broad H-alpha itself.
  30 free numbers therefore describe seventeen models.

Runtime: 10-20 minutes on one GPU with `NLIVE = 400, DLOGZ = 0.01`. For a quick look set
`QUICK = True` in the settings cell -- 120 live points and a looser stopping criterion --
and expect wider error bars."""),

    ("md", """## 1. Imports and settings

Everything a user might want to change is in the settings cell: the redshift, the
extinction, the wavelength window, and the two sampler settings. `QUICK = True` is the
one shortcut; everything else is the published configuration."""),

    ("extra", MAGIC),
    ("code", "imports"),
    ("code", "settings"),

    ("md", """## 2. The data

The FITS file that ships with GalSpec has flux only, no uncertainty, so the uncertainty
is measured from the data: subtract a running median, take the median absolute deviation
in a 30 A window (which ignores the few pixels belonging to emission lines), convert to a
Gaussian sigma, and smooth. That gives a wavelength-dependent noise level, shown in the
lower panel below, which rises from ~0.3% of the flux at 4700 A to ~4% at the red end.

On top of that, **5% of the flux itself** is added in quadrature. This is the photon noise
of the source: the counts scale with the flux, so the uncertainty does, and on the bright
continuum it dominates everything else. It also covers what the model does not describe
exactly (flux calibration, the shape of the Fe II template, departures from a pure power
law), which is why 5% is the honest number for a template fit rather than a fudge factor.

**No window is masked.** The masks an earlier version needed are gone with the wavelength
cut: the host galaxy's Ca II H+K absorption is below 4200 A, the Na D window is kept
because He I 5876 inside it is now fitted as a line, and the atmospheric O2 A band is
beyond 6800 A.

Units: fluxes in `1e-14 erg/s/cm2/A`, velocities in km/s."""),

    ("code", "load"),
    ("code", "step-1-data"),
    ("extra", PLOT_DATA),

    ("md", """## 3. The model

Seventeen models, added together:

1. a **power-law continuum** anchored at 5500 A -- this is what fixes the continuum level
   and slope from the line-free regions of the whole spectrum;
2. an **Fe II template** -- the forest of iron lines that fakes a broad bump blueward of
   H-beta and is routinely mistaken for continuum. It uses `IronTemplate_tied`: the
   template is broadened by the *same* multi-Gaussian kernel as broad H-alpha, tied
   component by component, so the iron has a single free parameter, its amplitude. The
   template is loaded through `galspec.iron_template.load_iron_template`, which resamples
   it onto a grid even in `ln(wavelength)` -- one channel is one velocity everywhere, so a
   width is one number -- and pads the tabulated range with zeros, so the model fades out
   where the template runs out instead of reflecting its edge flux. An earlier version
   fitted it with one velocity dispersion (`IronTemplate_new`); that is still available,
   and the two agree exactly when the kernel has one component. Tying it to the line is
   the more physical choice: the iron and the Balmer lines are emitted by the same gas, so
   they cannot have different kinematics;
3. **H-alpha, H-beta, H-gamma**, each described **twice**, as in
   `wrk_sagan_iras07546.ipynb`: `n_halpha` is a `Line_Gaussian` for the narrow component,
   and `b_halpha` is a `Line_MultiGauss` for the **broad component only** -- its component
   0 (`amp_c`, `dv_c`, `sigma_c`) is the broad core, not a narrow line, and
   `amp_w0`/`dv_w0`/`sigma_w0` add the broader wind. The wind amplitude is a *ratio to the
   core amplitude*, which is how the GalSpec models are written. The two models are simply
   added; neither knows about the other;
4. **He I 5876**, a narrow Gaussian (`n_hei`) plus **one** broad component (`b_hei`). It is
   what used to be inside the Na D mask, and it has to be modelled now that the mask is gone;
5. **He II 4686**, the same pattern: a narrow Gaussian (`HeII`) plus **one** broad Gaussian
   (`b_heii`). Broad He II is a standard feature of a narrow-line Seyfert 1 and it sits on
   the blue side of H-beta, where the iron and the Balmer wing overlap;
6. **[O III] 4959/5007** as a doublet with a narrow core and one broad wing;
7. **[N II], [S II], [O I]** doublets, plus a single narrow Gaussian for **[O III] 4363**.

The ties are what make this affordable:

* one `dv` and one `sigma` for **all** narrow lines, taken from the [O III] core (forbidden
  lines come from the same low-density gas);
* `dv_c`/`sigma_c` (the broad core) and `dv_w0`/`sigma_w0` (the wind) shared by the three
  broad Balmers, by He I and by He II: the two broad components are separate physical
  structures, so they get separate kinematics, but each is common to the lines. The Fe II
  kernel's two components are tied to these outright -- shifts, widths and the wind ratio
  alike;
* doublet ratios fixed by atomic physics for [O III] (2.98), [N II] (2.96) and [O I] (3.00)
  -- but **not** for [S II], whose 6716/6731 ratio is a density diagnostic and therefore
  stays free.

Note there is no `* multi` polynomial here: the reference notebook multiplies one in
because it uses a least-squares fitter with no error model, whereas here the 5% flux term
already absorbs flux-calibration error, and a high-order polynomial on top of a power law
plus an Fe II template would be strongly degenerate with both."""),

    ("code", "helpers"),
    ("code", "model"),

    ("md", """### Why the iron is broadened by the line profile

The kernel is a multi-Gaussian whose shifts, widths and weights are tied to the broad
components of H-alpha, and the normalisation keeps the iron scale separate from the line
amplitudes. The arithmetic is simple: subtract the template's own width in quadrature for
each component, then average the shifted profiles, on the 2 389-point template rather than
on the spectrum.

What it *changes* is the physics: the iron gets the same velocity structure as the Balmer
lines, including any asymmetry, so it is the right model when the two are known to share a
broad-line region and the wrong one when they are not. What it *costs* is one smoothing and
one interpolation per component of the template; `develop/dev_jax/iron_cpu_vs_jax.py`
measures that on the card you are using."""),

    ("md", """## 4. The priors -- broad, and justified by the data

This is the part that decides whether a fit is honest. Every range below comes from the
spectrum itself, never from a previous fit:

| what | range | why |
| --- | --- | --- |
| line peaks | 1e-3 to 3x the brightest pixel, **log-uniform** | four decades: from invisible to implausibly strong |
| wind/core ratio | 0-1, **uniform** | no wind is brighter than its own core -- and this is what breaks the core-wind degeneracy |
| narrow widths | 30-800 km/s | unresolved up to the widest known narrow-line region |
| [O III] wing width | 300-4000 km/s | a wing on the narrow core, not the broad-line wind |
| broad core | 300-8000 km/s | FWHM up to ~19000 km/s |
| broad wind | 800-25000 km/s | FWHM up to ~59000 km/s, for a line wing |
| continuum level | 0.2-4x the line-free windows | cannot be negative, cannot swamp the lines |
| iron amplitude | 1e-3 to 3x the continuum level | the only free iron parameter |
| [S II] 6716/6731 | both peaks 1e-3 to 3x peak | the ratio is measured, not assumed |

Two shapes do all the work. A parameter whose range is a plain box takes a *linear* prior,
and a linear prior over four decades would put 99% of its probability at the bright end, so
amplitudes get `priors.LogUniform`: every factor of ten is equally likely. `Uniform` is right
where the range is already the physically sensible one -- a ratio bounded by 1, a velocity
shift, a width.

One subtlety, and it is the reason the width ranges matter: the two components of a
`Line_MultiGauss` are **exchangeable**. Swapping them -- `amp_c' = amp_c * amp_w0`,
`sigma_c' = sigma_w0`, `dv_c' = dv_c + dv_w0`, `amp_w0' = 1/amp_w0`, `dv_w0' = -dv_w0`,
`sigma_w0' = sigma_c` -- leaves the spectrum identical to 3e-16, and the Fe II kernel follows
it because it is tied to the same profile. One number settles it: `amp_w0` is uniform on
[0, 1], so the swapped solution -- which needs `amp_w0' = 1/amp_w0 > 1` -- is outside the
prior, while the *widths* keep their full data-driven range. The fit below lands in one mode
with no posterior samples on the mirror branch;
`develop/dev_jax/io_test/iron_tied_io_test.html` measures the same thing on a mock."""),

    ("code", "priors"),
    ("code", "step-2-model"),

    ("md", """## 5. Run the sampler

`JNesty_Fit` compiles the log-likelihood and the prior transform with `jax.jit`, batches
likelihood evaluations on the GPU, and runs nested sampling until the remaining evidence is
below `dlogz`.

Nested sampling keeps `NLIVE` "live points" drawn from the prior and repeatedly replaces the
worst one with a point of higher likelihood: the volume shrinks, the likelihood rises, and the
integral of both is the evidence. The surviving points are a weighted sample of the posterior,
which is what we quote below.

JNesty draws its live points from the OS entropy, so a rerun is a fresh draw from the
same posterior rather than a bit-for-bit repeat: `log Z` moves by a few nats at 100 live
points and by a fraction of its error bar at 400, which is why the published number is
quoted with an uncertainty."""),

    ("code", "step-3-fitter"),

    ("md", """### Does the prior cover the data?

A useful sanity check, and the reason no least-squares seed is needed: draw parameter sets
from the prior, evaluate the model for each, and look at the spread. If the data fall inside
that band, the prior contains the answer. (The band is wide on purpose -- that is what "broad
prior" means.)"""),

    ("extra", PRIOR_SPREAD),

    ("code", "step-4-fit"),

    ("md", """## 6. What the fit says

The table is the answer: posterior median and the 16th/84th percentiles (a 68% credible
interval, the Bayesian counterpart of "1 sigma").

Read it with the ties in mind -- `OIII.dv_c` is the shift of *every* narrow line, and
`b_halpha.sigma_w0` is the width of the wind in *all three* Balmer lines, in He I and in the
Fe II kernel. The unit column is read off the parameter's suffix, so nothing has to be
remembered: `amp*` is a flux density, `dv*`/`sigma*` is km/s, `amp_w*` is a ratio to the core."""),

    ("code", "step-5-results"),

    ("md", """### Widths and fluxes

Widths and fluxes are what a paper quotes, and they are combinations of parameters, so they
are computed **per posterior sample** rather than from the medians -- that is how their error
bars inherit the correlations in the fit.

`FWHM = 2.3548 sigma`. The integrated flux of a Gaussian of peak `A` and width `sigma` at
wavelength `lambda` is `A sigma sqrt(2 pi) lambda / c`, converted here to the customary
`1e-17 erg/s/cm2`.

`FWHM` for the narrow components is the *observed* width (no LSF was removed)."""),

    ("extra", DERIVED),

    ("md", """## 7. How good is the fit?

`chi2/dof` near 1 means the model explains the data to within the error bars. Far below 1 would
mean the error bars were inflated; far above 1 means the model is missing something.

The 5% flux term sets the scale of this number, and it is worth knowing which way it cuts. A
chi2/dof **above** 1 would mean the model cannot reach the error bar -- something is missing.
What this fit gets is **0.32**, which says the opposite: the 5% term is *larger* than the true
pixel-to-pixel noise, so the residuals come out at ~0.6 sigma and the posterior is
correspondingly conservative -- parameter errors are wider than the photon statistics alone
would give, which is the safe direction for a template fit that also carries flux calibration
and template-shape uncertainty. Lower `PHOTON_FRACTION` to ~0.02 if you want error bars set by
pixel noise alone; raise it if you distrust the spectrophotometry more than that.

**The residuals are not white noise**, and a chi-square alone hides that: binned over a few
hundred pixels the mean residual wanders by more than pure noise would allow, and neighbouring
pixels are correlated. A 5% error bar cannot make that go away -- it is the model, not the
noise, that is coherent over tens of pixels. So the rms describes the pixel-to-pixel scatter,
not the accuracy of the model. Treat the error bars on individual parameters as statistical,
and add a systematic of order the worst binned residual when comparing with measurements from
other instruments."""),

    ("md", """## 8. The fit, drawn with GalSpec's own plotting function

`galspec.plot.plot_fit` takes the fitted JAX model directly and draws the data, the total
model, **every component**, the dashed sub-components of each multi-Gaussian line (so the
narrow Gaussian and the broad core and wind of each Balmer line can be told apart) and the
residual panel.

Two details of the figure are deliberate. The residual panel is drawn in **units of sigma**
(`residual_in_sigma=True`), so a residual of 3 means three error bars and the stated rms can be
checked by eye -- in raw flux units a single bright pixel sets the scale of the whole panel.
And the legend is placed **outside the axes**: seventeen components do not fit inside a panel
without covering the data."""),

    ("code", "step-6-figure"),

    ("md", """### Zooms

The whole-spectrum figure is crowded; the same function, called on a wavelength window, shows
the components where they matter."""),

    ("code", "step-7-zooms"),

    ("md", """### Correlations

Nested sampling gives the full posterior, not just error bars. An elongated blob means two
parameters trade off against each other -- a broader component with a smaller peak, say -- and
quoting them independently would be misleading."""),

    ("md", """### Marginal posteriors

The corner plot shows every pair; for the numbers people actually quote, one histogram each is
easier to read. `plot_posterior_hist` draws the marginal posterior, marks the median and the
16-84 band, and prints the value on top.

The **x range comes from the posterior, not from the prior**. That matters: the prior on a
width spans 300-8000 km/s while the posterior spans a few km/s, so an axis stretched to the
prior would draw the measurement as a spike one pixel wide and hide the shape that the panel
exists to show. A parameter that has genuinely filled its prior still looks flat, because the
histogram itself is flat -- and when a prior bound falls outside the plotted range its value is
written in the corner, so nothing is hidden by the zoom."""),

    ("code", "step-8-posteriors"),

    ("md", """## 9. What this does and does not show

* The blue end of the fitted range, 4200-4400 A, is the weakest part of the model: H-gamma is
  cut by the window edge on its blue wing, and there is no Balmer continuum, which rises
  towards its 3646 A limit and is not modelled at all. Trust H-gamma's broad flux less than
  H-alpha's.
* The two broad components are a *description* of a profile, not a physical decomposition.
  Real broad-line profiles have longer wings than any Gaussian, so read them as "a broad part
  and a broader part" and treat their individual fluxes as indicative. Because the iron kernel
  is tied to them, that caveat propagates to the iron profile too. They are also *exchangeable*
  (see the priors section), which is why the wind amplitude carries the bounded ratio prior.
* Without the LSF folded in, the narrow widths include the instrument (~150 km/s FWHM, small
  next to ~700 km/s but not zero). Subtract it in quadrature if you need the gas kinematics alone.
* The narrow lines share one width by construction -- the standard assumption for forbidden-line
  regions, but it means a genuinely different narrow kinematics per line could not be detected.
* The [S II] ratio is measured, not assumed, because it is a density diagnostic: I(6716)/I(6731)
  near 1.3 means low-density gas, near 0.5 high-density. Turning it into an electron density
  needs a temperature and the atomic data, which is beyond a line-profile fit.
* The Fe II *shape* is still a template: tying its kernel to broad H-alpha fixes the broadening,
  but the relative strengths of the iron lines come from the reference object (Mrk 493). A real
  AGN with a different iron excitation would need a different template, not a different width.
* He I 5876 is fitted with one broad component; the line is weak, so its broad fraction is the
  least well constrained number in the table.
* The evidence (`log Z`) is only comparable between runs made with the same sampler.

## Where to go next

* `python fit_iras07546_jax.py` -- the same fit as a batch script.
* `QUICK = True` in the settings cell -- the same script in a few minutes, for a look.
* `python fit_iras07546_jax_tied.py` -- releases the broad-line ties and asks whether H-beta and
  H-gamma really share H-alpha's profile.
* `python fit_iras07546_jax_toggle.py` -- puts a spike-and-slab prior on each wind amplitude and
  reports how many components the profile actually needs.
* `docs/source/jax_gpu.rst` -- the CPU/GPU benchmark built on a fit like this.
* `tests/test_jax_parity.py` -- proves the JAX models equal the astropy ones, the tied iron included.
* `tests/test_model_features.py` -- pins the iron template's grid and padding, the tied kernel, and
  the other model features this fit relies on.
* `tests/test_jax_io.py` -- proves the FITS round trip gives back the same fit."""),

    ("md", """## 10. Saving the fit so it can be reused

A nested-sampling run is expensive; the file should be the thing you keep. Two files are written:

* `output/iras07546_jax_posterior.npz` -- a plain array dump, handy for a quick numpy read;
* `output/iras07546_jax_fit.fits` -- everything: the posterior samples, their unit-cube
  coordinates, the evidence and its error, the parameter names, bounds and priors, the sampler's
  convergence trajectory, the model description and the spectrum that was fitted.

The FITS layout is the one `galspec.dynesty_fit_io` already uses (`SAMPLES`, `LOGWT`, `LOGL`,
`LOGVOL`, `SAMPLES_EQ`, `PARAM_NAMES`, `PARAM_BOUNDS`, `MODEL_PARAM`, `MODEL_STR`, ...), with
three JNesty-only extensions: `SAMPLES_U`, `DLZ_TRAJ` and `SCALE_TR`. The model is written by
`galspec.jax_models.serialization`, which records what a JAX model needs and an astropy model does
not -- the number of Gaussians per line, a template name or a polynomial degree -- and the ties as
*references* (`"OIII.dv_c"`), so they are restored no matter how they were written in the code."""),

    ("code", "step-9-save"),

    ("md", """### Reading it back

`load_jnesty_fit_from_fits` rebuilds the model and the fit, and every summary and plot works on the
result. The priors come back with it: each one is stored as its *type plus its arguments*
(`{"kind": "uniform", "args": {"lower": 800.0, "upper": 25000.0}}`), and a prior that names another
parameter stores that name too, so the file is enough to rebuild the prior transform and resume the
sampler. The single exception is a prior you wrote yourself as a plain Python function: that cannot
be written to a file, so the loader warns and the file keeps only the resulting bounds."""),

    ("extra", RELOAD),

    ("md", """### Handing the GPU back

JNesty's device memory is dominated by one buffer allocated once at `max_iterations` size (the
dead-point history), plus the live set and the compiled kernels. It does not grow from iteration to
iteration, but an XLA context never returns its high-water mark to the driver either, so a process
that has finished one fit keeps showing it in `nvidia-smi`. `release_device_buffers()` drops the
sampler and clears JAX's caches; the results, the model and the saved files are untouched."""),

    ("extra", FREE),
]


def _backup_executed(path: Path) -> None:
    """Keep a copy of the notebook being replaced, if it has been executed.

    Regenerating the notebook discards its outputs, and an executed notebook is the
    result of a sampler run -- ten minutes of GPU.  A copy next to it (ignored by git)
    means that cannot be lost by accident.
    """
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text())
    except Exception:                                   # pragma: no cover - defensive
        return
    if any(cell.get("outputs") for cell in existing.get("cells", [])):
        backup = path.with_name(path.stem + ".executed.ipynb")
        backup.write_text(path.read_text())
        print(f"kept the executed copy at {backup.name}")


def build(cells_in=None, out: Path = OUT) -> int:
    """Write the notebook; return its cell count."""
    cells_in = cells if cells_in is None else cells_in
    from_script = script_cells()
    used = {name for kind, name in cells_in if kind == "code"}
    missing = sorted(used - set(from_script))
    if missing:
        raise SystemExit(f"no such '# %%' block in {SCRIPT.name}: {missing}")
    empty = sorted(name for name in used if not from_script[name].strip())
    if empty:
        raise SystemExit(f"empty '# %%' block in {SCRIPT.name}: {empty}")

    out_cells = []
    for index, (kind, payload) in enumerate(cells_in):
        if kind == "md":
            text, cell_type = payload, "markdown"
        else:
            text, cell_type = (from_script[payload] if kind == "code" else payload), "code"
        text = text if text.endswith("\n") else text + "\n"
        cell = {"cell_type": cell_type, "id": f"cell-{index:02d}", "metadata": {},
                "source": text.splitlines(keepends=True)}
        if cell_type == "code":
            cell.update(execution_count=None, outputs=[])
        out_cells.append(cell)
    notebook = {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _backup_executed(out)
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {out} ({len(out_cells)} cells, {len(used)} cut from {SCRIPT.name})")
    return len(out_cells)


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())