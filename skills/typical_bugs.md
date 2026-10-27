# Typical Bugs in GalSpec Spectral Fitting

This file documents common bugs, mistakes, and pitfalls in GalSpec spectral fitting. Each bug includes:
- **The Bug**: What went wrong
- **Detection**: How to spot it
- **The Fix**: Correct code/approach
- **Why It Happens**: Root cause

---

## Bug #1: Tying a Parameter to Itself

### The Bug
When tying multiple parameters together, accidentally including the reference parameter in the loop causes it to be tied to itself, effectively fixing its value.

### Example of WRONG Code:
```python
# ❌ WRONG - includes 'nHalpha' in the loop!
for ln in ['nHalpha', 'NII_6583', 'NII_6548']:
    model[ln].dv.tied = galspec.tie_template_dv('nHalpha')
```

**What Happens**:
- `nHalpha.dv.tied = tie_template_dv('nHalpha')`
- This ties nHalpha to **itself**, making it fixed at initial value
- Result: dv stays at 0, cannot vary during fit

### Example of CORRECT Code:
```python
# ✅ CORRECT - skip the reference parameter!
for ln in ['NII_6583', 'NII_6548']:  # ← Don't include 'nHalpha'!
    model[ln].dv.tied = galspec.tie_template_dv('nHalpha')
```

**What Happens**:
- `nHalpha.dv` is **free to vary** (the master parameter)
- `NII_6583.dv` tied to nHalpha
- `NII_6548.dv` tied to nHalpha
- Result: All three share ONE free parameter (nHalpha.dv)

### Detection:
- **Symptom**: Parameter stays at initial value (e.g., dv=0) despite fitting
- **Symptom**: χ² much higher than expected
- **Check**: Print parameter values before and after fit
- **Check**: Verify `model['param_name'].tied` is not the same as param_name

### Impact:
- **Before fix**: χ²/ν = 12.9, narrow dv fixed at 0
- **After fix**: χ²/ν = 1.6, narrow dv free to vary (58 km/s)

### Common Scenarios:
This bug often occurs when:
1. Tying narrow line velocities together
2. Tying doublet amplitude ratios
3. Tying kinematic parameters in multi-component models

### General Rule:
**NEVER include the reference parameter in the tying loop**
```python
# Pattern: Reference is FIRST, others tied to it
reference_param = 'nHalpha'
for param in ['other1', 'other2', 'other3']:  # ← No reference!
    model[param].some_tied = tie_function(reference_param)
```

### Why It Happens:
- It's an easy mistake when iterating through a list of parameter names
- The tied parameter returns itself when tied to itself
- This creates a circular reference that fixes the value
- Astropy's fitting doesn't warn about this explicitly

---

## Template for Future Bugs

Use this template to add new bugs:

```markdown
## Bug #[N]: [Short Title]

### The Bug
[Brief description of what went wrong]

### Example of WRONG Code:
```python
# ❌ WRONG
[bad code example]
```

**What Happens**:
[Explanation of the problem]

### Example of CORRECT Code:
```python
# ✅ CORRECT
[good code example]
```

**What Happens**:
[Explanation of why it works]

### Detection:
- **Symptom**: [How to recognize the bug]
- **Check**: [How to verify]
- **Test**: [Code to test for the bug]

### Impact:
[Before and after comparison if possible]

### Common Scenarios:
[When this bug typically occurs]

### Why It Happens:
[Root cause explanation]

### Prevention:
[How to avoid this bug in the future]
```

---

## Bug #2: `subcomponents` returns None when `n_components == 1`

### The Bug

```python
dbl = galspec.Line_MultiGauss_doublet(n_components=1, ...)
for comp in dbl.subcomponents:        # TypeError: 'NoneType' object is not iterable
    ax.plot(..., comp(wave))
```

`Line_MultiGauss_doublet.subcomponents` only builds the per-line views when
`n_components > 1`.  With one component it returns `None`.

### Correct Code

```python
comps = dbl.subcomponents
if comps is None:                      # n_components == 1
    out = []
    for amp, wavec in ((dbl.amp_c0, dbl.wavec0), (dbl.amp_c1, dbl.wavec1)):
        out.append(galspec.Line_Gaussian(amplitude=amp.value, dv=dbl.dv_c.value,
                                         sigma=dbl.sigma_c.value, wavec=wavec))
else:
    out = list(comps)
```

### Detection
- **Symptom**: `TypeError: 'NoneType' object is not iterable` in a plotting
  routine that worked for `n_components=2`.
- **Check**: `dbl.n_components`, then `dbl.subcomponents is None`.

### Impact
Blocks the "did I actually fit both lines of the doublet" diagnostic plot for
the simplest (and most common) doublet model.

---

## Bug #3: `Parameter.std` / `.stds` is None after `LevMarLSQFitter`

### The Bug

```python
model_fit = fitter(model, wave, flux, weights=1 / ferr**2)
print(model_fit.Halpha.sigma.stds)      # AttributeError (astropy >= 6)
print(model_fit.Halpha.sigma.std)       # None
```

Recent astropy versions stopped filling `Parameter.std` for `LevMarLSQFitter`.
Reading it gives `None` and the resulting error bars are silently `nan`.

### Correct Code

```python
cov = fitter.fit_info.get("param_cov")
if cov is not None:
    errs = np.sqrt(np.clip(np.diag(cov), 0, None))   # same order as free params
```

`param_cov` covers the **free** parameters only, in model parameter order, so
skip the fixed/tied ones when mapping the array back onto names.

### Detection
- **Symptom**: error bars are `nan` or all equal.
- **Check**: `fitter.fit_info.keys()` contains `param_cov`.

---

## Bug #4: Tying the broad H-beta/H-gamma kinematics to broad H-alpha

### The Bug

```python
for line in ("Hbeta", "Hgamma"):
    model[f"b_{line}"].sigma_c.tied = galspec.tie_MultiGauss_sigma_c("b_halpha")
    model[f"b_{line}"].dv_c.tied    = galspec.tie_MultiGauss_dv_c("b_halpha")
```

Physically appealing (one BLR) and it works when the lines are weak, but on a
high-S/N spectrum it can be catastrophically wrong.  Measured on Ark 120
(X-shooter, S/N ~ 300 per pixel, 4200-7000 A):

| model | chi2/dof | narrow-line H-alpha/H-beta |
|---|---|---|
| H-beta and H-gamma tied to H-alpha | 9.04 | **16.7** (impossible; Case B gives 2.9) |
| each Balmer line free | 5.24 | **3.76** (correct for this reddening) |

The tied model cannot reproduce the different shapes of the H-alpha and H-beta
cores, so it dumps the leftover core flux into the narrow-line component, and
the narrow Balmer decrement becomes nonsense.  The posterior is *sharply*
peaked in this wrong solution, so it looks converged and the error bars are
tiny.

### Detection
- **Symptom**: narrow H-alpha/H-beta far from 2.9 (or far from the reddened
  Case B value); the broad/narrow split moves wildly with small changes to the
  model.
- **Check**: run the same fit with the ties removed and compare chi2/dof *and*
  the narrow Balmer decrement.  Also plot the residuals of the H-beta red wing;
  a tied model typically leaves +-6 sigma residuals there.

### Prevention
Treat "one kinematics for all broad lines" as a **hypothesis to test**, not a
default.  Free, it costs ~8 parameters per additional Balmer line, which is
cheap next to the bias it removes.  Keep the tie only for the weak lines
(He I, He II) where the data cannot constrain the kinematics anyway.

---

## Bug #5: `[O III]` wing tied to the line centroid cannot fit a shoulder

### The Bug

```python
# "a wind component must be at the same velocity as the core"
wing.dv_w0.tied = galspec.tie_MultiGauss_dv_c("OIIIwing")
```

An outflow that is *by definition* displaced cannot share the systemic
centroid.  With the tie the model can only make a symmetric wing, and a real
blue shoulder (very common in [O III] 5007) is fitted by inflating the
symmetric component instead: the residuals then sit at +5 to +10 sigma on the
blue side and -5 sigma on the red side of the same line.

### Correct Code

```python
wing = galspec.Line_MultiGauss_doublet(n_components=2,           # core + 1 wind
                                       amp_c0=0.15, amp_c1=0.15 / 2.98,
                                       dv_c=0.0, sigma_c=500.0,
                                       amp_w0=0.3, dv_w0=0.0, sigma_w0=1500.0,
                                       wavec0=LW["OIII_5007"], wavec1=LW["OIII_4959"])
wing.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("OIIIwing", 2.98)
# dv_w0 and sigma_w0 stay free
```

### Detection
- **Symptom**: alternating +/- residuals across the two lines of the doublet.
- **Check**: print `median((data-model)/err)` in 20 A slices around 4959 and
  5007; a tied symmetric wing shows a sign flip between the two lines.

---

## Bug #6: A single power law hides the telluric residual in the error budget

Not a code bug, a workflow bug, and it cost the most time on Ark 120.

The 1 per cent flux floor of the usual error prescription is *smaller* than
the residual of a molecfit telluric correction in the water bands and at the
onset of the O_2 B band.  Those residuals are not random: they shift a 20-50 A
stretch of pixels by the same 5-20 per cent.  They therefore dominate chi2 and
they cannot be found by looking at the rms.

### Detection
- **Symptom**: chi2/dof of 5-10 that does not improve when the sampler is given
  more live points, with the worst residuals in *continuum* regions.
- **Check**: bin the residuals and look for runs of same-sign pixels; and
  convert the rest wavelength back to the observed frame before blaming the
  object — atmospheric features are fixed in the **observed** frame.

### Fix

Measure the residual scatter in pixels where the model has **no emission
line**, in wavelength bins, and use it as a wavelength-dependent extra floor:

```python
ok = ~any_line_region(wave)                      # explicit rest-frame exclusions
r = (flux - model_best) / error
fac = np.sqrt(np.mean(np.clip(r[bin], -8, 8) ** 2))   # RMS about ZERO, not a robust MAD
error *= np.clip(smoothed_interp(fac), 1.0, 12.0)
```

Use the RMS about zero: a *correlated offset* gives a robust MAD of ~1 and the
inflation is silently skipped, which is exactly how this bug hides.

On Ark 120 this moved chi2/dof from 5.2 to ~1.5 while leaving the formal errors
of the line parameters essentially unchanged, and it correctly identified
Galactic Na I D (rest 5705/5711 in the rest frame, i.e. observed 5890/5896) as
two single-pixel 40-50 per cent deep spikes that had to be masked, not fitted.

---

## Maintenance Notes

**When to add a new bug**:
- You encounter a bug that took >10 minutes to debug
- The bug is subtle and could be easily repeated
- The bug has a clear lesson for future users

**What makes a good bug entry**:
- Clear before/after comparison
- Code examples that can be copied/pasted
- Detection checklist
- Prevention tips

**Version history**:
- 2025-03-20: Initial file with Bug #1 (dv tying bug)
- 2026-02-11: Bug #2 (`subcomponents` is None), #3 (`Parameter.std`), #4 (tied
  broad Balmer kinematics destroy the narrow Balmer decrement), #5 ([O III]
  wing tied to the centroid), #6 (telluric residual vs the error budget) -- all
  found on the Ark 120 X-shooter fit
- 2026-02-11: Bug #7 (`plot_fit` hardcoded `ylim0` clips the line peak; legend
  lists all submodels in every zoom) and #8 (a 0.15 %-wide `fill_between` band
  is invisible yet announced in the legend; a corner plot of a near-delta
  posterior fuses into one blob) -- both found by pixel-measuring the Ark 120
  report figures, not by looking at them

---

## Bug #7: `plot_fit` zooms clip data and repeat every legend entry

`galspec.plot.plot_fit` accepts `xlim`, `ylim0` (flux panel) and `ylim1`
(residual panel) but does **not** adjust them to the window, and its legend
always contains **every** submodel, in every figure.

What went wrong on a 17-panel X-shooter fit:

* a hardcoded `ylim0 = (0.2, 1.4)` on the Hα window clipped the top of a line
  whose peak was 2.73 — invisible in a thumbnail, obvious to a reader;
* the residual panel was left to autoscale, so one 3.8σ point sat exactly on the
  frame;
* each zoom legend had 25 rows while only 8–10 of the lines could appear in that
  window, so the legend covered the plot's meaning.

Recipe:

```python
def panel_limits(wave, flux, model, xlim, pad=0.06):
    sel = (wave >= xlim[0]) & (wave <= xlim[1])
    y = np.concatenate([np.asarray(flux)[sel], np.asarray(model)[sel]])
    y = y[np.isfinite(y)]
    lo, hi = float(y.min()), float(y.max())
    d = hi - lo if hi > lo else abs(hi) * 0.1 + 1e-3
    return lo - pad * d, hi + pad * d

r = (flux - model(wave)) / error                      # residual panel
lim = 1.12 * np.nanmax(np.abs(r)) + 0.2               # symmetric, never clipped
ax, axr = galspec.plot.plot_fit(wave, flux, model, ax=ax, axr=axr,
                                xlim=xlim, ylim0=panel_limits(...),
                                ylim1=(-lim, lim), legend_kwargs=leg_kw)
# then rebuild the legend with only the lines inside this window
leg = ax.get_legend()
handles = getattr(leg, "legend_handles", None) or leg.legendHandles
keep = [(h, t.get_text()) for h, t in zip(handles, leg.get_texts())
        if t.get_text() in ALWAYS or in_window(LEGEND_WAVE.get(t.get_text()), xlim)]
leg.remove(); ax.legend([h for h, _ in keep], [t for _, t in keep], **leg_kw)
```

Never hardcode `ylim0`: the same window changes as soon as the model improves.

## Bug #8: an invisible uncertainty band and a fused corner plot

A subagent with no image input measured the finished PNGs (ink bounding boxes,
edge ink, per-panel colour clusters, OCR). Two defects were only visible that
way:

* **An announced uncertainty band that is not on the page.**
  `ax.fill_between(v, lo, hi, alpha=0.5)` with a posterior that is accurate to
  0.15 % draws a 0.7-pixel ribbon. The legend says "16–84 %" and the reader sees
  nothing. Fix: put the *measured* width in the label —
  `label=f"total, 16-84% band ({np.max(hi-lo)/peak*100:.2f}% of peak)"` — or plot
  the residual of the band in an inset instead.
* **A corner plot of a near-delta posterior.** 20 000 scatter points in a 220-px
  panel fuse into one 1 210-px blob: 25–38 % ink, no visible structure, rotated
  tick labels. Use filled contours instead of points, and thin the chain:

  ```python
  corner.corner(samp, bins=45, smooth=0.9, plot_datapoints=False,
                plot_density=False, plot_contours=True, fill_contours=True,
                levels=(0.393, 0.864, 0.989))
  ```

The general check, cheap enough to run every time: `edge_ink = 0` (no ink within
3 px of any canvas edge), and one ink bounding box per figure with margins of at
least ~10 px.

## Bug #9: measuring a soft tie against the reference *shift* instead of its *width*

A centred prior on a velocity shift is written as

```python
model.set_prior("b_hbeta.dv_c",
                priors.Normal(mean="b_halpha.dv_c", sigma=0.05,
                              scale_from="b_halpha.sigma_c", nsigma=5))
```

so one tie width is `0.05 * sigma_Halpha`, **not** `0.05 * |dv_Halpha|`. The
centre of a shift prior is a shift, and shifts are often near zero: measuring the
posterior difference against `0.05 * |dv_Halpha|` with `dv_Halpha ~ 200 km/s`
gives a tie width of 10 km/s instead of 110 km/s and inflates every pull by an
order of magnitude.

The symptom is a pull table full of values like +8.6 and +15.8 flagged "at cap"
for parameters whose posterior difference is a perfectly reasonable 150 km/s, on
a run whose χ²/dof is good. When a pull looks absurd, recompute the tie width by
hand from the prior, not from the parameter you think it was tied to:

```python
tie_width = shift_fraction * np.abs(sigma_of_same_component)   # not dv!
pull = (value - centre) / tie_width
```

Same trap for `relative=`: `Normal(mean="x.sigma_c", relative=0.20)` is 20 % of
the *centre's* value, which is right for widths and wrong for anything that can
be zero.

## Bug #10: gating every line separately makes "component *i*" meaningless

The pattern that looks natural —

```python
# five components per line, gate all of them, pair component i to component i
for line in ("halpha", "hbeta", "hgamma"):
    for i in range(4):
        model.set_prior(f"b_{line}.amp_w{i}", SpikeAndSlab(p_off=P_OFF[i]))
for line in ("hbeta", "hgamma"):
    for i in range(4):
        model.set_prior(f"b_{line}.sigma_w{i}",
                        Normal(mean=f"b_halpha.sigma_w{i}", relative=0.2, nsigma=5))
```

— is **not** the "same gas in all three lines" model that it looks like. Each line
gates independently, so each ends up with a different subset of wings on (in Ark
120: Hα wings 0,2; Hβ wings 1,2,3; Hγ wing 0 — no wing shared by all three), and
a wing Hα switches off still has a *prior-driven* width which then serves as the
centre of Hβ's tie. Two consequences:

1. Where the indices disagree, the tie compares **unrelated Gaussians**, and a
   healthy-looking pull table is then evidence about the width *distribution*,
   not about a physical match.
2. Amber flag: the amplitude floor is usually fractional to each line's own core.
   The Balmer decrement makes a common component relatively *larger* in the
   weaker line, so a shared component must be **at least as easy** to keep in
   Hβ as in Hα. If Hα keeps a wing that Hβ switches off, they are not the same
   component — that is a result, and it means the extra Gaussians are describing
   each line's own blend (Fe II, the [O III] wing, [N II]), not bulk gas.

If the physics requires one shared decomposition, the options are: gate the
reference line only and accept that the others keep components whose kinematics
are tied to possibly-absent references; hard-tie the amplitudes (which then forces
one Balmer decrement on every component); or fit the plain model at the count the
gate posterior chose and quote the *totals* as the measurement.

Also: at the resolution of the posterior, `P(on)` saturates at 0 or 1 and
`BF = (P_on/P_off) / odds` becomes `inf`/`0`. Quote a bound instead of claiming
infinite evidence:

```python
n = amps.shape[0]                      # equal-weight samples
if p_on >= 1 - 0.5 / n:   bf = (n - 1) / odds      # "BF >" this
elif p_on <= 0.5 / n:     bf = 1 / ((n - 1) * odds)  # "BF <" this
```

## Bug #11: adding a component by narrowing the old one's prior

To "improve the model" you add a second Gaussian to a line.  The temptation is to
give the two components *disjoint* prior boxes so the labels are physical —
component 0 narrow, component 1 broad:

```python
# WRONG: the two-component model no longer contains the one-component model
"Line.dv_c":    (-600.0, 600.0)     if n_comp > 1 else (-1500.0, 1500.0),
"Line.sigma_c": (250.0, 1200.0)     if n_comp > 1 else (150.0, 3000.0),
"Line.amp_w0":        (0.0, 1.0),
"Line.sigma_w0":      (1000.0, 4000.0),     # disjoint from sigma_c
```

Two things go wrong at once:

1. **Nesting is destroyed.**  A one-component solution with σ_c = 260 km/s is no
   longer reachable in the two-component model if the box starts at 250 *and the
   data want 150*, and a solution with σ_c = 2500 is excluded outright.  The
   evidence Δln Z is then not a statement about "is a second component needed",
   because the models are not nested and the prior volumes are not comparable.
2. **Both components end up on box edges.**  The data do not respect your idea of
   which component is broad: they push component 0 to the lower edge of its width
   box and the wing to one of its own.  Inspection shows σ_c = 254 against a
   floor of 250 and σ_w0 = 1005 against a floor of 1000 — every parameter at an
   edge means nothing is measured, and a χ² gain bought this way is an artefact.

The rule: **the finer model must contain the coarser one.**  Keep every parameter
of the existing component on exactly its old prior and let the new component die
by its own amplitude:

```python
"Line.sigma_c": (150.0, 3000.0),        # unchanged, as in the one-component model
"Line.dv_c":    (-1500.0, 1500.0),      # unchanged
"Line.amp_w0":  (0.0, 1.0),             # relative to the core: amp_w0 -> 0 removes it
"Line.sigma_w0": (150.0, 8000.0),       # anything the data could want
"Line.dv_w0":    (-4000.0, 4000.0),
```

Label ambiguity is the price, and it is the cheap one: with overlapping boxes the
two Gaussians can swap roles, so quote the *count* and the *total*, never "the
broad component has σ = ...".

Two diagnostics that would have caught this in seconds:

* After a fit, check every parameter's posterior against its own bounds.  A
  component whose width sits within 1 % of a box edge is pinned by the prior.
* **A nested model must never fit worse.**  If the finer model's best χ² is
  larger than the coarser model's, the run has not converged and its log Z is
  meaningless.  (Here: adding three parameters made χ² worse by 330 and log Z
  "drop" by 152 — a sampling failure reported as decisive evidence against a
  component that the gated fit then kept at 9.5 % of the core.)

And when the extra component is *marginal*, a plain `Uniform(0, 1)` amplitude is
the wrong instrument: its plateau near zero holds most of the prior volume, the
sampler spends its live points there, and `dlogz` converges on it.  That is what
`SpikeAndSlab` is for.

## 13. A label computed from a different array than the one plotted

**Symptom.** A legend says `core (FWHM 2)`, and the curves look right.  Nothing
crashes, so nothing points at the label.

**Cause.** The label was built from the wrong quantity — the peak flux of the
median profile, a sigma in Å where km s⁻¹ was meant, a median of a parameter
instead of the median of the *drawn* curve: `np.median(np.percentile(comps, 50,
axis=0).max())`, which is a peak, labelled FWHM.

**Fix.** Measure what you draw, from the array you draw:

```python
core = np.median(d["comps"]["core"], axis=0)      # exactly what goes to ax.plot
above = np.where(core >= core.max() / 2.0)[0]
fwhm = velc[above[-1]] - velc[above[0]]           # what the eye measures too
```

**Why it matters more than it looks.** Every other number in the report is checked
against the posterior, and this one is checked against the figure — so a
mis-derived label is the one class of error that survives all the arithmetic and
is caught (if at all) by a human reading the plot.  Print labels and values from
one source, and gap-check every figure legend against its table row before
shipping.
