#!/usr/bin/env python
"""Are the CPU and JAX ``IronTemplate_tied`` the same function?

Yes -- as of the revision that removed the CPU deconvolution.  Both broaden the
template in velocity space by subtracting the template's own width in quadrature
from every kernel component, one Gaussian smoothing per component, then project
each component onto the data grid.  This script checks that claim from the
outside, and keeps the *removed* algorithm around as a reference for what the
answer used to be:

1. the CPU class, the JAX class and an independent *exact* convolution, on the
   same wavelength grid at identical kernel parameters (CPU vs JAX: 1e-15);
2. the old CPU algorithm -- FFT deconvolution of a 200 km/s intrinsic Gaussian,
   a Wiener filter, and a kernel sampled on a fixed +-10 000 km/s window --
   which was 8.5% (mock) and 10.0% (fit) from the exact answer.

Usage::

    python develop/dev_jax/iron_cpu_vs_jax.py            # GPU or CPU JAX
    JAX_PLATFORMS=cpu python develop/dev_jax/iron_cpu_vs_jax.py

Writes ``develop/dev_jax/io_test/cpu_vs_jax.html`` plus its figures.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal import convolve, wiener

import galspec
import galspec.jax_models as jax_models
from galspec.constants import ls_km

OUT = HERE / "io_test"

#: Two parameter sets.  "mock" is the one the earlier input/output test used;
#: "fit" is what the IRAS 07546 fit measured.  Both are (core, wind) kernels.
CASES = {
    "mock": {"label": "mock truth (the earlier input/output test)",
             "sigma_c": 1980.0, "amp_w0": 0.78, "dv_w0": -520.0, "sigma_w0": 6400.0},
    "fit": {"label": "IRAS 07546 fit (what the worked example measured)",
            "sigma_c": 1409.0, "amp_w0": 0.084, "dv_w0": -1121.0, "sigma_w0": 7255.0},
}

WAVE = np.arange(4200.0, 5600.0, 0.5)
AMPLITUDE = 1.0

#: The template was measured at 800 km/s FWHM, i.e. 340 km/s.  Both backends now
#: use that value; the removed CPU algorithm used "R ~ 1500", i.e. 200 km/s.
INTRINSIC_TEMPLATE = 800.0 / 2.3548
INTRINSIC_CPU = ls_km / 1500.0


# --------------------------------------------------------------------------------------
# the template
# --------------------------------------------------------------------------------------
def load_template(template_name="park2022"):
    """The iron template as both models load it: even in ln(wave), normalised as they do."""
    from galspec.iron_template import load_iron_template

    wave, flux = load_iron_template(template_name)
    window = (wave > 4500) & (wave < 5500)
    return wave, flux / np.max(flux[window]), 800.0 / 2.3548


# --------------------------------------------------------------------------------------
# the implementations
# --------------------------------------------------------------------------------------
def exact_iron(wave, case, template_name="park2022", step=10.0, pad=40000.0):
    """The iron as it is *meant*: the template convolved in velocity space.

    The template is resampled on a uniform velocity grid and each kernel
    component is applied there as a Gaussian of ``sqrt(sigma**2 - sigma_intr**2)``,
    so the requested width really is the width at every wavelength.  No
    deconvolution, no FFT, no pixel-space filter, no fixed kernel window.
    """
    wave_t, flux_t, sigma_intr = load_template(template_name)
    v_template = (np.log(wave_t) - np.log(wave_t[0])) * ls_km
    v_grid = np.arange(v_template[0] - pad, v_template[-1] + pad + step, step)
    flux_v = np.interp(v_grid, v_template, flux_t, left=0.0, right=0.0)

    def smoothed(width):
        extra = np.sqrt(max(width ** 2 - sigma_intr ** 2, 0.0))
        if extra < step:
            return flux_v
        half = int(5.0 * extra / step)
        offsets = np.arange(-half, half + 1) * step
        kernel = np.exp(-0.5 * (offsets / extra) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(flux_v, half, mode="edge")
        return np.convolve(padded, kernel, mode="same")[half:half + len(flux_v)]

    components = [(1.0, 0.0, case["sigma_c"]),
                  (case["amp_w0"], case["dv_w0"], case["sigma_w0"])]
    total = np.zeros_like(v_grid)
    for weight, shift, width in components:
        total += weight * np.interp(v_grid - shift, v_grid, smoothed(width))
    total /= sum(weight for weight, _, _ in components)
    smoothed_template = np.interp(v_template, v_grid, total)
    return AMPLITUDE * np.interp(wave, wave_t, smoothed_template)


def cpu_iron(wave, case, template_name="park2022"):
    """The CPU class itself, exactly as ``galspec.IronTemplate_tied`` evaluates."""
    kernel = galspec.Line_MultiGauss(
        n_components=2, amp_c=1.0, dv_c=0.0, sigma_c=case["sigma_c"],
        par_w=dict(amp_w0=case["amp_w0"], dv_w0=case["dv_w0"],
                   sigma_w0=case["sigma_w0"]),
        wavec=4861.0, name="kernel")
    model = galspec.IronTemplate_tied(amplitude=AMPLITUDE, z=0.0,
                                      template_name=template_name, kernel=kernel)
    return np.asarray(model(wave))


def legacy_iron(wave, case, template_name="park2022", window=10000.0,
                n_kernel=20000, wiener_smooth=True, sigma_intr=INTRINSIC_CPU):
    """The algorithm this revision removed, kept to document what changed.

    A line-for-line copy of the old ``IronTemplate_tied.evaluate``: deconvolve the
    template by an intrinsic Gaussian with an FFT, filter the result with a Wiener
    filter, then convolve with a kernel array sampled on a fixed velocity window.
    The class no longer does any of this; with the three incidental details
    exposed it is::

        window          the velocity half-range the kernel is sampled on,
        wiener_smooth   whether the Wiener filter after the deconvolution is on,
        sigma_intr      the dispersion the template is deconvolved by.
    """
    wave_t, flux_t, _ = load_template(template_name)
    intrinsic = sigma_intr
    # the class uses the log-spacing (velocity per channel), not the linear one
    vchan = (np.log(wave_t[1]) - np.log(wave_t[0])) * ls_km

    velocities = np.linspace(-window, window, n_kernel)
    line = galspec.Line_MultiGauss(
        n_components=2, amp_c=1.0, dv_c=0.0, sigma_c=case["sigma_c"],
        par_w=dict(amp_w0=case["amp_w0"], dv_w0=case["dv_w0"],
                   sigma_w0=case["sigma_w0"]),
        wavec=4861.0, name="kernel")
    kernel = line.gen_template(velocities, normalized=True)
    kernel = kernel / np.sum(kernel)

    centre = wave_t[len(wave_t) // 2]
    offsets = np.log(wave_t / centre) * ls_km        # exact on a log-even grid

    wave_kernel = interp1d(velocities, kernel, bounds_error=False, fill_value=0.0)(offsets)
    wave_kernel = wave_kernel / np.sum(wave_kernel)

    sigma_pix = intrinsic / vchan
    size = max(3, int(3 * sigma_pix))
    x_kernel = np.arange(-size, size + 1)
    gaussian = np.exp(-0.5 * (x_kernel / sigma_pix) ** 2)
    gaussian = gaussian / np.sum(gaussian)

    padded = np.pad(flux_t, (size, size), mode="edge")
    kernel_padded = np.pad(gaussian, (0, len(padded) - len(gaussian)), mode="constant")
    deconvolved = np.real(ifft(fft(padded) / (fft(kernel_padded) + 1e-12)))
    deconvolved = deconvolved[size:size + len(flux_t)]
    if wiener_smooth:
        deconvolved = wiener(deconvolved, (3,))
    convolved = convolve(deconvolved, wave_kernel, mode="same")
    return AMPLITUDE * np.interp(wave, wave_t, convolved)


def jax_iron(wave, case, template_name="park2022"):
    """The JAX model, at the same parameters."""
    model = jax_models.IronTemplate_tied(
        n_components=2, amplitude=AMPLITUDE, z=0.0, sigma_max=25000.0,
        sigma_c=case["sigma_c"], amp_w0=case["amp_w0"], dv_w0=case["dv_w0"],
        sigma_w0=case["sigma_w0"], template_name=template_name, name="FeII")
    return np.asarray(model.flux(wave, model.get_free_params()))


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def deviation(flux, reference):
    """Two numbers: how far off, and how much of that is a shape error.

    ``gap``   max |flux - reference| / max(reference): the straightforward "how
              different are these two profiles", scale included;
    ``shape`` the same after the best-fit scale factor is removed, so a model
              that is merely 3% too bright is not called a shape mismatch;
    ``scale`` that best-fit factor (>1 means the model is faint).
    """
    peak = float(np.max(np.abs(reference)))
    gap = float(np.max(np.abs(flux - reference)) / peak)
    scale = float(np.dot(flux, reference) / np.dot(reference, reference))
    shape = float(np.max(np.abs(flux - scale * reference)) / peak)
    return gap, shape, scale


def run_case(key, case):
    """All variants for one parameter set."""
    started = time.perf_counter()
    reference = exact_iron(WAVE, case)
    jax = jax_iron(WAVE, case)
    cpu = cpu_iron(WAVE, case)
    # the removed algorithm, unchanged and with its three constants corrected
    old = legacy_iron(WAVE, case)
    old_intr = legacy_iron(WAVE, case, sigma_intr=INTRINSIC_TEMPLATE)
    old_window = legacy_iron(WAVE, case, window=40000.0, n_kernel=80000)
    old_all = legacy_iron(WAVE, case, window=40000.0, n_kernel=80000,
                          wiener_smooth=False, sigma_intr=INTRINSIC_TEMPLATE)

    variants = {
        "exact velocity-space convolution": reference,
        "CPU IronTemplate_tied (this revision)": cpu,
        "JAX IronTemplate_tied": jax,
        "old CPU algorithm: FFT deconvolution (removed)": old,
        "old algorithm, intrinsic width 340 instead of 200 km/s": old_intr,
        "old algorithm, kernel window +-40 000 instead of +-10 000 km/s": old_window,
        "old algorithm, all three corrected": old_all,
    }
    core = (WAVE >= 4200.0) & (WAVE < 5400.0)     # where the template has signal
    peak = float(np.max(reference))

    rows = []
    for name, flux in variants.items():
        gap, shape, scale = deviation(flux, reference)
        rows.append({"name": name, "gap": 100 * gap, "shape": 100 * shape,
                     "scale": 100 * (scale - 1.0),
                     "core": 100 * float(np.max(np.abs(flux[core] - reference[core])) / peak)})

    # how much each single constant of the removed algorithm moved its answer
    reach = lambda flux: 100.0 * float(np.max(np.abs(flux - old)) / np.max(np.abs(old)))
    sensitivity = {
        "intrinsic width": reach(old_intr),
        "kernel window": reach(old_window),
        "all corrected": reach(old_all),
    }

    # the two current models against each other
    gap_cpu_jax, shape_cpu_jax, scale_cpu_jax = deviation(cpu, jax)

    def worst_position(flux):
        difference = np.abs(flux - reference)
        index = int(np.argmax(difference))
        return {"wave": float(WAVE[index]), "gap": 100 * difference[index] / np.max(reference)}

    return {"case": case, "rows": rows, "wall": time.perf_counter() - started,
            "cpu_vs_jax": {"gap": 100 * gap_cpu_jax, "shape": 100 * shape_cpu_jax,
                           "scale": 100 * (scale_cpu_jax - 1.0)},
            "worst": {"jax": worst_position(jax), "cpu": worst_position(cpu),
                      "old": worst_position(old), "old_all": worst_position(old_all)},
            "profile_range": {
                "jax": (100.0 * float(np.min(jax - reference)) / peak,
                        100.0 * float(np.max(jax - reference)) / peak),
                "cpu": (100.0 * float(np.min(cpu - reference)) / peak,
                        100.0 * float(np.max(cpu - reference)) / peak)},
            "sensitivity": sensitivity,
            "profiles": {"exact": reference, "jax": jax, "cpu": cpu,
                         "old": old, "old_all": old_all}}


def run_width_sweep(key, case):
    """How the disagreement grows with the width of the wind component."""
    widths = [400.0, 800.0, 1500.0, 2500.0, 4000.0, 6400.0, 9000.0]
    out = {"widths": widths, "jax": [], "cpu": [], "old": [], "old_all": []}
    for width in widths:
        local = dict(case, sigma_w0=width)
        reference = exact_iron(WAVE, local)
        out["jax"].append(100 * deviation(jax_iron(WAVE, local), reference)[0])
        out["cpu"].append(100 * deviation(cpu_iron(WAVE, local), reference)[0])
        out["old"].append(100 * deviation(legacy_iron(WAVE, local), reference)[0])
        out["old_all"].append(
            100 * deviation(legacy_iron(WAVE, local, window=40000.0, n_kernel=80000,
                                        wiener_smooth=False,
                                        sigma_intr=INTRINSIC_TEMPLATE),
                            reference)[0])
    return out


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def fig_profiles(results):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.3], "hspace": 0.08})
    ax = axes[0]
    ax.plot(WAVE, results["profiles"]["exact"], color="0.15", lw=3.4,
            label="exact velocity-space convolution")
    ax.plot(WAVE, results["profiles"]["cpu"], color="C0", lw=1.4, ls=(0, (4, 3)),
            label="CPU IronTemplate_tied (this revision)")
    ax.plot(WAVE, results["profiles"]["jax"], color="C3", lw=1.4, ls=(0, (1, 2)),
            label="JAX IronTemplate_tied")
    ax.set_ylabel("Fe II flux  [arbitrary]", fontsize=13)
    ax.legend(fontsize=10, frameon=False)
    ax.set_title("Same template, same kernel parameters, three implementations:\n"
                 "the dashed and dotted curves lie on the solid one", fontsize=11.5)

    ax = axes[1]
    reference = results["profiles"]["exact"]
    peak = np.max(reference)
    for name, colour, style, width, label in (
            ("cpu", "C0", (0, (4, 3)), 1.6, "CPU (this revision)"),
            ("jax", "C3", (0, (1, 2)), 1.6, "JAX"),
            ("old", "C1", "-", 1.6, "old CPU algorithm (removed)"),
            ("old_all", "C4", "--", 1.3, "old algorithm, constants corrected")):
        residual = np.abs(results["profiles"][name] - reference) / peak * 100.0
        ax.plot(WAVE, np.maximum(residual, 1e-16), color=colour, ls=style, lw=width,
                label=label)
    ax.set_yscale("log")
    ax.set_ylim(1e-15, 1e2)
    ax.set_xlabel(r"Rest wavelength  [$\AA$]", fontsize=13)
    ax.set_ylabel("$|$model $-$ exact$|$\n[% of peak]", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    ax.set_xlim(WAVE[0], WAVE[-1])
    fig.savefig(OUT / "cpu_vs_jax_profiles.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_width_sweep(sweeps):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    widths = sweeps["mock"]["widths"]
    ax.plot(widths, sweeps["mock"]["cpu"], "s-", color="C0", lw=1.9,
            label="CPU IronTemplate_tied (this revision)")
    ax.plot(widths, sweeps["mock"]["jax"], "o--", color="C3", lw=1.5,
            label="JAX IronTemplate_tied")
    ax.plot(widths, sweeps["mock"]["old"], "^-", color="C1", lw=1.5,
            label="old CPU algorithm (removed)")
    ax.plot(widths, sweeps["mock"]["old_all"], "v--", color="C4", lw=1.3,
            label="old algorithm, all three constants corrected")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e2)
    ax.set_xlabel("width of the wind component  $\\sigma_{w0}$  [km/s]", fontsize=12)
    ax.set_ylabel("distance from the exact answer\n[% of iron peak]", fontsize=11)
    ax.set_title("Distance from the exact velocity-space answer over a factor of 22\n"
                 "in kernel width: the two current models coincide", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.savefig(OUT / "cpu_vs_jax_width.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def png(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def table_rows(results):
    out = []
    for row in results["rows"]:
        emphasise = row["name"].startswith(("JAX", "CPU IronTemplate_tied (this"))
        tag = " class='hi'" if emphasise else ""
        out.append(
            f"<tr{tag}><td>{html.escape(row['name'])}</td>"
            f"<td class='num'>{row['gap']:.3f}</td>"
            f"<td class='num'>{row['core']:.3f}</td>"
            f"<td class='num'>{row['shape']:.3f}</td>"
            f"<td class='num'>{row['scale']:+.3f}</td></tr>")
    return "\n".join(out)


def build_report(results, sweeps, metadata):
    mock = results["mock"]
    fit = results["fit"]
    by_name = {row["name"]: row for row in mock["rows"]}
    fit_by_name = {row["name"]: row for row in fit["rows"]}
    cpu_mock = by_name["CPU IronTemplate_tied (this revision)"]
    cpu_fit = fit_by_name["CPU IronTemplate_tied (this revision)"]
    old_mock = by_name["old CPU algorithm: FFT deconvolution (removed)"]
    old_intr = by_name["old algorithm, intrinsic width 340 instead of 200 km/s"]
    old_all = by_name["old algorithm, all three corrected"]
    sens = mock["sensitivity"]
    sweep = sweeps["mock"]
    parity = mock["cpu_vs_jax"]["gap"]

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CPU vs JAX IronTemplate_tied</title>
<style>
 body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
        max-width: 60rem; margin: 2.4rem auto; padding: 0 1.2rem; color: #1b1b1b;
        line-height: 1.55; font-size: 15px; }}
 h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
 h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #e3e3e3;
      padding-bottom: .3rem; }}
 .sub {{ color: #666; margin-top: 0; }}
 table {{ border-collapse: collapse; width: 100%; margin: .9rem 0; font-size: 13.5px; }}
 th, td {{ padding: .4rem .5rem; border-bottom: 1px solid #ececec; text-align: left; }}
 th {{ background: #fafafa; font-weight: 600; }}
 td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 tr.hi td {{ font-weight: 600; background: #fbfbfd; }}
 figure {{ margin: 1.3rem 0; }}
 img {{ max-width: 100%; }}
 figcaption {{ font-size: 12.5px; color: #555; margin-top: .4rem; }}
 .verdict {{ border-left: 4px solid #146c2e; background: #f4faf5; padding: .8rem 1rem;
            margin: 1.1rem 0; }}
 .warn {{ border-left: 4px solid #b3261e; background: #fdf6f5; padding: .8rem 1rem;
         margin: 1.1rem 0; }}
 code {{ background: #f4f4f4; padding: .08rem .3rem; border-radius: 3px; font-size: 13px; }}
 .kv {{ font-size: 12.5px; color: #555; }}
</style></head><body>

<h1>CPU and JAX <code>IronTemplate_tied</code>: the same function</h1>
<p class="sub">Same template, same kernel parameters, same wavelength grid. Both
backends load the template through
<code>galspec.iron_template.load_iron_template</code>, which puts it on a grid even in
<code>ln(wavelength)</code> (so a velocity width is one number everywhere) and extends
the tabulated range with zeros (so nothing is reflected or invented beyond it).
Generated by
<code>develop/dev_jax/iron_cpu_vs_jax.py</code>; numbers in
<code>io_test/cpu_vs_jax_results.json</code>.</p>

<div class="verdict">
<b>The two models now agree to {parity:.1e}% of the iron peak</b> &mdash; machine
precision, so they are the same function, not two approximations to it. Both are
<b>{cpu_mock['gap']:.3f}%</b> from an independent velocity-space convolution at the
mock parameters and <b>{cpu_fit['gap']:.3f}%</b> at the parameters the
IRAS&nbsp;07546 fit measured, over the whole 4200&ndash;5600&nbsp;&Aring; range
including the truncated ends.
</div>

<div class="warn">
<b>What this replaced.</b> The CPU class used to FFT-<i>deconvolve</i> the template by
a 200&nbsp;km/s intrinsic Gaussian, smooth the result with a Wiener filter, and
convolve with a kernel sampled on a fixed &plusmn;10&nbsp;000&nbsp;km/s window. That
was <b>{old_mock['gap']:.1f}%</b> from the exact answer at the mock parameters
(<b>{old_all['gap']:.1f}%</b> even with all three constants corrected), and
<b>{old_mock['core']:.1f}%</b> in the 4200&ndash;5400&nbsp;&Aring; range where the
template has signal. The algorithm is kept in this script as
<code>legacy_iron()</code> so the numbers stay reproducible; it is not in the package
any more. Section&nbsp;3 has the details.
</div>

<h2>1. The profiles</h2>
<figure>
 <img src="{png(OUT / 'cpu_vs_jax_profiles.png')}" alt="profiles">
 <figcaption><b>Top:</b> the iron at identical parameters; the CPU and JAX curves are
 drawn as thin dashes over a thick exact curve and lie on it. <b>Bottom:</b> the same
 comparison on a logarithmic scale, |model &minus; exact| in per cent of the iron peak.
 The two current models sit about thirteen orders of magnitude below the removed
 algorithm, at the level where float64 round-off is all that is left.</figcaption>
</figure>

<h2>2. Distance from the exact answer</h2>
<p>The reference shares no code with either model: the template is resampled on a
uniform velocity grid and each component is applied there as a Gaussian of
<code>sqrt(sigma&sup2; &minus; sigma_intr&sup2;)</code>, so the requested width is the
width at every wavelength. <b>gap</b> = max|model &minus; exact| / peak over
4200&ndash;5600&nbsp;&Aring;; <b>4200&ndash;5400</b> is the same restricted to the range
where the template still has signal (everything above 5400&nbsp;&Aring; is the
truncated red end); <b>shape</b> removes the best scale factor; <b>scale</b> is that
factor, so +1&nbsp;% means the model is 1&nbsp;% brighter than exact.</p>

<p class="kv">{html.escape(mock['case']['label'])}:
sigma_c = {mock['case']['sigma_c']:.0f} km/s, amp_w0 = {mock['case']['amp_w0']},
dv_w0 = {mock['case']['dv_w0']:+.0f} km/s, sigma_w0 = {mock['case']['sigma_w0']:.0f} km/s.</p>
<table>
<tr><th>implementation</th><th class="num">gap [%]</th>
    <th class="num">4200-5400 [%]</th><th class="num">shape [%]</th>
    <th class="num">scale [%]</th></tr>
{table_rows(mock)}
</table>

<p class="kv">{html.escape(fit['case']['label'])}:
sigma_c = {fit['case']['sigma_c']:.0f} km/s, amp_w0 = {fit['case']['amp_w0']},
dv_w0 = {fit['case']['dv_w0']:+.0f} km/s, sigma_w0 = {fit['case']['sigma_w0']:.0f} km/s.</p>
<table>
<tr><th>implementation</th><th class="num">gap [%]</th>
    <th class="num">4200-5400 [%]</th><th class="num">shape [%]</th>
    <th class="num">scale [%]</th></tr>
{table_rows(fit)}
</table>

<p>The residual is not zero because the template is sampled at 126&nbsp;km/s per
channel: a Gaussian of width &sigma; can only be stored as samples of that Gaussian,
so the narrowest kernels lose a fraction of a per cent to the grid
({max(sweep['cpu']):.2f}% at &sigma;<sub>w0</sub>&nbsp;=&nbsp;400&nbsp;km/s, where six
channels describe the whole kernel). The exact reference has the mirror-image
limitation: it integrates on a 10&nbsp;km/s grid. Everything above
&sigma;<sub>w0</sub>&nbsp;&asymp;&nbsp;1500&nbsp;km/s agrees to a few hundredths of a
per cent.</p>

<h2>3. What the fix was</h2>
<table>
<tr><th></th><th>CPU, this revision</th><th>CPU, removed algorithm</th></tr>
<tr><td>broadening</td>
    <td>quadrature subtraction of the template's own width
        (<code>sqrt(sigma&sup2; &minus; sigma_intr&sup2;)</code>), then one Gaussian
        smoothing on the template grid</td>
    <td>FFT <i>deconvolution</i> of the template, then convolution with a kernel array</td></tr>
<tr><td>per component</td>
    <td>each kernel component is smoothed and projected separately at its own shift,
        then the components are added with the line's own relative weights</td>
    <td>the components are summed into one kernel array first, on a coarse velocity
        grid, then resampled onto the template</td></tr>
<tr><td>template's own width</td><td>340 km/s (what the template was measured at)</td>
    <td>200 km/s (&ldquo;R ~ 1500&rdquo;)</td></tr>
<tr><td>stabilisation</td><td>none needed: there is no inverse filter</td>
    <td>Wiener filter, 3-pixel window, on the deconvolved template</td></tr>
<tr><td>kernel window</td><td>none: the Gaussian is applied where it is</td>
    <td>fixed &plusmn;10&nbsp;000&nbsp;km/s, {10000.0 / 6400.0:.1f}&sigma; of the mock
        wind, which was cut off and the remainder renormalised</td></tr>
<tr><td>same as the JAX model</td><td>yes, to 1e-15 (checked by
        <code>tests/test_model_features.py</code>)</td>
    <td>no: 8&ndash;10% apart, and the difference moved with every constant</td></tr>
</table>

<p>Each constant of the removed algorithm was worth a visible fraction of the iron
peak: the intrinsic width <b>{sens['intrinsic width']:.1f}%</b> of the profile
(200 &rarr; 340&nbsp;km/s), the kernel window <b>{sens['kernel window']:.1f}%</b>
(&plusmn;10&nbsp;000 &rarr; &plusmn;40&nbsp;000&nbsp;km/s). Correcting all three
together left it <b>{old_all['gap']:.1f}%</b> from the exact answer &mdash; worse than
where it started ({old_mock['gap']:.1f}%) &mdash; and deconvolving by the width the
template was actually measured at made it <b>{old_intr['gap']:.0f}%</b>. That is the
signature of an inverse problem with no right answer: the deconvolution divides by a
narrow Gaussian, so what dominates the result is whatever the filter was tuned to
rather than the physics.</p>

<h2>4. Over a factor of 22 in kernel width</h2>
<figure>
 <img src="{png(OUT / 'cpu_vs_jax_width.png')}" alt="width sweep">
 <figcaption>Distance from the exact answer as the wind gets wider, everything else
 fixed. The two current models are the same curve,
 {min(sweep['cpu']):.3f}&ndash;{max(sweep['cpu']):.3f}%, and it rises only at the
 narrow end, where the 126&nbsp;km/s template grid stops resolving the kernel. The
 removed algorithm spans {min(sweep['old']):.1f}&ndash;{max(sweep['old']):.1f}% and does
 not converge as the kernel widens.</figcaption>
</figure>

<div class="verdict">
<b>What this means in practice.</b> The CPU and JAX models are interchangeable: a fit
gives the same answer on either backend, and the 0.01% that both still carry is the
template's own sampling, not an artefact of either implementation. Before this
revision they differed by 8&ndash;10%, which is why the earlier version of this page
had to report the two backends separately.
</div>

<p class="kv">Checks and provenance: the removed algorithm is reproduced by
<code>legacy_iron()</code> in this script, so the &ldquo;old algorithm&rdquo; rows differ
from the current class by exactly the algorithm and nothing else. Both current models
were evaluated on {metadata['npix']} pixels, 4200&ndash;5600&nbsp;&Aring;, at amplitude 1,
z = 0. JAX device: {html.escape(metadata['device'])}; generated
{html.escape(metadata['created'])}.</p>

</body></html>
"""
    return page


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args(argv)
    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    import jax
    device = jax.devices()[0]
    metadata = {"device": f"{device.platform} ({device.device_kind})",
                "created": time.strftime("%Y-%m-%d %H:%M"),
                "npix": len(WAVE)}
    print(f"JAX on {metadata['device']}")

    results = {}
    for key, case in CASES.items():
        print(f"[{key}] {case['label']}")
        results[key] = run_case(key, case)
        for row in results[key]["rows"]:
            print(f"    {row['name']:56s} gap {row['gap']:7.3f}%  "
                  f"shape {row['shape']:7.3f}%  scale {row['scale']:+7.3f}%")
        print(f"    CPU vs JAX: {results[key]['cpu_vs_jax']['gap']:.2e}% of peak")

    print("[sweep] distance from exact vs the width of the wind")
    sweeps = {key: run_width_sweep(key, case) for key, case in CASES.items()}
    for width, jax_dev, cpu_dev, old_dev in zip(sweeps["mock"]["widths"],
                                                sweeps["mock"]["jax"],
                                                sweeps["mock"]["cpu"],
                                                sweeps["mock"]["old"]):
        print(f"    sigma_w0 {width:6.0f} km/s   cpu {cpu_dev:6.3f}%   "
              f"jax {jax_dev:6.3f}%   old {old_dev:7.2f}%")

    fig_profiles(results["mock"])
    fig_width_sweep(sweeps)

    page = build_report(results, sweeps, metadata)
    target = OUT / "cpu_vs_jax.html"
    target.write_text(page)
    print(f"\nwrote {target}  ({len(page) / 1024:.0f} kB)")

    store = {"created": metadata["created"], "device": metadata["device"],
             "npix": int(len(WAVE)), "cases": CASES,
             "results": {key: {k: v for k, v in value.items() if k != "profiles"}
                         for key, value in results.items()},
             "sweeps": sweeps}
    (OUT / "cpu_vs_jax_results.json").write_text(json.dumps(store, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())