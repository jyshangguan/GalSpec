"""Running the pipeline: one source, or a whole sample.

``run_pipeline`` is the whole batch: resolve the spectra, fit each one, write
``<output_dir>/<source_id>/<source_id>_*`` and a ``run_report.json`` that
records what was fitted, with which settings, and what came out.  One source
that cannot be read or fitted is recorded as failed and the run continues --
a batch that throws away an hour of sampling because the last file has no
``ERROR`` column would be useless.

Everything here goes through the public :class:`galspec.jnesty_fit.JNesty_Fit`
API; the models are untouched.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import textwrap
import time
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .builder import build_model
from .errors import PipelineError
from .loader import (SourceRef, Spectrum, flux_statistics, read_spectrum,
                     resolve_sources)
from .schema import PipelineConfig

__all__ = ["run_pipeline", "run_one_source", "expected_outputs", "describe_run"]


def expected_outputs(config: PipelineConfig, source_id: str) -> List[Path]:
    """The files a completed fit of ``source_id`` leaves behind.

    Always ``<id>_fit.fits`` -- the model, the data and the whole posterior -- plus
    the figures in whichever container ``[output] figures`` asks for: one
    ``<id>_fit.pdf``, or one PNG per page (``_summary``, ``_fit``, ``_corner``,
    ``_histogram``).  There is no separate posterior archive, because every array
    such a file would hold (samples, weights, log-likelihoods, volumes, the LOGZ
    trajectory) is already in the FITS.
    """
    stem = Path(config.run.output_dir) / source_id / source_id
    files = [Path(f"{stem}_fit.fits")]
    if config.output.figures in ("pdf", "both"):
        files.append(Path(f"{stem}_fit.pdf"))
    if config.output.figures in ("png", "both"):
        files.append(Path(f"{stem}_summary.png"))
        if config.output.fit_plot:
            files.append(Path(f"{stem}_fit.png"))
        if config.output.corner:
            files.append(Path(f"{stem}_corner.png"))
        if config.output.histogram:
            files.append(Path(f"{stem}_histogram.png"))
    return files


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    try:
        import jax

        out["jax"] = jax.__version__
        out["jax_devices"] = ", ".join(
            f"{device.platform}:{device.device_kind}" for device in jax.devices())
    except Exception:                            # pragma: no cover - jax is required
        pass
    try:
        import jnesty

        out["jnesty"] = str(getattr(jnesty, "__version__", "unknown"))
    except Exception:                            # pragma: no cover - jnesty required
        pass
    try:
        from .. import __version__ as galspec_version

        out["galspec"] = str(galspec_version)
    except Exception:                            # pragma: no cover - defensive
        pass
    return out


def describe_run(config: PipelineConfig) -> Dict[str, Any]:
    """The settings block of ``run_report.json``."""
    return {
        "config_file": config.path,
        "config_sha256": config.digest,
        "versions": _versions(),
        "data": asdict(config.data),
        "fit": asdict(config.fit),
        "output": asdict(config.output),
        "run": asdict(config.run),
        "model": {
            "require_priors": config.model.require_priors,
            "components": [
                {"type": component.kind, "name": component.name,
                 "fixed": dict(component.fixed),
                 "bounds": {name: list(bounds)
                            for name, bounds in component.bounds.items()},
                 "ties": sorted(component.ties),
                 "priors": {name: spec.kind
                            for name, spec in component.priors.items()}}
                for component in config.model.components
            ],
        },
    }


# --------------------------------------------------------------------------------------
# One source
# --------------------------------------------------------------------------------------
def _make_fit(config: PipelineConfig, spectrum: Spectrum, model):
    from ..jnesty_fit import JNesty_Fit

    settings = config.fit
    return JNesty_Fit(
        model, spectrum.wave, spectrum.flux, spectrum.error,
        nlive=settings.nlive, bound=settings.bound, device=settings.device,
        queue_size=settings.queue_size, batch_size=settings.batch_size,
        rwalk_K=settings.rwalk_K, rwalk_step_scale=settings.rwalk_step_scale,
        max_ellipsoids=settings.max_ellipsoids, memory_frac=settings.memory_frac,
        min_eff=settings.min_eff, min_ncall=settings.min_ncall,
        share_compilation=settings.share_compilation,
    )


def run_one_source(config: PipelineConfig, ref: SourceRef, *,
                   progress: bool = False) -> Dict[str, Any]:
    """Read, model and fit one source; write its outputs.

    Returns
    -------
    dict
        The per-source entry of ``run_report.json``.  Exceptions propagate: the
        per-source failure isolation lives in :func:`run_pipeline`, so a caller
        that wants to see the traceback can.
    """
    from ..jnesty_fit_io import save_jnesty_fit_to_fits

    started = time.perf_counter()
    # the loader warns when a correction had to be assumed (no redshift, no
    # E(B-V)); collect them here so a sample prints one note per source instead
    # of one warning per file, and so the report can show them per source
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        spectrum = read_spectrum(config, ref)
    reading_warnings = [str(item.message) for item in caught]
    namespace = flux_statistics(spectrum, config.data.continuum_percentile)
    built = build_model(config, namespace, spectrum)
    model = built.model

    fit = _make_fit(config, spectrum, model)
    compile_started = time.perf_counter()
    fit.warmup()
    compile_seconds = time.perf_counter() - compile_started

    sample_started = time.perf_counter()
    with warnings.catch_warnings():
        # a posterior pressed against a bound is reported by prior_dominated()
        warnings.simplefilter("ignore", RuntimeWarning)
        fit.fit(progress=progress, dlogz=config.fit.dlogz,
                max_iterations=config.fit.max_iterations)
    sample_seconds = time.perf_counter() - sample_started

    # the maximum-likelihood sample, so THETA_BEST is in the file and the
    # plotting below shows the model the fit prefers
    fit.get_best_fit()

    out_dir = Path(config.run.output_dir) / spectrum.source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / spectrum.source_id

    fit_date = _now()
    header = {
        "OBJECT": (spectrum.source_id[:68], "source identifier from the config"),
        "SPECFILE": (os.path.basename(spectrum.path)[:68], "input spectrum"),
        "NDATA": (spectrum.n_pixels, "pixels fitted"),
        "REDSHIFT": (spectrum.redshift, "redshift used to de-shift the spectrum"),
        "ZSOURCE": (spectrum.z_source[:68], "where the redshift came from"),
        "CONFIG": (os.path.basename(config.path)[:68], "pipeline config file"),
        "CONFIGID": (config.digest[:16], "sha256 of the config file"),
        "FITDATE": (fit_date, "UTC date of the fit"),
        "COMPILER": ("galspec.pipeline", "how the fit was produced"),
    }
    if spectrum.flux_unit:
        header["FLUXUNIT"] = (str(spectrum.flux_unit)[:68], "flux unit from the file")
    header["EBV"] = (spectrum.e_bv, f"Galactic E(B-V) removed ({spectrum.e_bv_source})")
    header["EBVRV"] = (spectrum.extinction_rv, "R_V of the reddening law")
    if config.data.extinction:
        header["EXTINCT"] = (True, "flux and error de-reddened")
    if spectrum.wave_unit:
        header["WAVEUNIT"] = (str(spectrum.wave_unit)[:68], "wavelength unit read")
    dominated = fit.prior_dominated()
    summary = _parameter_summary(fit)
    fit_file = save_jnesty_fit_to_fits(fit, f"{stem}_fit.fits",
                                       thin=config.output.thin, extra_header=header)
    pages = _figures(
        fit, spectrum, built, namespace, config, summary, dominated,
        reading_warnings, timing={"compile": compile_seconds,
                                  "sample": sample_seconds,
                                  "total": time.perf_counter() - started},
        fit_date=fit_date)
    outputs = [fit_file]
    if config.output.figures in ("pdf", "both"):
        outputs.append(_write_pdf(f"{stem}_fit.pdf", pages))
    if config.output.figures in ("png", "both"):
        outputs.extend(_write_pngs(f"{stem}", pages))

    return {
        "status": "ok",
        "source_id": spectrum.source_id,
        "log_evidence": fit.log_evidence,
        "log_evidence_error": fit.log_evidence_err,
        "n_iterations": int(fit.n_iterations),
        "runtime_seconds": time.perf_counter() - started,
        "compile_seconds": compile_seconds,
        "sample_seconds": sample_seconds,
        "model": {
            "components": list(built.components),
            "free_parameters": len(fit.param_names),
            "parameters": list(fit.param_names),
            "default_priors": list(built.default_priors),
        },
        "parameters": summary,
        "prior_dominated": dominated,
        "outputs": [str(path) for path in outputs],
        "spectrum": spectrum.summary(),
        "tokens": {key: float(value) for key, value in namespace.items()},
        "reading_warnings": reading_warnings,
    }


def _flat_names(built, names, key: str):
    """Translate ``[output]`` names -- ``"Halpha.amp_c"`` or ``"amp_c_3"`` -- to flat ones.

    Both spellings are accepted because both are useful: the qualified name is what
    a config author writes (it is the one that does not depend on the summation
    order), and the flat name is what the FITS file stores.
    """
    if not names:
        return None
    resolved = []
    for name in names:
        flat = built.parameter_map.get(name, name)
        if flat not in built.free_parameters:
            available = ", ".join(built.free_parameters)
            raise PipelineError(
                f"[output] {key} names {name!r}, which is not a free parameter "
                f"of this model; the free parameters are: {available}")
        resolved.append(flat)
    return resolved


def _parameter_summary(fit) -> Dict[str, Dict[str, float]]:
    """Median and 16/84 percentiles of every free parameter.

    The samples are in the FITS already; this puts the numbers a sample fit is read
    for into the report, so comparing six spectra does not mean opening six result
    files.  ``theta_best`` is the maximum-likelihood sample.
    """
    samples = np.asarray(fit.results["samples"], dtype=float)
    if samples.size == 0:                      # a fit with no posterior output
        return {}
    equal_weight = getattr(fit, "samples_equal_weight", None)
    if equal_weight is not None and len(equal_weight) > 10:
        samples = np.asarray(equal_weight, dtype=float)
    low, median, high = np.percentile(samples, [15.865, 50.0, 84.135], axis=0)
    best = np.asarray(getattr(fit, "theta_best", np.full(samples.shape[1], np.nan)),
                      dtype=float)
    return {
        name: {"median": float(m), "lower": float(a), "upper": float(b),
               "best": float(v) if np.isfinite(v) else None}
        for name, m, a, b, v in zip(fit.param_names, median, low, high, best)
    }


def _pyplot():
    """matplotlib with a non-interactive backend, imported when a figure is needed.

    Importing :mod:`galspec.plot` imports pyplot, so the backend has to be chosen
    first; that is why every figure below goes through here.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _figures(fit, spectrum, built, namespace, config, summary, dominated,
             reading_warnings, timing, fit_date) -> List[Tuple[str, Any]]:
    """The pages to draw, named, in the order they are bound or written.

    The summary page is always first, because a figure cannot say which
    configuration produced it or whether a parameter is a measurement or a limit.
    The rest are the figures ``[output]`` asks for, in the order they are declared
    there: the fit, the corner plot, the marginal posteriors.  The name becomes the
    PNG suffix, so the same page is ``<id>_fit.pdf`` page 2 or ``<id>_fit.png``.
    """
    pages = [("summary", _summary_figure(
        fit, spectrum, built, namespace, config, summary, dominated,
        reading_warnings, timing, fit_date))]
    if config.output.fit_plot:
        pages.append(("fit", _fit_figure(fit, spectrum)))
    if config.output.corner:
        page = _corner_figure(
            fit, _flat_names(built, config.output.corner_params, "corner_params"),
            title=spectrum.source_id)
        if page is not None:
            pages.append(("corner", page))
    if config.output.histogram:
        pages.append(("histogram", _histogram_figure(
            fit, _flat_names(built, config.output.hist_params, "hist_params"),
            title=spectrum.source_id)))
    return pages


def _write_pdf(filename: str, pages: Sequence[Tuple[str, Any]]) -> str:
    """Bind the figures into one PDF, one page each, with a page number.

    Vector output: a 57k-pixel spectrum stays sharp at any zoom, which is the point
    of preferring a PDF to a directory of PNGs.  The figures are closed as they are
    written, so a twenty-source sample does not accumulate twenty open canvases.
    """
    plt = _pyplot()
    from matplotlib.backends.backend_pdf import PdfPages

    total = len(pages)
    with PdfPages(filename) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = Path(filename).stem
        metadata["Creator"] = "galspec.pipeline"
        for index, (_, figure) in enumerate(pages, start=1):
            figure.text(0.995, 0.004, f"page {index} of {total}", ha="right",
                        va="bottom", fontsize=7, color="0.45")
            pdf.savefig(figure, dpi=150, bbox_inches="tight")
            plt.close(figure)
    return str(filename)


def _write_pngs(stem: str, pages: Sequence[Tuple[str, Any]]) -> List[str]:
    """One PNG per page, for a talk: ``<id>_summary.png``, ``<id>_fit.png``, ...

    The same figures the PDF holds, at the resolution that goes into a slide; a
    raster page loses the zoom the vector one keeps, which is why the PDF is the
    default container.
    """
    plt = _pyplot()
    written = []
    for name, figure in pages:
        path = f"{stem}_{name}.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def _summary_figure(fit, spectrum, built, namespace, config, summary, dominated,
                    reading_warnings, timing, fit_date):
    """Page one: what was fitted, with what, and what came out.

    Plain text, because the questions a result file is opened for are not the ones
    a plot answers -- which configuration, how long, how many pixels, and whether a
    number is a measurement or a limit.  It repeats the report's numbers rather
    than replacing them: the report is the machine-readable copy, this is the one
    a person reads beside the figures.
    """
    plt = _pyplot()
    figure = plt.figure(figsize=(8.27, 11.69))       # A4 portrait
    figure.text(0.06, 0.972, spectrum.source_id, fontsize=15, weight="bold",
                va="top")
    components = list(built.components)
    slots = len(getattr(fit, "full_param_names", fit.param_names))
    free = len(fit.param_names)
    parameters = (f"{free} free parameters" if free == slots
                  else f"{free} free parameters of {slots} parameter slots")
    named = (f"{len(components)} "
             f"{'component' if len(components) == 1 else 'components'}: "
             f"{', '.join(components)}")
    subtitle = textwrap.wrap(named, width=100) + [
        f"{parameters}, {spectrum.n_pixels} pixels fitted"]
    figure.text(0.06, 0.945, "\n".join(subtitle), fontsize=9, va="top",
                color="0.25", linespacing=1.5)

    lines = _summary_lines(fit, spectrum, config, summary, dominated, timing,
                           reading_warnings, namespace, built.default_priors,
                           fit_date)
    # the text block has the page from 0.912 of its height down to the footer:
    # 8 pt with a 1.35 line spacing fits 64 lines of it, and a 60-parameter model
    # is still legible on one page rather than spilling onto a second
    size = 8.0 if len(lines) <= 64 else (7.2 if len(lines) <= 74 else 6.5)
    figure.text(0.06, 0.912, "\n".join(lines), family="monospace", fontsize=size,
                va="top", linespacing=1.35)
    return figure


def _summary_lines(fit, spectrum, config, summary, dominated, timing,
                   reading_warnings, namespace, default_priors,
                   fit_date) -> List[str]:
    """The text of the summary page, line by line.

    Monospace and column-aligned: this ends up in a PDF, so it has to line up by
    construction rather than by font luck.  ``*`` marks a parameter that took the
    pipeline's default prior (so its prior is a choice the pipeline made, not the
    file) and ``!`` one pressed against a prior limit, which is a limit rather than
    a measurement.
    """
    out: List[str] = []
    width = 96

    def block(title: str, rows: Sequence) -> None:
        out.append("")
        out.append(title.upper())
        for label, value in rows:
            wrapped = textwrap.wrap(str(value), width=width - 22) or [""]
            out.append(f"  {label:<20}{wrapped[0]}")
            out.extend(" " * 22 + extra for extra in wrapped[1:])

    results = getattr(fit, "results", None) or {}
    diagnostics = []
    information = results.get("information")
    if information is not None:
        diagnostics.append(f"H = {float(information):.2f} nat")
    acceptance = results.get("acceptance_rate")
    if acceptance is not None:
        diagnostics.append(f"acceptance {float(acceptance):.3f}")
    delta = results.get("delta_logz")
    if delta is not None:
        target = results.get("delta_logZ_threshold")
        diagnostics.append(f"final dlogZ = {float(delta):.3f}"
                           + (f" (target {float(target):g})"
                              if target is not None else ""))
    if results.get("converged") is not None:
        diagnostics.append("converged" if results["converged"] else "not converged")

    block("the run", [
        ("evidence", f"log Z = {fit.log_evidence:.3f} +/- {fit.log_evidence_err:.3f}"),
        ("iterations", f"{int(fit.n_iterations)} dead points, "
                       f"{int(getattr(fit, 'ncall', 0) or 0)} likelihood calls, "
                       f"nlive = {int(getattr(fit, 'nlive', 0) or 0)}"),
        ("sampler", f"JNesty, bound \"{getattr(fit, 'bound', 'multi')}\", "
                    f"thin {config.output.thin}"),
        ("device", f"{getattr(fit, 'device', 'unknown')}, "
                   f"{timing['total']:.1f} s total "
                   f"({timing['compile']:.1f} s compile, "
                   f"{timing['sample']:.1f} s sampling)"),
        ("configuration", f"{Path(config.path).name} "
                          f"(sha256 {config.digest[:16]})"),
        ("fit date", fit_date),
    ] + ([("diagnostics", ", ".join(diagnostics))] if diagnostics else []))

    block("the data", [
        ("file", os.path.basename(spectrum.path)),
        ("pixels", f"{spectrum.n_pixels} fitted "
                   f"(min_pixels {config.data.min_pixels})"),
        ("window", f"{spectrum.wave.min():.1f}-{spectrum.wave.max():.1f} A rest"
                   + (f", {len(config.data.mask)} masked window(s)"
                      if config.data.mask else "")),
        ("units", f"{spectrum.wave_unit or 'Angstrom'} wavelength, "
                  f"{spectrum.flux_unit or 'flux as stored'}"),
        ("redshift", f"{spectrum.redshift:.6g}  ({spectrum.z_source})"),
        ("extinction", f"E(B-V) = {spectrum.e_bv:.4g}  ({spectrum.e_bv_source}), "
                       f"R_V = {spectrum.extinction_rv:.2f}"
                       + (", removed from flux and error"
                          if config.data.extinction else ", not applied")),
        ("tokens", ", ".join(f"{name} = {value:.4g}"
                             for name, value in namespace.items())),
    ])

    out.append("")
    out.append("PARAMETERS   * default prior      ! pressed against a prior limit")
    out.append(f"   {'parameter':<32}{'median':>12}{'16th':>12}{'84th':>12}"
               f"{'best':>12}")
    full = list(getattr(fit, "full_param_names", None) or fit.param_names)
    defaults = set(default_priors or ())
    for name, flat in zip(full, fit.param_names):
        stats = summary.get(flat, {})
        best = stats.get("best")
        out.append(
            f"{'*' if flat in defaults else ' '}{'!' if flat in dominated else ' '}"
            f" {name:<32}"
            f"{stats.get('median', float('nan')):>12.5g}"
            f"{stats.get('lower', float('nan')):>12.5g}"
            f"{stats.get('upper', float('nan')):>12.5g}"
            f"{(best if best is not None else float('nan')):>12.5g}")

    out.append("")
    out.append("THE POSTERIOR IS IN THIS SOURCE'S FITS FILE")
    out.extend(textwrap.wrap(
        "SAMPLES holds every sample with LOGL, LOGWT and LOGVOL beside it; "
        "SAMPLES_EQ is the equal-weight resample the numbers above come from; "
        "SAMPLES_U is the same points in the unit cube; LOGZ/LOGZERR/DLZ_TRAJ/"
        "SCALE_TR are the run's trajectories, so the dynesty-style diagnostics can "
        "be rebuilt from the file alone.  PARAM_NAMES gives the column order, "
        "PARAM_BOUNDS the prior limits, THETA_BEST the maximum-likelihood sample.  "
        "No other file holds a copy of the posterior.",
        width=width, initial_indent="  ", subsequent_indent="  "))
    if reading_warnings:
        out.append("")
        out.append("WARNINGS RAISED WHILE READING")
        for message in reading_warnings:
            out.extend(textwrap.wrap(message, width=width, initial_indent="  ",
                                     subsequent_indent="  "))
    return out


def _fit_figure(fit, spectrum: Spectrum):
    """The figure :func:`galspec.plot.plot_fit_jax` draws, at pipeline size.

    Evaluation, ties and subcomponents are that function's job; the only decisions
    here are the layout -- the legend goes outside the axes once a model has more
    components than a panel can hold, the same choice
    ``example/fit_iras07546_jax.py`` makes -- and the title.
    """
    plt = _pyplot()
    from ..plot import plot_fit_jax

    crowded = len(getattr(fit.model, "submodel_names", ())) > 8
    figure = plt.figure(figsize=(16.0, 8.0) if crowded else (13.0, 7.5))
    ax = figure.add_axes([0.06, 0.20, 0.78 if crowded else 0.90, 0.72])
    axr = figure.add_axes([0.06, 0.07, 0.78 if crowded else 0.90, 0.13])
    legend_kwargs = ({"fontsize": 8, "ncol": 2, "loc": "center left",
                      "bbox_to_anchor": (1.005, 0.5), "frameon": False}
                     if crowded else {"fontsize": 9, "ncol": 1, "loc": "best"})
    ylabel = "flux" + (f" [{spectrum.flux_unit}]" if spectrum.flux_unit else "")
    plot_fit_jax(spectrum.wave, spectrum.flux, fit, ax=ax, axr=axr,
                 error=spectrum.error, residual_in_sigma=True,
                 xlabel="rest-frame wavelength [Angstrom]", ylabel=ylabel,
                 legend_kwargs=legend_kwargs)
    # plot_fit_jax draws at publication size; this is a per-source diagnostic
    for axis in (ax, axr):
        axis.tick_params(labelsize=10)
    ax.yaxis.label.set_fontsize(12)
    axr.xaxis.label.set_fontsize(12)
    axr.yaxis.label.set_fontsize(12)
    ax.set_title(f"{spectrum.source_id}: log Z = {fit.log_evidence:.2f} "
                 f"+/- {fit.log_evidence_err:.2f}, "
                 f"{fit.n_iterations} iterations", fontsize=11)
    return figure


def _corner_figure(fit, names=None, title=None):
    """A corner plot of ``[output] corner_params``, or of every parameter.

    Drawn by :func:`galspec.plot.plot_corner`, which slices the samples: a
    27-parameter corner is a square nobody reads.  A corner plot is decoration, so
    a parameter that never moved -- which happens when the sampler is stopped
    before it starts exploring -- skips this one figure instead of failing a fit
    that has already cost its runtime.
    """
    from ..plot import plot_corner

    indices = None
    if names:
        indices = [fit.param_names.index(name) for name in names]
    try:
        return plot_corner(fit, params=indices)
    except ValueError as exc:
        warnings.warn(f"no corner plot for {title or 'this source'}: {exc}",
                      RuntimeWarning, stacklevel=2)
        return None


def _histogram_figure(fit, names=None, title=None):
    """One panel per parameter, drawn by :func:`galspec.plot.plot_posterior_hist`.

    That function already marks the median and the 16-84 band, annotates the prior
    bounds and uses the qualified LaTeX labels, so this only chooses the grid and
    puts the parameters the sampler explored in log space on a log axis.
    """
    from ..plot import plot_posterior_hist

    indices = None
    if names:
        indices = [fit.param_names.index(name) for name in names]
    selected = indices if indices is not None else list(range(len(fit.param_names)))
    columns = len(selected)
    ncols = 2 if columns <= 4 else (3 if columns <= 8 else 4)
    flat = list(fit.param_names)
    full = list(getattr(fit, "full_param_names", None) or flat)
    log_sampled = set(getattr(fit, "_log_scale_params", ()) or ())
    log_scale = [full[i] for i in selected if flat[i] in log_sampled]
    figure, _ = plot_posterior_hist(
        fit, params=indices, ncols=ncols, bins=40, log_scale=log_scale or None,
        title=f"{title}: marginal posteriors" if title else None)
    return figure


# --------------------------------------------------------------------------------------
# The whole sample
# --------------------------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_report(config: PipelineConfig, payload: Mapping[str, Any]) -> str:
    out_dir = Path(config.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / "run_report.json"
    with filename.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return str(filename)


def _prepare(config: PipelineConfig) -> None:
    """Process-wide setup: the persistent compilation cache, once per process."""
    from ..jnesty_fit import enable_persistent_compilation_cache

    directory = config.fit.compilation_cache_dir
    if config.fit.share_compilation and directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        enable_persistent_compilation_cache(directory)


def _one_line(entry: Mapping[str, Any]) -> str:
    status = entry["status"]
    if status == "ok":
        return (f"{entry['source_id']}: log Z = {entry['log_evidence']:.3f} "
                f"+/- {entry['log_evidence_error']:.3f} "
                f"({entry['n_iterations']} iterations, "
                f"{entry['runtime_seconds']:.1f} s)")
    if status == "skipped":
        return (f"{entry['source_id']}: skipped "
                f"({entry.get('reason', 'already done')})")
    return f"{entry['source_id']}: FAILED: {entry.get('error', 'unknown error')}"


def _failed_entry(ref: SourceRef, exc: BaseException, position: int) -> Dict[str, Any]:
    return {
        "position": position,
        "status": "failed",
        "source_id": ref.source_id,
        "path": ref.path,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": (traceback.format_exc()
                      if os.environ.get("GALSPEC_PIPELINE_TRACE") else None),
    }


def _skipped_entry(config: PipelineConfig, ref: SourceRef,
                   position: int) -> Dict[str, Any]:
    return {"position": position, "status": "skipped", "source_id": ref.source_id,
            "path": ref.path, "reason": "outputs already exist",
            "outputs": [str(path) for path in expected_outputs(config, ref.source_id)]}


def wants_progress(config: PipelineConfig, workers: int = 1) -> bool:
    """Whether the sampler's progress bar should be drawn.

    A bar is only useful in front of a person: it is written to standard error with
    carriage returns, which is noise in a redirected log.  So the default
    (``[run] progress`` unset) is "a bar when standard error is a terminal", and the
    configuration can force either answer.  ``workers > 1`` never shows one: the
    bar belongs to one sampler, and a pool runs several.

    Parameters
    ----------
    config : PipelineConfig
    workers : int
        The number of processes the run will use.
    """
    if workers != 1:
        return False
    if config.run.progress is not None:
        return bool(config.run.progress)
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):             # pragma: no cover - closed stream
        return False


def run_pipeline(config: PipelineConfig, *,
                 sources: Optional[Sequence[SourceRef]] = None,
                 workers: Optional[int] = None, force: Optional[bool] = None,
                 log=None) -> Dict[str, Any]:
    """Fit every configured source; returns the ``run_report.json`` payload.

    Parameters
    ----------
    config : PipelineConfig
        Settings read by :func:`galspec.pipeline.schema.load_config`.
    sources : sequence of SourceRef, optional
        Overrides the ``[data]`` selection (used by the tests).
    workers : int, optional
        Overrides ``[run] workers``.  ``> 1`` fits in a spawn-based pool, so the
        configuration is pickled to fresh processes.
    force : bool, optional
        Overrides ``[run] force``: refit even when the outputs already exist.
    log : callable, optional
        ``log(message)`` sink; defaults to stderr.

    Notes
    -----
    A source that fails is recorded in the report and does not stop the run; the
    exception text is in the report's ``error`` field (set
    ``GALSPEC_PIPELINE_TRACE=1`` to keep the full traceback).
    """
    emit = log or (lambda message: print(message, file=sys.stderr, flush=True))
    _prepare(config)
    refs = list(sources) if sources is not None else resolve_sources(config)
    if not refs:
        raise PipelineError("no spectra to fit")
    workers = config.run.workers if workers is None else int(workers)
    force = config.run.force if force is None else bool(force)
    if workers < 1:
        raise PipelineError(f"workers must be at least 1, got {workers}")
    progress = wants_progress(config, workers)
    if config.run.progress and workers > 1:
        emit("note: [run] progress needs workers = 1; fitting without a progress bar")
    elif not progress:
        # a nested-sampling run is silent by design; say so, or a long fit looks hung
        emit(f"note: {config.fit.nlive} live points, quiet until it finishes "
             f"([run] progress = true draws the sampler bar; redirecting output "
             f"hides it)")

    started = _now()
    entries: Dict[int, Dict[str, Any]] = {}
    todo: List[tuple] = []
    for position, ref in enumerate(refs):
        if not force and all(path.is_file()
                             for path in expected_outputs(config, ref.source_id)):
            entries[position] = _skipped_entry(config, ref, position)
            emit(_one_line(entries[position]))
        else:
            todo.append((position, ref))

    if todo and workers == 1:
        for position, ref in todo:
            emit(f"[{position + 1}/{len(refs)}] {ref.source_id}: fitting")
            try:
                entry = run_one_source(config, ref, progress=progress)
                entry["position"] = position
            except Exception as exc:             # noqa: BLE001 - isolation is the point
                emit(f"[{position + 1}/{len(refs)}] {ref.source_id}: FAILED "
                     f"({type(exc).__name__}: {exc})")
                entry = _failed_entry(ref, exc, position)
            entries[position] = entry
            emit(_one_line(entry))
            for message in entry.get("reading_warnings") or ():
                emit(f"    note: {message}")
    elif todo:
        for position, entry in _fit_pool(config, todo, progress=progress,
                                         emit=emit, total=len(refs)):
            entries[position] = entry

    reports = [entries[position] for position in sorted(entries)]
    finished = _now()
    payload = {
        "started": started,
        "finished": finished,
        "settings": describe_run(config),
        "counts": {
            "total": len(refs),
            "ok": sum(1 for entry in reports if entry["status"] == "ok"),
            "failed": sum(1 for entry in reports if entry["status"] == "failed"),
            "skipped": sum(1 for entry in reports if entry["status"] == "skipped"),
        },
        "sources": reports,
    }
    payload["report_file"] = _write_report(config, payload)
    counts = payload["counts"]
    emit(f"{counts['ok']} fitted, {counts['failed']} failed, {counts['skipped']} "
         f"skipped; report in {payload['report_file']}")
    if counts["ok"] == 0 and counts["failed"] and todo:
        raise PipelineError(f"every source failed; see {payload['report_file']}")
    return payload


# --------------------------------------------------------------------------------------
# Worker pool
# --------------------------------------------------------------------------------------
def _worker(arguments):
    """One pool task: reload the config, fit one source, never raise."""
    config_path, ref, progress = arguments
    from .schema import load_config

    try:
        config = load_config(config_path)
        _prepare(config)
        return {"status": "ok", "entry": run_one_source(config, ref, progress=progress)}
    except Exception as exc:                     # noqa: BLE001 - isolation is the point
        import traceback as _traceback

        entry = _failed_entry(ref, exc, 0)
        if not os.environ.get("GALSPEC_PIPELINE_TRACE"):
            entry["traceback"] = None
        else:                                    # pragma: no cover - debugging aid
            entry["traceback"] = _traceback.format_exc()
        return {"status": "failed", "entry": entry}


def _fit_pool(config: PipelineConfig, todo: Sequence[tuple], *, progress: bool,
              emit, total: int):
    """Fit ``todo`` in ``workers`` fresh processes; yields ``(position, entry)``."""
    import multiprocessing

    emit(f"fitting {len(todo)} source(s) in {config.run.workers} process(es)")
    context = multiprocessing.get_context("spawn")
    tasks = [(config.path, ref, progress) for _, ref in todo]
    positions = {ref.source_id: position for position, ref in todo}
    with context.Pool(processes=config.run.workers) as pool:
        for result in pool.imap_unordered(_worker, tasks):
            entry = result["entry"]
            entry["position"] = positions.get(entry["source_id"], 0)
            emit(_one_line(entry))
            yield entry["position"], entry