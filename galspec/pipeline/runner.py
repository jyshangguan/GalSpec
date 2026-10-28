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
import time
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .builder import build_model
from .errors import PipelineError
from .loader import (SourceRef, Spectrum, flux_statistics, read_spectrum,
                     resolve_sources)
from .schema import PipelineConfig

__all__ = ["run_pipeline", "run_one_source", "expected_outputs", "describe_run"]


def expected_outputs(config: PipelineConfig, source_id: str) -> List[Path]:
    """The files a completed fit of ``source_id`` leaves behind."""
    stem = Path(config.run.output_dir) / source_id / source_id
    files = [Path(f"{stem}_fit.fits"), Path(f"{stem}_posterior.npz")]
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

    header = {
        "OBJECT": (spectrum.source_id[:68], "source identifier from the config"),
        "SPECFILE": (os.path.basename(spectrum.path)[:68], "input spectrum"),
        "NDATA": (spectrum.n_pixels, "pixels fitted"),
        "REDSHIFT": (spectrum.redshift, "redshift used to de-shift the spectrum"),
        "ZSOURCE": (spectrum.z_source[:68], "where the redshift came from"),
        "CONFIG": (os.path.basename(config.path)[:68], "pipeline config file"),
        "CONFIGID": (config.digest[:16], "sha256 of the config file"),
        "FITDATE": (_now(), "UTC date of the fit"),
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
    fit_file = save_jnesty_fit_to_fits(fit, f"{stem}_fit.fits",
                                       thin=config.output.thin, extra_header=header)

    posterior_file = _write_posterior(f"{stem}_posterior.npz", fit, spectrum)
    outputs = [fit_file, posterior_file]
    if config.output.fit_plot:
        outputs.append(_write_fit_plot(f"{stem}_fit.png", fit, spectrum, config))
    if config.output.corner:
        corner_file = _write_corner_plot(
            f"{stem}_corner.png", fit, config,
            _flat_names(built, config.output.corner_params, "corner_params"),
            title=spectrum.source_id)
        if corner_file is not None:
            outputs.append(corner_file)
    if config.output.histogram:
        outputs.append(_write_histogram(
            f"{stem}_histogram.png", fit, config,
            _flat_names(built, config.output.hist_params, "hist_params"),
            title=spectrum.source_id))

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
        "parameters": _parameter_summary(fit),
        "prior_dominated": fit.prior_dominated(),
        "outputs": [str(path) for path in outputs],
        "spectrum": spectrum.summary(),
        "tokens": {key: float(value) for key, value in namespace.items()},
        "reading_warnings": reading_warnings,
    }


def _flat_names(built, names, key: str):
    """Translate ``[output]`` names -- ``"Halpha.amp_c"`` or ``"amp_c_3"`` -- to flat ones.

    Both spellings are accepted because both are useful: the qualified name is what
    a config author writes (it is the one that does not depend on the summation
    order), and the flat name is what the posterior file stores.
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

    The samples are in the ``.npz`` already; this puts the numbers a sample fit
    is read for into the report, so comparing six spectra does not mean opening
    six archives.  ``theta_best`` is the maximum-likelihood sample.
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


def _write_posterior(filename: str, fit, spectrum: Spectrum) -> str:
    """The samples, in a format numpy reads without GalSpec."""
    results = fit.results
    payload = {
        "samples_equal_weight": np.asarray(fit.samples_equal_weight),
        "samples": np.asarray(results["samples"]),
        "logl": np.asarray(results["logl"]),
        "logwt": np.asarray(results["logwt"]),
        # plain unicode, not dtype=object, so np.load needs no allow_pickle
        "param_names": np.asarray(list(fit.param_names),
                                  dtype=f"<U{max(len(n) for n in fit.param_names)}"),
        "log_evidence": np.asarray(fit.log_evidence),
        "log_evidence_error": np.asarray(fit.log_evidence_err),
        "n_iterations": np.asarray(int(fit.n_iterations)),
        "redshift": np.asarray(spectrum.redshift),
        "wave": spectrum.wave,
        "flux": spectrum.flux,
        "error": spectrum.error,
    }
    np.savez_compressed(filename, **payload)
    return str(filename)


def _pyplot():
    """matplotlib with a non-interactive backend, imported when a figure is needed.

    Importing :mod:`galspec.plot` imports pyplot, so the backend has to be chosen
    first; that is why every figure below goes through here.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _write_fit_plot(filename: str, fit, spectrum: Spectrum, config) -> str:
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
    figure.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return str(filename)


def _write_corner_plot(filename: str, fit, config, names=None, title=None):
    """A corner plot of ``[output] corner_params``, or of every parameter.

    Drawn by :func:`galspec.plot.plot_corner`, which slices the samples: a
    27-parameter corner is a square nobody reads.  A corner plot is decoration, so
    a parameter that never moved -- which happens when the sampler is stopped
    before it starts exploring -- skips this one figure instead of failing a fit
    that has already cost its runtime.
    """
    plt = _pyplot()
    from ..plot import plot_corner

    indices = None
    if names:
        indices = [fit.param_names.index(name) for name in names]
    try:
        figure = plot_corner(fit, params=indices, filename=filename, dpi=110)
    except ValueError as exc:
        warnings.warn(f"no corner plot for {title or filename}: {exc}",
                      RuntimeWarning, stacklevel=2)
        return None
    plt.close(figure)
    return str(filename)


def _write_histogram(filename: str, fit, config, names=None, title=None) -> str:
    """One panel per parameter, drawn by :func:`galspec.plot.plot_posterior_hist`.

    That function already marks the median and the 16-84 band, annotates the prior
    bounds and uses the qualified LaTeX labels, so this only chooses the grid and
    puts the parameters the sampler explored in log space on a log axis.
    """
    plt = _pyplot()
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
        title=f"{title}: marginal posteriors" if title else None,
        filename=filename)
    plt.close(figure)
    return str(filename)


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