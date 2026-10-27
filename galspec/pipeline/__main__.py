"""``python -m galspec.pipeline CONFIG.toml`` -- fit a spectrum or a sample.

Examples
--------
.. code-block:: console

    # check the file, the data selection and every spectrum before a long run
    python -m galspec.pipeline iras07546.toml --dry-run

    # refit everything, overwriting what is there, on 8 processes
    python -m galspec.pipeline sample.toml --force --workers 8

    # what does the configuration select?
    python -m galspec.pipeline sample.toml --list-sources
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

from .errors import PipelineError
from .loader import flux_statistics, read_spectrum, resolve_sources
from .runner import run_pipeline
from .schema import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m galspec.pipeline",
        description="Fit GalSpec JAX models to one spectrum or a sample, as "
                    "described by a TOML file.",
        epilog="See docs/source/config_fit.rst for the file format.")
    parser.add_argument("config", nargs="?",
                        help="the TOML configuration file")
    parser.add_argument("--template", nargs="?", const="config.toml", default=None,
                        metavar="PATH",
                        help="write a commented starter configuration to PATH "
                             "(default config.toml) and exit; '-' prints it")
    parser.add_argument("--workers", type=int, default=None,
                        help="fit in this many processes (default: [run] workers)")
    parser.add_argument("--force", action="store_true", default=None,
                        help="refit even when the outputs already exist; with "
                             "--template, overwrite an existing file")
    parser.add_argument("--output", default=None, metavar="DIR",
                        help="write here instead of [run] output_dir")
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "gpu"),
                        help="override [fit] device")
    parser.add_argument("--dry-run", action="store_true",
                        help="read every spectrum and report what would be "
                             "fitted, without fitting")
    parser.add_argument("--list-sources", action="store_true",
                        help="print the source ids and files the config selects")
    parser.add_argument("--quiet", action="store_true",
                        help="only report failures and the final summary, and no "
                             "progress bar")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.template is not None:
        return _write_template(arguments.template, force=bool(arguments.force))
    if arguments.config is None:
        print("error: a configuration file is required "
              "(or --template to write a starter one)", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2
    try:
        config = load_config(arguments.config)
        if arguments.output is not None:
            config = replace(config, run=replace(config.run,
                                                 output_dir=str(Path(arguments.output).expanduser().resolve())))
        if arguments.device is not None:
            config = replace(config, fit=replace(config.fit, device=arguments.device))
        if arguments.list_sources:
            sources = resolve_sources(config)
            for source in sources:
                redshift = "" if source.redshift is None else f"  z={source.redshift}"
                print(f"{source.source_id}\t{source.path}{redshift}")
            print(f"{len(sources)} source(s)")
            return 0
        if arguments.quiet:
            # "quiet" has to mean quiet, and the sampler bar is the loudest thing here
            config = replace(config, run=replace(config.run, progress=False))
        if arguments.dry_run:
            return _dry_run(config, quiet=arguments.quiet)
        payload = run_pipeline(config, workers=arguments.workers,
                               force=arguments.force,
                               log=(None if not arguments.quiet else _quiet_log))
    except (ConfigError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:                    # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130
    counts = payload["counts"]
    return 0 if counts["failed"] == 0 else 1


def _write_template(destination: str, *, force: bool) -> int:
    """``--template [PATH]``: write the starter configuration and say what to do next."""
    from .template import template_text, write_template

    if str(destination) == "-":                       # to stdout, for reading first
        print(template_text(), end="")
        return 0
    try:
        path = write_template(destination, force=force)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    print("  edit [data] file, then")
    print(f"    python -m galspec.pipeline {path} --dry-run    # what would be fitted")
    print(f"    python -m galspec.pipeline {path}              # the fit")
    return 0


def _quiet_log(message: str) -> None:
    if "FAILED" in message or "report in" in message:
        print(message, file=sys.stderr, flush=True)


def _dry_run(config, quiet: bool = False) -> int:
    """Read every configured spectrum and report what a fit would see."""
    from .builder import build_model

    sources = resolve_sources(config)
    failures = 0
    print(f"config {config.path}")
    print(f"{len(sources)} source(s), outputs under {config.run.output_dir}")
    free_parameters = None
    for index, source in enumerate(sources, start=1):
        try:
            spectrum = read_spectrum(config, source)
            namespace = flux_statistics(spectrum, config.data.continuum_percentile)
            built = build_model(config, namespace)
        except (PipelineError, ConfigError) as exc:
            failures += 1
            print(f"[{index}/{len(sources)}] {source.source_id}: FAILED: {exc}",
                  file=sys.stderr)
            continue
        if free_parameters is None:
            free_parameters = built
        if not quiet:
            print(f"[{index}/{len(sources)}] {source.source_id}: "
                  f"{spectrum.n_pixels} pixels, "
                  f"{spectrum.wave.min():.1f}-{spectrum.wave.max():.1f} A rest, "
                  f"z={spectrum.redshift:g} ({spectrum.z_source}), "
                  f"peak_flux={namespace['peak_flux']:.4g}, "
                  f"{len(built.free_parameters)} free parameters")
    if free_parameters is not None:
        print("model: " + ", ".join(free_parameters.components))
        print(f"free parameters: {len(free_parameters.free_parameters)}")
    if failures:
        print(f"{failures} source(s) would fail", file=sys.stderr)
        return 1
    print("all sources read; nothing was fitted (--dry-run)")
    return 0


if __name__ == "__main__":                        # pragma: no cover - CLI entry
    sys.exit(main())