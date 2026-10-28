"""A configuration-driven fitting pipeline built on the GalSpec JAX models.

The pipeline is a *consumer* of :mod:`galspec.jax_models` and
:class:`galspec.jnesty_fit.JNesty_Fit`: it reads a TOML file that describes the
data, the model and the sampler, then fits one spectrum or a whole sample with
the same settings.  It adds no model code of its own, so a fit it produces is
the same fit a hand-written script would produce, and nothing here imports
until it is used -- ``import galspec.pipeline`` does not import jax.

Command line
------------
.. code-block:: console

    python -m galspec.pipeline --template config.toml  # write a starter file
    python -m galspec.pipeline config.toml            # fit every source
    python -m galspec.pipeline config.toml --dry-run  # check the data first
    python -m galspec.pipeline config.toml --workers 8

Library
-------
.. code-block:: python

    from galspec.pipeline import load_config, run_pipeline

    config = load_config("config.toml")
    report = run_pipeline(config)
    print(report["counts"])

See ``docs/source/config_fit.rst`` for the file format.
"""

from .builder import DEFAULT_PRIORS, BuiltModel, build_model
from .errors import PipelineError
from .loader import SourceRef, Spectrum, read_spectrum, resolve_sources
from .runner import (describe_run, expected_outputs, run_one_source, run_pipeline)
from .template import template_text, write_template
from .schema import (ComponentConfig, ConfigError, DataConfig, FitConfig,
                     ModelConfig, OutputConfig, PipelineConfig, PriorSpec,
                     RunConfig, TieSpec, TokenValue, load_config, resolve_value,
                     source_namespace)

__all__ = [
    "BuiltModel", "DEFAULT_PRIORS", "build_model",
    "ComponentConfig", "ConfigError", "DataConfig", "FitConfig", "ModelConfig",
    "OutputConfig", "PipelineConfig", "PipelineError", "PriorSpec", "RunConfig",
    "SourceRef", "Spectrum", "TieSpec", "TokenValue",
    "describe_run", "expected_outputs", "load_config", "read_spectrum",
    "resolve_sources", "resolve_value", "run_one_source", "run_pipeline",
    "source_namespace", "template_text", "write_template",
]