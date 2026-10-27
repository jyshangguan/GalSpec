"""Errors raised by :mod:`galspec.pipeline`.

Two kinds, because they are fixed in different places: a
:class:`~galspec.pipeline.schema.ConfigError` means the ``.toml`` file is wrong,
a :class:`PipelineError` means the configuration is fine but a spectrum cannot
be read, the model cannot be built from it, or a fit failed.
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """A source could not be read, modelled or fitted.

    The runner catches this per source, records the message in
    ``run_report.json`` and carries on with the next one, so one unreadable
    file does not throw away a night of fitting.
    """