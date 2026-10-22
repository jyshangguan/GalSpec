"""LaTeX labels for model parameters.

A fitter reports parameters as machine names -- ``amp_c0_7``, ``sigma_w1_3``,
``Continuum.amplitude`` -- which are unambiguous but unreadable in a plot.  This
module turns them into the symbols a paper uses: ``A_{c0}``, ``\\sigma_{w1}``,
``A``, prefixed by the model they belong to.

Every parameter name is split into a *base* (what the number means) and a
*subscript* (which component it belongs to), so a single table covers the whole
library:

===========  ==========  ==================================================
name         label       note
===========  ==========  ==================================================
amplitude    A           flux scale of a component
amp_w0       A_{w0}      broad component 0 of a multi-Gaussian line
sigma_c      \\sigma_{c}  narrow core width
dv_w1        \\Delta v_{w1}
wavec        \\lambda_{c}
alpha        \\alpha      continuum slope
c1           c_{1}       polynomial coefficient
===========  ==========  ==================================================

A model's display name is taken from ``model.label`` when set, otherwise from
``model.name``; the label may itself contain maths, so a broad H-alpha line
called ``Halpha`` can be displayed as ``BH$\\alpha$`` and its width as
``BH$\\alpha$:$\\sigma_{w0}$``.

The functions work on the JAX models (which know their submodel structure) and on
astropy compound models, so the same labels can be used for either backend.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

__all__ = [
    "PARAM_SYMBOLS",
    "parameter_symbol",
    "split_parameter_name",
    "model_label",
    "parameter_labels",
    "qualified_parameter_labels",
]

#: Base parameter name -> LaTeX symbol.  Longest match wins, so ``amplitude``
#: beats ``amp`` and ``x_break`` beats ``x``.
PARAM_SYMBOLS = {
    "amplitude": "A",
    "amp": "A",
    "stddev": r"\sigma",
    "sigma": r"\sigma",
    "dv": r"\Delta v",
    "wavec0": r"\lambda_{0}",
    "wavec1": r"\lambda_{1}",
    "wavec": r"\lambda",
    "x_0": "x_{0}",
    "x_break": "x_{\\rm br}",
    "alpha": r"\alpha",
    "z": "z",
    "w": "w",
    "tau0": r"\tau_{0}",
    "logtau0": r"\log_{10}\tau_{0}",
    "temperature": "T",
    "h3": "h_{3}",
    "h4": "h_{4}",
    "energy": "E",
    "epsilon": r"\epsilon",
    "c": "c",
}

#: Parameters whose *suffix* is not a subscript but part of the symbol
#: (``wavec0`` -> ``\lambda_0``), already handled by the table above.

_SUBSCRIPT_RE = re.compile(r"^[_.]?(?P<sub>[A-Za-z0-9]+)$")


def split_parameter_name(name: str):
    """Split ``amp_c0`` into ``('amp', 'c0')``.

    ``name`` may be a bare parameter name or a qualified one (``Halpha.amp_c0``);
    the qualifier is dropped.  Returns ``(base, subscript)`` where ``subscript``
    is an empty string when the parameter has none.
    """
    bare = name.rsplit(".", 1)[-1]
    bare = re.sub(r"_\d+$", "", bare) if bare not in PARAM_SYMBOLS else bare  # astropy index
    for base in sorted(PARAM_SYMBOLS, key=len, reverse=True):
        if bare == base:
            return base, ""
        if bare.startswith(base):
            match = _SUBSCRIPT_RE.match(bare[len(base):])
            if match:
                return base, match.group("sub")
    return bare, ""


def parameter_symbol(name: str, symbols: Optional[dict] = None) -> str:
    """LaTeX for one parameter, *without* math delimiters or model name.

    ``symbols`` overrides :data:`PARAM_SYMBOLS`; a model class can therefore
    declare its own vocabulary in a ``PARAM_SYMBOLS`` class attribute.
    """
    table = dict(PARAM_SYMBOLS)
    table.update(symbols or {})
    base, subscript = split_parameter_name(name)
    symbol = table.get(base)
    if symbol is None:
        return base.replace("_", "_{") + ("}" if "_" in base else "")
    return f"{symbol}_{{{subscript}}}" if subscript else symbol


def model_label(model) -> str:
    """Display name of a model: ``label`` if set, else ``name``."""
    label = getattr(model, "label", None)
    if label:
        return label
    name = getattr(model, "name", None)
    return name or type(model).__name__


def _submodel_of_parameter(model, index: int):
    """Return the submodel a flattened parameter belongs to, or ``None``."""
    # JAX compound models know the mapping exactly
    submodel_names = getattr(model, "submodel_names", None)
    qualified = getattr(model, "qualified_param_names", None)
    if submodel_names and qualified:
        try:
            return model[qualified[index].rsplit(".", 1)[0]]
        except (KeyError, IndexError, TypeError):
            return None
    # astropy compound models: the submodel index is the trailing number
    names = getattr(model, "param_names", ())
    if submodel_names and index < len(names):
        match = re.search(r"_(\d+)$", names[index])
        if match is not None:
            position = int(match.group(1))
            if position < len(submodel_names):
                try:
                    return model[submodel_names[position]]
                except (KeyError, IndexError, TypeError):
                    return None
    return None


def parameter_labels(model, with_model: bool = True) -> List[str]:
    """LaTeX label for every entry of ``model.param_names``.

    Parameters
    ----------
    model : JaxModel or astropy model
        Model whose parameters are labelled.
    with_model : bool
        Prefix each label with the model it belongs to, e.g.
        ``BH$\\alpha$:$\\sigma_{w0}$``.  Set to False for the bare symbol.

    Returns
    -------
    labels : list of str
        One mathtext string per parameter, aligned with ``model.param_names``.
    """
    names = list(getattr(model, "param_names", ()))
    labels = []
    for index, name in enumerate(names):
        submodel = _submodel_of_parameter(model, index)
        # a model may declare its own symbols: PARAM_SYMBOLS = {"w": "\\omega"}
        owner = submodel if submodel is not None else model
        override = getattr(type(owner), "PARAM_SYMBOLS", None)
        symbol = parameter_symbol(name, override)
        if not with_model:
            labels.append(f"${symbol}$")
            continue
        prefix = model_label(submodel) if submodel is not None else None
        labels.append(f"{prefix}:${symbol}$" if prefix else f"${symbol}$")
    return labels


def qualified_parameter_labels(model, names: Optional[Sequence[str]] = None) -> List[str]:
    """Labels for a list of *qualified* parameter names (``Halpha.sigma_w0``)."""
    if names is None:
        names = list(getattr(model, "qualified_param_names", ()))
    labels = []
    for name in names:
        base = name.rsplit(".", 1)[-1]
        prefix = name.rsplit(".", 1)[0] if "." in name else None
        submodel = None
        if prefix is not None:
            try:
                submodel = model[prefix]
            except (KeyError, IndexError, TypeError):
                submodel = None
        display = model_label(submodel) if submodel is not None else prefix
        owner = submodel if submodel is not None else model
        override = getattr(type(owner), "PARAM_SYMBOLS", None)
        labels.append(f"{display}:${parameter_symbol(base, override)}$" if display
                      else f"${parameter_symbol(base, override)}$")
    return labels


def labels_from_names(names: Iterable[str], displays: Optional[dict] = None) -> List[str]:
    """Labels from qualified names plus an optional ``{model_name: display}`` map."""
    displays = displays or {}
    out = []
    for name in names:
        base = name.rsplit(".", 1)[-1]
        prefix = name.rsplit(".", 1)[0] if "." in name else None
        display = displays.get(prefix, prefix) if prefix else None
        out.append(f"{display}:${parameter_symbol(base)}$" if display
                   else f"${parameter_symbol(base)}$")
    return out