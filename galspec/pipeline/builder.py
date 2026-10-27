"""Turning a ``[model]`` section into a JAX model, once per source.

The configuration describes the model declaratively -- a list of additive
components, their initial values, fixed flags, bounds, ties and priors -- and
this module builds the same :class:`~galspec.jax_models.JaxCompoundModel` a
Python script would have written by hand::

    m = PowerLaw1D(amplitude=3*peak_flux, x_0=5500, alpha=-1)
    m += Line_MultiGauss(n_components=2, wavec="@Halpha", ...)
    m["Halpha.amp_c1"].tied = ParamRefTie("Halpha", "amp_c0", scale=1/2.98)
    m["Halpha.amp_c0"].prior = Uniform(0.1*peak_flux, 5*peak_flux)

Nothing here changes the models; it only calls their public constructors and
the ``tied``/``prior`` setters, which is why a pipeline fit and a hand-written
fit of the same source produce the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .errors import PipelineError
from .schema import (ComponentConfig, ModelConfig, PriorSpec, TieSpec, TokenValue,
                     resolve_value)

__all__ = ["build_model", "describe_model"]


#: How each tie kind reads a reference parameter, as ``reference*scale+offset``.
#: The ratio conventions are the GalSpec ones (``tie_MultiGauss_doublet_ratio``
#: divides by the ratio; ``tie_Absorption_logtau0`` subtracts its logarithm).
_TIE_CONVENTIONS: Dict[str, Tuple[Optional[str], str]] = {
    # kind: (implied reference parameter, 'copy' | 'divide' | 'log-divide')
    "ref": (None, "copy"),
    "follow": (None, "copy"),
    "copy": (None, "copy"),
    "doublet_ratio": ("amp_c0", "divide"),
    "template_amplitude": ("amplitude", "divide"),
    "multigauss_amp_c": ("amp_c", "divide"),
    "logtau0_ratio": ("logtau0", "log-divide"),
    "template_dv": ("dv", "copy"),
    "starspectrum_sigma": ("sigma", "copy"),
    "starspectrum_deltaz": ("delta_z", "copy"),
    "absorption_sigma": ("sigma", "copy"),
    "absorption_dv": ("dv", "copy"),
    "absorption_cf": ("Cf", "copy"),
}


@dataclass(frozen=True)
class BuiltModel:
    """A model built for one source, with the names needed to report it."""

    model: Any                        # JaxCompoundModel or a single leaf
    components: Tuple[str, ...]       # submodel names, in sum order
    parameter_map: Mapping[str, str]  # 'Submodel.parameter' -> flat name
    kind_names: Mapping[str, str]     # submodel name -> model kind
    free_parameters: Tuple[str, ...] = ()   # flat names the sampler varies


def describe_model(model: Any) -> List[str]:
    """One line per additive component, for logs and the run report."""
    lines = []
    for name in getattr(model, "submodel_names", ()):
        leaf = model[name]
        lines.append(f"{name}: {type(leaf).__name__}"
                     + (f" ({leaf.model_kind})" if leaf.model_kind != type(leaf).__name__ else ""))
    return lines


# --------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------
def _number(value: Any, namespace: Mapping[str, float], path: str) -> float:
    """Resolve one configuration value (a number or an expression) to a float."""
    try:
        return float(resolve_value(value, namespace, path))
    except ValueError as exc:                   # ConfigError is a ValueError
        message = str(exc)
        prefix = "config: "
        if message.startswith(prefix):          # already names the TOML path
            message = message[len(prefix):]
        raise PipelineError(message) from None


def _bounds(pair, namespace, path) -> Tuple[Optional[float], Optional[float]]:
    low, high = pair
    return (_number(low, namespace, f"{path}[0]") if low is not None else None,
            _number(high, namespace, f"{path}[1]") if high is not None else None)


# --------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------
def build_model(config, namespace: Mapping[str, float]) -> BuiltModel:
    """Build the configured model for one source.

    Parameters
    ----------
    config : PipelineConfig
        A configuration read by :func:`galspec.pipeline.schema.load_config`.
    namespace : dict
        The per-source statistics from
        :func:`galspec.pipeline.schema.source_namespace`, used to resolve the
        expressions in the configuration.

    Returns
    -------
    BuiltModel

    Raises
    ------
    PipelineError
        If a component cannot be built, a tie or prior names something that does
        not exist, or a free parameter has no usable prior/bounds.
    """
    from ..jax_models import resolve_model_kind

    model_config: ModelConfig = config.model
    plans: List[Tuple[ComponentConfig, Any, str]] = []
    used_names: List[str] = []
    for index, component in enumerate(model_config.components):
        try:
            cls = resolve_model_kind(component.kind)
        except KeyError as exc:
            raise PipelineError(f"[model].component[{index}]: {exc}") from None
        name = component.name or cls.model_kind
        if not component.name:                  # derive a unique default
            base, counter = name, 2
            while name in used_names:
                name = f"{base}_{counter}"
                counter += 1
        elif name in used_names:
            raise PipelineError(
                f"[model].component[{index}]: duplicate component name {name!r}")
        used_names.append(name)
        plans.append((component, cls, name))

    base = Path(config.path).parent      # template_file is relative to the config
    leaves = [_build_component(component, cls, name, namespace, index, base)
              for index, (component, cls, name) in enumerate(plans)]

    model = leaves[0]
    for other in leaves[1:]:
        model = model + other

    # -- ties and priors, which need the composed model to resolve names --------
    built = BuiltModel(
        model=model,
        components=tuple(plans[i][2] for i in range(len(plans))),
        parameter_map={},
        kind_names={name: cls.model_kind for _, cls, name in plans},
    )
    for index, (component, cls, name) in enumerate(plans):
        label = f"[model].component[{index}] ({name})"
        for parameter, tie in component.ties.items():
            _apply_tie(model, name, parameter, tie, namespace, f"{label}.ties.{parameter}")
        for parameter, prior in component.priors.items():
            _apply_prior(model, name, parameter, prior, namespace,
                         f"{label}.priors.{parameter}")

    _check_free_parameters(model, model_config.require_priors)
    return replace(built, parameter_map=_parameter_map(model, plans),
                   free_parameters=tuple(model.free_param_names))


def _parameter_map(model, plans) -> Dict[str, str]:
    """``{'Halpha.amp_c0': 'amp_c0_2'}`` for the composed model."""
    if not hasattr(model, "n_submodels"):
        label = plans[0][2]
        return {f"{label}.{name}": name for name in model.param_names}
    mapping: Dict[str, str] = {}
    for index, leaf in enumerate(model.leaves):
        for base in leaf.param_names:
            mapping[f"{leaf.name}.{base}"] = f"{base}_{index}"
    return mapping


# --------------------------------------------------------------------------------------
# One component
# --------------------------------------------------------------------------------------
def _build_component(component: ComponentConfig, cls, name: str,
                     namespace: Mapping[str, float], index: int,
                     base: Optional[Path] = None):
    label = f"[model].component[{index}] ({name})"

    kwargs: Dict[str, Any] = {}
    deferred: Dict[str, Any] = {}          # values that may be expressions
    for key, value in component.values.items():
        if isinstance(value, float):
            kwargs[key] = value
        elif isinstance(value, TokenValue):
            deferred[key] = value
        elif isinstance(value, str):
            deferred[key] = value
        else:                               # list/dict constructor arguments
            kwargs[key] = value

    # A single template file may stand in for the two arrays the template
    # models require; the columns are (velocity, flux).
    template_file = kwargs.pop("template_file", None)
    if template_file is None:                   # a string ends up in `deferred`
        template_file = deferred.pop("template_file", None)
    if template_file is not None:
        array_kwargs = _template_arrays(_relative(template_file, base), label)
        for key, values in array_kwargs.items():
            if key in kwargs or key in deferred:
                raise PipelineError(f"{label}: give template_file or {key}, not both")
            kwargs[key] = values

    # Instantiate once without the deferred values to learn the parameter names;
    # those are the only keys that may hold an expression.
    try:
        probe = cls(name=name, fixed=dict(component.fixed),
                    bounds={k: _bounds(v, namespace, f"{label}.bounds.{k}")
                            for k, v in component.bounds.items()},
                    **kwargs)
    except (TypeError, ValueError, KeyError) as exc:
        raise PipelineError(f"{label}: cannot build {cls.model_kind}: {exc}") from None
    parameter_names = set(probe.param_names)

    for key, value in deferred.items():
        if key in parameter_names:
            kwargs[key] = _number(value, namespace, f"{label}.{key}")
        elif isinstance(value, str):
            kwargs[key] = value             # a string the constructor wants
        else:
            raise PipelineError(
                f"{label}.{key}: {value} is not the name of a parameter of "
                f"{cls.model_kind} (its parameters are "
                f"{', '.join(probe.param_names)}); an expression can only stand "
                "in for a parameter value")

    try:
        model = cls(name=name, fixed=dict(component.fixed),
                    bounds={k: _bounds(v, namespace, f"{label}.bounds.{k}")
                            for k, v in component.bounds.items()},
                    **kwargs)
    except (TypeError, ValueError, KeyError) as exc:
        raise PipelineError(f"{label}: cannot build {cls.model_kind}: {exc}") from None
    return model


def _relative(path: str, base: Optional[Path]) -> str:
    """``template_file`` as written in the config: absolute, or next to the config."""
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute() or base is None:
        return str(candidate)
    return str((base / candidate).resolve())


def _template_arrays(template_file: str, label: str) -> Dict[str, np.ndarray]:
    """Load a two-column template file into the arrays a template model needs."""
    path = Path(template_file).expanduser()
    if not path.is_absolute():
        raise PipelineError(f"{label}.template_file: {path} must be an absolute "
                            "path (the pipeline resolves it from the config file)")
    try:
        if path.suffix.lower() in (".npz", ".npy"):
            loaded = np.load(path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                keys = set(loaded.files)
                x_key = next((k for k in ("velc", "velocity", "wave", "x") if k in keys), None)
                y_key = next((k for k in ("flux", "y") if k in keys), None)
                if x_key is None or y_key is None:
                    raise PipelineError(f"{label}.template_file: {path} must hold "
                                        f"velocity/flux arrays or keys 'x'/'y'; it "
                                        f"has {sorted(keys)}")
                table = np.column_stack([np.asarray(loaded[x_key], dtype=float),
                                         np.asarray(loaded[y_key], dtype=float)])
            else:
                table = np.asarray(loaded, dtype=float)
        else:
            table = np.loadtxt(path, comments="#", dtype=float)
    except OSError as exc:
        raise PipelineError(f"{label}.template_file: cannot read {path}: {exc}") from None
    if table.ndim != 2 or table.shape[1] < 2:
        raise PipelineError(f"{label}.template_file: {path} must have two columns "
                            f"(velocity, flux); it has shape {table.shape}")
    order = np.argsort(table[:, 0], kind="stable")
    return {"template_velc": table[order, 0].copy(),
            "template_flux": table[order, 1].copy()}


# --------------------------------------------------------------------------------------
# Ties and priors
# --------------------------------------------------------------------------------------
def _parameter_handle(model, qualified: str, path: str):
    """The ``JaxParameter`` handle of ``"Submodel.parameter"``, or a clear error."""
    label, _, base = qualified.rpartition(".")
    if hasattr(model, "n_submodels"):
        leaf = model._by_name.get(label)
        if leaf is None or base not in leaf.param_names:
            available = ", ".join(model.qualified_param_names)
            raise PipelineError(
                f"{path}: {qualified!r} is not a parameter of this model; the "
                f"model has {available}")
        return getattr(leaf, base)
    if label and model.name and label != model.name:
        raise PipelineError(
            f"{path}: {label!r} is not a component of this model; it has only "
            f"{model.name!r}")
    if base not in model.param_names:
        raise PipelineError(
            f"{path}: {base!r} is not a parameter of {model.name!r}; its "
            f"parameters are {', '.join(model.param_names)}")
    return getattr(model, base)


def _apply_tie(model, component: str, parameter: str, tie: TieSpec,
               namespace: Mapping[str, float], path: str) -> None:
    from ..jax_models.serialization import ParamRefTie

    handle = _parameter_handle(model, f"{component}.{parameter}", path)
    implied_param, convention = _TIE_CONVENTIONS[tie.kind]
    ref_param = tie.param or implied_param
    if ref_param is None:
        raise PipelineError(
            f"{path}: kind = {tie.kind!r} needs param = \"<name of the reference "
            "parameter>\", or use a string tie \"Component.parameter\"")

    scale, offset = 1.0, 0.0
    if tie.scale is not None:
        scale = _number(tie.scale, namespace, f"{path}.scale")
    if tie.offset is not None:
        offset = _number(tie.offset, namespace, f"{path}.offset")
    if convention in ("divide", "log-divide"):
        if tie.ratio is None:                   # pragma: no cover - schema guards this
            raise PipelineError(f"{path}: kind = {tie.kind!r} needs ratio")
        ratio = _number(tie.ratio, namespace, f"{path}.ratio")
        if ratio == 0:
            raise PipelineError(f"{path}: ratio must not be zero")
        if convention == "divide":
            scale = (1.0 / ratio) if tie.scale is None else scale / ratio
        else:
            offset = -np.log10(ratio) if tie.offset is None else offset - np.log10(ratio)

    # the reference must exist and be readable, so a typo is a config error now
    _parameter_handle(model, f"{tie.ref}.{ref_param}", f"{path}.ref")

    tied = ParamRefTie(tie.ref, ref_param, scale=scale, offset=offset)
    try:
        handle.tied = tied
    except (AttributeError, ValueError) as exc:
        raise PipelineError(f"{path}: cannot tie {component}.{parameter}: {exc}") from None


def _apply_prior(model, component: str, parameter: str, spec: PriorSpec,
                 namespace: Mapping[str, float], path: str) -> None:
    from ..priors import _PRIOR_TYPES

    handle = _parameter_handle(model, f"{component}.{parameter}", path)
    matches = [cls for name, cls in _PRIOR_TYPES.items() if name.lower() == spec.kind]
    if not matches:
        raise PipelineError(f"{path}: unknown prior kind {spec.kind!r}")
    prior_class = matches[0]
    args = spec.resolve(namespace, path)
    try:
        prior = prior_class(**args)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{path}: cannot build {prior_class.__name__}: "
                            f"{exc}") from None
    try:
        handle.prior = prior
    except (AttributeError, ValueError, KeyError) as exc:
        raise PipelineError(f"{path}: cannot put a prior on {component}.{parameter}: "
                            f"{exc}") from None


# --------------------------------------------------------------------------------------
# Sanity checks that a batch run should fail in the first second
# --------------------------------------------------------------------------------------
def _check_free_parameters(model, require_priors: bool) -> None:
    """Every free parameter needs a finite sampling range, and a prior if asked.

    A parameter with an open bound and no prior would be sampled from
    ``lower + (upper - lower) * u`` with an infinite span, i.e. it would come
    back as ``inf`` after the first likelihood evaluation.  The configuration is
    the natural place to catch that: it is a second of validation instead of an
    hour of sampling.
    """
    from ..priors import CenteredPrior

    free = list(model.free_param_names)
    if not free:
        raise PipelineError("the configured model has no free parameter: every "
                            "parameter is fixed or tied, so there is nothing to fit")
    missing_bounds, missing_prior = [], []
    for name in free:
        spec = model.spec(name)
        prior = model.get_prior(name)
        if require_priors and prior is None:
            missing_prior.append(_full_name(model, name))
        low, high = spec.bounds
        if low is not None and high is not None:
            continue
        if prior is not None:
            prior_low = getattr(prior, "lower", None)
            prior_high = getattr(prior, "upper", None)
            if prior_low is not None and prior_high is not None:
                continue
            if not isinstance(prior, CenteredPrior):
                missing_bounds.append(_full_name(model, name) + " (prior has no limits)")
                continue
        missing_bounds.append(_full_name(model, name))
    if missing_prior:
        raise PipelineError(
            "[model] require_priors = true, but these free parameters have no "
            f"prior: {', '.join(missing_prior)}; give each one a "
            "[model.component.priors.<parameter>] table (or drop the flag)")
    if missing_bounds:
        raise PipelineError(
            "these free parameters have no finite sampling range: "
            f"{', '.join(missing_bounds)}; give each one [model.component.bounds.<parameter>] "
            "= [low, high], or a prior with finite limits "
            "(a centred prior such as Normal needs the bounds as well, because "
            "they truncate it)")


def _full_name(model, flat_name: str) -> str:
    """Map a flat ``amp_c0_2`` back to the ``Submodel.parameter`` spelling."""
    if not hasattr(model, "n_submodels"):
        return f"{model.name}.{flat_name}" if model.name else flat_name
    for index, leaf in enumerate(model.leaves):
        for base in leaf.param_names:
            if f"{base}_{index}" == flat_name:
                return f"{leaf.name}.{base}"
    return flat_name