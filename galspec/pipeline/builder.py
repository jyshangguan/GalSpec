"""Turning a ``[model]`` section into a JAX model, once per source.

The configuration describes the model declaratively -- a list of additive
components, their initial values, fixed flags, bounds, ties and priors -- and
this module builds the same :class:`~galspec.jax_models.JaxCompoundModel` a
Python script would have written by hand::

    m = PowerLaw1D(amplitude=3*peak_flux, x_0=5500, alpha=-1)
    m += Line_MultiGauss(n_components=2, wavec="@Halpha", ...)
    m["Halpha.amp_c1"].tied = ParamRefTie("Halpha", "amp_c0", scale=1/2.98)
    m["Halpha.amp_c0"].prior = Uniform(0.1*peak_flux, 5*peak_flux)

Nothing here changes the models; it only calls their public constructors and the
``tied``/``prior`` setters, which is why a pipeline fit and a hand-written fit of
the same source produce the same answer.

Nested sampling draws its live points from the prior, so a configuration does not
have to say where to start: an omitted parameter value falls back to the model
class's own default.  What it does have to say, for a free parameter, is how far
the sampler may stray -- and that too is optional: :data:`DEFAULT_PRIORS` gives
every free parameter the physically motivated prior of its kind (a logarithmic
amplitude, a few thousand km/s of velocity, hundreds to thousands of km/s of
width), scaled to the source where the unit would otherwise be a guess.  A prior
written in the configuration always wins, and the run report lists what was
defaulted, so nothing about the fit is silent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import PipelineError
from .schema import (ComponentConfig, ModelConfig, PriorSpec, TieSpec, TokenValue,
                     resolve_value)

__all__ = ["build_model", "describe_model", "DEFAULT_PRIORS"]


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
    #: ``'Halpha.sigma_c: uniform(300, 8000)'`` for every free parameter that
    #: took :data:`DEFAULT_PRIORS` instead of a prior written in the file.
    default_priors: Tuple[str, ...] = ()


def describe_model(model: Any) -> List[str]:
    """One line per additive component, for logs and the run report."""
    lines = []
    for name in getattr(model, "submodel_names", ()):
        leaf = model[name]
        lines.append(f"{name}: {type(leaf).__name__}"
                     + (f" ({leaf.model_kind})" if leaf.model_kind != type(leaf).__name__ else ""))
    return lines


# --------------------------------------------------------------------------------------
# The prior a free parameter gets when the file does not say
# --------------------------------------------------------------------------------------
#: ``(base-name prefixes, prior kind, lower, upper, scaled lower)``, tried in
#: order, for a free parameter that the configuration gives neither a prior nor
#: a bound.  The numbers describe the quantity rather than one source: a line
#: amplitude can be a thousandth of the spectrum or three times its peak (so the
#: prior is logarithmic), a velocity is a few thousand km/s, a width is hundreds
#: to thousands, a power-law index is a decade either way.
#:
#: The fifth entry is a second chance at the lower bound.  ``1e-3`` is a number
#: the *data* has to justify: in a file whose flux is not of order unity it can
#: sit above ``3*max_flux``, which leaves an empty interval.  When that happens
#: the default becomes ``1e-3*max_flux`` instead, and the run report says so.
DEFAULT_PRIORS: Tuple[Tuple[Tuple[str, ...], str, str, str, Optional[str]], ...] = (
    (("amp_w",), "uniform", "0.0", "1.0", None),
    (("amplitude", "amp"), "loguniform", "1e-3", "3*max_flux", "1e-3*max_flux"),
    (("dv_w",), "uniform", "-6000.0", "6000.0", None),
    (("dv",), "uniform", "-3000.0", "3000.0", None),
    (("sigma", "stddev"), "uniform", "300.0", "8000.0", None),
    (("alpha",), "uniform", "-5.0", "5.0", None),
    (("h3", "h4"), "uniform", "-0.5", "0.5", None),
    (("logtau",), "uniform", "-3.0", "3.0", None),
    (("cf",), "uniform", "0.0", "1.0", None),
    (("z", "redshift"), "uniform", "-0.1", "0.1", None),
    (("velscale",), "uniform", "10.0", "1000.0", None),
)

#: Parameters that say *which* feature a component describes, or where its
#: wavelength scale sits.  Their model default is not the line the user means
#: (``wavec = 5000``, ``x_0 = 1``), so a component that has one has to give it.
CENTRE_PARAMETERS: Tuple[str, ...] = ("wavec", "wavec0", "wavec1", "x_0")


def _default_prior_rule(base: str):
    """The rule that applies to the parameter ``base``, or ``None``."""
    lowered = base.lower()
    for prefixes, prior_kind, lower, upper, scaled in DEFAULT_PRIORS:
        if lowered.startswith(prefixes):
            return prior_kind, lower, upper, scaled
    return None


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
def build_model(config, namespace: Mapping[str, float],
                spectrum: Any = None) -> BuiltModel:
    """Build the configured model for one source.

    Parameters
    ----------
    config : PipelineConfig
        A configuration read by :func:`galspec.pipeline.schema.load_config`.
    namespace : dict
        The per-source statistics from
        :func:`galspec.pipeline.schema.source_namespace`, used to resolve the
        expressions in the configuration.
    spectrum : Spectrum, optional
        The source being fitted.  ``spectrum.narrow_template`` -- the
        ``(velocity, flux)`` table named by ``[data] narrow_template_key`` -- is
        the line profile of every component that takes one and does not name its
        own ``template_file``.

    Returns
    -------
    BuiltModel

    Raises
    ------
    PipelineError
        If a component cannot be built, a tie or prior names something that does
        not exist, a component that needs a line profile has none, or a free
        parameter has no usable sampling range.
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
    template = None if spectrum is None else getattr(spectrum, "narrow_template", None)
    leaves = [_build_component(component, cls, name, namespace, index, base, template)
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

    # a prior written in the file is what ``require_priors`` means by "a prior";
    # the defaults below fill in the rest of the parameters, not the flag's
    explicit = {f"{name}.{parameter}" for component, _cls, name in plans
                for parameter in component.priors}
    _check_required_priors(model, model_config.require_priors, explicit)
    defaults = _apply_default_priors(model, plans, namespace)
    _check_sampling_boxes(model)
    return replace(built, parameter_map=_parameter_map(model, plans),
                   free_parameters=tuple(model.free_param_names),
                   default_priors=defaults)


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
                     base: Optional[Path] = None,
                     template: Optional[Tuple[np.ndarray, np.ndarray]] = None):
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
    elif _template_arguments(cls):
        # no file of its own: the spectrum's own profile, if it carries one
        if template is None:
            raise PipelineError(
                f"{label}: {cls.model_kind} fits a measured line profile, so it "
                "needs one: give template_file = \"<velocity, flux table>\" here, "
                "or set [data] narrow_template_key = \"<HDU name>\" to read the "
                "profile out of each spectrum")
        kwargs["template_velc"], kwargs["template_flux"] = template

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
    _require_centre(component, probe, label)

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


@lru_cache(maxsize=None)
def _template_arguments(cls) -> Tuple[str, ...]:
    """The template-array arguments a model kind takes, if it takes any."""
    try:
        parameters = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):              # pragma: no cover - defensive
        return ()
    return tuple(key for key in ("template_velc", "template_flux") if key in parameters)


def _require_centre(component: ComponentConfig, probe, label: str) -> None:
    """A component whose model sits at a wavelength has to say where.

    Nested sampling does not care about an initial value, but two things do: a
    ``wavec``/``x_0`` names *which* line or pivot the component is, and its model
    default (5000 A, 1 A) is almost never that.  A bound counts as saying so,
    because it pins the window the sampler draws from.
    """
    names = set(probe.param_names)
    for key in CENTRE_PARAMETERS:
        if key not in names:
            continue
        if key in component.values or key in component.bounds or key in component.ties:
            continue
        raise PipelineError(
            f"{label}: {key} has no value, so the component does not say which "
            f"feature it describes (the model default is "
            f"{probe.spec(key).default!r}).  Write {key} = 6562.8, or one of the "
            "line names such as \"@Halpha\" or \"@OIII_5007\")")


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
def _apply_default_priors(model, plans, namespace: Mapping[str, float]
                          ) -> Tuple[str, ...]:
    """Give every free parameter the :data:`DEFAULT_PRIORS` prior of its kind.

    A prior or a bound written in the configuration always wins, so this only
    fills in what the file left unsaid -- which is what makes a minimal
    configuration possible.  Returns the resolved descriptions, for the report:
    they name the numbers the sampler actually drew from, not the rule.
    """
    from ..priors import LogUniform, Uniform

    configured = {f"{name}.{parameter}"
                  for component, _cls, name in plans
                  for parameter in list(component.priors) + list(component.bounds)}
    applied: List[str] = []
    for flat in model.free_param_names:
        full = _full_name(model, flat)
        if full in configured:
            continue
        if model.get_prior(flat) is not None:   # the model class set one itself
            continue
        rule = _default_prior_rule(full.rpartition(".")[2])
        if rule is None:
            continue
        prior_kind, lower, upper, scaled = rule
        path = f"[model] default prior of {full}"
        low = _number(lower, namespace, path)
        high = _number(upper, namespace, path)
        note = ""
        if not low < high:
            if scaled is None:                  # pragma: no cover - table is fixed
                raise PipelineError(
                    f"{path}: the default interval [{low:g}, {high:g}] is empty; "
                    "give the parameter a prior or bounds in the configuration")
            low = _number(scaled, namespace, path)
            if not low < high:
                raise PipelineError(
                    f"{path}: the default interval [{low:g}, {high:g}] is empty "
                    "for this spectrum; give the parameter a prior or bounds in "
                    "the configuration")
            note = " (scaled to the spectrum)"
        prior_class = LogUniform if prior_kind == "loguniform" else Uniform
        try:
            prior = prior_class(lower=low, upper=high)
            handle = _parameter_handle(model, full, "[model]")
            handle.prior = prior
        except (AttributeError, ValueError, KeyError) as exc:
            raise PipelineError(f"{path}: cannot build {prior_class.__name__}: "
                                f"{exc}") from None
        applied.append(f"{full}: {prior_class.__name__.lower()}({low:.6g}, "
                       f"{high:.6g}){note}")
    return tuple(applied)


def _check_required_priors(model, require_priors: bool, explicit) -> None:
    """``[model] require_priors = true`` asks for a prior written in the file.

    A default prior is not what the flag is about -- it exists to make a
    configuration that leaves the priors to the pipeline visible, so it counts
    only the priors the file states.
    """
    if not require_priors:
        return
    missing = [_full_name(model, name) for name in model.free_param_names
               if _full_name(model, name) not in explicit]
    if missing:
        raise PipelineError(
            "[model] require_priors = true, but these free parameters have no "
            f"prior written in the file: {', '.join(missing)}; give each one a "
            "[model.component.priors.<parameter>] table, or drop the flag")


def _check_sampling_boxes(model) -> None:
    """Every free parameter needs a finite interval for the sampler to draw in.

    The interval comes from a prior written in the file, or from
    :data:`DEFAULT_PRIORS` when the parameter is a named physical quantity.  What
    is *not* given a default is a parameter whose range depends on the data in a
    way the name cannot express -- a linear continuum's ``slope`` and
    ``intercept``, or a *free* ``wavec`` -- because a wrong box silently changes
    the evidence instead of failing.  Those still have to be written down, which
    is a second of validation instead of an hour of sampling.
    """
    from ..priors import CenteredPrior

    free = list(model.free_param_names)
    if not free:
        raise PipelineError("the configured model has no free parameter: every "
                            "parameter is fixed or tied, so there is nothing to fit")
    missing = []
    for name in free:
        spec = model.spec(name)
        prior = model.get_prior(name)
        low, high = spec.bounds
        if low is not None and high is not None:
            continue
        if prior is not None:
            prior_low = getattr(prior, "lower", None)
            prior_high = getattr(prior, "upper", None)
            if prior_low is not None and prior_high is not None:
                continue
            if not isinstance(prior, CenteredPrior):
                missing.append(_full_name(model, name) + " (prior has no limits)")
                continue
        missing.append(_full_name(model, name))
    if missing:
        raise PipelineError(
            "these free parameters have no finite sampling range: "
            f"{', '.join(missing)}; give each one [model.component.bounds.<parameter>] "
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