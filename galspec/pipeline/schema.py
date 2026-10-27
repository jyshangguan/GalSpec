"""The TOML configuration of the fitting pipeline.

This module turns one ``.toml`` file into the frozen dataclasses the rest of
:mod:`galspec.pipeline` consumes.  It is deliberately free of ``jax``: parsing,
validation and the per-source number resolution are pure Python, so a mistake in
a configuration file costs a millisecond instead of a compile.

Layout
------
.. code-block:: toml

    [run]        # where the output goes, how many workers, resume behaviour
    [data]       # where the spectra are and how to read them
    [fit]        # sampler settings (nlive, dlogz, device, ...)
    [output]     # what to write per source (thin, plots)
    [model]      # require_priors
    [[model.component]]   # one entry per additive component, in sum order

Expressions
-----------
Any *number* inside ``[model]`` may be written as an expression over the
per-source data statistics listed in ``TOKEN_NAMES`` -- ``"3*peak_flux"``,
``"0.5*continuum_level"`` -- which is what makes one configuration serve a whole
sample whose flux scale differs from spectrum to spectrum.  The expression
language is a deliberately small subset of Python: numbers, the token names,
``+ - * / **`` and parentheses.  Nothing else is accepted.
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:                                            # Python >= 3.11
    import tomllib as _toml
except ModuleNotFoundError:                     # pragma: no cover - 3.9/3.10
    import tomli as _toml                       # type: ignore[no-redef]


class ConfigError(ValueError):
    """A configuration file that cannot be used as written.

    The message always names the TOML path of the offending entry
    (``config: [model.component] #2 (Halpha).amp_c0: ...``), so the file can be
    fixed without re-reading the documentation.
    """


#: Statistics available to a ``[model]`` expression, computed from each spectrum.
#: ``peak_flux``/``median_flux``/``continuum_level`` are robust levels (percentile
#: based, so a bright emission line does not set the scale of a faint one);
#: ``rms`` is the standard deviation of the flux.
TOKEN_NAMES = ("peak_flux", "median_flux", "continuum_level", "rms",
               "min_flux", "max_flux", "n_pixels", "z")

#: Percentile used for ``continuum_level`` when ``[data] continuum_percentile``
#: is not given.
DEFAULT_CONTINUUM_PERCENTILE = 50.0


# --------------------------------------------------------------------------------------
# Values that may be an expression
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenValue:
    """A number written as an expression over the per-source statistics."""

    expression: str

    def __str__(self) -> str:
        return self.expression

    def evaluate(self, namespace: Mapping[str, float], path: str = "<value>") -> float:
        """Resolve the expression with this source's statistics."""
        return evaluate_token(self.expression, namespace, path)


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def compile_token(expression: str, path: str = "<value>") -> Any:
    """Parse an expression once and return it as a literal or evaluated tree.

    Parsing happens when the configuration is read, so a typo in
    ``"3*peak_flx"`` is reported before any spectrum is touched.
    """
    if not isinstance(expression, str):
        raise ConfigError(f"config: {path}: expected a number or an expression "
                          f"string, got {type(expression).__name__}")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ConfigError(f"config: {path}: cannot parse expression "
                          f"{expression!r} ({exc.msg})") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        if isinstance(node, ast.Name):
            if node.id not in TOKEN_NAMES:
                raise ConfigError(
                    f"config: {path}: unknown token {node.id!r} in {expression!r}; "
                    f"available tokens are {', '.join(TOKEN_NAMES)}")
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            continue
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            continue
        raise ConfigError(
            f"config: {path}: {expression!r} uses {type(node).__name__}, which the "
            "expression language does not allow; only numbers, the available "
            "tokens, + - * / ** and parentheses are accepted")
    return tree


def evaluate_token(expression: str, namespace: Mapping[str, float],
                   path: str = "<value>") -> float:
    """Evaluate a compiled expression with one source's statistics."""
    tree = compile_token(expression, path)

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            try:
                return float(namespace[node.id])
            except KeyError:                    # pragma: no cover - compile_token guards
                raise ConfigError(f"config: {path}: token {node.id!r} is not "
                                  "available for this source") from None
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            try:
                return _BINARY[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ConfigError(
                    f"config: {path}: {expression!r} divides by zero for this "
                    f"source (tokens: {dict(namespace)})") from None
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](walk(node.operand))
        raise ConfigError(f"config: {path}: unsupported expression {expression!r}")

    value = walk(tree)
    if not math.isfinite(value):
        raise ConfigError(f"config: {path}: {expression!r} evaluates to "
                          f"{value!r} for this source; check the tokens")
    return float(value)


def _as_number(value: Any, path: str) -> Any:
    """``18`` -> 18.0; ``"3*peak_flux"`` -> TokenValue; anything else is an error."""
    if isinstance(value, bool):
        raise ConfigError(f"config: {path}: expected a number, got a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        compile_token(value, path)
        return TokenValue(value)
    raise ConfigError(f"config: {path}: expected a number or an expression string, "
                      f"got {type(value).__name__}")


def resolve_value(value: Any, namespace: Mapping[str, float],
                  path: str = "<value>") -> float:
    """Resolve a configuration value -- a number or an expression -- to a float."""
    if isinstance(value, TokenValue):
        return value.evaluate(namespace, path)
    resolved = _as_number(value, path)
    if isinstance(resolved, TokenValue):
        return resolved.evaluate(namespace, path)
    return resolved


# --------------------------------------------------------------------------------------
# Per-component descriptions
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorSpec:
    """One prior, as written in the configuration (arguments may be expressions)."""

    kind: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self, namespace: Mapping[str, float], path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.args.items():
            if isinstance(value, TokenValue):
                out[key] = value.evaluate(namespace, f"{path}.{key}")
            else:
                out[key] = value
        return out


@dataclass(frozen=True)
class TieSpec:
    """One tie, as written in the configuration.

    ``ref`` names the submodel that owns the reference parameter.  ``kind``
    selects the meaning of the reference: a plain copy (``ref``/``follow``), or
    one of the GalSpec conventions that divides by a ratio
    (``doublet_ratio``, ``template_amplitude``, ``multigauss_amp_c``) or shifts a
    logarithm (``logtau0_ratio``).  All of them reduce to
    ``reference * scale + offset`` and are stored as such, so a saved fit never
    needs Python code to be understood.
    """

    kind: str
    ref: str
    param: Optional[str] = None
    ratio: Optional[Any] = None
    scale: Optional[Any] = None
    offset: Optional[Any] = None


@dataclass(frozen=True)
class ComponentConfig:
    """One additive component of the model."""

    kind: str                                   # resolved model class name
    name: Optional[str]                         # None -> derive from the kind
    values: Mapping[str, Any] = field(default_factory=dict)       # initial values
    ctor: Mapping[str, Any] = field(default_factory=dict)         # non-parameter args
    fixed: Mapping[str, bool] = field(default_factory=dict)
    bounds: Mapping[str, Tuple[Any, Any]] = field(default_factory=dict)
    ties: Mapping[str, TieSpec] = field(default_factory=dict)
    priors: Mapping[str, PriorSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    components: Tuple[ComponentConfig, ...]
    require_priors: bool = False


# --------------------------------------------------------------------------------------
# Run-level sections
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    source: str                                 # 'file' | 'files' | 'glob' | 'catalog'
    file: Optional[str] = None
    files: Tuple[str, ...] = ()
    glob: Optional[str] = None
    catalog: Optional[str] = None
    catalog_id_column: str = "id"
    catalog_file_column: str = "file"
    catalog_z_column: Optional[str] = None
    wave_key: str = "WAVELENGTH"
    flux_key: str = "FLUX"
    error_key: str = "ERROR"
    wave_hdu: Optional[str] = None
    flux_hdu: Optional[str] = None
    error_hdu: Optional[str] = None
    wave_unit: Optional[str] = None
    flux_unit: Optional[str] = None
    z_keys: Tuple[str, ...] = ("Z", "REDSHIFT", "ZBEST")
    z_default: float = 0.0
    #: True when the file says ``z_default`` itself.  A default the *file* does
    #: not mention is an assumption the pipeline warns about; one the file states
    #: is a decision, and warning about it would train the reader to ignore both.
    z_default_given: bool = False
    extinction: bool = True
    e_bv_keys: Tuple[str, ...] = ("E(B-V)", "EBV", "EBV_GAL", "A_V", "AV")
    extinction_rv: float = 3.1
    wave_min: Optional[float] = None
    wave_max: Optional[float] = None
    mask: Tuple[Tuple[float, float], ...] = ()
    min_pixels: int = 50
    continuum_percentile: float = DEFAULT_CONTINUUM_PERCENTILE


@dataclass(frozen=True)
class FitConfig:
    nlive: int = 500
    dlogz: float = 0.01
    max_iterations: int = 200000
    bound: str = "multi"
    device: str = "auto"
    queue_size: Optional[int] = None
    batch_size: Optional[int] = None
    rwalk_K: Optional[int] = None
    rwalk_step_scale: Optional[float] = None
    max_ellipsoids: int = 20
    memory_frac: float = 0.9
    min_eff: float = 10.0
    min_ncall: Optional[int] = None
    share_compilation: bool = True
    compilation_cache_dir: Optional[str] = None


@dataclass(frozen=True)
class OutputConfig:
    thin: int = 1
    fit_plot: bool = True
    corner: bool = False
    histogram: bool = False
    corner_params: Optional[Tuple[str, ...]] = None
    hist_params: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "."
    force: bool = False
    workers: int = 1
    #: ``None`` (the default) draws the sampler's progress bar when standard error
    #: is a terminal; ``True`` draws it always, ``False`` never.
    progress: Optional[bool] = None


@dataclass(frozen=True)
class PipelineConfig:
    """A validated configuration, with every path made absolute."""

    path: str
    digest: str
    data: DataConfig
    model: ModelConfig
    fit: FitConfig
    output: OutputConfig
    run: RunConfig

    @property
    def output_path(self) -> Path:
        return Path(self.run.output_dir)


# --------------------------------------------------------------------------------------
# Reading and validation
# --------------------------------------------------------------------------------------
_TOP_LEVEL = {"run", "data", "model", "fit", "output"}


def _fail(path: str, message: str) -> None:
    raise ConfigError(f"config: {path}: {message}")


def _check_keys(table: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        _fail(path, f"unknown key{'s' if len(unknown) > 1 else ''} "
                    f"{', '.join(repr(key) for key in unknown)}; "
                    f"allowed keys are {', '.join(allowed)}")


def _pool(values: Mapping, defaults: Mapping, path: str) -> Dict[str, Any]:
    """Validated ``{name: default}`` extraction with type coercion."""
    out = dict(defaults)
    for key in values:
        if key not in defaults:
            _fail(path, f"unknown key {key!r}; allowed keys are "
                        f"{', '.join(defaults)}")
    for key, default in defaults.items():
        if key not in values:
            continue
        value = values[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                _fail(f"{path}.{key}", f"expected true/false, got {value!r}")
            out[key] = value
        elif isinstance(default, int) and not isinstance(default, bool):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if not isinstance(value, int) or isinstance(value, bool):
                _fail(f"{path}.{key}", f"expected an integer, got {value!r}")
            out[key] = value
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _fail(f"{path}.{key}", f"expected a number, got {value!r}")
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _string_list(value: Any, path: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str)
                                                       for v in value):
        _fail(path, f"expected a table of strings, got {value!r}")
    return tuple(value)


def _resolve_path(value: Optional[str], base: Path) -> Optional[str]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path)


def load_config(filename) -> PipelineConfig:
    """Read and validate one ``.toml`` configuration file.

    Parameters
    ----------
    filename : str or pathlib.Path
        The configuration file.  Relative paths it contains are resolved against
        the directory of the file, not the current working directory.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    ConfigError
        Whenever the file is not exactly what the pipeline understands; the
        message names the TOML path of the problem.
    """
    path = Path(filename).expanduser()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    try:
        document = _toml.loads(raw_text)
    except _toml.TOMLDecodeError as exc:
        raise ConfigError(f"config: {path}: not valid TOML ({exc})") from None
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    base = path.parent

    _check_keys(document, sorted(_TOP_LEVEL), str(path))
    for section in ("data", "model"):
        if section not in document:
            raise ConfigError(f"config: {path}: missing required section "
                              f"[{section}]")

    data = _read_data(document.get("data") or {}, base)
    model = _read_model(document.get("model") or {})
    fit = _read_fit(document.get("fit") or {}, base)
    output = _read_output(document.get("output") or {})
    run = _read_run(document.get("run") or {}, base)
    return PipelineConfig(path=str(path.resolve()), digest=digest, data=data,
                          model=model, fit=fit, output=output, run=run)


def _read_data(table: Mapping[str, Any], base: Path) -> DataConfig:
    defaults = {
        "file": None, "files": (), "glob": None, "catalog": None,
        "catalog_id_column": "id", "catalog_file_column": "file",
        "catalog_z_column": None,
        "wave_key": "WAVELENGTH", "flux_key": "FLUX", "error_key": "ERROR",
        "wave_hdu": None, "flux_hdu": None, "error_hdu": None,
        "wave_unit": None, "flux_unit": None,
        "z_keys": ("Z", "REDSHIFT", "ZBEST"), "z_default": 0.0,
        "extinction": True,
        "e_bv_keys": ("E(B-V)", "EBV", "EBV_GAL", "A_V", "AV"),
        "extinction_rv": 3.1,
        "wave_min": None, "wave_max": None, "mask": (),
        "min_pixels": 50, "continuum_percentile": DEFAULT_CONTINUUM_PERCENTILE,
    }
    values = _pool(table, defaults, "[data]")
    chosen = [key for key in ("file", "files", "glob", "catalog")
              if table.get(key)]
    if len(chosen) != 1:
        _fail("[data]", "give exactly one of file, files, glob or catalog "
                        f"(got {', '.join(chosen) or 'none'})")
    source = chosen[0]

    mask = table.get("mask") or ()
    if isinstance(mask, list) and mask and isinstance(mask[0], (int, float)):
        mask = [mask]
    windows = []
    for index, window in enumerate(mask):
        if (not isinstance(window, (list, tuple)) or len(window) != 2
                or not all(isinstance(v, (int, float)) for v in window)):
            _fail(f"[data].mask[{index}]", f"expected [low, high], got {window!r}")
        low, high = float(window[0]), float(window[1])
        if not low < high:
            _fail(f"[data].mask[{index}]", f"need low < high, got [{low}, {high}]")
        windows.append((low, high))
    if values["wave_min"] is not None and values["wave_max"] is not None:
        if not values["wave_min"] < values["wave_max"]:
            _fail("[data]", "wave_min must be below wave_max")
    for key in ("min_pixels",):
        if values[key] < 1:
            _fail(f"[data].{key}", f"must be at least 1, got {values[key]}")
    if not 0.0 <= values["continuum_percentile"] <= 100.0:
        _fail("[data].continuum_percentile", "must be between 0 and 100")
    if values["extinction_rv"] <= 0.0:
        _fail("[data].extinction_rv", f"must be positive, got {values['extinction_rv']}")

    return DataConfig(
        source=source,
        file=_resolve_path(values["file"], base),
        files=tuple(_resolve_path(p, base) for p in _string_list(values["files"], "[data].files")),
        glob=values["glob"],
        catalog=_resolve_path(values["catalog"], base),
        catalog_id_column=values["catalog_id_column"],
        catalog_file_column=values["catalog_file_column"],
        catalog_z_column=values["catalog_z_column"],
        wave_key=values["wave_key"], flux_key=values["flux_key"],
        error_key=values["error_key"],
        wave_hdu=values["wave_hdu"], flux_hdu=values["flux_hdu"],
        error_hdu=values["error_hdu"],
        wave_unit=values["wave_unit"], flux_unit=values["flux_unit"],
        z_keys=_string_list(values["z_keys"], "[data].z_keys"),
        z_default=float(values["z_default"]),
        z_default_given="z_default" in table,
        extinction=bool(values["extinction"]),
        e_bv_keys=_string_list(values["e_bv_keys"], "[data].e_bv_keys"),
        extinction_rv=float(values["extinction_rv"]),
        wave_min=values["wave_min"], wave_max=values["wave_max"],
        mask=tuple(windows), min_pixels=int(values["min_pixels"]),
        continuum_percentile=float(values["continuum_percentile"]),
    )


#: Constructor arguments a config may set on a component, per model kind.
#: Anything not listed here and not a parameter is rejected, so a typo cannot
#: silently become an unused keyword argument.
RESERVED_COMPONENT_KEYS = ("type", "name", "params", "fixed", "bounds", "ties",
                           "priors", "label")

#: Parameter names that may be written as ``"@Halpha"`` -- resolved from
#: :data:`galspec.utils.line_wave_dict`.
WAVE_TOKENS = ("wavec", "wavec0", "wavec1", "x_0", "line_wave")


def _read_model(table: Mapping[str, Any]) -> ModelConfig:
    _check_keys(table, ("component", "components", "require_priors"), "[model]")
    entries = table.get("component", table.get("components"))
    if entries is None:
        _fail("[model]", "missing component table: list at least one "
                         "[[model.component]]")
    if not isinstance(entries, list):
        _fail("[model].component", "expected an array of tables "
                                   "([[model.component]]), got "
                                   f"{type(entries).__name__}")
    require_priors = table.get("require_priors", False)
    if not isinstance(require_priors, bool):
        _fail("[model].require_priors", "expected true/false")

    components = []
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            _fail(f"[model].component[{index}]", "expected a table")
        component = _read_component(entry, index)
        if component.name is not None:
            if component.name in seen:
                _fail(f"[model].component[{index}].name",
                      f"duplicate component name {component.name!r}; each component "
                      "needs a name that is unique in the model")
            seen.add(component.name)
        components.append(component)
    if not components:
        _fail("[model].component", "the model needs at least one component")
    return ModelConfig(components=tuple(components),
                       require_priors=bool(require_priors))


def _read_component(entry: Mapping[str, Any], index: int) -> ComponentConfig:
    label = f"[model].component[{index}]"
    kind = entry.get("type")
    if not isinstance(kind, str) or not kind:
        _fail(f"{label}.type", "every component needs a type, e.g. "
                               'type = "Line_MultiGauss"')
    name = entry.get("name")
    if name is not None and (not isinstance(name, str) or not name):
        _fail(f"{label}.name", "expected a non-empty string")
    label = f"[model].component[{index}] ({name or kind})"

    values: Dict[str, Any] = {}
    ctor: Dict[str, Any] = {}
    fixed = {}
    bounds = {}
    ties: Dict[str, TieSpec] = {}
    priors: Dict[str, PriorSpec] = {}

    for key, value in entry.items():
        if key in RESERVED_COMPONENT_KEYS:
            continue
        if key in WAVE_TOKENS and isinstance(value, str) and value.startswith("@"):
            from ..utils import line_wave_dict

            line = value[1:]
            if line not in line_wave_dict:
                _fail(f"{label}.{key}",
                      f"unknown line {line!r}; known lines are "
                      f"{', '.join(sorted(line_wave_dict))}")
            values[key] = float(line_wave_dict[line])
        elif isinstance(value, bool):
            fixed[key] = value
        elif isinstance(value, (int, float)):
            values[key] = float(value)
        elif isinstance(value, str):
            values[key] = _component_string(value, f"{label}.{key}",
                                            is_path=key.endswith("_file"))
        elif isinstance(value, (list, tuple)):
            ctor[key] = list(value)             # e.g. Star_types = [...]
        elif isinstance(value, dict):
            ctor[key] = value
        else:                                   # pragma: no cover - TOML has no other type
            _fail(f"{label}.{key}", f"unsupported value {value!r}")

    # -- the explicit parameter table ------------------------------------------
    params = entry.get("params") or {}
    if not isinstance(params, dict):
        _fail(f"{label}.params", "expected a table of per-parameter settings")
    for pname, settings in params.items():
        if not isinstance(settings, dict):
            _fail(f"{label}.params.{pname}", "expected a table")
        _check_keys(settings, ("value", "fixed", "bounds", "tie", "prior"),
                    f"{label}.params.{pname}")
        if "value" in settings:
            values[pname] = _as_number(settings["value"],
                                       f"{label}.params.{pname}.value")
        if "fixed" in settings:
            if not isinstance(settings["fixed"], bool):
                _fail(f"{label}.params.{pname}.fixed", "expected true/false")
            fixed[pname] = settings["fixed"]
        if "bounds" in settings:
            bounds[pname] = _read_bounds(settings["bounds"],
                                         f"{label}.params.{pname}.bounds")
        if "tie" in settings:
            ties[pname] = _read_tie(settings["tie"], f"{label}.params.{pname}.tie")
        if "prior" in settings:
            priors[pname] = _read_prior(settings["prior"],
                                        f"{label}.params.{pname}.prior")

    for pname, value in (entry.get("fixed") or {}).items():
        if not isinstance(value, bool):
            _fail(f"{label}.fixed.{pname}", "expected true/false")
        fixed[pname] = value
    for pname, value in (entry.get("bounds") or {}).items():
        bounds[pname] = _read_bounds(value, f"{label}.bounds.{pname}")
    for pname, value in (entry.get("ties") or {}).items():
        ties[pname] = _read_tie(value, f"{label}.ties.{pname}")
    for pname, value in (entry.get("priors") or {}).items():
        priors[pname] = _read_prior(value, f"{label}.priors.{pname}")

    return ComponentConfig(kind=kind, name=name, values=values, ctor=ctor,
                           fixed=fixed, bounds=bounds, ties=ties, priors=priors)


def _component_string(value: str, path: str, is_path: bool = False) -> Any:
    """Classify one string in a component table.

    A path-valued key (``template_file``), a bare identifier (``"park2022"``,
    ``"iron_template_optical"``) and anything that is not Python at all are
    strings a constructor wants, and are passed through.  Anything that *is* an
    expression -- a call, an operator, a number -- has to be a valid one, so a
    typo is reported when the file is read rather than after the spectrum has
    been loaded.

    ``template_file`` is exempt rather than guessed from its text: a relative
    path like ``data/template.txt`` is a division of two names as far as
    :mod:`ast` is concerned, and ``1.8/2.98`` is a division that *is* wanted --
    the key is the only thing that tells them apart.
    """
    if is_path:
        return value
    text = value.strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return value
    if isinstance(tree.body, ast.Name):
        # could be a token (``"peak_flux"``) or an opaque name; the builder
        # decides, because only it knows which keys are parameters
        return value
    return _as_number(value, path)


def _read_bounds(value: Any, path: str) -> Tuple[Any, Any]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2):
        _fail(path, f"expected [low, high], got {value!r}")
    low = _as_number(value[0], f"{path}[0]") if value[0] is not None else None
    high = _as_number(value[1], f"{path}[1]") if value[1] is not None else None
    return (low, high)


def _read_tie(value: Any, path: str) -> TieSpec:
    if isinstance(value, str):
        return _tie_from_reference(value, path)
    if not isinstance(value, dict):
        _fail(path, "expected a reference string (\"Comp.param\") or a table")
    _check_keys(value, ("ref", "kind", "param", "ratio", "scale", "offset"), path)
    kind = str(value.get("kind", "ref"))
    if "ref" not in value:
        _fail(path, "a tie needs ref = \"OtherComponent.parameter\"")
    ref = value["ref"]
    if not isinstance(ref, str):
        _fail(f"{path}.ref", "expected a string")
    param = value.get("param")
    if param is not None and not isinstance(param, str):
        _fail(f"{path}.param", "expected a string")
    ratio = (_as_number(value["ratio"], f"{path}.ratio")
             if "ratio" in value else None)
    scale = (_as_number(value["scale"], f"{path}.scale")
             if "scale" in value else None)
    offset = (_as_number(value["offset"], f"{path}.offset")
              if "offset" in value else None)
    if kind in TIE_KINDS_WITH_RATIO and ratio is None:
        _fail(path, f"kind = {kind!r} needs ratio = <number>")
    if kind not in TIE_KINDS:
        _fail(path, f"unknown tie kind {kind!r}; known kinds are "
                    f"{', '.join(TIE_KINDS)}")
    return TieSpec(kind=kind, ref=ref, param=param, ratio=ratio,
                   scale=scale, offset=offset)


def _tie_from_reference(value: str, path: str) -> TieSpec:
    if "." not in value:
        _fail(path, f"a tie written as a string must name a parameter, "
                    f"e.g. \"Halpha.amp_c0\"; got {value!r}")
    ref, _, param = value.rpartition(".")
    return TieSpec(kind="ref", ref=ref, param=param)


#: Tie kinds understood by the builder.  ``ref``/``follow`` copy a parameter;
#: the others are the GalSpec conventions that divide by (or shift a log by) a
#: ratio, and all are stored as ``reference * scale + offset``.
TIE_KINDS = ("ref", "follow", "copy", "doublet_ratio", "template_amplitude",
             "template_dv", "multigauss_amp_c", "logtau0_ratio",
             "starspectrum_sigma", "starspectrum_deltaz",
             "absorption_sigma", "absorption_dv", "absorption_cf")
TIE_KINDS_WITH_RATIO = ("doublet_ratio", "template_amplitude", "multigauss_amp_c",
                        "logtau0_ratio")

#: Prior kinds, matched against :data:`galspec.priors._PRIOR_TYPES` by name.
PRIOR_KINDS = ("uniform", "tophat", "loguniform", "spikeandslab", "normal",
               "clippednormal", "centeredprior")

#: Prior arguments that are numbers (and may therefore be expressions); every
#: other argument is a name or a flag and is passed through untouched.
PRIOR_NUMERIC_ARGS = ("lower", "upper", "scale", "ratio", "nsigma", "mini",
                      "maxi", "p_off", "sigma", "relative")


def _read_prior(value: Any, path: str) -> PriorSpec:
    if isinstance(value, (list, tuple)):        # a bare (lower, upper) box
        return PriorSpec(kind="uniform", args=_read_bounds(value, path))
    if not isinstance(value, dict):
        _fail(path, "expected a table such as { kind = \"uniform\", lower = 0.1, "
                    "upper = 10.0 }")
    if "kind" not in value:
        _fail(path, "a prior needs kind, e.g. kind = \"uniform\"")
    kind = str(value["kind"]).lower()
    if kind not in PRIOR_KINDS:
        _fail(f"{path}.kind", f"unknown prior kind {value['kind']!r}; known kinds "
                              f"are {', '.join(PRIOR_KINDS)}")
    args: Dict[str, Any] = {}
    for key, argument in value.items():
        if key == "kind":
            continue
        if key in PRIOR_NUMERIC_ARGS:
            args[key] = _as_number(argument, f"{path}.{key}")
        elif isinstance(argument, str):
            args[key] = argument                # e.g. mean = "Halpha.amp_c0"
        elif isinstance(argument, bool):
            args[key] = argument
        elif isinstance(argument, (int, float)):
            args[key] = float(argument)
        elif isinstance(argument, list):
            args[key] = list(argument)
        else:
            _fail(f"{path}.{key}", f"unsupported prior argument {argument!r}")
    return PriorSpec(kind=kind, args=args)


def _read_fit(table: Mapping[str, Any], base: Path) -> FitConfig:
    defaults = {
        "nlive": 500, "dlogz": 0.01, "max_iterations": 200000,
        "bound": "multi", "device": "auto", "queue_size": None,
        "batch_size": None, "rwalk_K": None, "rwalk_step_scale": None,
        "max_ellipsoids": 20, "memory_frac": 0.9, "min_eff": 10.0,
        "min_ncall": None, "share_compilation": True,
        "compilation_cache_dir": None,
    }
    values = _pool(table, defaults, "[fit]")
    if values["bound"] not in ("none", "single", "multi"):
        _fail("[fit].bound", f"expected none/single/multi, got {values['bound']!r}")
    if values["device"] not in ("auto", "gpu", "cpu"):
        _fail("[fit].device", f"expected auto/gpu/cpu, got {values['device']!r}")
    if values["nlive"] < 8:
        _fail("[fit].nlive", f"must be at least 8, got {values['nlive']}")
    if values["dlogz"] <= 0:
        _fail("[fit].dlogz", "must be positive")
    if not 0.0 < values["memory_frac"] <= 1.0:
        _fail("[fit].memory_frac", "must be in (0, 1]")
    for key in ("queue_size", "batch_size", "rwalk_K", "min_ncall"):
        if values[key] is not None and values[key] < 1:
            _fail(f"[fit].{key}", f"must be at least 1, got {values[key]}")
    return FitConfig(
        nlive=int(values["nlive"]), dlogz=float(values["dlogz"]),
        max_iterations=int(values["max_iterations"]),
        bound=values["bound"], device=values["device"],
        queue_size=values["queue_size"], batch_size=values["batch_size"],
        rwalk_K=values["rwalk_K"], rwalk_step_scale=values["rwalk_step_scale"],
        max_ellipsoids=int(values["max_ellipsoids"]),
        memory_frac=float(values["memory_frac"]), min_eff=float(values["min_eff"]),
        min_ncall=values["min_ncall"],
        share_compilation=bool(values["share_compilation"]),
        compilation_cache_dir=_resolve_path(values["compilation_cache_dir"], base),
    )


def _read_output(table: Mapping[str, Any]) -> OutputConfig:
    defaults = {"thin": 1, "fit_plot": True, "corner": False, "histogram": False,
                "corner_params": None, "hist_params": None}
    values = _pool(table, defaults, "[output]")
    if values["thin"] < 1:
        _fail("[output].thin", f"must be at least 1, got {values['thin']}")
    return OutputConfig(
        thin=int(values["thin"]), fit_plot=bool(values["fit_plot"]),
        corner=bool(values["corner"]), histogram=bool(values["histogram"]),
        corner_params=(_string_list(values["corner_params"], "[output].corner_params")
                       if values["corner_params"] else None),
        hist_params=(_string_list(values["hist_params"], "[output].hist_params")
                     if values["hist_params"] else None),
    )


def _read_run(table: Mapping[str, Any], base: Path) -> RunConfig:
    defaults = {"output_dir": ".", "force": False, "workers": 1, "progress": None}
    values = _pool(table, defaults, "[run]")
    if values["workers"] < 1:
        _fail("[run].workers", f"must be at least 1, got {values['workers']}")
    progress = values["progress"]
    if progress is not None and not isinstance(progress, bool):
        _fail("[run].progress", f"expected true or false, got {progress!r}")
    return RunConfig(output_dir=_resolve_path(values["output_dir"], base),
                     force=bool(values["force"]), workers=int(values["workers"]),
                     progress=progress)


# --------------------------------------------------------------------------------------
# Per-source statistics used by expressions
# --------------------------------------------------------------------------------------
def source_namespace(flux, wave=None, redshift: float = 0.0,
                     continuum_percentile: float = DEFAULT_CONTINUUM_PERCENTILE
                     ) -> Dict[str, float]:
    """The numbers an expression may refer to for one spectrum.

    Parameters
    ----------
    flux : array like
        Observed flux of the spectrum (any units; the expressions inherit them).
    wave : array like, optional
        Rest-frame wavelength, only used to report ``n_pixels``.
    redshift : float
        Redshift recorded for the source.
    continuum_percentile : float
        Percentile used for ``continuum_level``.

    Returns
    -------
    dict
        ``{name: value}`` for every name in :data:`TOKEN_NAMES`.
    """
    import numpy as np

    values = np.asarray(flux, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ConfigError("cannot compute the flux statistics of an empty spectrum")
    peak = float(np.max(np.abs(finite)))
    namespace = {
        "peak_flux": peak,
        "median_flux": float(np.median(finite)),
        "continuum_level": float(np.percentile(finite, continuum_percentile)),
        "rms": float(np.std(finite)),
        "min_flux": float(np.min(finite)),
        "max_flux": float(np.max(finite)),
        "n_pixels": float(np.size(finite)),
        "z": float(redshift),
    }
    for name, value in namespace.items():
        if not math.isfinite(value):
            raise ConfigError(f"the flux statistics of this spectrum are not "
                              f"finite ({name} = {value!r})")
    return namespace