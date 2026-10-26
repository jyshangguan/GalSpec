"""Core building blocks for the JAX spectral-model backend.

Design
------
* A model is a pure function ``evaluate(x, params) -> flux`` built from
  :mod:`jax.numpy` operations, so it can be ``jax.jit``-compiled and
  ``jax.vmap``-ed over batches of parameter vectors (this is what makes the
  nested samplers fast, on GPU or CPU).
* Parameter handling mirrors the astropy/GalSpec API used elsewhere in GalSpec:
  ``fixed``, ``tied`` (the very same ``galspec.tie_*`` classes may be reused),
  ``bounds`` and ``name``.
* Adding a model is two declarations: a ``PARAMS`` tuple of
  :class:`ParamSpec` and an ``evaluate`` method -- e.g.::

      class Line_Gaussian(JaxModel):
          PARAMS = (
              ParamSpec("amplitude", 1.0, bounds=(0.0, None)),
              ParamSpec("dv", 0.0, bounds=(-2000.0, 2000.0)),
              ParamSpec("sigma", 200.0, bounds=(20.0, 10000.0)),
              ParamSpec("wavec", 5000.0, fixed=True),
          )

          def evaluate(self, x, p):
              v = velocity(x, p["wavec"])
              return p["amplitude"] * jnp.exp(-0.5 * ((v - p["dv"]) / p["sigma"]) ** 2)

  See ``docs/source/jax_gpu.rst`` for a walk-through.

Parameter vectors
-----------------
``model.param_names`` lists every parameter (astropy-style ``name_index``
suffixes when the model is a compound), while ``model.free_param_names`` lists
only the adjustable ones -- the vector expected by ``model.flux(x, theta)``.
That ordering is what :class:`galspec.jnesty_fit.JNesty_Fit` feeds to the
sampler, and it matches ``Dynesty_Fit.param_names`` for the equivalent astropy
model, so the same ``bounds_dict`` works for both fitters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ParamSpec",
    "JaxParameter",
    "JaxModel",
    "JaxCompoundModel",
]


# --------------------------------------------------------------------------------------
# Parameter specification
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """Static description of one model parameter.

    Parameters
    ----------
    name : str
        Parameter name as used by the model's ``evaluate``.
    default : float
        Initial value.
    fixed : bool
        ``True`` to hold the parameter at its current value (astropy semantics).
    bounds : tuple
        ``(lower, upper)``; ``None`` means unbounded on that side.  Bounds are
        used by the fitter to build priors, mirroring ``Dynesty_Fit``.
    """

    name: str
    default: float = 0.0
    fixed: bool = False
    bounds: Tuple[Optional[float], Optional[float]] = (None, None)


class JaxParameter:
    """astropy-style handle on a single model parameter.

    Exposes ``value``, ``fixed``, ``bounds`` and ``tied`` exactly like an
    ``astropy.modeling.Parameter``, so existing GalSpec idioms such as::

        line.amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio("[O III]", 2.98)

    work unchanged on the JAX models.
    """

    __slots__ = ("_owner", "_name", "_qualified")

    def __init__(self, owner: "JaxModel", name: str, qualified: Optional[str] = None):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_qualified", qualified or name)

    @property
    def name(self) -> str:
        return self._qualified

    def __float__(self) -> float:
        """``float(model.parameter)``, as an astropy ``Parameter`` allows."""
        return float(self._owner._values[self._name])

    @property
    def value(self) -> float:
        return self._owner._values[self._name]

    @value.setter
    def value(self, value) -> None:
        self._owner.set_value(self._name, float(value))

    @property
    def default(self) -> float:
        return self._owner._specs[self._name].default

    @property
    def fixed(self) -> bool:
        return self._owner._specs[self._name].fixed

    @fixed.setter
    def fixed(self, flag: bool) -> None:
        self._owner.set_fixed(self._name, bool(flag))

    @property
    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return self._owner._specs[self._name].bounds

    @bounds.setter
    def bounds(self, bounds) -> None:
        self._owner.set_bounds(self._name, bounds)

    @property
    def tied(self):
        return self._owner._ties.get(self._name)

    @tied.setter
    def tied(self, func) -> None:
        self._owner.set_tied(self._name, func)

    @property
    def prior(self):
        """Prior distribution of this parameter (``None`` when it has none).

        This is the attribute astropy reserves on every
        ``astropy.modeling.Parameter`` for "user fitters"; GalSpec fills it in
        with a :class:`galspec.priors.Prior`::

            line.sigma_w0.prior = priors.Uniform(300.0, 8000.0)

            # a Gaussian around another parameter, hard-capped at five widths
            b_hbeta.sigma_w0.prior = priors.Normal(
                mean="b_halpha.sigma_w0", relative=0.20, nsigma=5)

        Assigning ``None`` removes the prior and leaves the parameter uniform
        over its bounds.  A prior may only be attached to a parameter that is
        free: a fixed or tied parameter takes its value elsewhere, so the prior
        would never be consulted.
        """
        return self._owner._priors.get(self._name)

    @prior.setter
    def prior(self, value) -> None:
        self._owner.set_prior(self._name, value)

    def __repr__(self) -> str:
        state = "fixed" if self.fixed else ("tied" if self.tied is not None else "free")
        prior = " prior" if self.prior is not None else ""
        return f"<JaxParameter {self._qualified}={self.value:g} ({state}{prior})>"


# --------------------------------------------------------------------------------------
# Leaf model
# --------------------------------------------------------------------------------------
class JaxModel:
    """Base class for a JAX spectral model (a leaf of a compound model)."""

    #: Declarative parameter list; subclasses either set this or override
    #: :meth:`_build_specs` when the parameter set is dynamic.
    PARAMS: Tuple[ParamSpec, ...] = ()

    #: Human-readable model kind, used only for messages/registries.
    model_kind: str = "JaxModel"

    def __init__(self, name: Optional[str] = None, fixed: Optional[Mapping[str, bool]] = None,
                 bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
                 **kwargs):
        self.name = name
        self._values: Dict[str, float] = {}
        self._specs: Dict[str, ParamSpec] = {}
        self._ties: Dict[str, Callable[[Any], Any]] = {}
        self._priors: Dict[str, Any] = {}
        self._belongs_to: Optional["JaxCompoundModel"] = None
        #: names explicitly supplied by the caller (used by ``_static_init`` hooks)
        self._provided = frozenset(kwargs)

        specs = list(self._build_specs(**kwargs))
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in {type(self).__name__}: {names}")

        fixed = dict(fixed or {})
        bounds = dict(bounds or {})
        unknown_fixed = set(fixed) - set(names)
        unknown_bounds = set(bounds) - set(names)
        if unknown_fixed:
            raise TypeError(
                f"{type(self).__name__}: `fixed` refers to unknown parameter(s) "
                f"{sorted(unknown_fixed)}; known: {names}")
        if unknown_bounds:
            raise TypeError(
                f"{type(self).__name__}: `bounds` refers to unknown parameter(s) "
                f"{sorted(unknown_bounds)}; known: {names}")

        for spec in specs:
            if spec.name in fixed:
                spec = replace(spec, fixed=bool(fixed[spec.name]))
            if spec.name in bounds:
                spec = replace(spec, bounds=_normalise_bounds(bounds[spec.name]))
            self._specs[spec.name] = spec
            self._values[spec.name] = float(spec.default)

        self._static_init()

    # -- construction helpers ------------------------------------------------------
    def _build_specs(self, **kwargs) -> Sequence[ParamSpec]:
        """Turn keyword overrides into parameter specs (default implementation)."""
        return _apply_kwargs(self.PARAMS, kwargs, type(self).__name__)

    def _static_init(self) -> None:
        """Hook for subclasses that need host-side precomputation after init."""

    # -- parameter access ----------------------------------------------------------
    @property
    def param_names(self) -> Tuple[str, ...]:
        return tuple(self._specs)

    @property
    def free_param_names(self) -> Tuple[str, ...]:
        return tuple(n for n in self._specs if self._is_free(n))

    def _is_free(self, name: str) -> bool:
        spec = self._specs[name]
        return (not spec.fixed) and (name not in self._ties)

    def spec(self, name: str) -> ParamSpec:
        return self._specs[name]

    def set_value(self, name: str, value: float) -> None:
        self._values[name] = float(value)

    def set_fixed(self, name: str, flag: bool) -> None:
        self._specs[name] = replace(self._specs[name], fixed=bool(flag))

    def set_bounds(self, name: str, bounds) -> None:
        self._specs[name] = replace(self._specs[name], bounds=_normalise_bounds(bounds))

    def set_tied(self, name: str, func) -> None:
        if func is None:
            self._ties.pop(name, None)
        else:
            self._ties[name] = func

    # -- priors --------------------------------------------------------------------
    def set_prior(self, name: str, prior) -> None:
        """Attach (or clear, with ``None``) the prior of one parameter.

        Parameters
        ----------
        name : str
            Parameter name, as accepted by :meth:`set_bounds`.
        prior : galspec.priors.Prior, callable or None
            A prior object; a plain ``u -> value`` callable is wrapped in
            :class:`galspec.priors.CallablePrior`; ``None`` removes the prior.

        Raises
        ------
        ValueError
            If the parameter is fixed or tied, in which case a prior would never
            be consulted.
        KeyError
            If the parameter does not exist.
        """
        from ..priors import as_prior

        if name not in self._specs:
            raise KeyError(f"unknown parameter {name!r}; known: {list(self._specs)}")
        if prior is None:
            self._priors.pop(name, None)
            return
        if self._specs[name].fixed:
            raise ValueError(
                f"cannot put a prior on {name!r}: the parameter is fixed, so its "
                "value never comes from the prior. Remove the fixed flag first.")
        if name in self._ties:
            raise ValueError(
                f"cannot put a prior on {name!r}: the parameter is tied to another "
                "one, so its value never comes from the prior. Remove the tie first.")
        self._priors[name] = as_prior(prior)

    def get_prior(self, name: str):
        """Prior of one parameter, or ``None``."""
        return self._priors.get(name)

    @property
    def priors(self) -> Dict[str, Any]:
        """``{parameter name: prior}`` for every parameter that has one."""
        return dict(self._priors)

    def param_dict(self) -> Dict[str, float]:
        """Current parameter values keyed by *unqualified* name."""
        return dict(self._values)

    def __getattr__(self, item: str) -> JaxParameter:
        # Only called when normal attribute lookup fails.
        if item.startswith("_") or item in ("name", "PARAMS", "model_kind"):
            raise AttributeError(item)
        specs = self.__dict__.get("_specs") or {}
        if item in specs:
            return JaxParameter(self, item)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")

    def __repr__(self) -> str:
        label = f"name={self.name!r}, " if self.name else ""
        body = ", ".join(f"{n}={self._values[n]:g}" for n in self._specs)
        return f"<{type(self).__name__}({label}{body})>"

    # -- evaluation ----------------------------------------------------------------
    def evaluate(self, x, p: Mapping[str, Any]):
        """Return the model flux; subclasses implement this with ``jnp`` ops."""
        raise NotImplementedError(f"{type(self).__name__}.evaluate is not implemented")

    def set_prior_limits(self, limits: Mapping[str, Tuple[Optional[float], Optional[float]]]) -> None:
        """Optional hook: tighten static kernel sizes from the fitter's priors.

        Called by :class:`galspec.jnesty_fit.JNesty_Fit` with the resolved
        ``{base_parameter_name: (lower, upper)}`` bounds.  Models whose
        evaluation smooths a template with a *fitted* width (e.g.
        ``IronTemplate``, ``StarSpectrum``) use it to size their static
        convolution kernel; the default is a no-op.
        """

    def prepare(self, x) -> None:
        """Optional hook: precompute ``x``-dependent static data before fitting.

        Called by :class:`galspec.jnesty_fit.JNesty_Fit` with the fitted
        wavelength grid.  Models whose evaluation needs an ``x``-dependent
        selection of a template (e.g. ``StarSpectrum``) override this; the
        default is a no-op.
        """

    # -- qualified names -----------------------------------------------------------
    def _compound_index(self) -> Optional[int]:
        parent = self._belongs_to
        if parent is None:
            return None
        return parent._leaf_index[self._leaf_key()]

    def _leaf_key(self) -> int:
        return id(self)

    def _qualified(self, param_name: str) -> str:
        idx = self._compound_index()
        return param_name if idx is None else f"{param_name}_{idx}"

    # -- composition ---------------------------------------------------------------
    def __add__(self, other):
        return JaxCompoundModel(self, "+", other)

    def __radd__(self, other):
        if other == 0:  # support sum([...])
            return self
        return JaxCompoundModel(other, "+", self)

    def __mul__(self, other):
        return JaxCompoundModel(self, "*", other)

    def __rmul__(self, other):
        return JaxCompoundModel(other, "*", self)

    # -- sampling interface (leaf standalone) --------------------------------------
    def flux(self, x, theta):
        """Evaluate the model for a flat vector of free parameters."""
        resolver = _Resolver(self, theta)
        return self._eval(jnp.asarray(x), resolver)

    def flux_batch(self, x, theta_batch):
        """Evaluate a batch of parameter vectors (``vmap`` over :meth:`flux`)."""
        x = jnp.asarray(x)
        return jax.vmap(lambda t: self.flux(x, t))(jnp.asarray(theta_batch))

    def compiled_flux(self, x):
        """Return a ``jit``-compiled ``theta -> flux`` callable for this grid."""
        x = jnp.asarray(x)
        return jax.jit(lambda t: self.flux(x, t))

    def __call__(self, x):
        """Evaluate with the *current* parameter values (numpy in/out).

        For a compound model the tied parameters are recomputed from the free
        ones; for a single submodel the stored values are used, matching how a
        submodel of an astropy compound model behaves.
        """
        resolver = _Resolver(self, None, use_current=True,
                             resolve_ties=isinstance(self, JaxCompoundModel))
        out = self._eval(jnp.asarray(np.asarray(x, dtype=float)), resolver)
        return np.asarray(out)

    # -- internals -----------------------------------------------------------------
    def _eval(self, x, resolver: "_Resolver"):
        root = resolver.root
        if isinstance(root, JaxCompoundModel):
            return root._eval_node(root, x, resolver)
        # Standalone evaluation: a leaf carries its own (possibly tie-resolved)
        # parameter values, exactly like evaluating an astropy submodel.
        params = {n: resolver.get(n) for n in self._specs}
        return self.evaluate(x, params)

    def get_free_params(self) -> np.ndarray:
        return np.asarray([self._values[n] for n in self.free_param_names], dtype=float)

    def set_free_params(self, theta) -> "JaxModel":
        for name, value in zip(self.free_param_names, np.asarray(theta, dtype=float)):
            self._values[name] = float(value)
        return self

    def copy(self) -> "JaxModel":
        """Shallow structural copy that keeps values, bounds, ties and buffers."""
        import copy as _copy

        new = _copy.copy(self)
        new._values = dict(self._values)
        new._specs = dict(self._specs)
        new._ties = dict(self._ties)
        new._priors = dict(self._priors)
        new._belongs_to = None
        return new


# --------------------------------------------------------------------------------------
# Compound model
# --------------------------------------------------------------------------------------
class JaxCompoundModel(JaxModel):
    """Sum or product of models, evaluated as an expression tree.

    Leaf parameters receive astropy-compatible ``_index`` suffixes in
    depth-first order, e.g. ``amplitude_0``, ``amp_c0_1``; that makes the
    parameter naming (and therefore ``bounds_dict`` keys) identical to the
    astropy equivalent.
    """

    def __init__(self, left: JaxModel, op: str, right: JaxModel):
        if op not in ("+", "*"):
            raise ValueError(f"unsupported compound operator {op!r}; use '+' or '*'")
        left = _as_model(left)
        right = _as_model(right)

        leaves = list(_flatten(left)) + list(_flatten(right))
        keys = [id(leaf) for leaf in leaves]
        if len(set(keys)) != len(keys):
            raise ValueError("the same model instance appears twice in a compound model; "
                             "use model.copy() for the second occurrence")

        self.op = op
        self.left = left
        self.right = right
        # astropy compounds carry a (usually empty) name; the LSF-convolution
        # wrapper reads it, so provide one here as well
        self.name: Optional[str] = None
        self._leaves = leaves
        self._leaf_index = {id(leaf): i for i, leaf in enumerate(leaves)}
        self._by_name: Dict[str, JaxModel] = {}

        for i, leaf in enumerate(leaves):
            owner = leaf._belongs_to
            # Re-parenting is allowed when the current owner is one of the two
            # operands (the usual ``a + b + c`` chain); reuse from an unrelated
            # compound is refused because the parameter suffixes would clash.
            if owner is not None and owner is not self and owner is not left and owner is not right:
                raise ValueError(
                    f"{type(leaf).__name__} instance is already used by another compound "
                    "model; use model.copy() to add it to a second compound")
            leaf._belongs_to = self

        # astropy leaves unnamed submodels as 'model_set' style names; keep a
        # readable fallback and enforce uniqueness for tie lookups.
        for i, leaf in enumerate(leaves):
            if not leaf.name:
                leaf.name = f"{leaf.model_kind}_{i}"
            if leaf.name in self._by_name:
                raise ValueError(
                    f"duplicate submodel name {leaf.name!r}; tie references need unique names")
            self._by_name[leaf.name] = leaf

    # -- structure -----------------------------------------------------------------
    @property
    def leaves(self) -> Tuple[JaxModel, ...]:
        return tuple(self._leaves)

    @property
    def n_submodels(self) -> int:
        return len(self._leaves)

    @property
    def submodel_names(self) -> Tuple[str, ...]:
        return tuple(leaf.name for leaf in self._leaves)

    def __len__(self) -> int:
        return len(self._leaves)

    def __iter__(self):
        return iter(self._leaves)

    # -- parameter names -----------------------------------------------------------
    @property
    def param_names(self) -> Tuple[str, ...]:
        return tuple(f"{p}_{i}" for i, leaf in enumerate(self._leaves) for p in leaf.param_names)

    @property
    def free_param_names(self) -> Tuple[str, ...]:
        names = []
        for i, leaf in enumerate(self._leaves):
            names.extend(f"{p}_{i}" for p in leaf.free_param_names)
        return tuple(names)

    def spec(self, name: str) -> ParamSpec:
        leaf, _, base = self._lookup_qualified(name)
        return leaf._specs[base]

    # -- priors, addressed by qualified name ---------------------------------------
    def _prior_target(self, name: str) -> Tuple["JaxModel", str]:
        """Resolve any spelling of a parameter name to ``(leaf, bare name)``.

        Accepts ``"Halpha.sigma_w0"`` (submodel label, as the prior of a centred
        prior quotes it), ``"sigma_w0_5"`` (the flat astropy spelling) and a bare
        ``"sigma_w0"`` when only one submodel has it.
        """
        if not isinstance(name, str):
            raise TypeError(f"parameter name must be a string, got {type(name).__name__}")
        if "." in name:
            label, _, base = name.rpartition(".")
            leaf = self._by_name.get(label)
            if leaf is not None and base in leaf._specs:
                return leaf, base
            raise KeyError(f"unknown parameter {name!r}")
        base, idx = _split_qualified(name)
        if idx is not None and 0 <= idx < len(self._leaves) and base in self._leaves[idx]._specs:
            return self._leaves[idx], base
        hits = [leaf for leaf in self._leaves if name in leaf._specs]
        if len(hits) == 1:
            return hits[0], name
        if len(hits) > 1:
            labels = [f"{leaf.name}.{name}" for leaf in hits]
            raise KeyError(f"parameter name {name!r} is ambiguous: it matches {labels}; "
                           "spell it with its submodel prefix")
        raise KeyError(f"unknown parameter {name!r}")

    def set_prior(self, name: str, prior) -> None:
        """Attach (or clear) the prior of the parameter ``"Submodel.parameter"``.

        Also accepts the flat spelling (``"sigma_w0_5"``) or a bare name that
        only one submodel provides.
        """
        leaf, base = self._prior_target(name)
        leaf.set_prior(base, prior)

    def get_prior(self, name: str):
        """Prior of the parameter ``"Submodel.parameter"``, or ``None``."""
        leaf, base = self._prior_target(name)
        return leaf.get_prior(base)

    @property
    def priors(self) -> Dict[str, Any]:
        """``{"Submodel.parameter": prior}`` for every parameter that has one."""
        out: Dict[str, Any] = {}
        for i, leaf in enumerate(self._leaves):
            label = leaf.name or self.submodel_names[i]
            for base, prior in leaf._priors.items():
                out[f"{label}.{base}"] = prior
        return out

    @property
    def qualified_param_names(self) -> Tuple[str, ...]:
        """``('Continuum.amplitude', 'OIII.amp_c0', ...)`` for every parameter.

        These are the ``<submodel name>.<parameter>`` strings an astropy
        compound model exposes, and are accepted as ``bounds_dict`` keys.
        """
        names = []
        for i, leaf in enumerate(self._leaves):
            label = leaf.name or self.submodel_names[i]
            names.extend(f"{label}.{p}" for p in leaf.param_names)
        return tuple(names)

    def _lookup_qualified(self, qualified: str) -> Tuple[JaxModel, int, str]:
        base, idx = _split_qualified(qualified)
        if idx is None or idx >= len(self._leaves):
            raise KeyError(f"unknown parameter {qualified!r}")
        leaf = self._leaves[idx]
        if base not in leaf._specs:
            raise KeyError(f"unknown parameter {qualified!r}")
        return leaf, idx, base

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._leaves[int(key)]
        if isinstance(key, str):
            if key in self._by_name:
                return self._by_name[key]
            for qualified in self.param_names:
                if qualified == key:
                    leaf, _, base = self._lookup_qualified(qualified)
                    return JaxParameter(leaf, base, qualified=qualified)
            base, idx = _split_qualified(key)
            if idx is not None and base in self._leaves[idx]._specs:
                leaf = self._leaves[idx]
                return JaxParameter(leaf, base, qualified=key)
        raise KeyError(f"{key!r} is neither a submodel name nor a parameter name")

    def __getattr__(self, item: str) -> JaxParameter:
        if item.startswith("_"):
            raise AttributeError(item)
        leaf_index = self.__dict__.get("_leaf_index")
        if leaf_index is None:
            raise AttributeError(item)
        base, idx = _split_qualified(item)
        leaves = self.__dict__.get("_leaves")
        if idx is not None and leaves is not None and idx < len(leaves):
            leaf = leaves[idx]
            if base in leaf._specs:
                return JaxParameter(leaf, base, qualified=item)
        raise AttributeError(f"no parameter {item!r} in this compound model")

    # -- hooks forwarded to every leaf ----------------------------------------------
    def prepare(self, x) -> None:
        for leaf in self._leaves:
            leaf.prepare(x)

    def set_prior_limits(self, limits: Mapping[str, Tuple[Optional[float], Optional[float]]]) -> None:
        for leaf in self._leaves:
            leaf.set_prior_limits(limits)

    # -- evaluation ----------------------------------------------------------------
    @property
    def free_param_values(self) -> np.ndarray:
        return np.asarray([self._value_of(q) for q in self.free_param_names], dtype=float)

    def _value_of(self, qualified: str) -> float:
        leaf, _, base = self._lookup_qualified(qualified)
        if base in leaf._ties and not leaf._specs[base].fixed:
            # evaluate the tie chain at the current values
            resolver = _Resolver(self, None, use_current=True)
            return float(np.asarray(resolver.get(qualified)))
        return leaf._values[base]

    def set_free_params(self, theta) -> "JaxCompoundModel":
        for name, value in zip(self.free_param_names, np.asarray(theta, dtype=float)):
            leaf, _, base = self._lookup_qualified(name)
            leaf._values[base] = float(value)
        return self

    def get_free_params(self) -> np.ndarray:
        return self.free_param_values

    def param_dict(self) -> Dict[str, float]:
        """Current values of every parameter, keyed by qualified name."""
        return {q: self._value_of(q) for q in self.param_names}

    def _eval_node(self, node, x, resolver: "_Resolver"):
        if isinstance(node, JaxCompoundModel):
            left = self._eval_node(node.left, x, resolver)
            right = self._eval_node(node.right, x, resolver)
            return left + right if node.op == "+" else left * right
        if node._belongs_to is not resolver.root:
            raise RuntimeError("evaluating a compound model with leaves owned by another compound")
        params = {n: resolver.get(node._qualified(n)) for n in node._specs}
        return node.evaluate(x, params)

    def copy(self) -> "JaxCompoundModel":
        """Copy the tree, duplicating leaves (ties/values/bounds preserved)."""
        left = self.left.copy() if isinstance(self.left, JaxCompoundModel) else self.left.copy()
        right = self.right.copy() if isinstance(self.right, JaxCompoundModel) else self.right.copy()
        return JaxCompoundModel(left, self.op, right)

    def __repr__(self) -> str:
        return f"<JaxCompoundModel {self.n_submodels} submodels, op={self.op!r}>"


# --------------------------------------------------------------------------------------
# Parameter binding / tie resolution
# --------------------------------------------------------------------------------------
class _LeafView:
    """Attribute-style access to one submodel's *values* inside tie callables."""

    __slots__ = ("_resolver", "_model")

    def __init__(self, resolver: "_Resolver", model: JaxModel):
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_model", model)

    def __getattr__(self, item: str):
        model = object.__getattribute__(self, "_model")
        if item not in model._specs:
            raise AttributeError(
                f"model {model.name!r} has no parameter {item!r}; "
                f"available: {list(model._specs)}")
        return self._resolver.get(model._qualified(item))


class _ModelView:
    """``model[tie_ref_name]`` interface handed to ``galspec.tie_*`` callables."""

    __slots__ = ("_resolver",)

    def __init__(self, resolver: "_Resolver"):
        object.__setattr__(self, "_resolver", resolver)

    def __getitem__(self, name: str):
        resolver = object.__getattribute__(self, "_resolver")
        return _LeafView(resolver, resolver.model_by_name(name))


class _Resolver:
    """Binds a parameter vector (or the current values) to qualified names."""

    def __init__(self, root: JaxModel, theta=None, use_current: bool = False,
                 resolve_ties: bool = True):
        self.root = root
        self.theta = None if theta is None else jnp.asarray(theta)
        self.use_current = bool(use_current)
        self.resolve_ties = bool(resolve_ties)
        self.cache: Dict[str, Any] = {}
        self.stack: List[str] = []
        names = root.free_param_names
        self._free_index = {name: i for i, name in enumerate(names)}
        if self.theta is not None:
            if jnp.ndim(self.theta) != 1:
                raise ValueError("theta must be a 1-D array of free parameters")
            if jnp.size(self.theta) != len(self._free_index):
                raise ValueError(
                    f"theta has {int(jnp.size(self.theta))} elements but the model has "
                    f"{len(self._free_index)} free parameters: {list(self._free_index)}")

    # -- model lookup --------------------------------------------------------------
    def model_by_name(self, name: str) -> JaxModel:
        root = self.root
        if isinstance(root, JaxCompoundModel):
            try:
                return root._by_name[name]
            except KeyError as exc:
                raise KeyError(
                    f"tie reference {name!r} does not match any submodel; available: "
                    f"{list(root._by_name)}") from exc
        if root.name == name:
            return root
        raise KeyError(
            f"tie reference {name!r} cannot be resolved: a standalone model has no "
            "submodels; build a compound model and use its name")

    # -- value resolution ----------------------------------------------------------
    def get(self, qualified: str):
        if qualified in self.cache:
            return self.cache[qualified]

        root = self.root
        if isinstance(root, JaxCompoundModel):
            leaf, _, base = root._lookup_qualified(qualified)
        else:
            leaf, base = root, qualified

        spec = leaf._specs[base]
        tie = leaf._ties.get(base)

        if tie is not None and not spec.fixed and self.resolve_ties:
            if qualified in self.stack:
                raise RuntimeError(f"cyclic tie definition involving {qualified!r}")
            self.stack.append(qualified)
            try:
                value = tie(_ModelView(self))
            finally:
                self.stack.pop()
            value = jnp.asarray(value)
        elif self.use_current or qualified not in self._free_index:
            value = jnp.asarray(leaf._values[base])
        else:
            value = jnp.asarray(self.theta[self._free_index[qualified]])

        self.cache[qualified] = value
        return value


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _normalise_bounds(bounds) -> Tuple[Optional[float], Optional[float]]:
    if bounds is None:
        return (None, None)
    if len(bounds) != 2:
        raise ValueError(f"bounds must have two entries, got {bounds!r}")
    lo, hi = bounds
    lo = None if lo is None else float(lo)
    hi = None if hi is None else float(hi)
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"bounds must be increasing, got {(lo, hi)}")
    return (lo, hi)


def _split_qualified(qualified: str) -> Tuple[str, Optional[int]]:
    base, sep, suffix = qualified.rpartition("_")
    if sep and suffix.isdigit():
        return base, int(suffix)
    return qualified, None


def _apply_kwargs(specs: Sequence[ParamSpec], kwargs: Mapping[str, Any], owner: str):
    known = {spec.name for spec in specs}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(
            f"{owner} got unexpected keyword argument(s) {sorted(unknown)}; "
            f"accepted: {sorted(known)}")
    out = []
    for spec in specs:
        out.append(replace(spec, default=float(kwargs[spec.name])) if spec.name in kwargs else spec)
    return out


def _flatten(model) -> Iterable[JaxModel]:
    if isinstance(model, JaxCompoundModel):
        for leaf in model._leaves:
            yield leaf
    else:
        yield model


def _as_model(model) -> JaxModel:
    if isinstance(model, JaxModel):
        return model
    raise TypeError(
        f"expected a JaxModel (e.g. galspec.jax_models.Line_Gaussian), got {type(model).__name__}")