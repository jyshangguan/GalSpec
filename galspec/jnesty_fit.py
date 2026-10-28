"""Nested sampling with JNesty and JAX/GPU spectral models.

:class:`JNesty_Fit` is the GPU counterpart of :class:`galspec.Dynesty_Fit`:

* the model is a :class:`galspec.jax_models.JaxModel` (compound models built
  with ``+``/``*`` and the usual ``fixed``/``tied``/``bounds``/``name``
  semantics);
* the likelihood and the prior transform are pure ``jax.numpy`` functions,
  ``jit``-compiled once -- no per-call ``setattr`` on astropy parameters, which
  is what dominates the CPU fitter's cost;
* :class:`jnesty.NestedSampler` evaluates likelihoods in parallel via ``vmap``,
  on GPU when available.

Priors are built with exactly the same rules as :class:`galspec.Dynesty_Fit`
(``bounds`` on the model parameter, then ``bounds_dict``, then per-parameter-type
defaults with log-uniform amplitudes), so the *same* ``bounds_dict`` gives the
*same* prior for both fitters -- a prerequisite for comparing evidences.

Example
-------
.. code-block:: python

    from galspec import jax_models as jxm
    from galspec.jnesty_fit import JNesty_Fit

    model = jxm.PowerLaw1D(..., name="Continuum") + jxm.Line_MultiGauss_doublet(...)
    fit = JNesty_Fit(model, wave, flux, error, nlive=200, bound="multi")
    fit.fit(dlogz=0.5, max_iterations=20000)
    best_model, names, theta = fit.get_best_fit()
"""

from __future__ import annotations

import hashlib
import time
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

try:  # pragma: no cover - optional dependency
    import jnesty
except ImportError:  # pragma: no cover
    jnesty = None

try:  # pragma: no cover
    import corner
except ImportError:  # pragma: no cover
    corner = None

from .dynesty_fit import Dynesty_Fit, _resolve_named_priors
from .priors import as_prior, resolve_prior_dependencies
from .param_labels import parameter_labels, qualified_parameter_labels

__all__ = ["JNesty_Fit", "jax_devices_summary", "select_device",
           "clear_shared_compilation_cache", "shared_compilation_cache_size",
           "enable_persistent_compilation_cache"]


# --------------------------------------------------------------------------------------
# Shared compilation
# --------------------------------------------------------------------------------------
#: Compiled likelihoods/prior transforms, keyed by the *traced program* they compile.
#: A plain closure over the data puts that data in the trace as constants, so every
#: spectrum produces a different program and pays a full XLA compile (~0.6 s for a
#: 37-parameter model on 20 000 pixels).  Passing the data as arguments instead makes
#: one program serve every spectrum of the same structure; this dictionary then lets
#: later fits reuse the executable XLA built for the first one.  JAX alone does not
#: help here: a second ``jax.jit`` object over an identical program recompiles it.
_SHARED_COMPILATION: Dict[str, Any] = {}


def _trace_key(fn, *example_args) -> str:
    """SHA-256 of the program ``fn`` traces to, for the given argument shapes."""
    program = jax.make_jaxpr(fn)(*example_args)
    return hashlib.sha256(str(program).encode("utf-8")).hexdigest()


def _shared_compiled(kind: str, fn, *example_args):
    """Return a compiled ``fn``, reusing the one already built for this program."""
    key = f"{kind}:{_trace_key(fn, *example_args)}"
    compiled = _SHARED_COMPILATION.get(key)
    if compiled is None:
        compiled = jax.jit(fn)
        _SHARED_COMPILATION[key] = compiled
    return compiled


def clear_shared_compilation_cache() -> int:
    """Forget every shared executable; returns how many were dropped.

    Only affects fits created with ``share_compilation=True``.  Useful in a long
    service process that switches between model structures, since each cached
    entry keeps its first model alive (its template arrays, mostly).
    """
    count = len(_SHARED_COMPILATION)
    _SHARED_COMPILATION.clear()
    return count


def shared_compilation_cache_size() -> int:
    """Number of compiled programs currently shared (diagnostic)."""
    return len(_SHARED_COMPILATION)


def enable_persistent_compilation_cache(directory) -> str:
    """Point JAX's on-disk compilation cache at ``directory``.

    With the data passed as arguments (:class:`JNesty_Fit` with
    ``share_compilation=True``), one program serves a whole sample, so XLA's
    cache serialises the compiled kernel once and later processes -- worker
    processes of a batch run, a resumed run, a second night -- load it instead
    of compiling.  Call this before the first fit in the process.

    Parameters
    ----------
    directory : str or pathlib.Path
        Cache directory; created if missing.

    Returns
    -------
    str
        The absolute path of the cache directory.
    """
    import pathlib

    path = pathlib.Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    # Cache small kernels too: a single likelihood or prior transform compiles in
    # well under the 1 s default, and those are exactly the ones a batch repeats.
    for option, value in (("jax_persistent_cache_min_compile_time_secs", 0.0),
                          ("jax_persistent_cache_min_entry_size_bytes", 0)):
        try:
            jax.config.update(option, value)
        except Exception:                        # pragma: no cover - newer/older jax
            pass
    return str(path)


#: Cached answer of :func:`_jnesty_context_supported`.
_CONTEXT_SUPPORT: Optional[bool] = None


def _jnesty_context_supported() -> bool:
    """Whether this jnesty's ``NestedSampler`` accepts a ``context`` argument.

    An older jnesty ignores the notion entirely: the fit then hands it the
    per-fit compiled callables instead, which produces the same numbers and
    recompiles inside the sampler as it always did.
    """
    global _CONTEXT_SUPPORT
    if _CONTEXT_SUPPORT is None:
        try:
            import inspect
            _CONTEXT_SUPPORT = "context" in inspect.signature(
                jnesty.NestedSampler.__init__).parameters
        except (TypeError, ValueError):          # pragma: no cover - exotic jnesty
            _CONTEXT_SUPPORT = False
    return _CONTEXT_SUPPORT


def _loglikelihood_fn(model):
    """The Gaussian likelihood with the data as explicit arguments (no literals).

    This is :meth:`JNesty_Fit._log_likelihood_jnp` with ``(x, flux, err,
    log_norm)`` lifted into the signature.  Same operations, same numbers; the
    difference is that the traced program no longer contains this spectrum.
    """
    def loglikelihood(theta, x, flux, err, log_norm):
        model_flux = model.flux(x, theta)
        residual = (flux - model_flux) / err
        value = -0.5 * jnp.sum(residual ** 2 + log_norm)
        return jnp.where(jnp.all(jnp.isfinite(model_flux)), value, -jnp.inf)

    return loglikelihood


def _prior_transform_fn(spec):
    """The unit-cube transform with its boxes as explicit arguments (no literals).

    Rebuilds :meth:`_PriorSpec.transform` with ``lower``/``upper`` lifted into the
    signature.  The *shape* of the prior (which parameters are centred on which,
    log-scaled, or come from a custom prior) stays a trace-time decision; only
    the numbers that change from spectrum to spectrum become arguments.

    ``center_lower``/``center_upper`` are the physical bounds a centred or custom
    prior is truncated to.  They equal ``lower``/``upper`` except for log-scaled
    parameters, whose box is kept in log10 units for the plain branch.
    """
    def transform(u, lower, upper, center_lower, center_upper):
        span = upper - lower
        values = {}
        for i in spec.order:
            ui = u[i]
            centered = spec.centered.get(i)
            if centered is not None:
                prior, center, scale_from = centered
                if not isinstance(center, float):
                    center = values[center]
                scale_value = None
                if scale_from is not None:
                    scale_value = (scale_from if isinstance(scale_from, float)
                                   else values[scale_from])
                # with_bounds() stores Python floats (repr/logpdf need them); here
                # the box is a traced argument, so bind it without conversion.
                local = prior.bind_bounds(center_lower[i], center_upper[i])
                values[i] = local.transform(ui, center, scale_value)
            elif spec.custom[i] is not None:
                prior = spec.custom[i]
                if prior.binds_bounds:
                    prior = prior.bind_bounds(center_lower[i], center_upper[i])
                values[i] = jnp.asarray(prior(ui))
            elif spec.log_scale[i]:
                values[i] = 10.0 ** (lower[i] + span[i] * ui)
            else:
                values[i] = lower[i] + span[i] * ui
        return jnp.stack([values[i] for i in range(len(spec.names))])

    return transform


def jax_devices_summary() -> str:
    """Human-readable description of the JAX backends in use."""
    if jax is None:
        return "jax not installed"
    parts = []
    for device in jax.devices():
        platform = getattr(device, "platform", "?")
        try:
            stats = device.memory_stats() or {}
            total = stats.get("bytes_limit")
        except Exception:  # pragma: no cover - backend specific
            total = None
        if total:
            parts.append(f"{platform}:{device.id} ({total / 1e9:.1f} GB)")
        else:
            parts.append(f"{platform}:{device.id}")
    return ", ".join(parts) if parts else "no devices"


def select_device(device: str = "auto") -> str:
    """Resolve ``device='auto'`` into ``'gpu'`` or ``'cpu'`` for JNesty."""
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError("device must be one of 'auto', 'gpu', 'cpu'")
    if device != "auto":
        return device
    if jax is None:
        return "cpu"
    for candidate in jax.devices():
        if getattr(candidate, "platform", "").lower() in ("gpu", "cuda", "rocm"):
            return "gpu"
    return "cpu"


class JNesty_Fit:
    """Fit a JAX spectral model with JNesty nested sampling (GPU accelerated).

    Parameters
    ----------
    share_compilation : bool
        Pass the spectrum to the compiled likelihood as an *argument* instead of
        closing over it, and reuse the compiled program across fits built from
        the same model structure (see :func:`clear_shared_compilation_cache`).
        Off by default, which keeps the pre-existing behaviour exactly.  Turning
        it on is what makes fitting a large sample affordable: the first fit
        compiles as before, every later fit of the same structure reuses the
        compiled kernel instead of paying ~0.5-1 s of XLA per spectrum.  Results
        agree with the default path to floating-point rounding (~1e-15 relative,
        far below the sampler's own run-to-run scatter).
    """

    #: identical prior rules as the CPU fitter
    DEFAULT_BOUNDS = Dynesty_Fit.DEFAULT_BOUNDS

    def __init__(self, model, wave_use, flux_use, ferr, bounds_dict: Optional[Mapping] = None,
                 default_bounds: Optional[Mapping] = None, nlive: int = 500,
                 bound: str = "multi", device: str = "auto", queue_size: Optional[int] = None,
                 batch_size: Optional[int] = None, rwalk_K: Optional[int] = None,
                 rwalk_step_scale: Optional[float] = None, max_ellipsoids: int = 20,
                 memory_frac: float = 0.9, min_eff: float = 10.0,
                 min_ncall: Optional[int] = None, custom_priors: Optional[Mapping] = None,
                 verbose: bool = False, share_compilation: bool = False):
        if jax is None or jnp is None:
            raise ImportError("JNesty_Fit requires jax; install it with `pip install jax`")
        if jnesty is None:
            raise ImportError(
                "JNesty_Fit requires jnesty; install it with "
                "`pip install -e /path/to/jnesty`")
        from .jax_models.core import JaxModel

        if bound not in ("none", "single", "multi"):
            raise ValueError("bound must be one of 'none', 'single', 'multi'")
        if not isinstance(model, JaxModel):
            raise TypeError(
                "JNesty_Fit expects a galspec.jax_models.JaxModel (e.g. "
                "jax_models.Line_Gaussian(...) + jax_models.PowerLaw1D(...)); "
                f"got {type(model).__name__}. For astropy models use galspec.Dynesty_Fit.")

        wave = np.asarray(wave_use, dtype=float)
        flux = np.asarray(flux_use, dtype=float)
        error = np.asarray(ferr, dtype=float)
        if wave.shape != flux.shape or wave.shape != error.shape:
            raise ValueError("wave_use, flux_use, and ferr must have the same shape")
        valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
        if not np.any(valid):
            raise ValueError("no valid data remain after filtering")
        if not np.all(valid):
            warnings.warn(f"Ignoring {valid.size - valid.sum()} invalid data samples")

        self.model = model
        self.wave_use = wave[valid]
        self.flux_use = flux[valid]
        self.ferr = error[valid]
        self.nlive = int(nlive)
        self.bound = bound
        self.n_iterations = None
        self.rwalk_K = rwalk_K
        self.rwalk_step_scale = rwalk_step_scale
        self.max_ellipsoids = int(max_ellipsoids)
        self.memory_frac = float(memory_frac)
        self.min_eff = float(min_eff)
        self.min_ncall = min_ncall
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.device = select_device(device)
        self.verbose = bool(verbose)
        self.bounds_dict = dict(bounds_dict or {})
        self.default_bounds = dict(self.DEFAULT_BOUNDS)
        self.default_bounds.update(default_bounds or {})
        # -- parameter bookkeeping ------------------------------------------------
        self.param_names = list(model.free_param_names)
        self.ndim = len(self.param_names)
        if self.ndim == 0:
            raise ValueError("model has no free parameters")
        self._specs = {name: model.spec(name) for name in self.param_names}
        # "Submodel.parameter" spelling of every free parameter, accepted as a
        # bounds_dict key exactly like Dynesty_Fit does.
        qualified = getattr(model, "qualified_param_names",
                            tuple(model.param_names))
        self.full_param_names = [q for q, flat in zip(qualified, model.param_names)
                                 if flat in set(self.param_names)]
        # LaTeX labels for plots: "Halpha.sigma_w0" -> "BH$\alpha$:$\sigma_{w0}$"
        # when the model's `label` is set, otherwise "Halpha:$\sigma_{w0}$".
        self.param_labels = qualified_parameter_labels(model, self.full_param_names)
        self.full_param_labels = list(self.param_labels)
        self._initial = dict(zip(self.param_names,
                                 np.asarray(model.get_free_params(), dtype=float)))
        self.custom_priors = _resolve_named_priors(
            dict(custom_priors or {}), self.param_names, self.full_param_names)
        # Priors set on the model win over the legacy dictionary: that is the
        # documented place to put them now.
        self.model_priors = self._collect_model_priors()
        self.custom_priors = {name: as_prior(prior)
                              for name, prior in self.custom_priors.items()}
        self.custom_priors.update(self.model_priors)
        self._log_scale_params = set()
        self.param_bounds = self.get_param_bounds()
        self.model.set_prior_limits(_limits_by_base_name(self.param_bounds, self.param_names))

        # -- data preparation (static shapes for jit) ------------------------------
        self.model.prepare(self.wave_use)
        self._x = jnp.asarray(self.wave_use)
        self._flux = jnp.asarray(self.flux_use)
        self._err = jnp.asarray(self.ferr)
        self._log_norm = jnp.log(2.0 * jnp.pi * self._err ** 2)

        self._build_prior_spec()
        #: ``share_compilation=True`` compiles the likelihood and the prior
        #: transform once *per model structure* and reuses them for every fit
        #: built from the same structure -- the difference between ~0.05 s and
        #: ~0.75 s of compilation per spectrum in a large sample.  The flag is
        #: off by default so existing code keeps its exact behaviour; the numbers
        #: agree to ~1e-15 relative (see the docstring of the class).
        self.share_compilation = bool(share_compilation)
        self._loglike_jax = None
        self._prior_jax = None
        self._batch_jax = None
        self._prior_shared = False
        if self.share_compilation:
            self._shared_loglike_fn = _loglikelihood_fn(self.model)
            self._shared_prior_fn = _prior_transform_fn(self._prior)
            self._loglike_args = (self._x, self._flux, self._err, self._log_norm)
            self._prior_args = tuple(jnp.asarray(a) for a in self._prior.transform_arrays())
        else:
            self._loglike_jax = jax.jit(self._log_likelihood_jnp)
            self._prior_jax = jax.jit(self._prior.transform)

        self.results = None
        self.samples_equal_weight = None
        self.log_evidence: Optional[float] = None
        self.log_evidence_err: Optional[float] = None
        self.theta_best: Optional[np.ndarray] = None
        self.runtime_seconds: Optional[float] = None
        self.ncall: Optional[int] = None
        self.sampler = None

    # -- priors --------------------------------------------------------------------
    def _collect_model_priors(self) -> Dict[str, object]:
        """Priors attached to the model's free parameters, keyed by flat name.

        The model is the primary place a prior is set
        (``line.sigma_w0.prior = priors.Uniform(...)``); this turns each one into
        the flat-name mapping the transform is built from.
        """
        out: Dict[str, object] = {}
        for flat, full in zip(self.param_names, self.full_param_names):
            prior = None
            for key in (full, flat):
                if key is None:
                    continue
                try:
                    prior = self.model.get_prior(key)
                except KeyError:
                    prior = None
                if prior is not None:
                    break
            if prior is not None:
                out[flat] = as_prior(prior)
        return out

    def _build_prior_spec(self) -> None:
        """(Re)build the unit-cube transform from the current priors and boxes."""
        self._prior = _PriorSpec(self.param_names, self.full_param_names,
                                 self.param_bounds, self._log_scale_params,
                                 self.custom_priors)
        self.prior_order = self._prior.order
        self.centered_priors = self._prior.centered

    def _base_name(self, param_name: str) -> str:
        """``amp_c0_1`` -> ``amp_c0`` (strip the astropy submodel suffix)."""
        return Dynesty_Fit._base_name(param_name)

    def _parameter_type(self, param_name: str) -> str:
        """The kind of quantity, from the shared table in :mod:`galspec.dynesty_fit`."""
        from .dynesty_fit import parameter_kind

        return parameter_kind(param_name)

    def _custom_bound(self, param_name: str):
        base = self._base_name(param_name)
        index = self.param_names.index(param_name)
        full = self.full_param_names[index] if index < len(self.full_param_names) else None
        for key in (param_name, full, base):
            if key is not None and key in self.bounds_dict:
                return self.bounds_dict[key]
        return None

    def get_param_bounds(self) -> list:
        """Resolve ``(lower, upper)`` for every free parameter.

        Precedence: the support of a prior set on the parameter (``Uniform``,
        ``LogUniform``, a ``Normal`` given ``mini``/``maxi``, ...), then
        ``bounds_dict``, then the model parameter's own finite bounds, then a
        per-type default (with log-uniform priors for amplitudes).

        A prior that carries its own range wins because that range *is* its
        definition: ``priors.Uniform(800, 25000)`` says more than ``bounds`` does,
        and writing the box twice is how the two used to drift apart.
        """
        bounds = []
        for name in self.param_names:
            prior = self.custom_priors.get(name)
            if prior is not None and prior.support is not None:
                bounds.append(_validate_bound(name, prior.support))
                continue
            custom = self._custom_bound(name)
            if custom is not None:
                bounds.append(_validate_bound(name, custom))
                continue
            lower, upper = self._specs[name].bounds
            if lower is not None and upper is not None:
                try:
                    bounds.append(_validate_bound(name, (lower, upper)))
                    continue
                except ValueError:
                    pass
            kind = self._parameter_type(name)
            default = _validate_bound(name, self.default_bounds[kind])
            if kind == "wavec":
                # scale the default window by the current line centre, as Dynesty_Fit does
                initial = self._initial.get(name, float(self._specs[name].default))
                default = tuple(sorted((initial * default[0], initial * default[1])))
            elif kind == "amplitude":
                default = (float(np.log10(default[0])), float(np.log10(default[1])))
                self._log_scale_params.add(name)
            bounds.append(default)
        return bounds

    # -- compilation ---------------------------------------------------------------
    def _shared_loglike(self):
        """The likelihood program, compiled once per model structure."""
        if self._loglike_jax is None:
            example = jnp.asarray(self.model.get_free_params(), dtype=self._x.dtype)
            self._loglike_jax = _shared_compiled(
                "loglike", self._shared_loglike_fn, example, *self._loglike_args)
        return self._loglike_jax

    def _shared_prior(self):
        """The prior-transform program, compiled once per prior structure.

        Falls back to a per-fit compile, with a warning, when a prior map cannot
        consume the box as a traced argument (a log-space centred prior converts
        its bounds to floats inside :meth:`~galspec.priors.Prior.transform`).
        """
        if self._prior_jax is None:
            example = jnp.full((self.ndim,), 0.5, dtype=self._prior_args[0].dtype)
            try:
                self._prior_jax = _shared_compiled(
                    "prior", self._shared_prior_fn, example, *self._prior_args)
                self._prior_shared = True
            except Exception as exc:      # pragma: no cover - prior-shape dependent
                warnings.warn(
                    "the prior transform of this model cannot be shared between "
                    f"fits ({type(exc).__name__}: {exc}); compiling it per fit",
                    RuntimeWarning)
                self._prior_jax = jax.jit(self._prior.transform)
                self._prior_shared = False
        return self._prior_jax

    def _loglike_value(self, theta):
        """(Compiled) log-likelihood at one parameter vector."""
        if self.share_compilation:
            return self._shared_loglike()(theta, *self._loglike_args)
        return self._loglike_jax(theta)

    def _prior_value(self, u):
        """(Compiled) prior transform at one unit-cube point."""
        if not self.share_compilation:
            return self._prior_jax(u)
        function = self._shared_prior()
        if self._prior_shared:
            return function(u, *self._prior_args)
        return function(u)                          # pragma: no cover - fallback branch

    # -- transforms ----------------------------------------------------------------
    def prior_transform(self, u):
        """Map the unit cube to physical parameters (numpy in/out)."""
        return np.asarray(self._prior_value(jnp.asarray(np.asarray(u, dtype=float))))

    def log_likelihood(self, theta) -> float:
        """Gaussian log-likelihood for one parameter vector (numpy in/out)."""
        return float(self._loglike_value(jnp.asarray(np.asarray(theta, dtype=float))))

    def _log_likelihood_jnp(self, theta):
        model_flux = self.model.flux(self._x, theta)
        residual = (self._flux - model_flux) / self._err
        value = -0.5 * jnp.sum(residual ** 2 + self._log_norm)
        return jnp.where(jnp.all(jnp.isfinite(model_flux)), value, -jnp.inf)

    def log_likelihood_batch(self, thetas) -> np.ndarray:
        """Vectorised log-likelihood for a batch of parameter vectors.

        The ``vmap``-ed function is compiled once and reused, so this is the
        entry point to use when feeding many points to the GPU.
        """
        batch = jnp.asarray(np.atleast_2d(thetas))
        if self.share_compilation:
            if self._batch_jax is None:
                # the data stay whole: only the leading (parameter-vector) axis moves
                self._batch_jax = _shared_compiled(
                    "loglike_batch",
                    jax.vmap(self._shared_loglike_fn,
                             in_axes=(0, None, None, None, None)),
                    batch, *self._loglike_args)
            return np.asarray(self._batch_jax(batch, *self._loglike_args))
        if self._batch_jax is None:
            self._batch_jax = jax.jit(jax.vmap(self._log_likelihood_jnp))
        return np.asarray(self._batch_jax(batch))

    def _raw_log_likelihood(self, theta):
        """jnp callable handed to JNesty (already jit-compiled)."""
        return self._loglike_value(theta)

    def _raw_prior_transform(self, u):
        return self._prior_value(u)

    def _context_log_likelihood(self, theta, context):
        """jnp callable for a JNesty that can carry a ``context``.

        The context is a broadcast argument of every traced function inside the
        sampler, so the program it compiles no longer contains this spectrum and
        serves the whole sample.
        """
        x, flux, err, log_norm = context[:4]
        return self._shared_loglike_fn(theta, x, flux, err, log_norm)

    def _context_prior_transform(self, u, context):
        lower, upper, center_lower, center_upper = context[4:]
        return self._shared_prior_fn(u, lower, upper, center_lower, center_upper)

    @property
    def _context(self):
        """The arrays a context-aware JNesty passes through as arguments."""
        return tuple(self._loglike_args) + tuple(self._prior_args)

    def warmup(self):
        """Compile the likelihood and prior transform up front.

        JAX compiles on the first call, which in a fresh process costs a few
        seconds (plus the CUDA context).  Calling this before :meth:`fit` keeps
        that one-off cost out of the sampling time, and lets a benchmark report
        it separately.  Safe to call repeatedly.
        """
        theta = jnp.asarray(self.model.get_free_params())
        self._loglike_value(theta).block_until_ready()
        self._prior_value(jnp.full((self.ndim,), 0.5)).block_until_ready()
        return self

    # -- sampling ------------------------------------------------------------------
    def fit(self, progress: bool = True, dlogz: float = 0.01,
            max_iterations: int = 100000, **run_kwargs):
        """Run JNesty nested sampling.

        Parameters
        ----------
        progress : bool
            Show the JNesty progress bar.
        dlogz : float
            Convergence threshold on the remaining evidence.
        max_iterations : int
            Safety cap on the number of NS iterations.
        **run_kwargs
            Forwarded to :meth:`jnesty.NestedSampler.run_nested`.
        """
        started = time.perf_counter()
        sampler_arguments = dict(
            ndim=self.ndim,
            nlive=self.nlive,
            rwalk_K=self.rwalk_K,
            rwalk_step_scale=self.rwalk_step_scale,
            device=self.device,
            verbose=self.verbose,
            bound=self.bound,
            max_ellipsoids=self.max_ellipsoids,
            batch_size=self.batch_size,
            queue_size=self.queue_size,
            memory_frac=self.memory_frac,
            min_eff=self.min_eff,
            min_ncall=self.min_ncall,
        )
        # Prefer handing JNesty a context: every function it traces then receives
        # the spectrum as an argument instead of a literal, so the program -- and
        # the XLA compilation -- is the same for the whole sample.
        contextual = False
        if self.share_compilation:
            self._shared_loglike()
            self._shared_prior()
            contextual = self._prior_shared and _jnesty_context_supported()
        if contextual:
            sampler_arguments["loglikelihood"] = self._context_log_likelihood
            sampler_arguments["prior_transform"] = self._context_prior_transform
            sampler_arguments["context"] = self._context
        else:
            sampler_arguments["loglikelihood"] = self._raw_log_likelihood
            sampler_arguments["prior_transform"] = self._raw_prior_transform
        self.sampler = jnesty.NestedSampler(**sampler_arguments)
        self.sampler.run_nested(max_iterations=int(max_iterations),
                                delta_logZ_threshold=float(dlogz),
                                print_progress=progress, **run_kwargs)
        self.runtime_seconds = time.perf_counter() - started
        self.results = self.sampler.results

        self.log_evidence = float(self.results["logz"])
        self.log_evidence_err = float(self.results["logzerr"])
        self.n_iterations = int(self.results["niter"])
        # JNesty does not expose an exact call count; use its own convention of
        # rwalk_K proposal evaluations per iteration plus the initial live set.
        rwalk_k = int(getattr(self.sampler, "rwalk_K", 0) or 0)
        self.ncall = int(self.n_iterations * rwalk_k + self.nlive) if rwalk_k else None
        self.samples_equal_weight = self._resample_equal(self.results)
        return self.samples_equal_weight, self.model, self.param_names

    def release_device_buffers(self) -> None:
        """Drop the sampler and the device copies of the run.

        JNesty keeps the live points, the frozen smoothing kernels and slices of
        the dead-point history on the GPU while the sampler object is alive, and
        an XLA/CUDA context never returns its high-water mark to the driver.  On a
        card shared with other jobs it is therefore worth releasing explicitly:

        * ``self.sampler`` is set to ``None``, which frees the buffers sized by
          ``max_iterations`` (``ndim * max_iterations * 8`` bytes for the history,
          plus the live set and the queue);
        * ``self.results``, ``samples_equal_weight`` and the model stay in host
          memory, so every summary, plot and ``get_quantiles`` call still works.

        Nothing is lost: run :func:`galspec.jnesty_fit_io.save_jnesty_fit_to_fits`
        first if you want to be able to reload the run.
        """
        self.sampler = None
        try:                                    # give the allocator a chance to reuse
            import jax
            jax.clear_caches()
        except Exception:                       # pragma: no cover - jax is a hard dep
            pass

    @staticmethod
    def _resample_equal(results, seed: int = 1234) -> np.ndarray:
        samples = np.asarray(results["samples"], dtype=float)
        logwt = np.asarray(results["logwt"], dtype=float)
        weights = np.exp(logwt - np.max(logwt))
        weights /= weights.sum()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(weights), size=len(weights), p=weights)
        return samples[idx]

    # -- results -------------------------------------------------------------------
    def get_best_fit(self):
        """Maximum-likelihood sample; also sets the model's current parameters."""
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        samples = np.asarray(self.results["samples"], dtype=float)
        logl = np.asarray(self.results["logl"], dtype=float)
        self.theta_best = samples[int(np.argmax(logl))]
        self.model.set_free_params(self.theta_best)
        return self.model, self.param_names, self.theta_best

    def get_quantiles(self, quantiles: Sequence[float] = (0.16, 0.5, 0.84)):
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        return np.quantile(self.samples_equal_weight, quantiles, axis=0)

    def prior_dominated(self, tolerance: float = 0.01) -> Dict[str, str]:
        """Parameters whose posterior is pressed against a prior limit.

        A parameter that sits within ``tolerance`` (fraction of the prior width)
        of an edge is only bounded from one side by the data -- it should be
        quoted as a limit, not as a measurement.  Returns
        ``{param: 'lower'|'upper'}`` for the offending parameters.
        """
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        median = np.median(self.samples_equal_weight, axis=0)
        flagged = {}
        for i, name in enumerate(self.param_names):
            lower, upper = self.param_bounds[i]
            span = upper - lower
            if span <= 0:
                continue
            if (median[i] - lower) / span < tolerance:
                flagged[name] = "lower"
            elif (upper - median[i]) / span < tolerance:
                flagged[name] = "upper"
        return flagged

    def get_evidence(self) -> Tuple[float, float]:
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        return self.log_evidence, self.log_evidence_err

    def plot_corner(self, params=None, **kwargs):
        """A corner plot, drawn by :func:`galspec.plot.plot_corner`.

        ``params`` selects a subset of the parameters -- a 27-parameter corner is
        not a figure anybody reads -- and takes the names that
        :func:`galspec.plot.plot_posterior_hist` takes.
        """
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        if self.theta_best is None:
            self.get_best_fit()
        if corner is not None:
            from .plot import plot_corner as _plot_corner

            return _plot_corner(self, params=params, **kwargs)
        return jnesty.plotting.cornerplot(self.results, **kwargs)

    def summary(self) -> dict:
        """Compact dictionary of the fit outcome (useful for benchmarks)."""
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        return {
            "log_evidence": self.log_evidence,
            "log_evidence_error": self.log_evidence_err,
            "iterations": self.n_iterations,
            "likelihood_calls": self.ncall,
            "runtime_seconds": self.runtime_seconds,
            "efficiency_percent": float(self.results.get("eff", float("nan")) or float("nan")),
            "converged": bool(self.results.get("converged", False)),
            "device": self.device,
            "nlive": self.nlive,
            "bound": self.bound,
            "share_compilation": bool(self.share_compilation),
        }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class _PriorSpec:
    """Static prior description compiled into a pure-jnp transform."""

    def __init__(self, names, full_names, bounds, log_scale_params, custom_priors):
        self.names = list(names)
        self.lower = np.asarray([b[0] for b in bounds], dtype=float)
        self.upper = np.asarray([b[1] for b in bounds], dtype=float)
        self.span = self.upper - self.lower
        self.log_scale = [name in log_scale_params for name in self.names]
        self.custom = [custom_priors.get(name) for name in self.names]
        # Priors centred on another parameter need their dependencies evaluated
        # first, and their truncation range is the box this parameter would have
        # had without one.
        self.order, self.centered = resolve_prior_dependencies(
            self.names, full_names, custom_priors)
        for index, (prior, center, scale_from) in list(self.centered.items()):
            if prior.lower is None or prior.upper is None:
                # a log-scaled parameter keeps its box in log10 units; the centred
                # prior works in physical units, so undo the logarithm here
                lower, upper = self.lower[index], self.upper[index]
                if self.log_scale[index]:
                    lower, upper = 10.0 ** lower, 10.0 ** upper
                self.centered[index] = (
                    prior.with_bounds(lower, upper), center, scale_from)

    def transform(self, u):
        """Unit cube -> physical parameters (dependency order, traced-safe)."""
        values = {}
        for i in self.order:
            ui = u[i]
            centered = self.centered.get(i)
            if centered is not None:
                prior, center, scale_from = centered
                if not isinstance(center, float):
                    center = values[center]
                scale_value = None
                if scale_from is not None:
                    scale_value = (scale_from if isinstance(scale_from, float)
                                   else values[scale_from])
                values[i] = prior.transform(ui, center, scale_value)
            elif self.custom[i] is not None:
                values[i] = jnp.asarray(self.custom[i](ui))
            elif self.log_scale[i]:
                values[i] = 10.0 ** (self.lower[i] + self.span[i] * ui)
            else:
                values[i] = self.lower[i] + self.span[i] * ui
        return jnp.stack([values[i] for i in range(len(self.names))])

    def transform_arrays(self):
        """The four per-parameter arrays :func:`_prior_transform_fn` takes.

        ``lower``/``upper`` are the boxes in the frame the plain branches use
        (log10 for a log-scaled amplitude); ``center_lower``/``center_upper`` are
        the same boxes in physical units, which is what a centred or custom prior
        is truncated to.
        """
        log_scale = np.asarray(self.log_scale, dtype=bool)
        center_lower = np.array(self.lower, dtype=float)
        center_upper = np.array(self.upper, dtype=float)
        if log_scale.any():
            # only the log-scaled entries are stored as log10; leave the rest alone
            # (10 ** 4500 for a line centre would only produce an overflow warning)
            center_lower[log_scale] = 10.0 ** self.lower[log_scale]
            center_upper[log_scale] = 10.0 ** self.upper[log_scale]
        return self.lower, self.upper, center_lower, center_upper


def _validate_bound(name, bound):
    if len(bound) != 2:
        raise ValueError(f"bounds for {name} must contain two values")
    lower, upper = map(float, bound)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"bounds for {name} must be finite and increasing")
    return lower, upper


def _limits_by_base_name(param_bounds, param_names) -> Dict[str, Tuple[float, float]]:
    """Aggregate priors by unqualified parameter name for ``set_prior_limits``."""
    out: Dict[str, Tuple[float, float]] = {}
    for name, (lower, upper) in zip(param_names, param_bounds):
        base = Dynesty_Fit._base_name(name)
        if base not in out:
            out[base] = (lower, upper)
        else:
            lo, hi = out[base]
            out[base] = (min(lo, lower), max(hi, upper))
    return out