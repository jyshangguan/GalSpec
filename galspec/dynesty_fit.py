"""Nested-sampling fits for Astropy compound models."""

from __future__ import annotations

import time
import warnings
from copy import deepcopy

import numpy as np
from astropy.modeling import CompoundModel

try:
    import dynesty
    from dynesty import utils as dyfunc
except ImportError:  # pragma: no cover
    dynesty = None
    dyfunc = None

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None

__all__ = ["Dynesty_Fit"]

_WORKER_FIT = None


def _initialize_likelihood_worker(fit):
    """Install one fitter copy per worker, avoiding per-call model serialization."""
    global _WORKER_FIT
    _WORKER_FIT = fit


def _worker_log_likelihood(theta):
    return _WORKER_FIT.log_likelihood(theta)


class _LikelihoodPool:
    """Expose a map API while replacing dynesty's large callable payload."""

    def __init__(self, pool):
        self._pool = pool

    def map(self, ignored_function, values):
        if getattr(ignored_function, "__name__", None) is not None:
            return self._pool.map(ignored_function, values)
        return self._pool.map(_worker_log_likelihood, values)


class Dynesty_Fit:
    """Fit an Astropy compound model with :mod:`dynesty`.

    Model bounds are used whenever both ends are finite. Otherwise a conservative
    parameter-type default is used. Default amplitude priors are log-uniform;
    explicitly supplied bounds are always linear-uniform.

    ``n_processes > 1`` parallelizes likelihood calls. Alternatively, ``pool``
    accepts a user-managed object with a ``map`` method; user pools are not closed.
    Invalid samples and samples with non-positive uncertainty are ignored.
    """

    DEFAULT_BOUNDS = {
        "amplitude": (1e-6, 1e6),
        "dv": (-10000.0, 10000.0),
        "sigma": (10.0, 20000.0),
        "stddev": (10.0, 20000.0),
        "h3": (-0.5, 0.5),
        "h4": (-0.5, 0.5),
        "logtau": (-3.0, 3.0),
        "cf": (0.0, 1.0),
        "wavec": (0.99, 1.01),
        "continuum": (-10.0, 10.0),
        "alpha": (-5.0, 5.0),
        "redshift": (-0.1, 0.1),
        "velscale": (10.0, 1000.0),
    }

    def __init__(self, model, wave_use, flux_use, ferr, bounds_dict=None,
                 default_bounds=None, sample_method="rwalk", nlive=500,
                 bound="multi", rstate=None, n_processes=1, pool=None,
                 queue_size=None):
        if dynesty is None:
            raise ImportError("dynesty is required; install GalSpec's dependencies")
        if not isinstance(model, CompoundModel):
            raise TypeError("model must be an Astropy CompoundModel")
        if pool is not None and n_processes != 1:
            raise ValueError("pass either pool or n_processes, not both")
        if int(n_processes) < 1:
            raise ValueError("n_processes must be at least 1")

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

        self.model = deepcopy(model)
        self.wave_use = wave[valid]
        self.flux_use = flux[valid]
        self.ferr = error[valid]
        self.sample_method = sample_method
        self.nlive = int(nlive)
        self.bound = bound
        self.rstate = rstate
        self.n_processes = int(n_processes)
        self.pool = pool
        self.queue_size = queue_size
        self.bounds_dict = dict(bounds_dict or {})
        self.default_bounds = self.DEFAULT_BOUNDS | dict(default_bounds or {})
        self.custom_priors = {}

        self._name_submodels()
        self.param_names = [name for name in self.model.param_names
                            if not getattr(self.model, name).fixed
                            and not getattr(self.model, name).tied]
        self.full_param_names = self._full_parameter_names()
        self.param_map = dict(zip(self.full_param_names, enumerate(self.param_names)))
        self._log_scale_params = set()
        self.param_bounds = self.get_param_bounds()
        self.theta_initial = np.asarray(
            [getattr(self.model, name).value for name in self.param_names], dtype=float)
        self.ndim = len(self.param_names)
        if self.ndim == 0:
            raise ValueError("model has no free parameters")

        self.results = None
        self.samples_equal_weight = None
        self.log_evidence = None
        self.log_evidence_err = None
        self.theta_best = None
        self.runtime_seconds = None
        self.ncall = None

    def _name_submodels(self):
        for index, submodel in enumerate(self.model):
            if submodel.name is None:
                submodel.name = self.model.submodel_names[index]

    def _full_parameter_names(self):
        names = []
        for model_name in self.model.submodel_names:
            submodel = self.model[model_name]
            for param_name in submodel.param_names:
                param = getattr(submodel, param_name)
                if not param.fixed and not param.tied:
                    names.append(f"{model_name}.{param_name}")
        return names

    @staticmethod
    def _base_name(param_name):
        parts = param_name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else param_name

    def _parameter_type(self, param_name):
        base = self._base_name(param_name).lower()
        if base.startswith("c") and base[1:].isdigit():
            return "continuum"
        if "amplitude" in base or base.startswith("amp"):
            return "amplitude"
        if base.startswith("sigma"):
            return "sigma"
        if base == "stddev":
            return "stddev"
        if base.startswith("dv"):
            return "dv"
        if base.startswith("wavec") or base in {"mean", "x_0"}:
            return "wavec"
        if base.startswith("logtau"):
            return "logtau"
        if base == "cf":
            return "cf"
        if base in {"h3", "h4", "alpha", "velscale"}:
            return base
        if base in {"z", "redshift"}:
            return "redshift"
        return "continuum"

    def _custom_bound(self, param_name, index):
        candidates = [param_name, self._base_name(param_name)]
        if index < len(self.full_param_names):
            candidates.insert(1, self.full_param_names[index])
        for key in candidates:
            if key in self.bounds_dict:
                return self.bounds_dict[key]
        return None

    @staticmethod
    def _validate_bound(name, bound):
        if len(bound) != 2:
            raise ValueError(f"bounds for {name} must contain two values")
        lower, upper = map(float, bound)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"bounds for {name} must be finite and increasing")
        return lower, upper

    def get_param_bounds(self):
        bounds = []
        for index, name in enumerate(self.param_names):
            param = getattr(self.model, name)
            custom = self._custom_bound(name, index)
            if custom is not None:
                bounds.append(self._validate_bound(name, custom))
                continue
            lower, upper = param.bounds
            if lower is not None and upper is not None:
                try:
                    bounds.append(self._validate_bound(name, (lower, upper)))
                    continue
                except ValueError:
                    pass
            kind = self._parameter_type(name)
            default = self._validate_bound(name, self.default_bounds[kind])
            if kind == "wavec":
                initial = float(param.value)
                default = tuple(sorted((initial * default[0], initial * default[1])))
            elif kind == "amplitude":
                default = (np.log10(default[0]), np.log10(default[1]))
                self._log_scale_params.add(name)
            bounds.append(default)
        return bounds

    def prior_transform(self, u):
        u = np.asarray(u, dtype=float)
        theta = np.empty_like(u)
        for index, name in enumerate(self.param_names):
            if name in self.custom_priors:
                theta[index] = self.custom_priors[name](u[index])
            else:
                lower, upper = self.param_bounds[index]
                value = lower + (upper - lower) * u[index]
                theta[index] = 10.0**value if name in self._log_scale_params else value
        return theta

    def set_model_params(self, theta):
        for name, value in zip(self.param_names, theta):
            setattr(self.model, name, value)
        for name in self.model.param_names:
            param = getattr(self.model, name)
            if param.tied:
                param.value = param.tied(self.model)
        return self.model

    def log_likelihood(self, theta):
        try:
            model_flux = np.asarray(self.set_model_params(theta)(self.wave_use), dtype=float)
        except (ArithmeticError, ValueError, RuntimeError):
            return -np.inf
        if model_flux.shape != self.flux_use.shape or not np.all(np.isfinite(model_flux)):
            return -np.inf
        residual = (self.flux_use - model_flux) / self.ferr
        norm = np.log(2.0 * np.pi * self.ferr**2)
        return float(-0.5 * np.sum(residual**2 + norm))

    def fit(self, progress=True, **run_nested_kwargs):
        """Run nested sampling; keywords are forwarded to ``run_nested``."""
        owned_pool = None
        active_pool = self.pool
        if active_pool is None and self.n_processes > 1:
            try:
                from multiprocess import Pool
            except ImportError as exc:  # pragma: no cover
                raise ImportError("multiprocess is required for n_processes > 1") from exc
            owned_pool = Pool(processes=self.n_processes,
                              initializer=_initialize_likelihood_worker,
                              initargs=(self,))
            active_pool = _LikelihoodPool(owned_pool)

        queue_size = self.queue_size
        if queue_size is None and active_pool is not None:
            queue_size = self.n_processes if owned_pool is not None else 1
        sampler_kwargs = dict(loglikelihood=self.log_likelihood,
                              prior_transform=self.prior_transform,
                              ndim=self.ndim, nlive=self.nlive,
                              bound=self.bound, sample=self.sample_method,
                              rstate=self.rstate)
        if active_pool is not None:
            sampler_kwargs.update(
                pool=active_pool,
                queue_size=queue_size,
                use_pool={"prior_transform": False,
                          "loglikelihood": True,
                          "propose_point": True,
                          "update_bound": False},
            )

        started = time.perf_counter()
        try:
            sampler = dynesty.NestedSampler(**sampler_kwargs)
            sampler.run_nested(print_progress=progress, **run_nested_kwargs)
            self.results = sampler.results
        finally:
            if owned_pool is not None:
                owned_pool.close()
                owned_pool.join()
        self.runtime_seconds = time.perf_counter() - started
        self.ncall = int(np.sum(self.results.ncall))
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        self.samples_equal_weight = dyfunc.resample_equal(
            self.results.samples, weights, rstate=self.rstate)
        self.log_evidence = float(self.results.logz[-1])
        self.log_evidence_err = float(self.results.logzerr[-1])
        return self.samples_equal_weight, self.model, self.param_names

    def get_best_fit(self):
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        self.theta_best = np.asarray(self.results.samples[np.argmax(self.results.logl)])
        self.set_model_params(self.theta_best)
        return self.model, self.param_names, self.theta_best

    def get_quantiles(self, quantiles=(0.16, 0.5, 0.84)):
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        return np.quantile(self.samples_equal_weight, quantiles, axis=0)

    def get_evidence(self):
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        return self.log_evidence, self.log_evidence_err

    def plot_corner(self, **kwargs):
        if corner is None:
            raise ImportError("corner is required for plotting")
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        if self.theta_best is None:
            self.get_best_fit()
        return corner.corner(self.samples_equal_weight, labels=self.param_names,
                             truths=self.theta_best, **kwargs)
