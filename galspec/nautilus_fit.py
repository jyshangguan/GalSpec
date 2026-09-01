"""Importance nested-sampling fits for Astropy compound models."""

from __future__ import annotations

import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import CompoundModel

from .dynesty_fit import Dynesty_Fit

try:
    from nautilus import Sampler
except ImportError:  # pragma: no cover
    Sampler = None

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None

__all__ = ["Nautilus_Fit"]

_WORKER_FIT = None


def _initialize_likelihood_worker(fit):
    global _WORKER_FIT
    _WORKER_FIT = fit


def _worker_log_likelihood(theta):
    return _WORKER_FIT.log_likelihood(theta)


class Nautilus_Fit(Dynesty_Fit):
    """Fit an Astropy compound model with :mod:`nautilus`.

    The parameter discovery, default bounds, custom bounds and prior transform
    follow :class:`~galspec.Dynesty_Fit`. ``n_processes > 1`` parallelizes
    likelihood evaluations with ``multiprocess``, which also supports locally
    defined Astropy tie functions. Nautilus' native HDF5 checkpoint is enabled
    by passing ``filepath``.
    """

    def __init__(self, model, wave_use, flux_use, ferr, bounds_dict=None,
                 default_bounds=None, n_live=2000, n_processes=1, pool=None,
                 seed=None, filepath=None, resume=True, **sampler_kwargs):
        if Sampler is None:
            raise ImportError(
                "nautilus-sampler is required; install GalSpec's dependencies")
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

        self.model = deepcopy(model)
        self.wave_use, self.flux_use, self.ferr = wave[valid], flux[valid], error[valid]
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
        if self.ndim < 2:
            raise ValueError("Nautilus requires at least two free parameters")

        self.n_live = int(n_live)
        self.nlive = self.n_live  # compatibility with the dynesty-facing API
        self.n_processes = int(n_processes)
        self.pool = pool
        self.seed = seed
        self.filepath = filepath
        self.resume = bool(resume)
        self.sampler_kwargs = dict(sampler_kwargs)
        self.sampler = None
        self.samples = None
        self.log_weights = None
        self.log_likelihood_values = None
        self.samples_equal_weight = None
        self.theta_best = None
        self.log_evidence = None
        self.log_evidence_err = None
        self.runtime_seconds = None
        self.ncall = None

    def fit(self, progress=True, **run_kwargs):
        """Run Nautilus; keywords are forwarded to :meth:`Sampler.run`."""
        owned_pool = None
        active_pool = self.pool
        if active_pool is None and self.n_processes > 1:
            try:
                from multiprocess import Pool
            except ImportError as exc:  # pragma: no cover
                raise ImportError("multiprocess is required for parallel fits") from exc
            owned_pool = Pool(processes=self.n_processes,
                              initializer=_initialize_likelihood_worker,
                              initargs=(self,))
            active_pool = owned_pool

        options = dict(self.sampler_kwargs)
        options.update(
            n_dim=self.ndim, n_live=self.n_live, pool=active_pool,
            seed=self.seed, filepath=self.filepath, resume=self.resume,
            pass_dict=False,
        )
        started = time.perf_counter()
        try:
            likelihood = (_worker_log_likelihood if owned_pool is not None
                          else self.log_likelihood)
            self.sampler = Sampler(self.prior_transform, likelihood, **options)
            self.sampler.run(verbose=progress, **run_kwargs)
            self.samples, self.log_weights, self.log_likelihood_values = (
                self.sampler.posterior(equal_weight=False, return_as_dict=False))
            self.samples_equal_weight = self.sampler.posterior(
                equal_weight=True, return_as_dict=False)[0]
            self.log_evidence = float(self.sampler.log_z)
            self.ncall = int(self.sampler.n_like)
        finally:
            if owned_pool is not None:
                owned_pool.close()
                owned_pool.join()
        self.runtime_seconds = time.perf_counter() - started
        return self.samples_equal_weight, self.model, self.param_names

    @property
    def results(self):
        """Alias for the underlying Nautilus sampler."""
        return self.sampler

    def get_best_fit(self):
        if self.samples is None:
            raise RuntimeError("fit() must be run first")
        self.theta_best = np.asarray(
            self.samples[np.argmax(self.log_likelihood_values)], dtype=float)
        self.set_model_params(self.theta_best)
        return self.model, self.param_names, self.theta_best

    def get_quantiles(self, quantiles=(0.16, 0.5, 0.84)):
        """Return weighted marginal posterior quantiles."""
        if self.samples is None:
            raise RuntimeError("fit() must be run first")
        probabilities = np.exp(self.log_weights)
        output = np.empty((len(quantiles), self.ndim))
        for column in range(self.ndim):
            order = np.argsort(self.samples[:, column])
            values = self.samples[order, column]
            cumulative = np.cumsum(probabilities[order])
            cumulative /= cumulative[-1]
            output[:, column] = np.interp(quantiles, cumulative, values)
        return output

    def get_evidence(self):
        if self.log_evidence is None:
            raise RuntimeError("fit() must be run first")
        return self.log_evidence, self.log_evidence_err

    def _plot_selection(self, parameters):
        if parameters is None:
            indices = np.arange(self.ndim)
        else:
            missing = set(parameters) - set(self.param_names)
            if missing:
                raise KeyError(f"unknown parameters: {sorted(missing)}")
            indices = np.asarray([self.param_names.index(name) for name in parameters])
        return indices, [self.param_names[index] for index in indices]

    def plot_corner(self, parameters=None, **kwargs):
        if corner is None:
            raise ImportError("corner is required for plotting")
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        if self.theta_best is None:
            self.get_best_fit()
        indices, labels = self._plot_selection(parameters)
        return corner.corner(self.samples_equal_weight[:, indices], labels=labels,
                             truths=self.theta_best[indices], **kwargs)

    def plot_trace(self, parameters=None, max_points=5000, **kwargs):
        """Plot posterior values against likelihood-call order."""
        if self.samples is None:
            raise RuntimeError("fit() must be run first")
        indices, labels = self._plot_selection(parameters)
        stride = max(1, len(self.samples) // int(max_points))
        fig, axes = plt.subplots(len(indices), 1, figsize=(10, 2.2 * len(indices)),
                                 sharex=True, squeeze=False)
        defaults = {"s": 3, "alpha": 0.35, "color": "tab:blue"}
        defaults.update(kwargs)
        x = np.arange(len(self.samples))[::stride]
        for axis, index, label in zip(axes[:, 0], indices, labels):
            axis.scatter(x, self.samples[::stride, index], **defaults)
            axis.set_ylabel(label)
        axes[-1, 0].set_xlabel("Posterior sample index")
        fig.tight_layout()
        return fig, axes[:, 0]
