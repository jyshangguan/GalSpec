#!/usr/bin/env python3
"""Focused Nautilus fitter regression checks."""

from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.modeling import models

import galspec


def make_problem(n_processes=1):
    wave = np.linspace(-4, 4, 180)
    truth = models.Const1D(0.15, name="continuum") + models.Gaussian1D(
        1.2, 0.3, 0.7, name="line")
    error = np.full(wave.shape, 0.05)
    flux = truth(wave) + np.random.default_rng(4).normal(0, error)
    model = models.Const1D(0.1, name="continuum") + models.Gaussian1D(
        1.0, 0.0, 1.0, name="line")
    bounds = {"amplitude_0": (-0.1, 0.4), "amplitude_1": (0.5, 2.0),
              "mean_1": (-1.0, 1.0), "stddev_1": (0.2, 1.5)}
    return galspec.Nautilus_Fit(
        model, wave, flux, error, bounds_dict=bounds, n_live=60,
        n_processes=n_processes, seed=10, n_networks=2)


def main():
    fit = make_problem()
    assert np.allclose(fit.prior_transform(np.zeros(fit.ndim)),
                       [bound[0] for bound in fit.param_bounds])
    samples, _, names = fit.fit(progress=False, n_eff=150, n_like_max=2500)
    model, _, best = fit.get_best_fit()
    assert samples.shape[1] == len(names) == 4
    assert np.isfinite(fit.log_evidence)
    assert fit.ncall > 0
    assert np.mean(((fit.flux_use - model(fit.wave_use)) / fit.ferr) ** 2) < 2
    assert abs(best[names.index("mean_1")] - 0.3) < 0.15
    assert fit.get_quantiles().shape == (3, 4)
    fit.plot_corner(parameters=["amplitude_0", "mean_1"])
    fit.plot_trace(parameters=["amplitude_0", "mean_1"])

    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / "fit.fits"
        galspec.save_nautilus_fit_to_fits(fit, filename)
        loaded = galspec.load_nautilus_fit_from_fits(filename)
        assert loaded.samples.shape == fit.samples.shape
        assert loaded.param_names == fit.param_names
        assert np.allclose(loaded.theta_best, fit.theta_best)

    parallel = make_problem(n_processes=2)
    parallel.fit(progress=False, n_eff=20, n_like_max=250)
    assert parallel.ncall > 0
    print("All Nautilus regression checks passed.")


if __name__ == "__main__":
    main()
