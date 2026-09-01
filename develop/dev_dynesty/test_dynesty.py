"""Focused regression tests; run directly with the project Python environment."""

import numpy as np
from astropy.modeling import models

import galspec


def make_problem(workers=1):
    wave = np.linspace(-4, 4, 300)
    truth = models.Const1D(0.15, name="continuum") + models.Gaussian1D(
        1.2, 0.3, 0.7, name="line")
    error = np.full(wave.shape, 0.04)
    flux = truth(wave) + np.random.default_rng(4).normal(0, error)
    model = models.Const1D(0.1, name="continuum") + models.Gaussian1D(
        1.0, 0.0, 1.0, name="line")
    bounds = {"amplitude_0": (-0.1, 0.4), "amplitude_1": (0.5, 2.0),
              "mean_1": (-1.0, 1.0), "stddev_1": (0.2, 1.5)}
    return galspec.Dynesty_Fit(model, wave, flux, error, bounds_dict=bounds,
                              nlive=40, sample_method="rwalk",
                              n_processes=workers,
                              rstate=np.random.default_rng(10))


def main():
    fit = make_problem()
    assert np.allclose(fit.prior_transform(np.zeros(fit.ndim)),
                       np.array([bound[0] for bound in fit.param_bounds]))
    samples, _, names = fit.fit(progress=False, dlogz=2.0)
    model, _, best = fit.get_best_fit()
    assert samples.shape[1] == len(names) == 4
    assert np.isfinite(fit.log_evidence)
    assert np.mean(((fit.flux_use - model(fit.wave_use)) / fit.ferr) ** 2) < 2
    assert abs(best[names.index("mean_1")] - 0.3) < 0.15

    parallel = make_problem(workers=2)
    parallel.fit(progress=False, maxiter=20, dlogz=100)
    assert parallel.ncall > 0
    print("All dynesty regression checks passed.")


if __name__ == "__main__":
    main()
