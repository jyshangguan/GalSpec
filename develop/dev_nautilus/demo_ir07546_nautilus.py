#!/usr/bin/env python3
"""Fit the full IRAS 07546+3928 optical spectrum with Nautilus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import fitting

import galspec
from develop.dev_dynesty.demo_ir07546_full_spectrum import (
    build_model, load_spectrum, local_bounds)


def install_serializable_ties(model):
    """Replace notebook-local functions with FITS-serializable tie objects."""
    model["[O III]"].amp_c1.tied = galspec.tie_MultiGauss_doublet_ratio(
        "[O III]", 2.98)
    tied_sigma = galspec.tie_MultiGauss_sigma_c("[O III]")
    tied_dv = galspec.tie_MultiGauss_dv_c("[O III]")
    model["[S II]"].sigma_c.tied = tied_sigma
    model["[S II]"].dv_c.tied = tied_dv
    for name in ("narrow Ha", "narrow Hb", "narrow Hg", "narrow He2", "[O III] 4363"):
        model[name].sigma.tied = galspec.tie_MultiGauss_sigma_c("[O III]")
        model[name].dv.tied = galspec.tie_MultiGauss_dv_c("[O III]")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--n-live", type=int, default=400)
    parser.add_argument("--n-eff", type=int, default=3000,
                        help="target effective posterior sample size")
    parser.add_argument("--n-like-max", type=float, default=np.inf,
                        help="optional likelihood-call ceiling")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    wave_all, flux_all, error_all, mask = load_spectrum()
    initial = install_serializable_ties(build_model())
    initialized = fitting.LevMarLSQFitter()(
        initial, wave_all, flux_all, weights=mask.astype(float), maxiter=10000)
    wave, flux, error = wave_all[mask], flux_all[mask], error_all[mask]
    bounds = local_bounds(initialized)

    checkpoint = HERE / "iras07546_nautilus_checkpoint.hdf5"
    fit = galspec.Nautilus_Fit(
        initialized, wave, flux, error, bounds_dict=bounds,
        n_live=args.n_live, n_processes=args.cores, seed=args.seed,
        filepath=checkpoint, resume=args.resume)
    fit.fit(progress=True, n_eff=args.n_eff, n_like_max=args.n_like_max)
    best_model, names, theta_best = fit.get_best_fit()

    residual = (flux - best_model(wave)) / error
    reduced_chi2 = float(np.sum(residual**2) / (wave.size - fit.ndim))
    summary = {
        "target": "IRAS 07546+3928", "fitted_pixels": int(wave.size),
        "free_parameters": fit.ndim, "cores": args.cores,
        "n_live": args.n_live, "n_eff_target": args.n_eff,
        "likelihood_calls": fit.ncall, "runtime_seconds": fit.runtime_seconds,
        "log_evidence": fit.log_evidence, "reduced_chi2": reduced_chi2,
        "multiplicative_polynomial": False,
        "parameters": dict(zip(names, map(float, theta_best))),
    }
    (HERE / "iras07546_nautilus_results.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    galspec.save_nautilus_fit_to_fits(
        fit, HERE / "iras07546_nautilus_fit.fits")

    fig, (axis, residual_axis) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    axis.step(wave_all, flux_all, where="mid", color="0.35", lw=0.45,
              label="IRAS 07546+3928")
    axis.plot(wave_all, best_model(wave_all), color="tab:red", lw=1,
              label="Nautilus best fit")
    axis.axvspan(5770, 6000, color="0.85", label="Excluded")
    axis.set_ylabel("Scaled flux")
    axis.legend()
    residual_all = (flux_all - best_model(wave_all)) / error_all
    residual_axis.plot(wave_all, residual_all, color="0.35", lw=0.4)
    residual_axis.axhline(0, color="tab:red", lw=0.8)
    residual_axis.axvspan(5770, 6000, color="0.85")
    residual_axis.set(xlabel="Rest wavelength (Å)", ylabel="Residual / σ",
                      ylim=(-8, 8))
    fig.suptitle(f"Nautilus full-spectrum fit; reduced χ²={reduced_chi2:.2f}")
    fig.tight_layout()
    fig.savefig(HERE / "iras07546_nautilus_fit.png", dpi=160)
    plt.close(fig)

    selected = [name for name in names if any(
        token in name for token in ("alpha", "amplitude_0", "sigma_c_2", "dv_c_2"))][:6]
    if len(selected) >= 2 and len(fit.samples_equal_weight) > len(selected):
        corner_figure = fit.plot_corner(
            parameters=selected, show_titles=True, quantiles=[0.16, 0.5, 0.84])
        corner_figure.savefig(HERE / "iras07546_nautilus_corner.png", dpi=140)
        plt.close(corner_figure)
    else:
        print("Corner plot skipped: increase --n-eff for more posterior samples.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
