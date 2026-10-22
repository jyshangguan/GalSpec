#!/usr/bin/env python3
"""Run one fit of the SDSS multi-region AGN model and save everything the report needs.

    python develop/dev_jax/run_sdss_fit.py --config jnesty-gpu
    python develop/dev_jax/run_sdss_fit.py --config dynesty-serial
    python develop/dev_jax/run_sdss_fit.py --config dynesty-parallel --processes 8

Each configuration writes ``results/sdss_<config>.json`` (timing, evidence,
parameter table) and ``results/sdss_<config>.npz`` (predicted flux, residuals,
posterior samples for the corner plot).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULTS = HERE / "results"


def _parameter_table(fit, names) -> dict:
    """Best fit, posterior median and 1-sigma bounds for every parameter."""
    quantiles = fit.get_quantiles()            # shape (3, ndim): 16th, 50th, 84th
    _, _, theta_best = fit.get_best_fit()
    return {
        name: {
            "best": float(theta_best[i]),
            "median": float(quantiles[1][i]),
            "minus": float(quantiles[0][i]),
            "plus": float(quantiles[2][i]),
        }
        for i, name in enumerate(names)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        choices=["dynesty-serial", "dynesty-parallel",
                                 "jnesty-cpu", "jnesty-gpu"])
    parser.add_argument("--nlive", type=int, default=100)
    parser.add_argument("--dlogz", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    import sdss_agn

    wave, flux, error = sdss_agn.load_sdss_spectrum()
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"{args.config}: {wave.size} pixels, nlive={args.nlive}, dlogz={args.dlogz}",
          flush=True)

    result = {
        "config": args.config,
        "n_pixels": int(wave.size),
        "nlive": args.nlive,
        "dlogz": args.dlogz,
        "processes": 1 if args.config == "dynesty-serial" else (
            args.processes if args.config == "dynesty-parallel" else 1),
    }

    if args.config.startswith("dynesty"):
        from galspec.dynesty_fit import Dynesty_Fit

        model = sdss_agn.build_cpu()
        fit = Dynesty_Fit(model, wave, flux, error, bounds_dict=sdss_agn.BOUNDS,
                          nlive=args.nlive, sample_method="rwalk", bound="multi",
                          n_processes=result["processes"],
                          rstate=np.random.default_rng(args.seed))
        result["device"] = "cpu (astropy model)"
        maxiter = args.maxiter or 100000
        started = time.perf_counter()
        fit.fit(progress=args.progress, dlogz=args.dlogz, maxiter=maxiter)
        result["runtime_seconds"] = time.perf_counter() - started
        result["setup_seconds"] = 0.0
        names = list(fit.param_names)
        model_out, _, theta_best = fit.get_best_fit()
        model_out.set_free_params(theta_best) if hasattr(model_out, "set_free_params") \
            else None
        predicted = np.asarray(model_out(wave))
        samples = np.asarray(fit.samples_equal_weight, dtype=float)
    else:
        from galspec.jnesty_fit import JNesty_Fit

        model = sdss_agn.build_jax()
        device = "auto" if args.config == "jnesty-gpu" else "cpu"
        setup = time.perf_counter()
        fit = JNesty_Fit(model, wave, flux, error, bounds_dict=sdss_agn.BOUNDS,
                         nlive=args.nlive, bound="multi", device=device, verbose=False)
        fit.warmup()                              # compile before timing
        result["setup_seconds"] = time.perf_counter() - setup
        result["device"] = ("gpu" if args.config == "jnesty-gpu" else "cpu") + " (JAX model)"
        maxiter = args.maxiter or 100000
        started = time.perf_counter()
        fit.fit(progress=args.progress, dlogz=args.dlogz, max_iterations=maxiter)
        result["runtime_seconds"] = time.perf_counter() - started
        names = list(fit.param_names)
        model_out, _, theta_best = fit.get_best_fit()
        predicted = np.asarray(model_out.flux(wave, theta_best))
        samples = np.asarray(fit.samples_equal_weight, dtype=float)

    residual = (flux - predicted) / error
    chi2 = float(np.sum(residual ** 2))
    result.update({
        "log_evidence": float(fit.log_evidence),
        "log_evidence_error": float(fit.log_evidence_err),
        "chi2": chi2,
        "dof": int(wave.size - len(names)),
        "reduced_chi2": chi2 / (wave.size - len(names)),
        "n_free_parameters": len(names),
        "n_iterations": int(getattr(fit, "n_iterations", 0) or 0),
        "ncall": int(getattr(fit, "ncall", 0) or 0),
        # dynesty exposes the leftover evidence as ``dlogz``; JNesty sets its own flag
        "converged": bool(
            getattr(fit.results, "dlogz", None) is not None
            and float(fit.results.dlogz) <= float(args.dlogz)
        ) if args.config.startswith("dynesty") else bool(
            getattr(fit.results, "get", lambda *_: True)("converged", True)),
        "parameters": _parameter_table(fit, names),
        "residual_sigma": float(np.std(residual)),
        "max_abs_residual_sigma": float(np.max(np.abs(residual))),
    })

    tag = suffix
    np.savez_compressed(RESULTS / f"sdss_{args.config}{tag}.npz",
                        wave=wave, flux=flux, error=error, predicted=predicted,
                        residual=residual, samples=samples[-4000:],
                        parameter_names=np.array(names))
    (RESULTS / f"sdss_{args.config}{tag}.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"  time      : {result['runtime_seconds']:.1f} s"
          f"  (setup {result['setup_seconds']:.1f} s)")
    print(f"  logZ      : {result['log_evidence']:.2f} +- {result['log_evidence_error']:.2f}")
    print(f"  chi2/dof  : {chi2:.1f}/{result['dof']} = {result['reduced_chi2']:.2f}")
    print(f"  residual  : rms {result['residual_sigma']:.2f} sigma, "
          f"worst {result['max_abs_residual_sigma']:.1f} sigma")
    print(f"  written   : results/sdss_{args.config}{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())