#!/usr/bin/env python3
"""Benchmark: likelihood throughput, astropy/CPU vs JAX (GPU or CPU).

Measures the cost of *one likelihood evaluation*, which is what nested sampling
spends all its time on:

* ``cpu``   -- ``galspec.Dynesty_Fit.log_likelihood`` (astropy model: per-call
  ``setattr`` of the parameters plus a numpy evaluation), i.e. exactly what
  dynesty calls;
* ``jax``   -- ``JNesty_Fit`` likelihood compiled with ``jax.jit``, called one
  point at a time;
* ``batch`` -- the same function ``vmap``-ed over 64 / 512 / 2048 parameter
  vectors, which is how JNesty feeds the GPU.

Every configuration is validated by comparing the log-likelihoods with the CPU
reference, so a speed number never comes at the price of a wrong answer.

Examples
--------
    python develop/dev_jax/bench_likelihood.py                    # both problems
    python develop/dev_jax/bench_likelihood.py --problem full --oversample 1
    JAX_PLATFORMS=cpu python develop/dev_jax/bench_likelihood.py   # CPU-only JAX
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

import common  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from galspec.dynesty_fit import Dynesty_Fit  # noqa: E402
from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary  # noqa: E402


def build_problem(name: str, oversample: int):
    """Return ``(wave, flux, error, bounds_dict, cpu_model, jax_model)``."""
    if name == "o3":
        wave, flux, error = common.load_spectrum(oversample=oversample)
        return wave, flux, error, common.O3_BOUNDS, common.build_o3_cpu(), common.build_o3_jax()
    if name == "full":
        wave_all, flux_all, error_all, mask = common.load_full_spectrum()
        return (wave_all[mask], flux_all[mask], error_all[mask], None,
                common.build_full_cpu(), common.build_full_jax())
    raise ValueError(f"unknown problem {name!r}")


def time_calls(fn, arg, repeats: int, device_sync=None) -> float:
    """Seconds per call (first call excluded so compilation is not measured)."""
    fn(arg)
    if device_sync is not None:
        device_sync()
    started = time.perf_counter()
    for _ in range(repeats):
        fn(arg)
    if device_sync is not None:
        device_sync()
    return (time.perf_counter() - started) / repeats


def benchmark_problem(name: str, oversample: int, batches, repeats: int) -> dict:
    wave, flux, error, bounds, cpu_model, jax_model = build_problem(name, oversample)
    common.sync_params(cpu_model, jax_model)

    cpu_fit = Dynesty_Fit(cpu_model, wave, flux, error, bounds_dict=bounds, nlive=50)
    jax_fit = JNesty_Fit(jax_model, wave, flux, error, bounds_dict=bounds, nlive=50,
                         verbose=False)

    # The parameter point used for timing is the model's own starting point, which
    # sits in the region the sampler actually explores.
    theta = np.asarray(jax_model.get_free_params(), dtype=float)
    # A handful of prior draws, used to validate the speed across parameter space
    rng = np.random.default_rng(20260901)
    initial = np.asarray(jax_model.get_free_params(), dtype=float)
    sample = np.vstack([initial,
                        np.array([cpu_fit.prior_transform(rng.uniform(0.0, 1.0, cpu_fit.ndim))
                                  for _ in range(8)])])

    # ---- CPU (astropy) reference -------------------------------------------------
    cpu_t = time_calls(cpu_fit.log_likelihood, theta, repeats)
    cpu_logl = np.array([cpu_fit.log_likelihood(t) for t in sample])

    # ---- JAX single point --------------------------------------------------------
    single = jax.jit(jax_fit._log_likelihood_jnp)
    theta_j = jnp.asarray(theta)
    jax_t = time_calls(single, theta_j, repeats, device_sync=lambda: jnp.asarray(0.0).block_until_ready())

    # ---- JAX batched (vmap) ------------------------------------------------------
    batched = jax.jit(jax.vmap(jax_fit._log_likelihood_jnp))
    batch_results = []
    for batch_size in batches:
        batch_theta = jnp.asarray(np.tile(theta, (batch_size, 1)))
        n_calls = max(3, min(200, int(5000 / batch_size)))
        batch_t = time_calls(batched, batch_theta, n_calls,
                             device_sync=lambda: jnp.asarray(0.0).block_until_ready())
        per_point = batch_t / batch_size
        batch_results.append({
            "batch_size": int(batch_size),
            "seconds_per_batch": batch_t,
            "seconds_per_likelihood": per_point,
            "evaluations_per_second": 1.0 / per_point,
            "speedup_vs_cpu": cpu_t / per_point,
        })

    # ---- numerical agreement -----------------------------------------------------
    jax_logl = np.asarray(batched(jnp.asarray(sample)))
    diff = np.abs(jax_logl - cpu_logl)
    max_abs_diff = float(diff[0])          # at the model's own parameter values
    max_rel_diff = float(np.max(diff / np.maximum(np.abs(cpu_logl), 1.0)))
    all_bits = bool(np.array_equal(jax_logl, cpu_logl))

    result = {
        "problem": name,
        "oversample": oversample if name == "o3" else 1,
        "n_pixels": int(len(wave)),
        "n_free_parameters": int(cpu_fit.ndim),
        "device": jax_devices_summary(),
        "cpu": {
            "source": "galspec.Dynesty_Fit.log_likelihood (astropy model)",
            "seconds_per_likelihood": cpu_t,
            "evaluations_per_second": 1.0 / cpu_t,
        },
        "jax_single": {
            "source": "galspec.jnesty_fit.JNesty_Fit log-likelihood, jax.jit",
            "seconds_per_likelihood": jax_t,
            "evaluations_per_second": 1.0 / jax_t,
            "speedup_vs_cpu": cpu_t / jax_t,
        },
        "jax_batched": batch_results,
        "agreement": {
            "at_starting_parameters_abs_logl_difference": max_abs_diff,
            "over_random_prior_draws_max_rel_logl_difference": max_rel_diff,
            "bitwise_identical": all_bits,
        },
    }

    print(f"\n=== {name} problem: {len(wave)} pixels, {cpu_fit.ndim} free parameters ===")
    print(f"CPU  astropy : {cpu_t*1e3:9.3f} ms/eval  ({1/cpu_t:10.1f} evals/s)")
    print(f"JAX  jit     : {jax_t*1e3:9.3f} ms/eval  ({1/jax_t:10.1f} evals/s)  "
          f"{cpu_t/jax_t:6.1f}x")
    for entry in batch_results:
        print(f"JAX  b{entry['batch_size']:<6d}: {entry['seconds_per_likelihood']*1e3:9.4f} ms/eval  "
              f"({entry['evaluations_per_second']:10.1f} evals/s)  "
              f"{entry['speedup_vs_cpu']:6.1f}x")
    print(f"agreement    : |dlogL| = {max_abs_diff:.3e} at the starting parameters, "
          f"max rel {max_rel_diff:.3e} over prior draws")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=["o3", "full", "both"], default="both")
    parser.add_argument("--oversample", type=int, default=8,
                        help="cost multiplier for the [O III] problem (demo default: 8)")
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--repeats", type=int, default=200,
                        help="single-point timing repeats (batch repeats are derived)")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "bench_likelihood.json")
    args = parser.parse_args()

    print("JAX devices:", jax_devices_summary())

    results = {}
    for name in (["o3", "full"] if args.problem == "both" else [args.problem]):
        results[name] = benchmark_problem(name, args.oversample, args.batches, args.repeats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())