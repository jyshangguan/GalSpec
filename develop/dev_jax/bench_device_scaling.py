#!/usr/bin/env python3
"""Benchmark: where the GPU actually pays off (device and precision scaling).

`bench_likelihood.py` measures one likelihood evaluation and `bench_jnesty.py`
measures complete fits.  This script answers the practical question *when does
the GPU help*, by measuring, for a small and a large spectrum:

1. the cost of one batched likelihood evaluation,
2. the cost per likelihood call inside a JNesty run,
3. JNesty's own per-iteration overhead (measured with a trivial likelihood),

for four configurations: JAX-CPU and JAX-GPU, in float64 and float32.

float64/float32 are measured in separate subprocesses because
``jax_enable_x64`` is process-wide.

    python develop/dev_jax/bench_device_scaling.py
    python develop/dev_jax/bench_device_scaling.py --sizes 1 8 --iterations 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMBOS = (
    ("cpu", True), ("gpu", True),     # (device, float64)
    ("cpu", False), ("gpu", False),
)


def _label(cfg) -> str:
    device, x64 = cfg
    return f"{device}-{'f64' if x64 else 'f32'}"


# --------------------------------------------------------------------------------------
# Child: measure one (device, precision) configuration
# --------------------------------------------------------------------------------------
def measure_single(child_args) -> int:
    import common
    from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary

    import jax

    device, x64 = child_args.device, bool(child_args.x64)
    assert jax.config.read("jax_enable_x64") == x64, "precision must be set by the environment"

    results = {"configuration": _label((device, x64)), "device": device,
               "float64": x64, "jax_devices": jax_devices_summary(), "sizes": {}}

    for oversample in child_args.sizes:
        wave, flux, error = common.load_spectrum(oversample=oversample)
        model = common.build_o3_jax()

        def make_fit():
            return JNesty_Fit(model, wave, flux, error, bounds_dict=common.O3_BOUNDS,
                              nlive=child_args.nlive, device=device,
                              queue_size=0, batch_size=child_args.batch_size,
                              verbose=False)

        fit = make_fit()
        fit.warmup()
        theta = np.asarray(model.get_free_params(), dtype=float)

        # (1) batched likelihood throughput
        batch = np.tile(theta, (256, 1))
        fit.log_likelihood_batch(batch)
        started = time.perf_counter()
        for _ in range(child_args.batch_repeats):
            fit.log_likelihood_batch(batch)
        batch_seconds = (time.perf_counter() - started) / child_args.batch_repeats / 256

        # (2) JNesty cost per likelihood call (warm: a short fit absorbs compilation)
        probe = make_fit()
        probe.fit(progress=False, dlogz=1e9, max_iterations=50)
        timed = make_fit()
        timed.warmup()
        started = time.perf_counter()
        timed.fit(progress=False, dlogz=1e-4, max_iterations=child_args.iterations)
        wall = time.perf_counter() - started
        niter = int(timed.results["niter"])
        calls = max(int(timed.ncall or 0), 1)

        # (3) JNesty's own overhead: same sampler, trivial likelihood
        import jax.numpy as jnp
        import jnesty

        trivial = jnesty.NestedSampler(
            loglikelihood=lambda t: -0.5 * jnp.sum(t ** 2),
            prior_transform=lambda u: u, ndim=int(fit.ndim), nlive=child_args.nlive,
            bound="none", device=device, verbose=False)
        started = time.perf_counter()
        trivial.run_nested(max_iterations=child_args.iterations,
                           delta_logZ_threshold=1e-4, print_progress=False)
        overhead_wall = time.perf_counter() - started
        overhead_iterations = int(trivial.results["niter"])

        results["sizes"][int(wave.size)] = {
            "n_pixels": int(wave.size),
            "n_free_parameters": int(fit.ndim),
            "batched_likelihood_seconds_per_point": batch_seconds,
            "batched_likelihood_per_second": 1.0 / batch_seconds,
            "jnesty_seconds": wall,
            "jnesty_iterations": niter,
            "jnesty_likelihood_calls": calls,
            "jnesty_seconds_per_call": wall / calls,
            "jnesty_milliseconds_per_iteration": wall / max(niter, 1) * 1e3,
            "jnesty_overhead_milliseconds_per_iteration":
                overhead_wall / max(overhead_iterations, 1) * 1e3,
        }
        print(f"  {_label((device, x64)):10s} pixels={wave.size:7d} "
              f"batch={batch_seconds*1e6:6.1f} us/point "
              f"jnesty={wall/max(niter,1)*1e3:6.1f} ms/iter "
              f"({wall/calls*1e3:5.2f} ms/call)", flush=True)

    out = child_args.output
    out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


# --------------------------------------------------------------------------------------
# Parent
# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8],
                        help="oversample factors (1 -> 4382 pixels, 8 -> 35056)")
    parser.add_argument("--nlive", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batch-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path,
                        default=HERE / "results" / "bench_device_scaling.json")
    # child mode
    parser.add_argument("--run", dest="run", action="store_true",
                        help="internal: measure a single configuration")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--x64", type=int, default=1)
    args = parser.parse_args()

    if args.run:
        return measure_single(args)

    print(f"pixel scales (oversample): {args.sizes}, nlive={args.nlive}, "
          f"iterations={args.iterations}")
    results = {}
    for device, x64 in COMBOS:
        print(f"[{_label((device, x64))}]", flush=True)
        child_output = args.output.parent / f"_device_{_label((device, x64))}.json"
        child_output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--run",
                   "--device", device, "--x64", str(int(x64)),
                   "--nlive", str(args.nlive), "--iterations", str(args.iterations),
                   "--batch-size", str(args.batch_size),
                   "--batch-repeats", str(args.batch_repeats),
                   "--sizes", *[str(s) for s in args.sizes],
                   "--output", str(child_output)]
        environment = dict(os.environ, GALSPEC_JAX_X64="1" if x64 else "0")
        completed = subprocess.run(command, capture_output=True, text=True,
                                   env=environment)
        if completed.returncode != 0:
            print(completed.stdout[-3000:])
            print(completed.stderr[-3000:])
            return completed.returncode
        print(completed.stdout.rstrip())
        results[_label((device, x64))] = json.loads(child_output.read_text())

    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print("\ncost per likelihood evaluation (microseconds):")
    header = "  ".join(f"{name:>22s}" for name in results)
    print(f"{'pixels':>8s}  {header}")
    first = next(iter(results.values()))["sizes"]
    for size in sorted(first, key=int):
        row = []
        for name, payload in results.items():
            entry = payload["sizes"].get(str(size)) or payload["sizes"][size]
            row.append(f"{entry['batched_likelihood_seconds_per_point']*1e6:10.1f} "
                       f"{entry['jnesty_seconds_per_call']*1e3:9.2f} ms")
        print(f"{int(size):8d}  " + "  ".join(f"{value:>22s}" for value in row))
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())