#!/usr/bin/env python3
"""Benchmark: end-to-end nested sampling, dynesty (CPU) vs JNesty (JAX/GPU).

Runs the *same* inference problem -- same data, model, priors, live points and
convergence threshold -- through

* ``dynesty-serial``    -- ``galspec.Dynesty_Fit``, ``n_processes=1``
* ``dynesty-parallel``  -- ``galspec.Dynesty_Fit``, ``n_processes=<cores>``
* ``jnesty-cpu``        -- ``galspec.JNesty_Fit`` on the CPU backend
* ``jnesty-gpu``        -- ``galspec.JNesty_Fit`` on the GPU (auto device)

and reports wall-clock time, likelihood calls, evidence, best-fit quality and
the parameter agreement between the samplers.

Each configuration runs in its **own subprocess**: dynesty parallelises with
``os.fork``, which is unsafe in a process that has already initialised JAX, and
separate processes also give every configuration a cold cache and a clean
timing baseline.

Examples
--------
    python develop/dev_jax/bench_jnesty.py                        # [O III], demo config
    python develop/dev_jax/bench_jnesty.py --nlive 200 --dlogz 0.5
    python develop/dev_jax/bench_jnesty.py --problem full --bin-pixels 4000 --maxiter 4000
    python develop/dev_jax/bench_jnesty.py --runs jnesty-gpu --tag quick
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

ALL_RUNS = ("dynesty-serial", "dynesty-parallel", "jnesty-cpu", "jnesty-gpu")


# --------------------------------------------------------------------------------------
# Problem definition (used by the children; imports galspec but not necessarily JAX)
# --------------------------------------------------------------------------------------
def load_problem(name: str, oversample: int, bin_pixels=None):
    """Return ``(wave, flux, error, bounds_dict)``."""
    import common

    if name == "o3":
        wave, flux, error = common.load_spectrum(oversample=oversample)
        bounds = dict(common.O3_BOUNDS)
    elif name == "agn":
        wave, flux, error = common.load_spectrum(oversample=oversample)
        bounds = dict(common.AGN_BOUNDS)
    elif name == "full":
        wave_all, flux_all, error_all, mask = common.load_full_spectrum()
        wave, flux, error = wave_all[mask], flux_all[mask], error_all[mask]
        bounds = None
    else:
        raise ValueError(f"unknown problem {name!r}")

    if bin_pixels and bin_pixels < wave.size:
        index = np.clip(np.searchsorted(np.linspace(wave[0], wave[-1], bin_pixels + 1),
                                        wave) - 1, 0, bin_pixels - 1)
        weight = 1.0 / error ** 2
        total = np.bincount(index, weights=weight, minlength=bin_pixels)
        occupied = total > 0          # masks (e.g. the Na D gap) leave empty bins
        wave = (np.bincount(index, weights=wave * weight, minlength=bin_pixels)[occupied] /
                total[occupied])
        flux = (np.bincount(index, weights=flux * weight, minlength=bin_pixels)[occupied] /
                total[occupied])
        error = np.sqrt(1.0 / total[occupied])
    return wave, flux, error, bounds


def cpu_model(problem):
    import common
    return {"o3": common.build_o3_cpu, "agn": common.build_agn_cpu,
            "full": common.build_full_cpu}[problem]()


def jax_model(problem):
    import common
    return {"o3": common.build_o3_jax, "agn": common.build_agn_jax,
            "full": common.build_full_jax}[problem]()


def chi2_and_params(fit, wave, flux, error):
    model, names, theta = fit.get_best_fit()
    if hasattr(model, "set_free_params"):
        model.set_free_params(theta)
    predicted = np.asarray(model(wave))
    chi2 = float(np.sum(((flux - predicted) / error) ** 2))
    return chi2, {name: float(value) for name, value in zip(names, theta)}, predicted


# --------------------------------------------------------------------------------------
# Single-configuration runs (executed in a child process)
# --------------------------------------------------------------------------------------
def _run_dynesty(config, args):
    import common
    from galspec.dynesty_fit import Dynesty_Fit

    wave, flux, error, bounds = load_problem(args.problem, args.oversample, args.bin_pixels)
    model = cpu_model(args.problem)
    processes = 1 if config == "dynesty-serial" else args.cores
    fit = Dynesty_Fit(model, wave, flux, error, bounds_dict=bounds, nlive=args.nlive,
                      sample_method="rwalk", bound="multi", n_processes=processes,
                      rstate=np.random.default_rng(args.seed))
    started = time.perf_counter()
    fit.fit(progress=False, dlogz=args.dlogz, maxiter=args.maxiter)
    wall = time.perf_counter() - started
    chi2, params, predicted = chi2_and_params(fit, wave, flux, error)
    niter = int(getattr(fit.results, "niter", 0) or 0)
    payload = {
        "sampler": "dynesty",
        "processes": processes,
        "runtime_seconds": wall,
        "iterations": niter,
        "partial": bool(args.maxiter and niter >= args.maxiter),
        "likelihood_calls": int(fit.ncall),
        "log_evidence": float(fit.log_evidence),
        "log_evidence_error": float(fit.log_evidence_err),
        "chi2": chi2,
        "reduced_chi2": chi2 / (wave.size - len(params)),
        "parameters": params,
        "n_pixels": int(wave.size),
    }
    samples = np.asarray(fit.samples_equal_weight)
    return payload, (wave, flux, error, predicted, samples)


def _run_jnesty(config, args):
    import common
    from galspec.jnesty_fit import JNesty_Fit

    wave, flux, error, bounds = load_problem(args.problem, args.oversample, args.bin_pixels)
    model = jax_model(args.problem)
    device = "cpu" if config == "jnesty-cpu" else "auto"
    setup_started = time.perf_counter()
    fit = JNesty_Fit(model, wave, flux, error, bounds_dict=bounds, nlive=args.nlive,
                     bound="multi", device=device, queue_size=args.queue_size,
                     batch_size=args.batch_size, verbose=False)
    fit.warmup()                      # compile the likelihood/prior, initialise CUDA
    # JNesty compiles its own sampler internals on the first run_nested call.  A
    # short discarded fit absorbs that one-off cost so the reported number is the
    # steady-state sampling time (the cold-start cost is measured separately).
    cold = None
    if args.warm_first:
        probe = JNesty_Fit(model, wave, flux, error, bounds_dict=bounds, nlive=args.nlive,
                           bound="multi", device=device, queue_size=args.queue_size,
                           batch_size=args.batch_size, verbose=False)
        cold_started = time.perf_counter()
        probe.fit(progress=False, dlogz=1e9, max_iterations=max(2 * args.nlive, 50))
        cold = time.perf_counter() - cold_started
    setup = time.perf_counter() - setup_started
    started = time.perf_counter()
    fit.fit(progress=False, dlogz=args.dlogz, max_iterations=args.maxiter)
    wall = time.perf_counter() - started
    chi2, params, predicted = chi2_and_params(fit, wave, flux, error)
    summary = fit.summary()
    payload = {
        "sampler": f"jnesty-{fit.device}",
        "processes": 1,
        "runtime_seconds": wall,
        "setup_seconds": setup,
        "first_run_seconds": cold,
        "total_seconds": setup + wall,
        "likelihood_calls": summary["likelihood_calls"],
        "log_evidence": summary["log_evidence"],
        "log_evidence_error": summary["log_evidence_error"],
        "converged": summary["converged"],
        "partial": bool(args.maxiter and summary["iterations"] >= args.maxiter),
        "chi2": chi2,
        "reduced_chi2": chi2 / (wave.size - len(params)),
        "parameters": params,
        "n_pixels": int(wave.size),
    }
    samples = np.asarray(fit.samples_equal_weight)
    return payload, (wave, flux, error, predicted, samples)


def run_one(config, args) -> int:
    if config.startswith("dynesty"):
        payload, data = _run_dynesty(config, args)
    else:
        payload, data = _run_jnesty(config, args)
    wave, flux, error, predicted, samples = data
    out = args.raw_dir / f"raw_{config}_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), wave=wave, flux=flux, error=error,
                        predicted=predicted, samples=samples)
    return 0


# --------------------------------------------------------------------------------------
# Orchestration, comparison, figure
# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=["o3", "agn", "full"], default="o3")
    parser.add_argument("--oversample", type=int, default=8,
                        help="cost multiplier applied to the [O III] spectrum")
    parser.add_argument("--bin-pixels", type=int, default=None,
                        help="bin the spectrum down to this many pixels (all runs)")
    parser.add_argument("--nlive", type=int, default=80)
    parser.add_argument("--dlogz", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=200000)
    parser.add_argument("--cores", type=int, default=min(8, os.cpu_count() or 1),
                        help="worker processes for the parallel dynesty run")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--warm-first", dest="warm_first", action="store_true", default=True,
                        help="absorb JNesty's compile cost in a short discarded fit (default)")
    parser.add_argument("--no-warm-first", dest="warm_first", action="store_false",
                        help="time the very first run, compilation included")
    parser.add_argument("--queue-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--runs", nargs="+", default=list(ALL_RUNS), choices=list(ALL_RUNS))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--raw-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run", default=None, choices=list(ALL_RUNS),
                        help="internal: execute a single configuration")
    args = parser.parse_args()
    args.tag = args.tag or args.problem

    if args.run:                     # child mode
        return run_one(args.run, args)

    output = args.output or (HERE / "results" / f"bench_jnesty_{args.tag}.json")
    print(f"problem={args.problem} nlive={args.nlive} dlogz={args.dlogz} "
          f"maxiter={args.maxiter} oversample={args.oversample} bin={args.bin_pixels}")

    runs, data = {}, {}
    for index, config in enumerate(args.runs, start=1):
        print(f"[{index}/{len(args.runs)}] {config} ...", end=" ", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--run", config,
                   "--problem", args.problem, "--oversample", str(args.oversample),
                   "--nlive", str(args.nlive), "--dlogz", str(args.dlogz),
                   "--maxiter", str(args.maxiter), "--cores", str(args.cores),
                   "--seed", str(args.seed), "--tag", args.tag,
                   "--raw-dir", str(args.raw_dir)]
        if args.bin_pixels:
            command += ["--bin-pixels", str(args.bin_pixels)]
        if args.queue_size:
            command += ["--queue-size", str(args.queue_size)]
        if args.batch_size:
            command += ["--batch-size", str(args.batch_size)]
        if not args.warm_first:
            command += ["--no-warm-first"]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print("FAILED")
            print(completed.stdout[-4000:])
            print(completed.stderr[-4000:])
            return completed.returncode
        raw_path = args.raw_dir / f"raw_{config}_{args.tag}.json"
        runs[config] = json.loads(raw_path.read_text())
        with np.load(args.raw_dir / f"raw_{config}_{args.tag}.npz") as archive:
            data[config] = {key: archive[key] for key in archive.files}
        print(f"{runs[config]['runtime_seconds']:.1f} s, "
              f"{runs[config]['likelihood_calls']} calls, "
              f"logZ={runs[config]['log_evidence']:.2f}")

    # -- comparison ----------------------------------------------------------------
    baseline = runs.get("dynesty-serial") or next(iter(runs.values()))
    reference = runs.get("dynesty-parallel") or baseline
    reference_name = "dynesty-parallel" if "dynesty-parallel" in runs else \
        ("dynesty-serial" if "dynesty-serial" in runs else next(iter(runs)))
    for name, run in runs.items():
        run["speedup_vs_dynesty_serial"] = baseline["runtime_seconds"] / run["runtime_seconds"]
        run["speedup_vs_dynesty_parallel"] = (
            runs["dynesty-parallel"]["runtime_seconds"] / run["runtime_seconds"]
            if "dynesty-parallel" in runs else None)
        shared = sorted(set(run["parameters"]) & set(reference["parameters"]))
        run["reference"] = reference_name
        run["parameter_difference_vs_reference"] = {
            p: run["parameters"][p] - reference["parameters"][p] for p in shared}
        run["max_parameter_difference_vs_reference"] = max(
            (abs(v) for v in run["parameter_difference_vs_reference"].values()), default=0.0)
        # parameter difference in units of the posterior width (GPU vs reference);
        # the sample columns follow the parameter order of the reference run
        if name in data and reference_name in data:
            sigma = np.std(data[reference_name]["samples"], axis=0)
            order = list(reference["parameters"])
            ratios = []
            for param in shared:
                index = order.index(param)
                width = sigma[index] if index < sigma.size else 0.0
                if width > 0:
                    ratios.append(abs(run["parameters"][param] -
                                      reference["parameters"][param]) / width)
            run["max_parameter_difference_in_sigma"] = float(max(ratios, default=0.0))

    summary = {
        "problem": args.problem,
        "n_pixels": int(baseline["n_pixels"]),
        "n_free_parameters": len(baseline["parameters"]),
        "nlive": args.nlive,
        "dlogz": args.dlogz,
        "maxiter": args.maxiter,
        "oversample": args.oversample,
        "bin_pixels": args.bin_pixels,
        "cores": args.cores,
        "reference": reference_name,
        "runs": runs,
        "fastest": min(runs, key=lambda k: runs[k]["runtime_seconds"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 104)
    print(f"{'configuration':20s} {'sampling':>9s} {'setup':>7s} {'vs serial':>10s} "
          f"{'vs par.':>9s} {'calls':>8s} {'logZ':>11s} {'+-':>6s} {'chi2/dof':>9s} "
          f"{'dpar/σ':>7s}")
    print("-" * 104)
    for name, run in runs.items():
        parallel = run["speedup_vs_dynesty_parallel"]
        print(f"{name:20s} {run['runtime_seconds']:9.2f} "
              f"{run.get('setup_seconds', 0.0):7.2f} "
              f"{run['speedup_vs_dynesty_serial']:9.1f}x "
              f"{(f'{parallel:8.1f}x' if parallel else '       -')} "
              f"{(run['likelihood_calls'] or 0):8d} {run['log_evidence']:11.2f} "
              f"{run['log_evidence_error']:6.2f} {run['reduced_chi2']:9.3f} "
              f"{run.get('max_parameter_difference_in_sigma', float('nan')):7.2f}")
    print("=" * 104)
    truncated = [name for name, run in runs.items() if run.get("partial")]
    if truncated:
        print(f"NOTE: {', '.join(truncated)} hit the maxiter limit "
              f"({args.maxiter} iterations): the samplers did equal *work*, so the "
              f"timings are comparable, but no run is converged -- the evidences, "
              f"chi2 and parameter columns are not meaningful.")
    else:
        spread = max(run["log_evidence"] for run in runs.values()) - \
            min(run["log_evidence"] for run in runs.values())
        print(f"logZ spread across samplers: {spread:.2f} "
              f"(median reported error "
              f"{np.median([r['log_evidence_error'] for r in runs.values()]):.2f})")
    print(f"fastest: {summary['fastest']}  |  written: {output}")

    _make_figure(runs, data, args.tag, HERE / "results" / f"bench_jnesty_{args.tag}.png")
    return 0


def _make_figure(runs, data, tag, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(iter(data.values()))
    wave, flux, error = first["wave"], first["flux"], first["error"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.step(wave, flux, color="0.25", lw=0.6, where="mid", label="data")
    styles = {"dynesty-serial": ("tab:blue", "--"), "dynesty-parallel": ("tab:green", "-."),
              "jnesty-cpu": ("tab:orange", ":"), "jnesty-gpu": ("tab:red", "-")}
    for name, run in runs.items():
        color, style = styles.get(name, ("k", "-"))
        ax.plot(wave, data[name]["predicted"], color=color, ls=style, lw=1.1,
                label=f"{name}: {run['runtime_seconds']:.1f} s, "
                      f"logZ={run['log_evidence']:.1f} "
                      f"({run['speedup_vs_dynesty_serial']:.1f}x)")
        axr.plot(wave, (flux - data[name]["predicted"]) / error, color=color, ls=style,
                 lw=0.6)
    ax.set_ylabel("Scaled flux")
    ax.legend(fontsize=8)
    axr.axhline(0, color="0.5", lw=0.8)
    # scale the residual panel to the data (these fits reach a reduced chi2 < 0.2,
    # so a fixed +-8 sigma window would be mostly empty)
    worst = max(float(np.max(np.abs((flux - d["predicted"]) / error))) for d in data.values())
    limit = max(3.5, 1.2 * worst)
    axr.set(xlabel="Rest wavelength (Å)", ylabel="Residual / σ", ylim=(-limit, limit))
    for axes in (ax, axr):
        axes.tick_params(direction="out")
    ax.set_title(f"Nested sampling benchmark ({tag}): {wave.size} pixels, "
                 f"{len(next(iter(runs.values()))['parameters'])} free parameters")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"figure: {path}")


if __name__ == "__main__":
    raise SystemExit(main())