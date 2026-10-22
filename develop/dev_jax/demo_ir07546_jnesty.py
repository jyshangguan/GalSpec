#!/usr/bin/env python3
"""Fit the IRAS 07546+3928 [O III] complex with JNesty on the GPU.

This is the GPU/JAX counterpart of ``develop/dev_dynesty/demo_ir07546.py``: the
same data, the same model, the same prior and the same sampler settings, but
the model is a :mod:`galspec.jax_models` model and the sampler is
:class:`jnesty.NestedSampler` running on the GPU.

Expected output: the fit converges in a few seconds and returns a posterior
that agrees with the CPU/dynesty run (see ``bench_jnesty.py`` for the
side-by-side numbers).

    python develop/dev_jax/demo_ir07546_jnesty.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

from galspec.jnesty_fit import JNesty_Fit, jax_devices_summary  # noqa: E402


def main() -> int:
    print("JAX devices:", jax_devices_summary())

    # ---- data --------------------------------------------------------------------
    wave, flux, error = common.load_spectrum(oversample=4)
    print(f"data: {wave.size} pixels, {wave.min():.1f}-{wave.max():.1f} A")

    # ---- model -------------------------------------------------------------------
    model = common.build_o3_jax()
    print(f"model: {len(list(model))} components, {len(model.free_param_names)} "
          f"free parameters out of {len(model.param_names)}")

    fit = JNesty_Fit(model, wave, flux, error, bounds_dict=common.O3_BOUNDS,
                     nlive=200, bound="multi", device="auto")

    print("\nfree parameters and priors:")
    for name, (lower, upper) in zip(fit.param_names, fit.param_bounds):
        print(f"  {name:12s} [{lower:9.4g}, {upper:9.4g}]")

    # ---- sampling ----------------------------------------------------------------
    print("\nrunning JNesty ...")
    # the progress bar is only useful on a terminal (tqdm floods pipes/files)
    fit.fit(progress=sys.stdout.isatty(), dlogz=0.5, max_iterations=100000)

    summary = fit.summary()
    print("\nresults")
    print(f"  log evidence   : {summary['log_evidence']:.3f} +- "
          f"{summary['log_evidence_error']:.3f}")
    print(f"  iterations     : {summary['iterations']}")
    print(f"  likelihood cals: ~{summary['likelihood_calls']}")
    print(f"  wall time      : {summary['runtime_seconds']:.2f} s on {summary['device']}")

    best_model, names, theta = fit.get_best_fit()
    quantiles = fit.get_quantiles((0.16, 0.5, 0.84))
    print("\n  parameter        best fit         median       -1sigma     +1sigma")
    for i, name in enumerate(names):
        print(f"  {name:14s} {theta[i]:10.4f} {quantiles[1, i]:12.4f} "
              f"{quantiles[0, i]:12.4f} {quantiles[2, i]:12.4f}")

    # a parameter pressed against a prior edge is only bounded from one side
    flagged = fit.prior_dominated()
    if flagged:
        print("\n  WARNING: prior-dominated parameters (quote as limits):")
        for name, edge in flagged.items():
            print(f"    {name} is at its {edge} prior limit")

    predicted = np.asarray(best_model(wave))
    chi2 = np.sum(((flux - predicted) / error) ** 2)
    dof = wave.size - len(names)
    print(f"\n  chi2/dof        : {chi2:.1f}/{dof} = {chi2 / dof:.3f}")

    # ---- comparison with the input model -----------------------------------------
    print("\n  line measurements from the posterior (median +- 1 sigma):")
    for label, index, unit in (("fwhm(core) [km/s]", names.index("sigma_c_1") if
                                "sigma_c_1" in names else None, 2.3548),
                               ("dv(core) [km/s]", names.index("dv_c_1") if
                                "dv_c_1" in names else None, 1.0)):
        if index is None:
            continue
        median = quantiles[1, index] * (unit if unit != 1.0 else 1.0)
        lo = (quantiles[1, index] - quantiles[0, index]) * (unit if unit != 1.0 else 1.0)
        hi = (quantiles[2, index] - quantiles[1, index]) * (unit if unit != 1.0 else 1.0)
        print(f"    {label:20s} {median:9.1f} +{hi:.1f} / -{lo:.1f}")

    # ---- plots -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        ax.step(wave, flux, where="mid", color="0.3", lw=0.6, label="data")
        ax.plot(wave, predicted, color="tab:red", lw=1.2, label="JNesty best fit")
        ax.set_ylabel("Scaled flux")
        ax.legend()
        axr.plot(wave, (flux - predicted) / error, color="tab:red", lw=0.6)
        axr.axhline(0, color="0.5", lw=0.8)
        limit = max(3.5, 1.2 * float(np.max(np.abs((flux - predicted) / error))))
        axr.set(xlabel="Rest wavelength (A)", ylabel="Residual / sigma",
                ylim=(-limit, limit))
        for axes in (ax, axr):
            axes.tick_params(direction="out")
        ax.set_title(f"JNesty GPU fit, logZ = {summary['log_evidence']:.2f}")
        fig.tight_layout()
        out = HERE / "results" / "demo_ir07546_jnesty_fit.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\nfigure: {out}")

        corner_out = HERE / "results" / "demo_ir07546_jnesty_corner.png"
        fit.plot_corner()
        plt.savefig(corner_out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"figure: {corner_out}")
    except ImportError as exc:  # pragma: no cover
        print(f"(plotting skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())