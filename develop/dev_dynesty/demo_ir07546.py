#!/usr/bin/env python3
"""Fit the IRAS 07546+3928 [O III] region and benchmark parallel dynesty."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling import models
from extinction import ccm89, remove

import galspec
from galspec.utils import line_wave_dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_FILE = ROOT / "example" / "data" / "ir07546sed.fits"


def load_spectrum(oversample=1):
    with fits.open(DATA_FILE) as hdul:
        header = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=float) * 1e14
    wave = header["CRVAL1"] + header["CD1_1"] * np.arange(flux.size)
    flux = remove(ccm89(wave, 0.176, r_v=3.1, unit="aa"), flux)
    wave = wave / (1.0 + 0.0953)
    use = (wave > 4900.0) & (wave < 5100.0)
    wave = wave[use]
    flux = flux[use]
    error = np.maximum(0.05 * np.abs(flux), 1e-4)
    if oversample > 1:
        # Preserve approximately the same integrated statistical weight while
        # emulating the cost of a larger/full-spectrum likelihood.
        dense_wave = np.linspace(wave[0], wave[-1], wave.size * oversample)
        flux = np.interp(dense_wave, wave, flux)
        error = np.interp(dense_wave, wave, error) * np.sqrt(oversample)
        wave = dense_wave
    return wave, flux, error


def build_model():
    continuum = models.PowerLaw1D(
        amplitude=0.5, x_0=5000.0, alpha=1.0,
        fixed={"x_0": True}, name="Continuum")
    line = galspec.Line_MultiGauss_doublet(
        n_components=2, amp_c0=1.7, amp_c1=1.7 / 2.98,
        dv_c=-43.0, sigma_c=250.0,
        wavec0=line_wave_dict["OIII_5007"],
        wavec1=line_wave_dict["OIII_4959"], name="OIII",
        par_w={"amp_w0": 0.35, "dv_w0": -280.0, "sigma_w0": 340.0})

    def tie_ratio(model):
        return model["OIII"].amp_c0 / 2.98

    line.amp_c1.tied = tie_ratio
    return continuum + line


BOUNDS = {
    "amplitude_0": (0.2, 1.0),
    "alpha_0": (-2.0, 4.0),
    "amp_c0_1": (0.2, 3.5),
    "dv_c_1": (-350.0, 200.0),
    "sigma_c_1": (80.0, 600.0),
    "amp_w0_1": (0.01, 1.0),
    "dv_w0_1": (-900.0, 200.0),
    "sigma_w0_1": (120.0, 1400.0),
}


def run_fit(wave, flux, error, workers, nlive, dlogz, seed):
    fit = galspec.Dynesty_Fit(
        build_model(), wave, flux, error, bounds_dict=BOUNDS,
        nlive=nlive, sample_method="rwalk", bound="multi",
        n_processes=workers, rstate=np.random.default_rng(seed))
    fit.fit(progress=False, dlogz=dlogz)
    model, names, theta = fit.get_best_fit()
    chi2 = float(np.sum(((flux - model(wave)) / error) ** 2))
    return fit, model, {
        "workers": workers,
        "runtime_seconds": fit.runtime_seconds,
        "likelihood_calls": fit.ncall,
        "log_evidence": fit.log_evidence,
        "log_evidence_error": fit.log_evidence_err,
        "chi2": chi2,
        "parameters": dict(zip(names, map(float, theta))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlive", type=int, default=80)
    parser.add_argument("--dlogz", type=float, default=1.0)
    parser.add_argument("--cores", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--oversample", type=int, default=8,
                        help="cost multiplier for a stable parallel benchmark")
    args = parser.parse_args()
    if args.cores < 2:
        parser.error("--cores must be at least 2 for the benchmark")

    wave, flux, error = load_spectrum(args.oversample)
    serial_fit, serial_model, serial = run_fit(
        wave, flux, error, 1, args.nlive, args.dlogz, 20260831)
    parallel_fit, parallel_model, parallel = run_fit(
        wave, flux, error, args.cores, args.nlive, args.dlogz, 20260831)
    speedup = serial["runtime_seconds"] / parallel["runtime_seconds"]
    summary = {
        "data_file": str(DATA_FILE), "data_points": len(wave),
        "nlive": args.nlive, "dlogz": args.dlogz,
        "oversample": args.oversample,
        "serial": serial, "parallel": parallel, "speedup": speedup,
        "parallel_faster": speedup > 1.0,
    }
    (HERE / "demo_results.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, (ax, residual_ax) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.step(wave, flux, color="0.25", lw=0.7, where="mid", label="IRAS 07546+3928")
    ax.plot(wave, parallel_model(wave), color="tab:red", lw=1.3,
            label=f"dynesty ({args.cores} cores)")
    ax.set_ylabel("Scaled flux")
    ax.legend()
    residual_ax.plot(wave, (flux - parallel_model(wave)) / error,
                     color="0.25", lw=0.6)
    residual_ax.axhline(0, color="tab:red", lw=0.8)
    residual_ax.set(xlabel="Rest wavelength (Å)", ylabel="Residual / σ")
    fig.suptitle(f"[O III] fit; serial {serial['runtime_seconds']:.1f}s, "
                 f"{args.cores}-core {parallel['runtime_seconds']:.1f}s ({speedup:.2f}×)")
    fig.tight_layout()
    fig.savefig(HERE / "demo_ir07546_fit.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
