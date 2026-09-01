#!/usr/bin/env python3
"""Full-spectrum IRAS 07546+3928 dynesty demo, without a polynomial."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from extinction import ccm89, remove
import galspec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_FILE = ROOT / "example/data/ir07546sed.fits"


def load_spectrum():
    with fits.open(DATA_FILE) as hdul:
        hdr, flux = hdul[0].header, np.asarray(hdul[0].data, float) * 1e14
    wave = hdr["CRVAL1"] + hdr["CD1_1"] * np.arange(flux.size)
    flux = remove(ccm89(wave, 0.176, r_v=3.1, unit="aa"), flux)
    wave /= 1 + 0.0953
    keep = (wave > 4200) & (wave < 6800)
    wave, flux = wave[keep], flux[keep]
    error = np.maximum(0.05 * np.abs(flux), 1e-4)
    fit_mask = ~((wave > 5770) & (wave < 6000))
    return wave, flux, error, fit_mask


def build_model():
    """Exact additive model from wrk_sagan_iras07546.ipynb."""
    w, label = galspec.utils.line_wave_dict, galspec.utils.line_label_dict
    pl = models.PowerLaw1D(amplitude=.55, x_0=5500, alpha=1,
                           fixed={"x_0": True}, name="Continuum")
    iron = galspec.IronTemplate(amplitude=.2, stddev=900, z=0, name="Fe II")
    bha = galspec.Line_MultiGauss(n_components=2, amp_c=2.22, dv_c=280,
        sigma_c=830, wavec=w["Halpha"], name=label["Halpha"],
        amp_w0=.26, dv_w0=-185, sigma_w0=2400)
    bhb = galspec.Line_MultiGauss(n_components=2, amp_c=.8, dv_c=-17,
        sigma_c=850, wavec=w["Hbeta"], name=label["Hbeta"],
        amp_w0=.3, dv_w0=-120, sigma_w0=2700,
        bounds={"sigma_w0": (100, 4000)})
    bhg = galspec.Line_MultiGauss(n_components=1, amp_c=.4, dv_c=80,
        sigma_c=1200, wavec=w["Hgamma"], name=label["Hgamma"])
    bhe2 = galspec.Line_MultiGauss(n_components=1, amp_c=.06, dv_c=85,
        sigma_c=2300, wavec=w["HeII_4686"], name=label["HeII_4686"],
        bounds={"sigma_c": (100, 4000), "dv_c": (-500, 500)})
    o3 = galspec.Line_MultiGauss_doublet(n_components=3, amp_c0=1.7,
        amp_c1=.6, dv_c=-43, sigma_c=250, wavec0=w["OIII_5007"],
        wavec1=w["OIII_4959"], name="[O III]", amp_w0=.35,
        dv_w0=-280, sigma_w0=340, amp_w1=.05, dv_w1=500, sigma_w1=1000)
    s2 = galspec.Line_MultiGauss_doublet(n_components=1, amp_c0=.1,
        amp_c1=.1, wavec0=w["SII_6716"], wavec1=w["SII_6731"], name="[S II]")
    nha = galspec.Line_Gaussian(amplitude=1.1, wavec=w["Halpha"], name="narrow Ha")
    nhb = galspec.Line_Gaussian(amplitude=.4, wavec=w["Hbeta"], name="narrow Hb")
    nhg = galspec.Line_Gaussian(amplitude=.12, wavec=w["Hgamma"], name="narrow Hg")
    nhe2 = galspec.Line_Gaussian(amplitude=.08, wavec=w["HeII_4686"], name="narrow He2")
    no3 = galspec.Line_Gaussian(amplitude=.05, wavec=w["OIII_4363"], name="[O III] 4363")
    model = pl + iron + bha + nha + bhb + nhb + bhg + nhg + bhe2 + nhe2 + o3 + s2 + no3

    def tie_o3(m): return m["[O III]"].amp_c0 / 2.98
    def tie_sigma(m): return m["[O III]"].sigma_c
    def tie_dv(m): return m["[O III]"].dv_c
    o3.amp_c1.tied = tie_o3
    s2.sigma_c.tied, s2.dv_c.tied = tie_sigma, tie_dv
    for line in (nha, nhb, nhg, nhe2, no3):
        line.sigma.tied, line.dv.tied = tie_sigma, tie_dv
    return model


def local_bounds(model):
    """Finite priors around the notebook's deterministic initialization."""
    result = {}
    for name in model.param_names:
        p = getattr(model, name)
        if p.fixed or p.tied: continue
        value, base = float(p.value), galspec.Dynesty_Fit._base_name(name).lower()
        if "amplitude" in base or base.startswith("amp"):
            lo, hi = max(0, value*.35), max(value*2, value+.05)
        elif base.startswith("sigma") or base == "stddev":
            lo, hi = max(10, value*.5), max(30, value*1.6)
        elif base.startswith("dv"): lo, hi = value-500, value+500
        elif base == "alpha": lo, hi = value-1.5, value+1.5
        elif base in {"z", "redshift"}: lo, hi = value-.003, value+.003
        else:
            width = max(abs(value)*.5, .1); lo, hi = value-width, value+width
        lower, upper = p.bounds
        if lower is not None and np.isfinite(lower): lo = max(lo, float(lower))
        if upper is not None and np.isfinite(upper): hi = min(hi, float(upper))
        if not lo < hi:
            width = max(abs(value)*.1, 1e-3); lo, hi = value-width, value+width
        result[name] = (lo, hi)
    return result


def run(model, wave, flux, error, workers, bounds, args):
    fit = galspec.Dynesty_Fit(model, wave, flux, error, bounds_dict=bounds,
        sample_method="rwalk", bound="multi", nlive=args.nlive,
        n_processes=workers, rstate=np.random.default_rng(args.seed))
    run_options = {"dlogz": args.dlogz}
    if args.maxiter is not None:
        run_options["maxiter"] = args.maxiter
    fit.fit(progress=True, **run_options)
    best, names, theta = fit.get_best_fit()
    return fit, best, {"workers": workers, "runtime_seconds": fit.runtime_seconds,
        "likelihood_calls": fit.ncall, "iterations": len(fit.results.logl),
        "log_evidence": fit.log_evidence, "log_evidence_error": fit.log_evidence_err,
        "chi2": float(np.sum(((flux-best(wave))/error)**2)),
        "parameters": dict(zip(names, map(float, theta)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--nlive", type=int, default=80)
    ap.add_argument("--maxiter", type=int, default=None,
                    help="optional safety limit; default runs to convergence")
    ap.add_argument("--dlogz", type=float, default=5.)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--parallel-only", action="store_true",
                    help="run only the converged multicore science fit")
    ap.add_argument("--output-tag", default="full_spectrum",
                    help="tag used for JSON and plot output names")
    args = ap.parse_args()
    wave_all, flux_all, error_all, mask = load_spectrum()
    initial = build_model()
    initialized = fitting.LevMarLSQFitter()(initial, wave_all, flux_all,
        weights=mask.astype(float), maxiter=10000)
    wave, flux, error = wave_all[mask], flux_all[mask], error_all[mask]
    bounds = local_bounds(initialized)
    if args.nlive <= len(bounds): ap.error(f"--nlive must exceed {len(bounds)}")
    serial_fit = None
    serial = None
    if not args.parallel_only:
        serial_fit, _, serial = run(initialized, wave, flux, error, 1, bounds, args)
    parallel_fit, parallel_model, parallel = run(initialized, wave, flux, error, args.cores, bounds, args)
    speedup = None if serial is None else serial["runtime_seconds"] / parallel["runtime_seconds"]
    residual_fit = (flux-parallel_model(wave))/error
    reduced_chi2 = parallel["chi2"] / (len(wave)-len(bounds))
    at_bound = []
    for name, value in parallel["parameters"].items():
        lo, hi = bounds[name]
        if min(value-lo, hi-value) < 0.01*(hi-lo): at_bound.append(name)
    summary = {"data_file": str(DATA_FILE), "fitted_data_points": len(wave),
        "free_parameters": len(bounds), "multiplicative_polynomial": False,
        "masked_range_angstrom": [5770, 6000], "nlive": args.nlive,
        "maxiter": args.maxiter, "dlogz": args.dlogz,
        "stopped_at_iteration_limit": (args.maxiter is not None and
            len(parallel_fit.results.logl) >= args.maxiter),
        "serial": serial, "parallel": parallel, "speedup": speedup,
        "parallel_faster": None if speedup is None else speedup > 1,
        "validation": {"reduced_chi2": reduced_chi2,
            "residual_mean_sigma": float(np.mean(residual_fit)),
            "residual_std_sigma": float(np.std(residual_fit)),
            "parameters_within_1pct_of_prior_bound": at_bound}}
    (HERE/f"demo_{args.output_tag}_results.json").write_text(
        json.dumps(summary, indent=2)+"\n")
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax.step(wave_all, flux_all, color=".3", lw=.45, where="mid", label="Data")
    ax.plot(wave_all, parallel_model(wave_all), color="tab:red", lw=1,
            label=f"Full additive model ({args.cores} cores)")
    ax.axvspan(5770, 6000, color=".85", label="Excluded"); ax.legend(); ax.set_ylabel("Scaled flux")
    residual = (flux_all-parallel_model(wave_all))/error_all
    axr.plot(wave_all, residual, color=".3", lw=.4); axr.axhline(0, color="tab:red", lw=.8)
    axr.axvspan(5770, 6000, color=".85"); axr.set(xlabel="Rest wavelength (Å)", ylabel="Residual / σ", ylim=(-8,8))
    timing = (f"{args.cores} cores {parallel['runtime_seconds']:.1f}s" if serial is None else
              f"1 core {serial['runtime_seconds']:.1f}s, {args.cores} cores "
              f"{parallel['runtime_seconds']:.1f}s ({speedup:.2f}×)")
    fig.suptitle(f"Full spectrum: {timing}; reduced χ²={reduced_chi2:.2f}")
    fig.tight_layout(); fig.savefig(HERE/f"demo_ir07546_{args.output_tag}.png", dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
