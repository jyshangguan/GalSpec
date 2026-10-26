"""Save and load :class:`galspec.jnesty_fit.JNesty_Fit` objects to/from FITS.

The layout mirrors :mod:`galspec.dynesty_fit_io` -- same extension names, same
column names, same header keywords wherever the meaning is the same -- so a file
written here can be read with the habits (and the tooling) built up for the
dynesty files.  Three things are JNesty-specific and have no counterpart there:

* ``SAMPLES_U``: the unit-cube coordinates of the dead points, which is what the
  sampler actually proposes and what a prior-sensitivity study needs;
* ``DLZ_TRAJ`` and ``SCALE_TR``: the remaining-evidence estimate and the
  random-walk step scale as functions of iteration, i.e. the convergence record;
* the model is a **JAX** model, so the model tree is serialised by
  :mod:`galspec.jax_models.serialization` (which knows about ``n_components``,
  template names and convolution wrappers) rather than by the astropy walker.

What a loaded fit can and cannot do
-----------------------------------
``custom_priors`` are Python callables -- in the documented examples they are
closures defined in ``__main__`` -- and no file format can preserve those.  The
FITS file stores the *resolved bounds* (``PARAM_BOUNDS``) and records which
parameters had a custom prior (``CUSTOM_PRIORS``).  A loaded fit therefore
reproduces the posterior, the evidence, the best-fit model, the credible
intervals and every evaluation of the model, but it cannot rebuild the original
prior transform, so it must not be used to resume sampling and would give
different priors if re-run.  ``load_jnesty_fit_from_fits`` warns when a file has
custom priors for exactly this reason.

Example
-------
>>> from galspec.jnesty_fit_io import save_jnesty_fit_to_fits, load_jnesty_fit_from_fits
>>> save_jnesty_fit_to_fits(fit, "iras07546.fits")          # doctest: +SKIP
>>> loaded = load_jnesty_fit_from_fits("iras07546.fits")    # doctest: +SKIP
>>> loaded.get_quantiles((0.16, 0.5, 0.84))[1][:3]          # doctest: +SKIP
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Optional

import numpy as np
from astropy.io import fits

# The JAX/jnesty imports are deliberately inside the two functions below: this
# module must stay importable on an install without JAX, where only the CPU
# samplers are used.

__all__ = ["save_jnesty_fit_to_fits", "load_jnesty_fit_from_fits"]


def _results_mapping(results) -> Dict[str, Any]:
    """JNesty results as a plain dict (it may be a mapping or an object)."""
    if results is None:
        return {}
    if isinstance(results, dict):
        return dict(results)
    out = {}
    for key in ("samples", "samples_u", "logl", "logvol", "logwt", "logz", "logzerr",
                "niter", "ncall", "logz_trajectory", "logzerr_trajectory",
                "delta_logZ_trajectory", "scale_trajectory", "information"):
        try:
            value = results[key]
        except (KeyError, TypeError, IndexError):
            value = getattr(results, key, None)
        if value is not None:
            out[key] = value
    return out


def save_jnesty_fit_to_fits(jnesty_fit, filename, thin: int = 1) -> str:
    """Write a :class:`~galspec.jnesty_fit.JNesty_Fit` into a FITS file.

    Parameters
    ----------
    jnesty_fit : JNesty_Fit
        A fit that has been run (`fit.fit(...)`).
    filename : str or pathlib.Path
        Output file; overwritten if it exists.
    thin : int
        Save every ``thin``-th posterior sample.  ``thin > 1`` shrinks the file
        for very long runs; the factor is recorded in the ``THIN`` keyword.

    Returns
    -------
    str
        The filename that was written.
    """
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if getattr(jnesty_fit, "samples_equal_weight", None) is None:
        raise RuntimeError("this JNesty_Fit has no posterior yet; run .fit() first")

    hdul = fits.HDUList()

    hdr = fits.Header()
    hdr["SAMPLER"] = ("JNESTY", "nested sampler that produced this file")
    hdr["NDIM"] = int(getattr(jnesty_fit, "ndim", 0))
    hdr["NLIVE"] = int(getattr(jnesty_fit, "nlive", 0))
    hdr["BOUND"] = str(getattr(jnesty_fit, "bound", "multi"))
    hdr["THIN"] = int(thin)
    hdr["LOGZ"] = float(getattr(jnesty_fit, "log_evidence", np.nan))
    hdr["LOGZERR"] = float(getattr(jnesty_fit, "log_evidence_err", np.nan))
    hdr["NITER"] = int(getattr(jnesty_fit, "n_iterations", 0) or 0)
    hdr["NCALL"] = int(getattr(jnesty_fit, "ncall", 0) or 0)
    hdr["RUNTIME"] = float(getattr(jnesty_fit, "runtime_seconds", np.nan))
    hdr["DEVICE"] = str(getattr(jnesty_fit, "device", "unknown"))
    for key, attr in (("RWALK_K", "rwalk_K"), ("QUEUE_SZ", "queue_size"),
                      ("MAXELLIP", "max_ellipsoids"), ("MINEFF", "min_eff")):
        value = getattr(jnesty_fit, attr, None)
        if value is not None:
            hdr[key] = value
    try:
        import jax
        hdr["X64"] = bool(jax.config.jax_enable_x64)
        hdr["JAXVER"] = str(jax.__version__)
    except Exception:                               # pragma: no cover - jax is a hard dep
        pass
    try:
        import jnesty
        hdr["JNESTYV"] = str(getattr(jnesty, "__version__", "unknown"))
    except Exception:                               # pragma: no cover
        pass
    hdul.append(fits.PrimaryHDU(header=hdr))

    # -- the data the fit was made against -------------------------------------
    for name, attr in (("WAVE_USE", "wave_use"), ("FLUX_USE", "flux_use"),
                       ("FERR", "ferr")):
        value = getattr(jnesty_fit, attr, None)
        if value is None and attr == "ferr":
            value = getattr(jnesty_fit, "error", None)
        if value is not None:
            hdul.append(fits.ImageHDU(np.asarray(value, dtype=float), name=name))

    initial = getattr(jnesty_fit, "_initial", None)
    if initial:
        order = list(jnesty_fit.param_names)
        hdul.append(fits.ImageHDU(
            np.asarray([initial[name] for name in order], dtype=float), name="THETA_INIT"))
    theta_best = getattr(jnesty_fit, "theta_best", None)
    if theta_best is not None:
        hdul.append(fits.ImageHDU(np.asarray(theta_best, dtype=float), name="THETA_BEST"))

    # -- sampler output --------------------------------------------------------
    results = _results_mapping(getattr(jnesty_fit, "results", None))
    for hdu_name, key in (("SAMPLES", "samples"), ("SAMPLES_U", "samples_u"),
                          ("LOGL", "logl"), ("LOGWT", "logwt"), ("LOGVOL", "logvol")):
        value = results.get(key)
        if value is not None:
            hdul.append(fits.ImageHDU(np.asarray(value, dtype=float)[::thin],
                                      name=hdu_name))
    for hdu_name, key in (("LOGZ", "logz_trajectory"), ("LOGZERR", "logzerr_trajectory"),
                          ("DLZ_TRAJ", "delta_logZ_trajectory"),
                          ("SCALE_TR", "scale_trajectory")):
        value = results.get(key)
        if value is None and key in ("logz_trajectory", "logzerr_trajectory"):
            value = results.get(key.replace("_trajectory", ""))
        if value is not None:
            hdul.append(fits.ImageHDU(np.asarray(value, dtype=float), name=hdu_name))

    samples_eq = np.asarray(jnesty_fit.samples_equal_weight, dtype=float)
    hdul.append(fits.ImageHDU(samples_eq[::thin], name="SAMPLES_EQ"))

    # -- parameter bookkeeping -------------------------------------------------
    param_names = [str(name) for name in jnesty_fit.param_names]
    hdul.append(fits.BinTableHDU.from_columns(
        [fits.Column(name="PARAM_NAME", format="A100",
                     array=np.array(param_names, dtype="U100"))],
        name="PARAM_NAMES"))
    full_names = [str(name) for name in getattr(jnesty_fit, "full_param_names", param_names)]
    hdul.append(fits.BinTableHDU.from_columns(
        [fits.Column(name="FULL_PARAM_NAME", format="A100",
                     array=np.array(full_names, dtype="U100"))],
        name="FULL_PARAM"))

    bounds = getattr(jnesty_fit, "param_bounds", None)
    if bounds:
        hdul.append(fits.BinTableHDU.from_columns(
            [fits.Column(name="BOUND_LOWER", format="D",
                         array=np.array([b[0] for b in bounds], dtype=float)),
             fits.Column(name="BOUND_UPPER", format="D",
                         array=np.array([b[1] for b in bounds], dtype=float))],
            name="PARAM_BOUNDS"))

    log_scale = getattr(jnesty_fit, "_log_scale_params", None)
    if log_scale is not None:
        hdul.append(fits.BinTableHDU.from_columns(
            [fits.Column(name="LOG_SCALE", format="L",
                         array=np.array([name in log_scale for name in param_names]))],
            name="LOG_SCALE_PARAMS"))

    custom = dict(getattr(jnesty_fit, "custom_priors", {}) or {})
    if custom:
        # Priors given as galspec.priors objects are stored whole (kind + the
        # arguments of their constructor) and rebuilt on load.  A bare callable
        # has no data form, so only its repr is kept and loading reports it.
        full_by_flat = dict(zip(getattr(jnesty_fit, "param_names", []),
                                getattr(jnesty_fit, "full_param_names", [])))
        names, kinds, payloads, reprs = [], [], [], []
        for key, prior in custom.items():
            name = full_by_flat.get(key, key)
            names.append(str(name))
            reprs.append(repr(prior)[:200])
            try:
                data = prior.to_dict()
            except TypeError:
                kinds.append("callable")
                payloads.append("")
            else:
                kinds.append(str(data["kind"]))
                payloads.append(json.dumps(data["args"], sort_keys=True))
        hdul.append(fits.BinTableHDU.from_columns(
            [fits.Column(name="PARAM_NAME", format="A100",
                         array=np.array(names, dtype="U100")),
             fits.Column(name="KIND", format="A32",
                         array=np.array(kinds, dtype="U32")),
             fits.Column(name="ARGS", format="A400",
                         array=np.array(payloads, dtype="U400")),
             fits.Column(name="REPR", format="A200",
                         array=np.array(reprs, dtype="U200"))],
            name="CUSTOM_PRIORS"))

    # -- the model -------------------------------------------------------------
    from .jax_models.serialization import serialize_jax_model_tree

    tree, arrays = serialize_jax_model_tree(jnesty_fit.model)
    array_meta = {}
    for index, (key, value) in enumerate(arrays.items()):
        extname = f"ARR{index:03d}"[:8]
        hdul.append(fits.ImageHDU(np.asarray(value), name=extname))
        array_meta[key] = extname
    hdul.append(fits.ImageHDU(
        np.frombuffer(json.dumps(tree).encode("utf-8"), dtype=np.uint8), name="MODEL_STR"))
    if array_meta:
        hdul.append(fits.ImageHDU(
            np.frombuffer(json.dumps(array_meta).encode("utf-8"), dtype=np.uint8),
            name="MODEL_ARR"))

    # -- the model at a glance, for humans and for quick checks ----------------
    model = jnesty_fit.model
    rows_names, rows_values, rows_fixed, rows_lo, rows_hi, rows_tied = [], [], [], [], [], []
    for name in model.param_names:
        param = getattr(model, name)
        rows_names.append(name)
        rows_values.append(float(param.value))
        rows_fixed.append(bool(param.fixed))
        lower, upper = param.bounds
        rows_lo.append(-np.inf if lower is None else float(lower))
        rows_hi.append(np.inf if upper is None else float(upper))
        rows_tied.append("" if param.tied is None else str(param.tied))
    hdul.append(fits.BinTableHDU.from_columns(
        [fits.Column(name="PARAM_NAME", format="A100",
                     array=np.array(rows_names, dtype="U100")),
         fits.Column(name="VALUE", format="D", array=np.array(rows_values, dtype=float)),
         fits.Column(name="FIXED", format="L", array=np.array(rows_fixed, dtype=bool)),
         fits.Column(name="BOUND_LOW", format="D", array=np.array(rows_lo, dtype=float)),
         fits.Column(name="BOUND_UP", format="D", array=np.array(rows_hi, dtype=float)),
         fits.Column(name="TIED", format="A100",
                     array=np.array(rows_tied, dtype="U100"))],
        name="MODEL_PARAM"))

    submodels = list(getattr(model, "submodel_names", ()) or ())
    hdul.append(fits.BinTableHDU.from_columns(
        [fits.Column(name="SUBMODEL_NAME", format="A100",
                     array=np.array(submodels, dtype="U100"))],
        name="SUBMODELS"))

    hdul.writeto(filename, overwrite=True)
    print(f"JNesty_Fit object saved to {filename}")
    return str(filename)


def _decode_json_hdu(hdul, name: str):
    if name not in hdul:
        return None
    return json.loads(hdul[name].data.tobytes().decode("utf-8"))


def load_jnesty_fit_from_fits(filename, wave: Optional[np.ndarray] = None):
    """Read a file written by :func:`save_jnesty_fit_to_fits`.

    Parameters
    ----------
    filename : str or pathlib.Path
        The FITS file.
    wave : sequence, optional
        Override the stored wavelength grid.  The model is prepared on the stored
        grid by default; pass this only if you know the model was built for a
        different one.

    Returns
    -------
    JNesty_Fit
        An object with ``model``, posterior samples, bounds, evidence and every
        method of the original fit except those that need the sampler itself
        (``fit.fit`` must not be called again -- see the module notes about
        ``custom_priors``).

    Warns
    -----
    UserWarning
        If the file records custom priors, because their callables cannot be
        stored and the loaded fit therefore cannot re-derive the prior
        transform.
    """
    from .jnesty_fit import JNesty_Fit, _limits_by_base_name
    from .jax_models.serialization import deserialize_jax_model_tree

    with fits.open(filename) as hdul:
        header = hdul[0].header
        data: Dict[str, Any] = {
            "ndim": int(header.get("NDIM", 0)),
            "nlive": int(header.get("NLIVE", 0)),
            "bound": header.get("BOUND", "multi"),
            "thin": int(header.get("THIN", 1)),
            "log_evidence": float(header.get("LOGZ", np.nan)),
            "log_evidence_err": float(header.get("LOGZERR", np.nan)),
            "n_iterations": int(header.get("NITER", 0)),
            "ncall": int(header.get("NCALL", 0) or 0),
            "runtime_seconds": float(header.get("RUNTIME", np.nan)),
            "device": header.get("DEVICE", "unknown"),
        }
        if data["thin"] > 1:
            print(f"Note: samples were saved with thinning factor {data['thin']}")

        for key, name in (("wave_use", "WAVE_USE"), ("flux_use", "FLUX_USE"),
                          ("ferr", "FERR")):
            if name in hdul:
                data[key] = np.asarray(hdul[name].data, dtype=float)

        results: Dict[str, Any] = {}
        for key, name in (("samples", "SAMPLES"), ("samples_u", "SAMPLES_U"),
                          ("logl", "LOGL"), ("logwt", "LOGWT"), ("logvol", "LOGVOL"),
                          ("logz_trajectory", "LOGZ"), ("logzerr_trajectory", "LOGZERR"),
                          ("delta_logZ_trajectory", "DLZ_TRAJ"),
                          ("scale_trajectory", "SCALE_TR")):
            if name in hdul:
                results[key] = np.asarray(hdul[name].data, dtype=float)
        results.setdefault("logz", data["log_evidence"])
        results.setdefault("logzerr", data["log_evidence_err"])
        results["niter"] = data["n_iterations"]
        data["results"] = results
        data["samples_equal_weight"] = (np.asarray(hdul["SAMPLES_EQ"].data, dtype=float)
                                        if "SAMPLES_EQ" in hdul else None)
        data["theta_best"] = (np.asarray(hdul["THETA_BEST"].data, dtype=float)
                              if "THETA_BEST" in hdul else None)

        data["param_names"] = ([str(v) for v in hdul["PARAM_NAMES"].data["PARAM_NAME"]]
                               if "PARAM_NAMES" in hdul else [])
        data["full_param_names"] = (
            [str(v) for v in hdul["FULL_PARAM"].data["FULL_PARAM_NAME"]]
            if "FULL_PARAM" in hdul else list(data["param_names"]))
        if "PARAM_BOUNDS" in hdul:
            table = hdul["PARAM_BOUNDS"].data
            data["param_bounds"] = [(float(lo), float(up))
                                    for lo, up in zip(table["BOUND_LOWER"],
                                                      table["BOUND_UPPER"])]
        else:
            data["param_bounds"] = None
        if "LOG_SCALE_PARAMS" in hdul:
            flags = hdul["LOG_SCALE_PARAMS"].data["LOG_SCALE"]
            data["_log_scale_params"] = {name for name, flag in
                                         zip(data["param_names"], flags) if flag}
        else:
            data["_log_scale_params"] = set()
        custom_names: list = []
        custom_priors: dict = {}
        if "CUSTOM_PRIORS" in hdul:
            table = hdul["CUSTOM_PRIORS"].data
            custom_names = [str(v) for v in table["PARAM_NAME"]]
            if "KIND" in table.names:
                from .priors import Prior

                for row in table:
                    kind = str(row["KIND"]).strip()
                    if not kind or kind == "callable":
                        continue
                    try:
                        args = json.loads(str(row["ARGS"]) or "{}")
                        custom_priors[str(row["PARAM_NAME"])] = Prior.from_dict(
                            {"kind": kind, "args": args})
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        warnings.warn(f"{filename}: could not rebuild the prior of "
                                      f"{row['PARAM_NAME']} ({exc})", stacklevel=2)

        tree = _decode_json_hdu(hdul, "MODEL_STR")
        array_meta = _decode_json_hdu(hdul, "MODEL_ARR") or {}
        arrays = {key: np.asarray(hdul[extname].data) for key, extname in array_meta.items()
                  if extname in hdul}

    if tree is None:
        raise ValueError(f"{filename} has no MODEL_STR extension; is it a JNesty file?")

    wave_grid = np.asarray(wave, dtype=float) if wave is not None else data.get("wave_use")
    model = deserialize_jax_model_tree(tree, arrays, wave=wave_grid)

    from .jax_models.serialization import translate_parameter_name

    stored = data.get("param_names") or []
    translated = [translate_parameter_name(name) for name in stored]
    if any(old != new for old, new in zip(stored, translated)):
        renamed = {old: new for old, new in zip(stored, translated) if old != new}
        warnings.warn(f"{filename} uses the old parameter name(s) {renamed}; they are "
                      "renamed on load, the values are unchanged", stacklevel=2)
        data["param_names"] = translated
        data["full_param_names"] = [f"{name.rsplit('.', 1)[0]}.{translate_parameter_name(name.rsplit('.', 1)[-1])}"
                                    if '.' in name else translate_parameter_name(name)
                                    for name in data.get("full_param_names", translated)]
    current = [str(name) for name in model.free_param_names]
    if translated and translated != current:
        raise ValueError(
            f"{filename} was written with parameters {stored}, but the model in it now "
            f"has {current}. The sample columns would be misaligned, so the file is not "
            "loaded; refit with the current GalSpec, or load it with the version that "
            "wrote it.")

    if data["param_bounds"] and data["param_names"]:
        model.set_prior_limits(_limits_by_base_name(data["param_bounds"],
                                                    data["param_names"]))
        model.prepare(wave_grid)

    fit = JNesty_Fit.__new__(JNesty_Fit)
    fit.model = model
    for key, value in data.items():
        if key == "thin":
            continue
        setattr(fit, key, value)
    fit.custom_priors = {}
    fit.bounds_dict = {}
    fit.default_bounds = dict(getattr(JNesty_Fit, "DEFAULT_BOUNDS", {}))
    fit._initial = {name: float(value) for name, value in
                    zip(fit.param_names,
                        np.asarray(model.get_free_params(), dtype=float))}
    fit.sampler = None
    fit.param_map = {full: (index, name) for index, (full, name)
                     in enumerate(zip(fit.full_param_names, fit.param_names))}
    from .param_labels import qualified_parameter_labels
    fit.param_labels = qualified_parameter_labels(fit.model, fit.full_param_names)
    fit.full_param_labels = list(fit.param_labels)

    # -- priors: put the stored ones back on the model, then rebuild the map -----
    unrestored = [name for name in custom_names if name not in custom_priors]
    for name, prior in custom_priors.items():
        try:
            model.set_prior(name, prior)
        except (KeyError, ValueError) as exc:
            unrestored.append(name)
            warnings.warn(f"{filename}: the stored prior of {name} could not be "
                          f"attached ({exc}); it is reported, not restored",
                          stacklevel=2)
    fit.model_priors = fit._collect_model_priors()
    fit.custom_priors = dict(fit.model_priors)
    fit._build_prior_spec()

    if unrestored:
        warnings.warn(
            "this file records custom priors for "
            f"{sorted(set(unrestored))}, but those cannot be stored in FITS (a bare "
            "Python callable has no data form): the posterior, evidence and model are "
            "restored, the prior transform of those parameters is uniform instead. "
            "Do not resume sampling from this object.", stacklevel=2)

    print(f"JNesty_Fit object loaded from {filename}")
    return fit