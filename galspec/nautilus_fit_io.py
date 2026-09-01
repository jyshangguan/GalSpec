"""Save and load :class:`galspec.Nautilus_Fit` objects as FITS files."""

from __future__ import annotations

import json
import numpy as np
from astropy.io import fits

from .mcmc_fit_io import serialize_model_tree, deserialize_model_tree

__all__ = ["save_nautilus_fit_to_fits", "load_nautilus_fit_from_fits"]


def _text_hdu(value, name):
    return fits.ImageHDU(np.frombuffer(json.dumps(value).encode(), dtype=np.uint8), name=name)


def _read_text(hdu):
    return json.loads(hdu.data.tobytes().decode())


def save_nautilus_fit_to_fits(fit, filename, thin=1):
    """Persist input data, model, priors, posterior samples and run metadata."""
    if thin < 1:
        raise ValueError("thin must be >= 1")
    header = fits.Header()
    header["SAMPLER"] = "NAUTILUS"
    header["NDIM"] = fit.ndim
    header["NLIVE"] = fit.n_live
    header["THIN"] = thin
    header["LOGZ"] = fit.log_evidence if fit.log_evidence is not None else np.nan
    header["RUNTIME"] = fit.runtime_seconds if fit.runtime_seconds is not None else np.nan
    header["NCALL"] = fit.ncall if fit.ncall is not None else 0
    hdus = [fits.PrimaryHDU(header=header),
            fits.ImageHDU(fit.wave_use, name="WAVE_USE"),
            fits.ImageHDU(fit.flux_use, name="FLUX_USE"),
            fits.ImageHDU(fit.ferr, name="FERR"),
            fits.ImageHDU(fit.theta_initial, name="THETA_INIT")]
    for data, name in ((fit.samples, "SAMPLES"),
                       (fit.log_weights, "LOG_WEIGHT"),
                       (fit.log_likelihood_values, "LOGL"),
                       (fit.samples_equal_weight, "SAMPLES_EQ"),
                       (fit.theta_best, "THETA_BEST")):
        if data is not None:
            hdus.append(fits.ImageHDU(np.asarray(data)[::thin] if name != "THETA_BEST"
                                      else np.asarray(data), name=name))
    hdus.append(_text_hdu(fit.param_names, "PARAMS"))
    hdus.append(_text_hdu(fit.full_param_names, "FULLPAR"))
    hdus.append(_text_hdu(fit.param_bounds, "PRIORS"))
    hdus.append(_text_hdu(sorted(fit._log_scale_params), "LOGSCALE"))

    nodes, special = serialize_model_tree(fit.model)
    special_meta = {}
    for node, values in special.items():
        special_meta[str(node)] = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                extname = f"N{node}_{key}".upper()[:8]
                hdus.append(fits.ImageHDU(np.atleast_1d(value), name=extname))
                special_meta[str(node)][key] = {
                    "_is_array": True, "extname": extname,
                    "shape": list(value.shape)}
            else:
                special_meta[str(node)][key] = value
    hdus.append(_text_hdu(nodes, "MODELSTR"))
    hdus.append(_text_hdu(special_meta, "MODELSPC"))
    params = {}
    for name in fit.model.param_names:
        p = getattr(fit.model, name)
        params[name] = {"value": float(p.value), "fixed": bool(p.fixed),
                        "bounds": [p.bounds[0], p.bounds[1]]}
    hdus.append(_text_hdu(params, "MODELPAR"))
    fits.HDUList(hdus).writeto(filename, overwrite=True)


def load_nautilus_fit_from_fits(filename):
    """Recover a fitted object without rerunning Nautilus."""
    from .nautilus_fit import Nautilus_Fit
    with fits.open(filename) as hdul:
        header = hdul[0].header
        nodes = _read_text(hdul["MODELSTR"])
        params = _read_text(hdul["MODELPAR"])
        special = {}
        for node, values in _read_text(hdul["MODELSPC"]).items():
            special[int(node)] = {}
            for key, value in values.items():
                if isinstance(value, dict) and value.get("_is_array"):
                    array = np.asarray(hdul[value["extname"]].data)
                    special[int(node)][key] = array.reshape(value["shape"])
                else:
                    special[int(node)][key] = value
        param_values = {name: {"value": value["value"], "fixed": value["fixed"],
                               "bounds": tuple(value["bounds"])}
                        for name, value in params.items()}
        model = deserialize_model_tree(nodes, special, param_values)
        obj = Nautilus_Fit.__new__(Nautilus_Fit)
        obj.model = model
        obj.wave_use = np.array(hdul["WAVE_USE"].data)
        obj.flux_use = np.array(hdul["FLUX_USE"].data)
        obj.ferr = np.array(hdul["FERR"].data)
        obj.theta_initial = np.array(hdul["THETA_INIT"].data)
        obj.param_names = _read_text(hdul["PARAMS"])
        obj.full_param_names = _read_text(hdul["FULLPAR"])
        obj.param_bounds = [tuple(v) for v in _read_text(hdul["PRIORS"])]
        obj._log_scale_params = set(_read_text(hdul["LOGSCALE"]))
        obj.ndim = int(header["NDIM"])
        obj.n_live = obj.nlive = int(header["NLIVE"])
        obj.log_evidence = float(header["LOGZ"])
        obj.runtime_seconds = float(header["RUNTIME"])
        obj.ncall = int(header["NCALL"])
        obj.samples = np.array(hdul["SAMPLES"].data) if "SAMPLES" in hdul else None
        obj.log_weights = np.array(hdul["LOG_WEIGHT"].data) if "LOG_WEIGHT" in hdul else None
        obj.log_likelihood_values = np.array(hdul["LOGL"].data) if "LOGL" in hdul else None
        obj.samples_equal_weight = np.array(hdul["SAMPLES_EQ"].data) if "SAMPLES_EQ" in hdul else None
        obj.theta_best = np.array(hdul["THETA_BEST"].data) if "THETA_BEST" in hdul else None
    obj.param_map = dict(zip(obj.full_param_names, enumerate(obj.param_names)))
    obj.custom_priors = {}
    obj.bounds_dict = {}
    obj.default_bounds = obj.DEFAULT_BOUNDS.copy()
    obj.n_processes, obj.pool, obj.seed = 1, None, None
    obj.filepath, obj.resume, obj.sampler_kwargs = None, False, {}
    obj.sampler = None
    obj.log_evidence_err = None
    return obj
