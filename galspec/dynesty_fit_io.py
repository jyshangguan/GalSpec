"""
Save and Load Dynesty_Fit objects to/from FITS files.

This module handles serialization of Dynesty_Fit objects, including
the model structure, posterior samples, and evidence.
"""

import numpy as np
from astropy.io import fits
from astropy.modeling import models, CompoundModel
from copy import deepcopy
import json
import operator
import importlib


__all__ = ['save_dynesty_fit_to_fits', 'load_dynesty_fit_from_fits']


# Import serialization functions from mcmc_fit_io
from .mcmc_fit_io import (
    _is_json_serializable, _to_json_compatible, _resolve_qualname,
    _serialize_tied_callable, _deserialize_tied_callable,
    _is_compound, _check_convolution, serialize_model_tree,
    deserialize_model_tree, apply_convolution, deserialize_single_model
)


def save_dynesty_fit_to_fits(dynesty_fit, filename, thin=1):
    """
    Save a Dynesty_Fit object entirely into a FITS file.

    Parameters
    ----------
    dynesty_fit : Dynesty_Fit
        The Dynesty_Fit object to save.
    filename : str
        The output FITS filename.
    thin : int, optional
        Thinning factor for samples arrays.
        If thin > 1, only every Nth sample will be saved.
        Default is 1 (no thinning).
    """
    if thin < 1:
        raise ValueError("thin must be >= 1")

    # Create HDU list
    hdul = fits.HDUList()

    # Primary HDU with basic information
    primary_hdr = fits.Header()
    primary_hdr['NDIM'] = getattr(dynesty_fit, 'ndim', 0)
    primary_hdr['NLIVE'] = getattr(dynesty_fit, 'nlive', 500)
    primary_hdr['SAMPLE_METHOD'] = getattr(dynesty_fit, 'sample_method', 'rwalk')
    primary_hdr['BOUND'] = getattr(dynesty_fit, 'bound', 'multi')
    primary_hdr['THIN'] = thin  # Record the thinning factor
    primary_hdr['LOG_EVIDENCE'] = getattr(dynesty_fit, 'log_evidence', np.nan)
    primary_hdr['LOG_EVIDENCE_ERR'] = getattr(dynesty_fit, 'log_evidence_err', np.nan)
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)
    hdul.append(primary_hdu)

    # Save data arrays
    hdul.append(fits.ImageHDU(dynesty_fit.wave_use, name='WAVE_USE'))
    hdul.append(fits.ImageHDU(dynesty_fit.flux_use, name='FLUX_USE'))
    hdul.append(fits.ImageHDU(dynesty_fit.ferr, name='FERR'))

    if hasattr(dynesty_fit, 'theta_initial') and dynesty_fit.theta_initial is not None:
        hdul.append(fits.ImageHDU(dynesty_fit.theta_initial, name='THETA_INIT'))

    if hasattr(dynesty_fit, 'theta_best') and dynesty_fit.theta_best is not None:
        hdul.append(fits.ImageHDU(dynesty_fit.theta_best, name='THETA_BEST'))

    # Save dynesty results if they exist
    if hasattr(dynesty_fit, 'results') and dynesty_fit.results is not None:
        results = dynesty_fit.results

        # Save samples (raw, not resampled)
        if hasattr(results, 'samples'):
            samples_thinned = results.samples[::thin]
            hdul.append(fits.ImageHDU(samples_thinned, name='SAMPLES'))
            if thin > 1:
                print(f"Thinned samples from {results.samples.shape[0]} to {samples_thinned.shape[0]} (thin={thin})")

        # Save log weights
        if hasattr(results, 'logwt'):
            logwt_thinned = results.logwt[::thin]
            hdul.append(fits.ImageHDU(logwt_thinned, name='LOGWT'))

        # Save log likelihoods
        if hasattr(results, 'logl'):
            logl_thinned = results.logl[::thin]
            hdul.append(fits.ImageHDU(logl_thinned, name='LOGL'))

        # Save log vol (cumulative volume)
        if hasattr(results, 'logvol'):
            logvol_thinned = results.logvol[::thin]
            hdul.append(fits.ImageHDU(logvol_thinned, name='LOGVOL'))

        # Save log evidence evolution
        if hasattr(results, 'logz'):
            hdul.append(fits.ImageHDU(results.logz, name='LOGZ'))

        # Save log evidence errors
        if hasattr(results, 'logzerr'):
            hdul.append(fits.ImageHDU(results.logzerr, name='LOGZERR'))

        # Save information matrix
        if hasattr(results, 'information'):
            hdul.append(fits.ImageHDU(results.information, name='INFORMATION'))

    # Save equal-weighted samples (for corner plots, etc.)
    if hasattr(dynesty_fit, 'samples_equal_weight') and dynesty_fit.samples_equal_weight is not None:
        samples_eq_thinned = dynesty_fit.samples_equal_weight[::thin]
        hdul.append(fits.ImageHDU(samples_eq_thinned, name='SAMPLES_EQ'))
        if thin > 1:
            print(f"Thinned equal-weight samples from {dynesty_fit.samples_equal_weight.shape[0]} to {samples_eq_thinned.shape[0]} (thin={thin})")

    # Save parameter names as a table
    if hasattr(dynesty_fit, 'param_names') and dynesty_fit.param_names is not None:
        param_names_array = np.array(dynesty_fit.param_names, dtype='U100')
        col1 = fits.Column(name='PARAM_NAME', format='A100', array=param_names_array)
        param_table = fits.BinTableHDU.from_columns([col1], name='PARAM_NAMES')
        hdul.append(param_table)

    # Save full parameter names
    if hasattr(dynesty_fit, 'full_param_names') and dynesty_fit.full_param_names is not None:
        full_param_names_array = np.array(dynesty_fit.full_param_names, dtype='U100')
        col2 = fits.Column(name='FULL_PARAM_NAME', format='A100', array=full_param_names_array)
        full_param_table = fits.BinTableHDU.from_columns([col2], name='FULL_PARAM')
        hdul.append(full_param_table)

    # Save parameter bounds
    if hasattr(dynesty_fit, 'param_bounds') and dynesty_fit.param_bounds is not None:
        n_params = len(dynesty_fit.param_bounds)
        bounds_lower = np.array([b[0] for b in dynesty_fit.param_bounds])
        bounds_upper = np.array([b[1] for b in dynesty_fit.param_bounds])

        col_lower = fits.Column(name='BOUND_LOWER', format='D', array=bounds_lower)
        col_upper = fits.Column(name='BOUND_UPPER', format='D', array=bounds_upper)
        bounds_table = fits.BinTableHDU.from_columns([col_lower, col_upper], name='PARAM_BOUNDS')
        hdul.append(bounds_table)

    # Save log-scale parameter indicators
    if hasattr(dynesty_fit, '_log_scale_params') and dynesty_fit._log_scale_params is not None:
        log_scale_flags = [pn in dynesty_fit._log_scale_params for pn in dynesty_fit.param_names]
        col_logscale = fits.Column(name='LOG_SCALE', format='L', array=np.array(log_scale_flags))
        logscale_table = fits.BinTableHDU.from_columns([col_logscale], name='LOG_SCALE_PARAMS')
        hdul.append(logscale_table)

    # Save model information (reuse from mcmc_fit_io)
    model_nodes, model_special_data = serialize_model_tree(dynesty_fit.model)

    # Save special data arrays for each model
    model_special_data_for_json = {}
    for node_id, special_data in model_special_data.items():
        model_special_data_for_json[str(node_id)] = {}
        for key, value in special_data.items():
            if isinstance(value, np.ndarray):
                extname = f'N{node_id}_{key}'.upper()[:8]
                value_to_save = np.atleast_1d(value)
                hdul.append(fits.ImageHDU(value_to_save, name=extname))
                model_special_data_for_json[str(node_id)][key] = {
                    '_is_array': True,
                    'extname': extname,
                    'shape': list(value.shape),
                    'dtype': str(value.dtype)
                }
            else:
                if isinstance(value, (list, tuple)):
                    model_special_data_for_json[str(node_id)][key] = list(value)
                else:
                    model_special_data_for_json[str(node_id)][key] = value

    # Save model structure as JSON
    model_json_str = json.dumps(model_nodes, indent=2)
    model_json_bytes = model_json_str.encode('utf-8')
    model_hdu = fits.ImageHDU(np.frombuffer(model_json_bytes, dtype=np.uint8), name='MODEL_STR')
    hdul.append(model_hdu)

    # Save special data structure as JSON
    if model_special_data_for_json:
        special_json_str = json.dumps(model_special_data_for_json, indent=2)
        special_json_bytes = special_json_str.encode('utf-8')
        special_hdu = fits.ImageHDU(np.frombuffer(special_json_bytes, dtype=np.uint8), name='MODEL_SPC')
        hdul.append(special_hdu)

    # Save the model parameters in a human-readable format
    model_param_names = []
    model_param_values = []
    model_param_fixed = []
    model_param_bounds_lower = []
    model_param_bounds_upper = []
    model_param_tied = []

    for param_name in dynesty_fit.model.param_names:
        param = getattr(dynesty_fit.model, param_name)
        model_param_names.append(param_name)
        model_param_values.append(param.value)
        model_param_fixed.append(param.fixed)

        lower, upper = param.bounds
        model_param_bounds_lower.append(lower if lower is not None else -np.inf)
        model_param_bounds_upper.append(upper if upper is not None else np.inf)

        if param.tied:
            model_param_tied.append(str(param.tied))
        else:
            model_param_tied.append('')

    model_param_names_array = np.array(model_param_names, dtype='U100')
    col_names = fits.Column(name='PARAM_NAME', format='A100', array=model_param_names_array)
    col_values = fits.Column(name='VALUE', format='D', array=model_param_values)
    col_fixed = fits.Column(name='FIXED', format='L', array=model_param_fixed)
    col_lower = fits.Column(name='BOUND_LOW', format='D', array=model_param_bounds_lower)
    col_upper = fits.Column(name='BOUND_UP', format='D', array=model_param_bounds_upper)
    col_tied = fits.Column(name='TIED', format='A100', array=np.array(model_param_tied, dtype='U100'))

    model_param_table = fits.BinTableHDU.from_columns(
        [col_names, col_values, col_fixed, col_lower, col_upper, col_tied],
        name='MODEL_PARAM'
    )
    hdul.append(model_param_table)

    # Save submodel information
    if hasattr(dynesty_fit.model, 'submodel_names'):
        submodel_names = np.array(dynesty_fit.model.submodel_names, dtype='U100')
        col_submodel = fits.Column(name='SUBMODEL_NAME', format='A100', array=submodel_names)
        submodel_table = fits.BinTableHDU.from_columns([col_submodel], name='SUBMODELS')
        hdul.append(submodel_table)
    else:
        submodel_names = np.array([], dtype='U100')
        col_submodel = fits.Column(name='SUBMODEL_NAME', format='A100', array=submodel_names)
        submodel_table = fits.BinTableHDU.from_columns([col_submodel], name='SUBMODELS')
        hdul.append(submodel_table)

    # Write to file
    hdul.writeto(filename, overwrite=True)
    print(f"Dynesty_Fit object saved to {filename}")


def load_dynesty_fit_from_fits(filename):
    """
    Load a FITS file and recover a Dynesty_Fit object.

    Parameters
    ----------
    filename : str
        The input FITS filename.

    Returns
    -------
    dynesty_fit : Dynesty_Fit
        The recovered Dynesty_Fit object.

    Notes
    -----
    If the data was saved with thinning (thin > 1), the loaded samples
    will be the thinned versions. The THIN keyword in the header
    records the thinning factor used.
    """
    from .dynesty_fit import Dynesty_Fit

    # Simple results container to mimic dynesty results
    class DynestyResults:
        """Container for dynesty results loaded from FITS."""
        pass

    # Open FITS file
    with fits.open(filename) as hdul:
        # Read basic information
        primary_hdr = hdul[0].header
        ndim = primary_hdr.get('NDIM', 0)
        nlive = primary_hdr.get('NLIVE', 500)
        sample_method = primary_hdr.get('SAMPLE_METHOD', 'rwalk')
        bound = primary_hdr.get('BOUND', 'multi')
        thin = primary_hdr.get('THIN', 1)
        log_evidence = primary_hdr.get('LOG_EVIDENCE', np.nan)
        log_evidence_err = primary_hdr.get('LOG_EVIDENCE_ERR', np.nan)

        if thin > 1:
            print(f"Note: Data was saved with thinning factor = {thin}")

        # Read data arrays
        wave_use = hdul['WAVE_USE'].data
        flux_use = hdul['FLUX_USE'].data
        ferr = hdul['FERR'].data

        theta_initial = None
        if 'THETA_INIT' in hdul:
            theta_initial = hdul['THETA_INIT'].data

        theta_best = None
        if 'THETA_BEST' in hdul:
            theta_best = hdul['THETA_BEST'].data

        # Read dynesty results
        results = None
        if 'SAMPLES' in hdul:
            results = DynestyResults()
            results.samples = hdul['SAMPLES'].data

            if 'LOGWT' in hdul:
                results.logwt = hdul['LOGWT'].data
            if 'LOGL' in hdul:
                results.logl = hdul['LOGL'].data
            if 'LOGVOL' in hdul:
                results.logvol = hdul['LOGVOL'].data
            if 'LOGZ' in hdul:
                results.logz = hdul['LOGZ'].data
            if 'LOGZERR' in hdul:
                results.logzerr = hdul['LOGZERR'].data
            if 'INFORMATION' in hdul:
                results.information = hdul['INFORMATION'].data

        # Read equal-weighted samples
        samples_equal_weight = None
        if 'SAMPLES_EQ' in hdul:
            samples_equal_weight = hdul['SAMPLES_EQ'].data

        # Read parameter names
        param_names = []
        if 'PARAM_NAMES' in hdul:
            param_names = list(hdul['PARAM_NAMES'].data['PARAM_NAME'])

        full_param_names = []
        if 'FULL_PARAM' in hdul:
            full_param_names = list(hdul['FULL_PARAM'].data['FULL_PARAM_NAME'])

        # Read parameter bounds
        param_bounds = None
        if 'PARAM_BOUNDS' in hdul:
            bounds_table = hdul['PARAM_BOUNDS'].data
            bounds_lower = bounds_table['BOUND_LOWER']
            bounds_upper = bounds_table['BOUND_UPPER']
            param_bounds = list(zip(bounds_lower, bounds_upper))

        # Read log-scale parameter indicators
        log_scale_params = set()
        if 'LOG_SCALE_PARAMS' in hdul:
            logscale_flags = hdul['LOG_SCALE_PARAMS'].data['LOG_SCALE']
            log_scale_params = {pn for pn, flag in zip(param_names, logscale_flags) if flag}

        # Reconstruct model from structure
        model_struct_data = hdul['MODEL_STR'].data
        model_json_bytes = model_struct_data.tobytes()
        model_json_str = model_json_bytes.decode('utf-8')
        model_nodes = json.loads(model_json_str)

        # Load special data if it exists
        model_special_data = {}
        if 'MODEL_SPC' in hdul:
            special_data = hdul['MODEL_SPC'].data
            special_json_bytes = special_data.tobytes()
            special_json_str = special_json_bytes.decode('utf-8')
            model_special_data_meta = json.loads(special_json_str)

            for node_id_str, special_dict in model_special_data_meta.items():
                node_id = int(node_id_str)
                model_special_data[node_id] = {}
                for key, value in special_dict.items():
                    if isinstance(value, dict) and value.get('_is_array'):
                        extname = value['extname']
                        if extname in hdul:
                            array_data = hdul[extname].data
                            original_shape = tuple(value['shape'])
                            if original_shape == ():
                                array_data = np.asarray(array_data).item()
                            model_special_data[node_id][key] = array_data
                    else:
                        model_special_data[node_id][key] = value

        # Read the parameter table
        param_value_dict = {}
        if 'MODEL_PARAM' in hdul:
            model_param_table = hdul['MODEL_PARAM'].data
            for i, param_name in enumerate(model_param_table['PARAM_NAME']):
                param_value_dict[param_name] = {
                    'value': float(model_param_table['VALUE'][i]),
                    'fixed': bool(model_param_table['FIXED'][i]),
                    'bounds': (
                        None if (np.isinf(model_param_table['BOUND_LOW'][i]) and model_param_table['BOUND_LOW'][i] < 0)
                        else float(model_param_table['BOUND_LOW'][i]),
                        None if (np.isinf(model_param_table['BOUND_UP'][i]) and model_param_table['BOUND_UP'][i] > 0)
                        else float(model_param_table['BOUND_UP'][i])
                    )
                }

        # Reconstruct the model
        model = deserialize_model_tree(model_nodes, model_special_data, param_value_dict)

    # Create a dummy Dynesty_Fit object
    dynesty_fit = Dynesty_Fit.__new__(Dynesty_Fit)

    # Manually set all attributes
    dynesty_fit.model = model
    dynesty_fit.wave_use = wave_use
    dynesty_fit.flux_use = flux_use
    dynesty_fit.ferr = ferr
    dynesty_fit.ndim = ndim
    dynesty_fit.nlive = nlive
    dynesty_fit.sample_method = sample_method
    dynesty_fit.bound = bound
    dynesty_fit.param_names = param_names
    dynesty_fit.full_param_names = full_param_names
    dynesty_fit.theta_initial = theta_initial
    dynesty_fit.theta_best = theta_best
    dynesty_fit.param_bounds = param_bounds
    dynesty_fit._log_scale_params = log_scale_params
    dynesty_fit.custom_priors = {}

    # Set up param_map
    if full_param_names and param_names:
        dynesty_fit.param_map = dict(zip(full_param_names,
                                        [(ii, pn) for ii, pn in enumerate(param_names)]))
    else:
        dynesty_fit.param_map = {}

    # Restore dynesty results
    dynesty_fit.results = results
    dynesty_fit.samples_equal_weight = samples_equal_weight
    dynesty_fit.log_evidence = log_evidence
    dynesty_fit.log_evidence_err = log_evidence_err

    print(f"Dynesty_Fit object loaded from {filename}")
    return dynesty_fit
