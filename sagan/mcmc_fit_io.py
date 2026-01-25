import numpy as np
from astropy.io import fits
from astropy.modeling import models, CompoundModel
from copy import deepcopy
import pickle
import operator
import importlib


__all__ = ['save_mcmc_fit_to_fits', 'load_mcmc_fit_from_fits']


# Operator mappings
_OP_TO_STR = {
    operator.add: "+",
    operator.mul: "*",
    operator.sub: "-",
    operator.truediv: "/",
    operator.pow: "**",
    "+": "+",
    "*": "*",
    "-": "-",
    "/": "/",
    "**": "**",
}

_STR_TO_OP = {
    "+": operator.add,
    "*": operator.mul,
    "-": operator.sub,
    "/": operator.truediv,
    "**": operator.pow,
}


def save_mcmc_fit_to_fits(mcmc_fit, filename):
    """
    Save an MCMC_Fit object entirely into a FITS file.
    
    Parameters
    ----------
    mcmc_fit : MCMC_Fit
        The MCMC_Fit object to save.
    filename : str
        The output FITS filename.
    """
    # Create HDU list
    hdul = fits.HDUList()
    
    # Primary HDU with basic information (only scalar values)
    primary_hdr = fits.Header()
    primary_hdr['NWALKERS'] = getattr(mcmc_fit, 'nwalkers', 0)
    primary_hdr['NSTEPS'] = getattr(mcmc_fit, 'nsteps', 0)
    primary_hdr['NBURN'] = getattr(mcmc_fit, 'nburn', 0)
    primary_hdr['NDIM'] = getattr(mcmc_fit, 'ndim', 0)
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)
    hdul.append(primary_hdu)
    
    # Save data arrays
    hdul.append(fits.ImageHDU(mcmc_fit.wave_use, name='WAVE_USE'))
    hdul.append(fits.ImageHDU(mcmc_fit.flux_use, name='FLUX_USE'))
    hdul.append(fits.ImageHDU(mcmc_fit.ferr, name='FERR'))
    
    if hasattr(mcmc_fit, 'theta_initial') and mcmc_fit.theta_initial is not None:
        hdul.append(fits.ImageHDU(mcmc_fit.theta_initial, name='THETA_INIT'))
    
    if hasattr(mcmc_fit, 'pos') and mcmc_fit.pos is not None:
        hdul.append(fits.ImageHDU(mcmc_fit.pos, name='POS'))
    
    if hasattr(mcmc_fit, 'theta_best') and mcmc_fit.theta_best is not None:
        hdul.append(fits.ImageHDU(mcmc_fit.theta_best, name='THETA_BEST'))
    
    # Save MCMC results if they exist
    if hasattr(mcmc_fit, 'flat_samples') and mcmc_fit.flat_samples is not None:
        hdul.append(fits.ImageHDU(mcmc_fit.flat_samples, name='FLAT_SAMP'))
    
    if hasattr(mcmc_fit, 'log_prob') and mcmc_fit.log_prob is not None:
        hdul.append(fits.ImageHDU(mcmc_fit.log_prob, name='LOG_PROB'))
    
    if hasattr(mcmc_fit, 'samples_initial') and mcmc_fit.samples_initial is not None:
        hdul.append(fits.ImageHDU(mcmc_fit.samples_initial, name='SAMP_INIT'))
    
    # Save parameter names as a table
    if hasattr(mcmc_fit, 'param_names') and mcmc_fit.param_names is not None:
        param_names_array = np.array(mcmc_fit.param_names, dtype='U100')
        col1 = fits.Column(name='PARAM_NAME', format='A100', array=param_names_array)
        param_table = fits.BinTableHDU.from_columns([col1], name='PARAM_NAMES')
        hdul.append(param_table)
    
    # Save full parameter names
    if hasattr(mcmc_fit, 'full_param_names') and mcmc_fit.full_param_names is not None:
        full_param_names_array = np.array(mcmc_fit.full_param_names, dtype='U100')
        col2 = fits.Column(name='FULL_PARAM_NAME', format='A100', array=full_param_names_array)
        full_param_table = fits.BinTableHDU.from_columns([col2], name='FULL_PARAM')
        hdul.append(full_param_table)
    
    # Save model information
    # Serialize model structure with all special data
    model_nodes, model_special_data = serialize_model_tree(mcmc_fit.model)
    
    # Save special data arrays for each model (like template arrays)
    # Create a version for pickling where arrays are replaced with metadata
    model_special_data_for_pickle = {}
    for node_id, special_data in model_special_data.items():
        model_special_data_for_pickle[node_id] = {}
        for key, value in special_data.items():
            if isinstance(value, np.ndarray):
                # Save array to FITS HDU with shorter names to avoid truncation
                extname = f'N{node_id}_{key}'.upper()[:8]  # FITS extension name limit
                # Ensure array is at least 1D
                value_to_save = np.atleast_1d(value)
                hdul.append(fits.ImageHDU(value_to_save, name=extname))
                # Store metadata instead of array
                model_special_data_for_pickle[node_id][key] = {
                    '_is_array': True,
                    'extname': extname,
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            else:
                # Not an array, can be pickled directly
                model_special_data_for_pickle[node_id][key] = value
    
    # Save model structure as pickle
    model_struct_pickle = pickle.dumps(model_nodes)
    model_hdu = fits.ImageHDU(np.frombuffer(model_struct_pickle, dtype=np.uint8), name='MODEL_STR')
    hdul.append(model_hdu)
    
    # Save special data structure as pickle (with array metadata)
    if model_special_data_for_pickle:
        special_pickle = pickle.dumps(model_special_data_for_pickle)
        special_hdu = fits.ImageHDU(np.frombuffer(special_pickle, dtype=np.uint8), name='MODEL_SPC')
        hdul.append(special_hdu)
    
    # Save the model parameters in a human-readable format
    model_param_names = []
    model_param_values = []
    model_param_fixed = []
    model_param_bounds_lower = []
    model_param_bounds_upper = []
    model_param_tied = []
    
    for param_name in mcmc_fit.model.param_names:
        param = getattr(mcmc_fit.model, param_name)
        model_param_names.append(param_name)
        model_param_values.append(param.value)
        model_param_fixed.append(param.fixed)
        
        # Handle bounds
        lower, upper = param.bounds
        model_param_bounds_lower.append(lower if lower is not None else -np.inf)
        model_param_bounds_upper.append(upper if upper is not None else np.inf)
        
        # Handle tied parameters (store as string representation)
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
    submodel_names = np.array(mcmc_fit.model.submodel_names, dtype='U100')
    col_submodel = fits.Column(name='SUBMODEL_NAME', format='A100', array=submodel_names)
    submodel_table = fits.BinTableHDU.from_columns([col_submodel], name='SUBMODELS')
    hdul.append(submodel_table)
    
    # Write to file
    hdul.writeto(filename, overwrite=True)
    print(f"MCMC_Fit object saved to {filename}")


def load_mcmc_fit_from_fits(filename):
    """
    Load a FITS file and recover an MCMC_Fit object.
    
    Parameters
    ----------
    filename : str
        The input FITS filename.
    
    Returns
    -------
    mcmc_fit : MCMC_Fit
        The recovered MCMC_Fit object.
    """
    from .mcmc_fit import MCMC_Fit
    
    # Open FITS file
    with fits.open(filename) as hdul:
        # Read basic information
        primary_hdr = hdul[0].header
        nwalkers = primary_hdr.get('NWALKERS', 50)
        nsteps = primary_hdr.get('NSTEPS', 1000)
        nburn = primary_hdr.get('NBURN', 0)
        ndim = primary_hdr.get('NDIM', 0)
        
        # Read data arrays
        wave_use = hdul['WAVE_USE'].data
        flux_use = hdul['FLUX_USE'].data
        ferr = hdul['FERR'].data
        
        theta_initial = None
        if 'THETA_INIT' in hdul:
            theta_initial = hdul['THETA_INIT'].data
        
        pos = None
        if 'POS' in hdul:
            pos = hdul['POS'].data
        
        theta_best = None
        if 'THETA_BEST' in hdul:
            theta_best = hdul['THETA_BEST'].data
        
        flat_samples = None
        if 'FLAT_SAMP' in hdul:
            flat_samples = hdul['FLAT_SAMP'].data
        
        log_prob = None
        if 'LOG_PROB' in hdul:
            log_prob = hdul['LOG_PROB'].data
        
        samples_initial = None
        if 'SAMP_INIT' in hdul:
            samples_initial = hdul['SAMP_INIT'].data
        
        # Read parameter names
        param_names = []
        if 'PARAM_NAMES' in hdul:
            param_names = list(hdul['PARAM_NAMES'].data['PARAM_NAME'])
        
        full_param_names = []
        if 'FULL_PARAM' in hdul:
            full_param_names = list(hdul['FULL_PARAM'].data['FULL_PARAM_NAME'])
        
        # Reconstruct model from structure
        model_struct_data = hdul['MODEL_STR'].data
        model_struct_bytes = model_struct_data.tobytes()
        model_nodes = pickle.loads(model_struct_bytes)
        
        # Load special data if it exists
        model_special_data = {}
        if 'MODEL_SPC' in hdul:
            special_data = hdul['MODEL_SPC'].data
            special_bytes = special_data.tobytes()
            model_special_data_meta = pickle.loads(special_bytes)
            
            # Load the actual arrays from HDUs
            for node_id, special_dict in model_special_data_meta.items():
                model_special_data[node_id] = {}
                for key, value in special_dict.items():
                    if isinstance(value, dict) and value.get('_is_array'):
                        # This is array metadata, load the actual array
                        extname = value['extname']
                        if extname in hdul:
                            array_data = hdul[extname].data
                            # Restore original shape if needed
                            original_shape = value['shape']
                            if original_shape == ():
                                # Was a scalar, convert back
                                array_data = np.asarray(array_data).item()
                            model_special_data[node_id][key] = array_data
                    else:
                        # Not an array, use directly
                        model_special_data[node_id][key] = value
        
        # Read the parameter table BEFORE reconstructing model
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
    
    # Create a dummy MCMC_Fit object
    mcmc_fit = MCMC_Fit.__new__(MCMC_Fit)
    
    # Manually set all attributes
    mcmc_fit.model = model
    mcmc_fit.wave_use = wave_use
    mcmc_fit.flux_use = flux_use
    mcmc_fit.ferr = ferr
    mcmc_fit.nwalkers = nwalkers
    mcmc_fit.nsteps = nsteps
    mcmc_fit.nburn = nburn
    mcmc_fit.ndim = ndim
    mcmc_fit.param_names = param_names
    mcmc_fit.full_param_names = full_param_names
    mcmc_fit.theta_initial = theta_initial
    mcmc_fit.pos = pos
    mcmc_fit.theta_best = theta_best
    
    # Set up param_map
    if full_param_names and param_names:
        mcmc_fit.param_map = dict(zip(full_param_names, 
                                       [(ii, pn) for ii, pn in enumerate(param_names)]))
    else:
        mcmc_fit.param_map = {}
    
    # Restore MCMC results
    if flat_samples is not None:
        mcmc_fit.flat_samples = flat_samples
    
    if log_prob is not None:
        mcmc_fit.log_prob = log_prob
    
    if samples_initial is not None:
        mcmc_fit.samples_initial = samples_initial
    
    # Set log_prior_func
    mcmc_fit.log_prior_func = mcmc_fit.log_prior
    
    print(f"MCMC_Fit object loaded from {filename}")
    return mcmc_fit


def _is_compound(m):
    """Check if a model is a compound model."""
    return hasattr(m, "left") and hasattr(m, "right") and hasattr(m, "op")


def _check_convolution(m):
    """
    Check if a model (compound or single) has convolution applied.
    Returns (has_convolution, convolution_info)
    """
    conv_tag = getattr(m, '_gaussconv1d_callswap', None)
    meta = getattr(m, 'meta', {}) or {}
    
    if conv_tag and isinstance(conv_tag, dict):
        info = {
            'conv_tag': True,
            'conv_sigx': float(conv_tag.get('sigma_x', 0))
        }
        if 'orig_cls' in conv_tag:
            info['orig_cls'] = conv_tag['orig_cls']
        
        # Also check meta for LSF info
        if meta.get('lsf_convolved'):
            info['lsf_conv'] = True
            if 'lsf_sigma_x' in meta:
                info['lsf_sigx'] = float(meta['lsf_sigma_x'])
            if 'lsf_info' in meta:
                lsf_info = meta['lsf_info']
                info['lsf_wavec'] = float(lsf_info.get('wavec', 0))
                info['lsf_R'] = float(lsf_info.get('R', 0))
        
        return True, info
    
    return False, {}


def serialize_model_tree(model, node_id=0):
    """
    Serialize model tree into nodes with special data handling.
    
    Returns
    -------
    nodes : list of dict
        Model structure information
    special_data : dict
        Dictionary mapping node_id to special initialization data
    """
    nodes = []
    special_data = {}
    counter = [node_id]  # Use list to allow modification in nested function
    
    def serialize_node(m, nid):
        counter[0] += 1
        
        # Check if this model (compound or single) has convolution
        has_conv, conv_info = _check_convolution(m)
        
        if _is_compound(m):
            # Compound model
            op_str = _OP_TO_STR.get(m.op, '+')
            
            # If this compound model has convolution, we need to unwrap it to get the underlying model
            # The underlying model's left/right are the actual compound model parts
            left_nodes, left_special = serialize_model_tree(m.left, counter[0])
            counter[0] = max([n['node_id'] for n in left_nodes]) if left_nodes else counter[0]
            
            right_nodes, right_special = serialize_model_tree(m.right, counter[0] + 1)
            counter[0] = max([n['node_id'] for n in right_nodes]) if right_nodes else counter[0]
            
            node = {
                'node_id': nid,
                'type': 'compound',
                'operator': op_str,
                'name': getattr(m, 'name', None),
                'left_id': left_nodes[0]['node_id'] if left_nodes else None,
                'right_id': right_nodes[0]['node_id'] if right_nodes else None,
            }
            
            # Save convolution info for compound model if present
            if has_conv:
                special_data[nid] = conv_info
                node['has_convolution'] = True
            else:
                node['has_convolution'] = False
            
            nodes.append(node)
            nodes.extend(left_nodes)
            nodes.extend(right_nodes)
            special_data.update(left_special)
            special_data.update(right_special)
            
        else:
            # Single model - check if it has convolution applied
            # Get the original class if convolution was applied
            if has_conv and 'orig_cls' in conv_info:
                orig_cls = conv_info['orig_cls']
                class_name = orig_cls.__name__
                class_module = orig_cls.__module__
            else:
                class_name = m.__class__.__name__
                class_module = m.__class__.__module__
            
            node = {
                'node_id': nid,
                'type': 'single',
                'class_name': class_name,
                'class_module': class_module,
                'name': getattr(m, 'name', None),
                'parameters': {},
                'has_convolution': has_conv
            }
            
            # Save parameters
            for param_name in m.param_names:
                param = getattr(m, param_name)
                node['parameters'][param_name] = {
                    'value': param.value,
                    'fixed': param.fixed,
                    'bounds': param.bounds,
                    'default': param.default
                }
            
            # Save special model-specific attributes
            special = extract_special_data(m, has_conv, conv_info)
            if special:
                special_data[nid] = special
                node['has_special_data'] = True
            else:
                node['has_special_data'] = False
            
            nodes.append(node)
        
        return nodes, special_data
    
    return serialize_node(model, node_id)


def extract_special_data(model, has_conv=False, conv_info=None):
    """
    Extract special initialization data from model.
    
    This handles models that need non-parameter data for initialization.
    Returns the actual arrays/data (not None placeholders).
    Uses short key names to avoid FITS extension name truncation issues.
    """
    special = {}
    
    # Add convolution info if present
    if has_conv and conv_info:
        special.update(conv_info)
    
    # Get the original class name for checking the model type
    # If convolution was applied, we need to check the original class
    if has_conv and conv_info and 'orig_cls' in conv_info:
        class_name = conv_info['orig_cls'].__name__
    else:
        class_name = model.__class__.__name__
    
    # Line_template: needs template_velc and template_flux
    # Use short names to avoid FITS extension truncation
    if class_name == 'Line_template':
        if hasattr(model, '_template_velc'):
            special['velc'] = np.asarray(model._template_velc)
        if hasattr(model, '_template_flux'):
            special['flux'] = np.asarray(model._template_flux)
        # Also save computed attributes
        if hasattr(model, '_vmin'):
            special['vmin'] = float(model._vmin)
        if hasattr(model, '_vmax'):
            special['vmax'] = float(model._vmax)
    
    # Line_MultiGauss: needs n_components
    if class_name in ['Line_MultiGauss', 'Line_MultiGauss_doublet']:
        if hasattr(model, 'n_components'):
            special['ncomp'] = model.n_components
    
    # StarSpectrum: needs templates and computed values
    if class_name == 'StarSpectrum':
        if hasattr(model, 'wave_temp'):
            special['wave'] = np.asarray(model.wave_temp)
        if hasattr(model, 'flux_temp'):
            special['flux'] = np.asarray(model.flux_temp)
        if hasattr(model, 'ln_lam'):
            special['lnlam'] = np.asarray(model.ln_lam)
    
    # Multi_StarSpectrum: needs velscale and Star_types
    if class_name == 'Multi_StarSpectrum':
        if hasattr(model, 'velscale'):
            special['velsc'] = model.velscale
        if hasattr(model, 'Star_types'):
            special['stypes'] = model.Star_types
    
    # IronTemplate: needs template arrays and computed values
    if class_name == 'IronTemplate':
        if hasattr(model, '_wave_temp'):
            special['wave'] = np.asarray(model._wave_temp)
        if hasattr(model, '_flux_temp'):
            special['flux'] = np.asarray(model._flux_temp)
        if hasattr(model, '_stddev_intr'):
            special['stdintr'] = float(model._stddev_intr)
        if hasattr(model, '_vchan'):
            special['vchan'] = float(model._vchan)
        if hasattr(model, '_vmin'):
            special['vmin'] = float(model._vmin)
        if hasattr(model, '_vmax'):
            special['vmax'] = float(model._vmax)
    
    return special


def deserialize_model_tree(nodes, special_data=None, param_values=None):
    """
    Reconstruct model from serialized nodes.
    
    Parameters
    ----------
    nodes : list
        Serialized node information
    special_data : dict
        Special initialization data for each node
    param_values : dict
        Parameter values from MODEL_PARAM table
    """
    if special_data is None:
        special_data = {}
    if param_values is None:
        param_values = {}
    
    # Build node dictionary
    node_dict = {node['node_id']: node for node in nodes}
    
    # Find root (first node or node with no parent reference)
    root_node = nodes[0]
    
    def build_model(node):
        if node['type'] == 'compound':
            # Reconstruct compound model
            left_node = node_dict[node['left_id']]
            right_node = node_dict[node['right_id']]
            
            left_model = build_model(left_node)
            right_model = build_model(right_node)
            
            op_str = node['operator']
            op = _STR_TO_OP.get(op_str, operator.add)
            
            model = op(left_model, right_model)
            if node['name']:
                model.name = node['name']
            
            # Apply convolution to compound model if it had it
            if node.get('has_convolution', False):
                node_special = special_data.get(node['node_id'], {})
                if node_special.get('lsf_conv') or node_special.get('conv_tag'):
                    model = apply_convolution(model, node_special)
            
            return model
        
        else:
            # Reconstruct single model
            return deserialize_single_model(node, special_data.get(node['node_id'], {}), param_values)
    
    return build_model(root_node)


def apply_convolution(model, conv_info):
    """
    Apply LSF convolution to a model based on saved convolution info.
    """
    try:
        from .convolution import convolve_lsf
        
        # Restore LSF convolution using the saved parameters
        if 'lsf_wavec' in conv_info and 'lsf_R' in conv_info:
            wavec = conv_info['lsf_wavec']
            resolving_power = conv_info['lsf_R']
            # Re-apply the convolution to the model
            model = convolve_lsf(model, wavec=wavec, resolving_power=resolving_power)
            return model
    except ImportError:
        print("Warning: Could not import convolve_lsf, skipping LSF convolution")
    except Exception as e:
        print(f"Warning: Failed to apply LSF convolution: {e}")
    
    return model


def deserialize_single_model(node, special_data=None, param_values=None):
    """
    Deserialize a single model with special data handling.
    
    Parameters
    ----------
    node : dict
        Node information
    special_data : dict
        Special initialization data
    param_values : dict
        Global parameter values to apply after initialization
    """
    from scipy.interpolate import interp1d
    
    if special_data is None:
        special_data = {}
    if param_values is None:
        param_values = {}
    
    # Import the model class (should be the original class now, not the convolved one)
    module = importlib.import_module(node['class_module'])
    ModelClass = getattr(module, node['class_name'])
    
    class_name = node['class_name']
    
    # Prepare initialization parameters based on model type
    # For models that need special data, DON'T pass parameter values during init
    init_params = {}
    
    # Handle special initialization requirements
    if class_name == 'Line_template':
        # Requires template_velc and template_flux
        # DO NOT pass parameter values here, only template data
        if 'velc' in special_data and 'flux' in special_data:
            init_params['template_velc'] = special_data['velc']
            init_params['template_flux'] = special_data['flux']
        else:
            raise ValueError("Line_template requires template velocity and flux data")
        # Add name if present
        if node['name']:
            init_params['name'] = node['name']
    
    elif class_name in ['Line_MultiGauss', 'Line_MultiGauss_doublet']:
        # Requires n_components
        if 'ncomp' in special_data:
            init_params['n_components'] = special_data['ncomp']
        # For these models, add name
        if node['name']:
            init_params['name'] = node['name']
    
    elif class_name == 'StarSpectrum':
        # May have specific velscale
        if node['name']:
            init_params['name'] = node['name']
    
    elif class_name == 'Multi_StarSpectrum':
        if 'velsc' in special_data:
            init_params['velscale'] = special_data['velsc']
        if 'stypes' in special_data:
            init_params['Star_types'] = special_data['stypes']
        if node['name']:
            init_params['name'] = node['name']
    
    else:
        # For standard astropy models, can pass parameter values during init
        for param_name, param_dict in node['parameters'].items():
            if param_name in param_values:
                init_params[param_name] = param_values[param_name]['value']
            else:
                init_params[param_name] = param_dict['value']
        if node['name']:
            init_params['name'] = node['name']
    
    # Try to create the model
    try:
        model = ModelClass(**init_params)
    except TypeError as e:
        print(f"Warning: Could not initialize {class_name} with params: {list(init_params.keys())}")
        print(f"Error: {e}")
        raise ValueError(f"Cannot initialize {class_name}. Missing required initialization data.")
    
    # NOW set all parameter values from param_values (after model creation)
    for param_name in model.param_names:
        if param_name in param_values:
            param = getattr(model, param_name)
            param.value = param_values[param_name]['value']
            param.fixed = param_values[param_name]['fixed']
            param.bounds = param_values[param_name]['bounds']
        elif param_name in node['parameters']:
            # Fallback to node parameters if not in param_values
            param = getattr(model, param_name)
            param.value = node['parameters'][param_name]['value']
            param.fixed = bool(node['parameters'][param_name]['fixed'])
            param.bounds = node['parameters'][param_name]['bounds']
    
    # Restore/recreate computed private attributes
    if class_name == 'Line_template':
        # Verify the interp1d object was created properly during __init__
        # If not, recreate it explicitly
        if not hasattr(model, '_model') or model._model is None:
            if hasattr(model, '_template_velc') and hasattr(model, '_template_flux'):
                model._model = interp1d(model._template_velc, model._template_flux)
        # Restore vmin/vmax if they exist in special_data
        if 'vmin' in special_data:
            model._vmin = special_data['vmin']
        if 'vmax' in special_data:
            model._vmax = special_data['vmax']
    
    elif class_name == 'StarSpectrum':
        if 'lnlam' in special_data:
            model.ln_lam = special_data['lnlam']
        if 'wave' in special_data:
            model.wave_temp = special_data['wave']
        if 'flux' in special_data:
            model.flux_temp = special_data['flux']
    
    elif class_name == 'IronTemplate':
        if 'vchan' in special_data:
            model._vchan = special_data['vchan']
        if 'vmin' in special_data:
            model._vmin = special_data['vmin']
        if 'vmax' in special_data:
            model._vmax = special_data['vmax']
        if 'stdintr' in special_data:
            model._stddev_intr = special_data['stdintr']
        if 'wave' in special_data:
            model._wave_temp = special_data['wave']
        if 'flux' in special_data:
            model._flux_temp = special_data['flux']
    
    # Apply convolution to single model if it had it
    if node.get('has_convolution', False):
        if special_data.get('lsf_conv') or special_data.get('conv_tag'):
            model = apply_convolution(model, special_data)
    
    return model