import numpy as np
from .constants import ls_km

__all__ = ['line_emission_flux', 'line_absorption_ew', 'line_fwhm']

def line_emission_flux(model, model_names, wave_obs):
    '''
    Calculate the integrated flux of a line model.
    '''
    flux_list = []
    for model_name in model_names:
        flux_list.append(model[model_name](wave_obs))
    flux = np.sum(flux_list, axis=0)
    int_flux = np.trapz(flux, wave_obs)
    return int_flux

def line_absorption_ew(model, model_names, wave_rest):
    '''
    Calculate the equivalent width of an absorption line model.
    '''
    flux_list = []
    for model_name in model_names:
        flux_list.append(model[model_name](wave_rest))
    flux = np.sum(flux_list, axis=0)
    ew = np.trapz((1 - flux), wave_rest)
    return ew

def line_fwhm(model, model_names, wave_obs, wavec):
    '''
    Calculate the FWHM of a line model.
    '''
    flux_list = []
    for model_name in model_names:
        flux_list.append(model[model_name](wave_obs))
    flux = np.sum(flux_list, axis=0)
    half_max = np.max(flux) / 2
    fltr = flux >= half_max
    wave_fwhm = wave_obs[fltr]
    fwhm = (wave_fwhm.max() - wave_fwhm.min()) / wavec * ls_km
    return fwhm
