import numpy as np
from .constants import ls_km

__all__ = ['line_emission_flux', 'line_absorption_ew', 'line_emission_fwhm', 'cont_flux', 'line_emission_ew']

def line_emission_flux(model, model_names, wave_obs, wave_range=None):
    '''
    Calculate the integrated flux of a line model.
    '''
    flux_list = []
    for model_name in model_names:
        flux_list.append(model[model_name](wave_obs))
    flux = np.sum(flux_list, axis=0)

    if wave_range is not None:
        fltr = (wave_obs >= wave_range[0]) & (wave_obs <= wave_range[1])
        wave_obs = wave_obs[fltr]
        flux = flux[fltr]

    int_flux = np.trapz(flux, wave_obs)
    return int_flux

def line_emission_fwhm(model, model_names, wave_rest, wavec):
    '''
    Calculate the FWHM of a line model.
    '''
    flux_list = []
    for model_name in model_names:
        flux_list.append(model[model_name](wave_rest))
    flux = np.sum(flux_list, axis=0)
    half_max = np.max(flux) / 2
    fltr = flux >= half_max
    wave_fwhm = wave_rest[fltr]
    fwhm = (wave_fwhm.max() - wave_fwhm.min()) / wavec * ls_km
    return fwhm

def line_emission_ew(model, line_model_names, cont_model_names, wave_rest, wavec=None, wave_range=None):
    '''
    Calculate the equivalent width of an emission line model.

    Parameters
    ----------
    model : dict
        A dictionary containing the line and continuum models as callables.
    line_model_names : list of str
        List of keys in the model dict corresponding to line models.
    cont_model_names : list of str
        List of keys in the model dict corresponding to continuum models.
    wave_rest : array like
        Wavelength array in the rest frame.
    wavec : float, optional
        Central wavelength. If not provided, the mean of wave_range will be used.
    wave_range : tuple of float, optional
        Wavelength range (min, max) to calculate the equivalent width. If wavec is not provided,
        this must be provided.
    
    Returns
    -------
    ew : float
        Equivalent width of the emission line.

    Notes
    -----
    For the EW of Fe II peudocontinuum, the wave_range should be 4434 and 4684 Angstroms.
    '''
    flux_list = []
    for model_name in line_model_names:
        flux_list.append(model[model_name](wave_rest))
    line_flux = np.sum(flux_list, axis=0)

    if wave_range is not None:
        fltr = (wave_rest >= wave_range[0]) & (wave_rest <= wave_range[1])
        wave_rest = wave_rest[fltr]
        line_flux = line_flux[fltr]

    int_line_flux = np.trapz(line_flux, wave_rest)

    cont_flux_list = []
    for model_name in cont_model_names:
        cont_flux_list.append(model[model_name](wave_rest))
    cont_flux = np.sum(cont_flux_list, axis=0)

    if wavec is None:
        assert wave_range is not None, "Either wavec or wave_range must be provided."
        fltr = (wave_rest >= wave_range[0]) & (wave_rest <= wave_range[1])
        wavec = np.mean(wave_rest[fltr])

    cont_flux_x0 = np.interp(wavec, wave_rest, cont_flux)
    ew = int_line_flux / cont_flux_x0
    return ew

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

def cont_flux(model, model_names, wave_rest, x0):
    '''
    Calculate the continuum flux at a given wavelength.
    '''
    flux_list = []
    for model_name in model_names:
        flux_list.append(model[model_name](wave_rest))
    flux = np.sum(flux_list, axis=0)
    flux_x0 = np.interp(x0, wave_rest, flux)
    return flux_x0
