import os
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
from .constants import ls_km
from scipy.ndimage import gaussian_filter1d
from spectres import spectres

__all__ = ['package_path', 'splitter', 'line_wave_dict', 'line_label_dict',
           'wave_to_velocity', 'velocity_to_wave', 
           'down_spectres']


if platform == "linux" or platform == "linux2":  # Linux
    splitter = '/'
elif platform == "darwin":  # Mac
    splitter = '/'
elif platform == "win32":  # Windows
    splitter = '\\'

pathList = os.path.abspath(__file__).split(splitter)
package_path = splitter.join(pathList[:-1])

# Wavelengths of the typical emission lines
# Based on http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
line_wave_dict = {
    'Halpha': 6562.819,
    'Hbeta': 4861.333,
    'Hgamma': 4340.471,
    'OIII_4959': 4958.911,
    'OIII_5007': 5006.843,
    'SII_6718': 6716.440,
    'SII_6733': 6730.810,
    'NII_6548': 6548.050,
    'NII_6583': 6583.460,
    'HeII_4686': 4685.710,
    'OIII_4363': 4363.210,
    'HeI_4471': 4471.479,
    'HeI_4713': 4713,
    'HeI_4922': 4922,
    'HeI_5016': 5016,
    'NI_5199': 5199,
    'NI_5201': 5201,
    'FeVI_5176': 5176,
    }

# Labels of the typical emission lines
line_label_dict = {
    'Halpha': r'H$\alpha$',
    'Hbeta': r'H$\beta$',
    'Hgamma': r'H$\gamma$',
    'OIII_4959': r'[O III] 4959',
    'OIII_5007': r'[O III] 5007',
    'SII_6718': r'[S II] 6718',
    'SII_6733': r'[S II] 6733',
    'NII_6548': r'[N II] 6548',
    'NII_6583': r'[N II] 6583',
    'HeII_4686': r'He II 4686',
    'HeI_4471': r'He I 4471',
    'HeI_4713': r'He I 4713',
    'HeI_4922': r'He I 4922',
    'HeI_5016': r'He I 5016',
    'NI_5199': r'NI 5199',
    'NI_5201': r'NI 5201',
    'FeVI_5176': r'Fe VI 5176',
    'OIII_4363': r'[O III] 4363',
}


def wave_to_velocity(wave, wave0):
    '''
    Convert wavelength to velocity.

    Parameters
    ----------
    wave : array like
        Wavelength.
    wave0 : float
        Reference wavelength.

    Returns
    -------
    velocity : array like
        Velocity.
    '''
    return (wave - wave0) / wave0 * ls_km


def velocity_to_wave(velocity, wave0):
    '''
    Convert velocity to wavelength.

    Parameters
    ----------
    velocity : array like
        Velocity.
    wave0 : float
        Reference wavelength.

    Returns
    -------
    wave : array like
        Wavelength.
    '''
    return (velocity / ls_km + 1 )* wave0


def down_spectres(wave, flux, R_org, R_new, wave_new=None, wavec=None, dw=None, 
                  spec_errs=None, fill=None, verbose=False):
    '''
    Degrade the spectral resolution of a spectrum according to the resolving power.

    Parameters
    ----------
    wave, flux : 1d arrays
        Wavelength and flux of the spectrum.
    R_org : float
        Original resolving power.
    R_new : float
        New resolving power.
    wave_new : 1d array, optional
        New wavelength grid.
    wavec : float, optional
        Central wavelength. If not provided, the mean of the input wavelength 
        will be used.
    dw : float, optional
        Wavelength dispersion. If not provided, the median of the input 
        wavelength dispersion will be used.
    spec_errs : 1d array, optional
        Spectrum errors as an input for SpectRes.spectres. It is only used 
        when wave_new is provided.
    fill : float, optional
        Fill value for the output spectrum. It is only used when wave_new is 
        provided.
    verbose : bool, optional
        Print the parameters for SpectRes.spectres.
    '''
    if wavec is None:
        wavec = np.mean(wave)

    if dw is None:
        dw = np.median(np.diff(wave))

    sigma = wavec * np.sqrt(R_new**-2 - R_org**-2) / 2.3548 / dw
    flux_new = gaussian_filter1d(flux, sigma)

    if wave_new is not None:
        flux_new = spectres(wave_new, wave, flux_new, spec_errs=spec_errs, fill=fill, verbose=verbose)

    return flux_new
