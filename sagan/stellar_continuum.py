import os
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from .utils import splitter, package_path
from .constants import ls_km


__all__ = ['stellar_11Gyr', 'stellar_300Myr']


spec_11gyr = np.loadtxt('{0}{1}data{1}sed_bc03_11Gyr.dat'.format(package_path, splitter))
wave_11gyr = spec_11gyr[:, 0]
flux_11gyr = spec_11gyr[:, 1]
logw_11gyr = np.log(wave_11gyr)
logw_even_11gyr = np.linspace(logw_11gyr[0], logw_11gyr[-1], 5*len(logw_11gyr))
logf_even_11gyr = np.interp(logw_even_11gyr, logw_11gyr, np.log(flux_11gyr))
wave_even_11gyr = np.exp(logw_even_11gyr)
flux_even_11gyr = np.exp(logf_even_11gyr)
flux_even_11gyr /= np.max(flux_even_11gyr)

spec_300myr = np.loadtxt('{0}{1}data{1}sed_bc03_300Myr.dat'.format(package_path, splitter))
wave_300myr = spec_300myr[:, 0]
flux_300myr = spec_300myr[:, 1]
logw_300myr = np.log(wave_300myr)
logw_even_300myr = np.linspace(logw_300myr[0], logw_300myr[-1], 5*len(logw_300myr))
logf_even_300myr = np.interp(logw_even_300myr, logw_300myr, np.log(flux_300myr))
wave_even_300myr = np.exp(logw_even_300myr)
flux_even_300myr = np.exp(logf_even_300myr)
flux_even_300myr /= np.max(flux_even_300myr)


class stellar_11Gyr(Fittable1DModel):
    '''
    The stellar continuum with 11 Gyr single stellar population.

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    fmax : float
        The maximum flux of the stellar continuum.
    sigma : float
        The velocity dispersion, units: km s^-1.
    z : float
        The redshift.

    Returns
    -------
    flux : array like
        The SED flux of the stellar continuum, units: unit: per Angstrom.
    '''
    fmax = Parameter(default=1, bounds=(0, None))
    sigma = Parameter(default=200, bounds=(20, 10000))
    z = Parameter(default=0, bounds=(0, 10))

    @staticmethod
    def evaluate(x, fmax, sigma, z):
        '''
        Stellar (11 Gyr) model function.
        '''
        x = x / (1 + z)
        assert (np.min(x) >= wave_even_11gyr[0]) & (np.max(x) <= wave_even_11gyr[-1]), \
               'The wavelength is out of the supported range ({0:.0f}-{1:.0f})!'.format(wave_even_11gyr[0], wave_even_11gyr[-1])
        s = sigma / ls_km
        nsig = s / (logw_even_11gyr[1] - logw_even_11gyr[0])
        flux_conv = gaussian_filter(flux_even_11gyr, nsig)
        f = np.interp(x, wave_even_11gyr, fmax * flux_conv)

        return f


class stellar_300Myr(Fittable1DModel):
    '''
    The stellar continuum with 11 Gyr single stellar population.

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    fmax : float
        The maximum flux of the stellar continuum.
    sigma : float
        The velocity dispersion, units: km s^-1.
    z : float
        The redshift.

    Returns
    -------
    flux : array like
        The SED flux of the stellar continuum, units: unit: per Angstrom.
    '''
    fmax = Parameter(default=1, bounds=(0, None))
    sigma = Parameter(default=200, bounds=(20, 10000))
    z = Parameter(default=0, bounds=(0, 10))

    @staticmethod
    def evaluate(x, fmax, sigma, z):
        '''
        Stellar (300 Myr) model function.
        '''
        x = x / (1 + z)
        assert (np.min(x) >= wave_even_300myr[0]) & (np.max(x) <= wave_even_300myr[-1]), \
               'The wavelength is out of the supported range ({0:.0f}-{1:.0f})!'.format(wave_even_300myr[0], wave_even_300myr[-1])
        s = sigma / ls_km
        nsig = s / (logw_even_300myr[1] - logw_even_300myr[0])
        flux_conv = gaussian_filter(flux_even_300myr, nsig)
        f = np.interp(x, wave_even_300myr, fmax * flux_conv)

        return f