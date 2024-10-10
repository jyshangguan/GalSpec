import os
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.io import fits
from scipy.interpolate import interp1d
from .utils import splitter, package_path
from .constants import ls_km


__all__ = ['stellar_11Gyr', 'stellar_300Myr', 'cstar_KpA']


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

hdul_K0III = fits.open(f'{package_path}{splitter}data{splitter}K0III_HD002952.fits')
hdul_A0V = fits.open(f'{package_path}{splitter}data{splitter}A0V_HD065900.fits')
wave_K0III = hdul_K0III[1].data['WAVE'][0, :]
flux_K0III = hdul_K0III[1].data['FLUX'][0, :]
flux_A0V = hdul_A0V[1].data['FLUX'][0, :]
logw_star = np.log(wave_K0III)
logw_even_kpa = np.linspace(logw_star[0], logw_star[-1], 5*len(logw_star))
logf_star_K0III = np.interp(logw_even_kpa, logw_star, np.log(flux_K0III))
logf_star_A0V = np.interp(logw_even_kpa, logw_star, np.log(flux_A0V))
wave_even_kpa = np.exp(logw_even_kpa)
flux_even_K0III = np.exp(logf_star_K0III)
flux_even_A0V = np.exp(logf_star_A0V)

class cstar_KpA(Fittable1DModel):
    '''
    The stellar continuum with a K star plus an A star.

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    fnorm : float
        The normalization flux of the stellar continuum.
    frac : float
        The fraction of the K star flux at the normalization wavelength.
    sigma : float
        The velocity dispersion, units: km s^-1.
    z : float
        The redshift.
    x0 : float
        The normalization wavelength, units: Angstrom.

    Returns
    -------
    flux : array like
        The SED flux of the stellar continuum, units: unit: per Angstrom.
    '''
    fnorm = Parameter(default=1, bounds=(0, None))
    frac = Parameter(default=0.9, bounds=(0, 1))
    sigma = Parameter(default=1, bounds=(1, 500))
    z = Parameter(default=0, bounds=(0, 10))
    x0 = Parameter(default=5000, bounds=(0, None), fixed=True)

    @staticmethod
    def evaluate(x, fnorm, frac, sigma, z, x0):
        '''
        Stellar (KpA) model function.
        '''
        x = x / (1 + z)
        assert (np.min(x) >= wave_even_kpa[0]) & (np.max(x) <= wave_even_kpa[-1]), \
               'The wavelength is out of the supported range ({0:.0f}-{1:.0f})!'.format(wave_even_kpa[0], wave_even_kpa[-1])

        s = sigma / ls_km
        nsig = s / (logw_even_kpa[1] - logw_even_kpa[0])
        flux_interp_K0III = interp1d(wave_even_kpa, gaussian_filter(flux_even_K0III, nsig))
        flux_interp_A0V = interp1d(wave_even_kpa, gaussian_filter(flux_even_A0V, nsig))
        
        fnorm_k = fnorm * frac
        fnorm_a = fnorm * (1 - frac)
        f_k = flux_interp_K0III(x) / flux_interp_K0III(x0) * fnorm_k
        f_a = flux_interp_A0V(x) / flux_interp_A0V(x0) * fnorm_a
        f = f_k + f_a

        return f


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