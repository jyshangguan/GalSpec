import os
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.io import fits
from scipy.interpolate import interp1d
from .utils import splitter, package_path
from .constants import ls_km
import pandas as pd

#__all__ = ['stellar_11Gyr', 'stellar_300Myr', 'cstar_KpA']
__all__ = ['StarSpectrum','Multi_StarSpectrum']

star_data_cache = {}
star_velscale_cache = {}

star_list = {
            'A': 'HD_97633.txt',
            'F': 'HD_89254.txt',
            'G': 'HD_140027.txt',
            'K': 'HD_49520.txt',
            'M': 'HD_44478.txt'
            }

def cache_star_data():
    
    for star_type, file_name in star_list.items():
        spec = pd.read_csv(f'{package_path}{splitter}data{splitter}{file_name}', sep='\s+', comment='#', header=None, names=['lam', 'Flux'])
        lam  = np.array(spec['lam'])
        flux = np.array(spec['Flux'])
        #ins = (lam>3900)&(lam<7000)
        #star_data_cache[star_type] = (lam[ins], flux[ins])
        star_data_cache[star_type] = (lam, flux)
        
cache_star_data()

def get_star(star_type, velscale):
    '''
    Retrieves and processes the star data for the specified star type.

    Parameters
    ----------
    star_type : str
        The type of star ('A', 'F', 'G', 'K', or 'M').
    velscale: 
        The desired velocity scale for the re-binned spectrum.
    Returns
    -------
    tuple
        The processed star data: original wavelength, flux, rebinned wavelength, and normalized flux.
    '''
    if star_type in star_velscale_cache and velscale in star_velscale_cache[star_type]:
        return star_velscale_cache[star_type][velscale]
    
    lam, flux = star_data_cache[star_type]
    
    # Rebin the spectrum to the desired velocity scale
    flux_rebin, ln_lam_temp = log_rebin(lam, flux, velscale=velscale)[:2]
    lam_rebin = np.exp(ln_lam_temp)
    #lam_rebin, flux_rebin = lam, flux
    # Normalize the rebinned flux
    f_rebin_norm = flux_rebin / np.max(flux_rebin)
    
    # Cache the result for this star_type and velscale
    if star_type not in star_velscale_cache:
        star_velscale_cache[star_type] = {}
    star_velscale_cache[star_type][velscale] = (lam, flux, lam_rebin, f_rebin_norm)
    
    return lam, flux, lam_rebin, f_rebin_norm
    

def log_rebin(lam, spec, velscale, oversample=1, flux=False):
    '''
    Logarithmically re-bins the spectrum to match the desired velocity scale.

    Parameters
    ----------
    lam : np.ndarray
        The wavelength array.
    spec : np.ndarray
        The flux or spectrum array.
    velscale : float
        The desired velocity scale for the re-binned spectrum. If None, it is calculated.
    oversample : int, optional
        Oversampling factor for the re-binning.
    flux : bool, optional
        If True, return flux. If False, return the wavelength.

    Returns
    -------
    tuple
        The rebinned spectrum, logarithmic wavelength array, and velocity scale.
    '''
    
    lam, spec = np.asarray(lam, dtype=float), np.asarray(spec, dtype=float)
    assert np.all(np.diff(lam) > 0), '`lam` must be monotonically increasing'
    n = len(spec)
    assert lam.size in [2, n], "`lam` must be either a 2-elements range or a vector with the length of `spec`"

    if lam.size == 2:
        dlam = np.diff(lam)/(n - 1)             # Assume constant dlam
        lim = lam + [-0.5, 0.5]*dlam
        borders = np.linspace(*lim, n + 1)
    else:
        lim = 1.5*lam[[0, -1]] - 0.5*lam[[1, -2]]
        borders = np.hstack([lim[0], (lam[1:] + lam[:-1])/2, lim[1]])
        dlam = np.diff(borders)

    ln_lim = np.log(lim)
    c = 299792.458                          # Speed of light in km/s

    if velscale is None:
        m = int(n*oversample)               # Number of output elements
        velscale = c*np.diff(ln_lim)/m      # Only for output (eq. 8 of Cappellari 2017, MNRAS)
        velscale = velscale.item()          # Make velscale a scalar
    else:
        ln_scale = velscale/c
        m = int(np.diff(ln_lim)/ln_scale)   # Number of output pixels

    newBorders = np.exp(ln_lim[0] + velscale/c*np.arange(m + 1))

    if lam.size == 2:
        k = ((newBorders - lim[0])/dlam).clip(0, n-1).astype(int)
    else:
        k = (np.searchsorted(borders, newBorders) - 1).clip(0, n-1)

    specNew = np.add.reduceat((spec.T*dlam).T, k)[:-1]    # Do analytic integral of step function
    specNew.T[...] *= np.diff(k) > 0                      # fix for design flaw of reduceat()
    specNew.T[...] += np.diff(((newBorders - borders[k]))*spec[k].T)    # Add to 1st dimension

    if not flux:
        specNew.T[...] /= np.diff(newBorders)   # Divide 1st dimension

    # Output np.log(wavelength): natural log of geometric mean
    ln_lam = 0.5*np.log(newBorders[1:]*newBorders[:-1])

    return specNew, ln_lam

class StarSpectrum(Fittable1DModel):
    """
    A class to represent a star spectrum model that can be fitted to data.

    Parameters
    ----------
    amplitude : float
        Amplitude of the star, units: arbitrary. Default is 1.
    sigma : float
        Velocity dispersion of the star, units: km/s. Default is 200.
    velscale : float
        The desired velocity scale for the rebinned spectrum. Default is 69.
    Star_type : str
        Type of the star ('A', 'F', 'G', 'K', 'M'). Default is 'A'.
    """
    
    amplitude = Parameter(default=1, bounds=(0, None))
    sigma = Parameter(default=200, bounds=(20, 6000))
    delta_z = Parameter(default=0, bounds=(-0.001, 0.001), fixed=False)
    
    def __init__(self, amplitude=amplitude, sigma=sigma, velscale=None, delta_z=delta_z, Star_type='A', **kwargs):
        """
        Initializes the StarSpectrum model with the given parameters.

        Parameters
        ----------
        amplitude : float, optional
            Amplitude of the star, default is 1.
        sigma : float, optional
            Velocity dispersion, default is 200.
        velscale : float, optional
            Velocity scale for the rebinned spectrum, default is 69.
        star_type : str, optional
            Type of star to load data for, default is 'A'.
        """
        
        super().__init__(amplitude=amplitude, sigma=sigma, **kwargs)
        
        Star_x, Star_y, Star_x_rebin, Star_y_rebin_norm = get_star(Star_type, velscale)
        
        self.wave_temp = Star_x_rebin
        self.flux_temp = Star_y_rebin_norm
        self.ln_lam = np.log(Star_x_rebin)
        
    def evaluate(self, x, amplitude, sigma, delta_z):
        '''
        Stellar model function.
        '''
        
        #if np.min(x) < self.wave_temp[0] or np.max(x) > self.wave_temp[-1]:
        #    raise ValueError(f'The wavelength is out of the supported range ({model_x_rebin[0]:.0f}-{model_x_rebin[-1]:.0f})!')
        
        s = sigma / ls_km
        
        nsig = s / (self.ln_lam[1] - self.ln_lam[0])
        nsig = max(nsig, 1e-6)
        
        ins     = (self.wave_temp>=np.min(x)-10)&(self.wave_temp<=np.max(x)+10)
        model_x = self.wave_temp[ins]/(1+delta_z)
        model_y = self.flux_temp[ins]
        
        flux_convolved = np.interp(x, model_x, gaussian_filter(model_y, nsig))
        #flux_convolved = np.interp(x, self.wave_temp, gaussian_filter(self.flux_temp, nsig))
        #flux_convolved = interp1d(self.wave_temp, gaussian_filter(self.flux_temp, nsig), fill_value="extrapolate")(x)
        
        return amplitude * flux_convolved


class Multi_StarSpectrum(Fittable1DModel):
    
    amp_0 = Parameter(default=1, bounds=(0, None))
    amp_1 = Parameter(default=1, bounds=(0, None))
    amp_2 = Parameter(default=1, bounds=(0, None))
    amp_3 = Parameter(default=1, bounds=(0, None))
    amp_4 = Parameter(default=1, bounds=(0, None))
    sigma = Parameter(default=200, bounds=(20, 6000))
    
    def __init__(self, amp_0=amp_0, amp_1=amp_1, amp_2=amp_2, amp_3=amp_3, amp_4=amp_4, sigma=sigma, velscale=69, Star_types=['A', 'F', 'G', 'K', 'M'], **kwargs):
        """
        Initializes the StarSpectrum model with the given parameters.

        Parameters
        ----------
        amplitude : float, optional
            Amplitude of the star, default is 1.
        sigma : float, optional
            Velocity dispersion, default is 200.
        velscale : float, optional
            Velocity scale for the rebinned spectrum, default is 69.
        star_type : str, optional
            Type of star to load data for, default is 'A'.
        """
        
        super().__init__(amp_0=amp_0, amp_1=amp_1, amp_2=amp_2, amp_3=amp_3, amp_4=amp_4, sigma=sigma, **kwargs)
        
        self.velscale = velscale
        self.Star_types = Star_types
        # Pre-calculate Star_x_rebin and Star_y_rebin_norm for all Star_types
        self.star_data = {star_type: get_star(star_type, velscale) for star_type in Star_types}
        
    def evaluate(self, x, amp_0, amp_1, amp_2, amp_3, amp_4, sigma):
        '''
        Stellar model function.
        '''
        
        s = sigma / ls_km
        
        ln_lam = np.log(self.star_data[self.Star_types[0]][2])
        nsig = s / (ln_lam[1] - ln_lam[0])
        nsig = max(nsig, 1e-6)
        
        flux_convolved = amp_0 * interp1d(self.star_data[self.Star_types[0]][2], 
                                          gaussian_filter(self.star_data[self.Star_types[0]][3], nsig))(x)+\
                         amp_1 * interp1d(self.star_data[self.Star_types[1]][2], 
                                          gaussian_filter(self.star_data[self.Star_types[1]][3], nsig))(x)+\
                         amp_2 * interp1d(self.star_data[self.Star_types[2]][2], 
                                          gaussian_filter(self.star_data[self.Star_types[2]][3], nsig))(x)+\
                         amp_3 * interp1d(self.star_data[self.Star_types[3]][2], 
                                          gaussian_filter(self.star_data[self.Star_types[3]][3], nsig))(x)+\
                         amp_4 * interp1d(self.star_data[self.Star_types[4]][2], 
                                          gaussian_filter(self.star_data[self.Star_types[4]][3], nsig))(x)
        return flux_convolved


# Deprecated

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