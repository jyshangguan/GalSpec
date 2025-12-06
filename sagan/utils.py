import os
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
from .constants import ls_km
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from astropy.modeling import CompoundModel

try: 
    from sfdmap2 import sfdmap
except ImportError:
    sfdmap = None

try: 
    from PyAstronomy import pyasl
except ImportError:
    pyasl = None

try:
    from spectres import spectres
except ImportError:
    spectres = None

try:
    import emcee
except ImportError:
    emcee = None

try:
    import corner
except ImportError:
    corner = None

__all__ = ['package_path', 'splitter', 'line_wave_dict', 'line_label_dict',
           'wave_to_velocity', 'velocity_to_wave', 'down_spectres', 
           'ReadSpectrum']


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
    'Hdelta': 4101.742,
    'OIII_4959': 4958.911,
    'OIII_5007': 5006.843,
    'SII_6716': 6716.440,
    'SII_6731': 6730.810,
    'NII_6548': 6548.050,
    'NII_6583': 6583.460,
    'HeII_4686': 4685.710,
    'OIII_4363': 4363.210,
    'HeI_4471': 4471.479,
    'OII_3726': 3726.032,
    'OII_3728': 3728.815,
    'OI_6300': 6300.304,
    'ArIV_4740': 4740.120,
#    'HeI_4713': 4713,
#    'HeI_4922': 4922,
#    'HeI_5016': 5016,
#    'NI_5199': 5199,
#    'NI_5201': 5201,
#    'FeVI_5176': 5176,
    }

# Labels of the typical emission lines
line_label_dict = {
    'Halpha': r'H$\alpha$',
    'Hbeta': r'H$\beta$',
    'Hgamma': r'H$\gamma$',
    'OIII_4959': r'[O III] 4959',
    'OIII_5007': r'[O III] 5007',
    'SII_6716': r'[S II] 6716',
    'SII_6731': r'[S II] 6731',
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
    if spectres is None:
        raise ImportError("spectres package is not installed. Please install it to use down_spectres function.")

    if wavec is None:
        wavec = np.mean(wave)

    if dw is None:
        dw = np.median(np.diff(wave))

    sigma = wavec * np.sqrt(R_new**-2 - R_org**-2) / 2.3548 / dw
    flux_new = gaussian_filter1d(flux, sigma)

    if wave_new is not None:
        flux_new = spectres(wave_new, wave, flux_new, spec_errs=spec_errs, fill=fill, verbose=verbose)

    return flux_new


class ReadSpectrum:
    def __init__(self, is_sdss=True, hdu=None, lam=None, flux=None, ferr=None, z=0.0, ra=None, dec=None):
        """
        Initialize the ReadSpectrum class with input parameters.

        Parameters:
        - is_sdss: bool, whether the spectrum is from SDSS
        - hdu: HDU object for SDSS spectrum (used only if is_sdss=True)
        - lam: numpy array, custom spectrum wavelengths
        - flux: numpy array, custom spectrum flux values
        - ferr: numpy array, custom spectrum errors
        - z: float, redshift value
        - ra: float, right ascension (required for custom spectrum)
        - dec: float, declination (required for custom spectrum)
        """
        if sfdmap is None:
            raise ImportError("sfdmap2 package is not installed. Please install it to use ReadSpectrum class.")
        
        if pyasl is None:
            raise ImportError("PyAstronomy package is not installed. Please install it to use ReadSpectrum class.")

        self.dustmap_path = '{0}{1}data{1}sfddata/'.format(package_path, splitter)
        self.sfd_map = sfdmap.SFDMap(self.dustmap_path)

        # Validate and initialize SDSS or custom spectrum data
        self.is_sdss = is_sdss
        if self.is_sdss:
            if hdu is None:
                raise ValueError("For SDSS spectra, please provide an HDU object.")
            try:
                self.lam = np.asarray(10 ** hdu[1].data['loglam'], dtype=np.float64)
                self.flux = np.asarray(hdu[1].data['flux'], dtype=np.float64)
                self.ferr = np.asarray(1 / np.sqrt(hdu[1].data['ivar']), dtype=np.float64)
                if z==0.0:
                    self.z = hdu[2].data['z'][0]
                else:
                    self.z = z
                self.ra = hdu[0].header['plug_ra']
                self.dec = hdu[0].header['plug_dec']
            except KeyError as e:
                raise ValueError(f"Missing key in HDU data: {e}")
        else:
            # Custom spectrum
            self.lam = lam
            self.flux = flux
            self.ferr = ferr
            self.z = z
            self.ra = ra
            self.dec = dec

        # Validate custom spectrum parameters
        if self.lam is None or self.flux is None or self.ferr is None:
            raise ValueError("Please provide valid wavelength, flux, and error arrays.")
        if self.z is None or self.ra is None or self.dec is None:
            raise ValueError("Please provide redshift (z), right ascension (ra), and declination (dec).")

        # Filter valid data
        valid = (self.ferr > 0) & np.isfinite(self.ferr) & (self.flux != 0) & np.isfinite(self.flux)
        self.lam, self.flux, self.ferr = self.lam[valid], self.flux[valid], self.ferr[valid]

    
    def de_redden(self):
        """
        Perform dust extinction correction on the spectrum.

        Returns:
        - flux_unred: numpy array, flux values after extinction correction
        - err_unred: numpy array, errors after extinction correction
        """
        ebv = self.sfd_map.ebv(self.ra, self.dec)
        zero_flux = np.where(self.flux == 0, True, False)
        self.flux[zero_flux] = 1e-10
        flux_unred = pyasl.unred(self.lam, self.flux, ebv)
        err_unred = self.ferr * flux_unred / self.flux
        flux_unred[zero_flux] = 0
        return flux_unred, err_unred
    
    def rest_frame(self):
        """
        Convert spectrum data to the rest frame.

        Returns:
        - lam_res: numpy array, rest-frame wavelengths
        - flux_res: numpy array, rest-frame flux values
        - err_res: numpy array, rest-frame errors
        """
        lam_res = self.lam / (1 + self.z)
        flux_res = self.flux * (1 + self.z)
        err_res = self.ferr * (1 + self.z)
        valid = (err_res > 0) & (flux_res > err_res)
        return lam_res[valid], flux_res[valid], err_res[valid]
        
    def unredden_res(self):
        """
        Process the spectrum data.

        Returns:
        - lam_res_unred: numpy array, rest-frame wavelengths after extinction correction
        - flux_res_unred: numpy array, rest-frame flux values after extinction correction
        - err_res_unred: numpy array, rest-frame errors after extinction correction
        """
        # Perform dust extinction correction
        flux_unred, err_unred = self.de_redden()
        # Delete the original flux and ferr attributes
        del self.flux, self.ferr
        self.flux = flux_unred
        self.ferr = err_unred
        # Convert to rest frame
        lam_res_unred, flux_res_unred, err_res_unred = self.rest_frame()

        return lam_res_unred, flux_res_unred, err_res_unred