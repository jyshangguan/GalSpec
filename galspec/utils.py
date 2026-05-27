import os
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
from .constants import ls_km
from scipy.ndimage import gaussian_filter1d, median_filter
from copy import deepcopy
from astropy.modeling import CompoundModel
from urllib.request import urlretrieve
import warnings

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
           'ReadSpectrum', 'calculate_bic', 'detect_noise_mask']


if platform == "linux" or platform == "linux2":  # Linux
    splitter = '/'
elif platform == "darwin":  # Mac
    splitter = '/'
elif platform == "win32":  # Windows
    splitter = '\\'

pathList = os.path.abspath(__file__).split(splitter)
package_path = splitter.join(pathList[:-1])

def _download_sfd_data():
    """Download SFD dust extinction map data if missing."""
    sfddata_dir = os.path.join(package_path, 'data', 'sfddata')
    os.makedirs(sfddata_dir, exist_ok=True)

    # Files to download from https://github.com/kbarbary/sfddata
    base_url = "https://raw.githubusercontent.com/kbarbary/sfddata/master/"
    files = [
        'SFD_dust_4096_ngp.fits',
        'SFD_dust_4096_sgp.fits',
        'SFD_mask_4096_ngp.fits',
        'SFD_mask_4096_sgp.fits'
    ]

    missing_files = []
    for f in files:
        file_path = os.path.join(sfddata_dir, f)
        if not os.path.exists(file_path):
            missing_files.append(f)

    if not missing_files:
        return

    warnings.warn(f"Downloading missing SFD dust map data files: {missing_files}")
    for f in missing_files:
        url = base_url + f
        dest = os.path.join(sfddata_dir, f)
        try:
            urlretrieve(url, dest)
            print(f"Downloaded: {f}")
        except Exception as e:
            warnings.warn(f"Failed to download {f}: {e}")

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
    'OI_6364': 6363.776,
    'ArIV_4740': 4740.120,
    'HeI_6678': 6678.1517,
    'FeII_6369': 6369.462,
    'HeI_5876': 5875.624,
    'NaD_5890': 5891.583,
    'NaD_5896': 5897.558,
    'FeVII_6087': 6087.000,
    'FeVII_5720': 5720.700,
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


def calculate_bic(model, wave, flux, error=None):
    '''
    Calculate the Bayesian Information Criterion (BIC) for a fitted model.

    BIC = χ² + k * ln(n)

    where:
    - χ² = sum(((flux - model(wave)) / error)²)  [or sum((flux - model(wave))²) if error=None]
    - k = number of free parameters (not fixed and not tied)
    - n = number of data points

    Lower BIC indicates a better model, accounting for both fit quality
    and model complexity (penalizing extra parameters).

    Parameters
    ----------
    model : astropy.modeling.Model
        Fitted model (can be a compound model).
    wave : array like
        Wavelength array.
    flux : array like
        Flux array.
    error : array like, optional
        Error array. If not provided, χ² is calculated without error weighting.

    Returns
    -------
    bic : float
        Bayesian Information Criterion value.
    chi2 : float
        Total χ² value.
    n_params : int
        Number of free parameters.

    Examples
    --------
    >>> from sagan.utils import calculate_bic
    >>> # Fit a model
    >>> model_fit = fitter(model_init, wave, flux, weights=1/error**2)
    >>> # Calculate BIC
    >>> bic, chi2, n_params = calculate_bic(model_fit, wave, flux, error)
    >>> print(f"BIC: {bic:.1f}, χ²: {chi2:.1f}, Parameters: {n_params}")

    >>> # Compare two models
    >>> bic1, _, _ = calculate_bic(model1_fit, wave, flux, error)
    >>> bic2, _, _ = calculate_bic(model2_fit, wave, flux, error)
    >>> delta_bic = bic2 - bic1
    >>> if delta_bic < -10:
    ...     print("Strong evidence for model 2")
    >>> elif delta_bic > 10:
    ...     print("Strong evidence for model 1")
    ... else:
    ...     print("Weak evidence, prefer simpler model")
    '''
    # Calculate χ²
    model_flux = model(wave)
    if error is not None:
        chi2 = np.sum(((flux - model_flux) / error)**2)
    else:
        chi2 = np.sum((flux - model_flux)**2)

    # Count free parameters (not fixed and not tied)
    n_params = 0
    seen = set()

    # Traverse the compound model tree
    for item in model.traverse_postorder():
        for param_name in item.param_names:
            # Create unique identifier for this parameter
            param_id = f"{item.name}.{param_name}"
            if param_id not in seen:
                seen.add(param_id)
                p = getattr(item, param_name)
                if not (p.fixed or p.tied):
                    n_params += 1

    # Number of data points
    n_data = len(wave)

    # Calculate BIC
    bic = chi2 + n_params * np.log(n_data)

    return bic, chi2, n_params


def detect_noise_mask(wave, flux, threshold=20.0, protection_margin=10.0,
                      critical_lines=None, verbose=False):
    '''
    Detect and mask noise spikes in a spectrum while protecting emission lines.

    This function uses a three-tier strategy:
    1. Detect high S/N spikes using median filtering
    2. Group spikes into continuous regions
    3. Exclude regions near known emission lines
    4. Return mask for remaining noise regions

    Parameters
    ----------
    wave : array like
        Rest-frame wavelength array (in Angstroms).
    flux : array like
        Flux array (same length as wave).
    threshold : float, optional
        Detection threshold in units of residual standard deviation.
        Default is 20.0 (very conservative).
    protection_margin : float, optional
        Protection margin around each emission line in Angstroms.
        Default is 10.0 Å.
    critical_lines : dict, optional
        Dictionary of emission line names and wavelengths to protect.
        If None, uses CRITICAL_LINES from emission_lines module.
        Format: {'LINE_NAME': wavelength, ...}
    verbose : bool, optional
        Print detailed information about detection process.
        Default is False.

    Returns
    -------
    noise_mask : boolean array
        Boolean mask where True indicates pixels to mask (same length as wave).
    candidate_regions : list of tuples
        List of (wmin, wmax) tuples for all detected spike regions.
    final_mask_regions : list of tuples
        List of (wmin, wmax) tuples for regions that will be masked
        (excluding those near emission lines).
    excluded_regions : list of tuples
        List of (wmin, wmax, line_name) tuples for regions excluded
        because they are near protected emission lines.

    Examples
    --------
    >>> from sagan.utils import detect_noise_mask
    >>> # Load spectrum
    >>> wave, flux = load_spectrum()
    >>> # Detect noise
    >>> noise_mask, candidates, final, excluded = detect_noise_mask(
    ...     wave, flux, threshold=20.0, protection_margin=10.0, verbose=True
    ... )
    >>> # Apply mask
    >>> wave_clean = wave[~noise_mask]
    >>> flux_clean = flux[~noise_mask]

    Notes
    -----
    - The detection uses median filtering with a 21-pixel window
    - Residuals are calculated as flux - median(flux)
    - Detection threshold is applied to absolute residuals
    - Regions are grouped if they are within 2 pixels of each other
    - All wavelengths should be in the rest frame

    References
    ----------
    Emission line wavelengths from:
    http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
    '''
    from .emission_lines import CRITICAL_LINES

    # Use default critical lines if not provided
    if critical_lines is None:
        critical_lines = CRITICAL_LINES

    # ========================================
    # Step 1: Calculate local statistics
    # ========================================
    window = 21
    flux_median = median_filter(flux, size=window)
    residual = flux - flux_median

    # Calculate residual std in continuum region (4200-4300 Å if available)
    continuum_mask = (wave > 4200) & (wave < 4300)
    if np.sum(continuum_mask) > 0:
        residual_std = np.std(residual[continuum_mask])
    else:
        # Fallback: use global std
        residual_std = np.std(residual)

    # ========================================
    # Step 2: Find high S/N spikes
    # ========================================
    high_sn_spikes = np.abs(residual) > threshold * residual_std

    # Also find isolated single-pixel spikes (conservative)
    gradient = np.abs(np.gradient(flux))
    if np.sum(continuum_mask) > 0:
        gradient_threshold = 10 * np.median(gradient[continuum_mask])
    else:
        gradient_threshold = 10 * np.median(gradient)
    single_pixel_spikes = gradient > gradient_threshold

    # Combine
    potential_noise = high_sn_spikes | single_pixel_spikes

    if verbose:
        print(f"\n[Noise Detection]")
        print(f"  Residual std: {residual_std:.4f}")
        print(f"  Threshold: {threshold:.1f}σ")
        print(f"  Spikes detected: {np.sum(potential_noise)} pixels "
              f"({100*np.sum(potential_noise)/len(wave):.2f}%)")

    # ========================================
    # Step 3: Group into continuous regions
    # ========================================
    noise_indices = np.where(potential_noise)[0]
    candidate_regions = []

    if len(noise_indices) > 0:
        current_start = noise_indices[0]
        current_end = noise_indices[0]

        for i in range(1, len(noise_indices)):
            if noise_indices[i] - noise_indices[i-1] <= 2:
                current_end = noise_indices[i]
            else:
                w_start = wave[current_start]
                w_end = wave[current_end]
                candidate_regions.append((w_start - 2, w_end + 2))
                current_start = noise_indices[i]
                current_end = noise_indices[i]

        # Add last region
        w_start = wave[current_start]
        w_end = wave[current_end]
        candidate_regions.append((w_start - 2, w_end + 2))

    candidate_regions.sort()

    if verbose:
        print(f"  Grouped into {len(candidate_regions)} regions")

    # ========================================
    # Step 4: Filter out regions near emission lines
    # ========================================
    final_mask_regions = []
    excluded_regions = []

    for wmin, wmax in candidate_regions:
        near_line = False
        nearby_line_name = None

        for name, wave_c in critical_lines.items():
            # Check if region overlaps with protected zone around line
            line_min = wave_c - protection_margin
            line_max = wave_c + protection_margin

            # Overlap check
            if not (wmax < line_min or wmin > line_max):
                near_line = True
                nearby_line_name = name
                break

        if near_line:
            excluded_regions.append((wmin, wmax, nearby_line_name))
        else:
            final_mask_regions.append((wmin, wmax))

    if verbose:
        print(f"\n[Filtering Results]")
        print(f"  Candidate regions: {len(candidate_regions)}")
        print(f"  Excluded (near lines): {len(excluded_regions)}")
        print(f"  Final mask regions: {len(final_mask_regions)}")

        if len(excluded_regions) > 0:
            print(f"\n  Excluded regions:")
            for i, (wmin, wmax, line) in enumerate(excluded_regions):
                print(f"    {i+1}. {wmin:.1f}-{wmax:.1f} Å (protected - {line})")

        if len(final_mask_regions) > 0:
            print(f"\n  Final mask regions:")
            for i, (wmin, wmax) in enumerate(final_mask_regions):
                print(f"    {i+1}. {wmin:.1f}-{wmax:.1f} Å")

    # ========================================
    # Step 5: Create final mask
    # ========================================
    noise_mask = np.zeros_like(wave, dtype=bool)
    for wmin, wmax in final_mask_regions:
        noise_mask |= (wave >= wmin) & (wave <= wmax)

    if verbose:
        print(f"\n  Pixels to mask: {np.sum(noise_mask)} "
              f"({100*np.sum(noise_mask)/len(wave):.2f}%)")

    return noise_mask, candidate_regions, final_mask_regions, excluded_regions


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

        # Download dust map data if missing
        _download_sfd_data()

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