import numpy as np
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from .utils import splitter, package_path
from .constants import ls_km
from .stellar_continuum import log_rebin


__all__ = ['IronTemplate', 'IronTemplate_new', 'IronTemplate_tied']


#: Template files, and the wavelength and flux column of each.
_TEMPLATE_FILES = {
    'park2022': ('irontemplate_park2022.dat', 'cds', 'Wave', 'Flux'),
    'boroson1992': ('irontemplate_boroson1992.ipac', 'ipac', 'Spectral_axis', 'Intensity'),
}


def resample_log_even(wave, flux, velscale=None, oversample=1):
    """Resample a spectrum onto a grid that is even in the natural log of wavelength.

    Both Fe II templates are tabulated on a grid that is even in *wavelength*
    (``park2022``: exactly 2.0 A from 4000 to 5600 A), so one channel spans
    150 km/s at the blue end and 107 km/s at the red end.  Broadening such a
    template means convolving it in velocity, and that needs a grid whose
    spacing is constant in velocity -- ``dln(lambda) = dv / c`` -- otherwise a
    Gaussian of a given ``sigma`` in km/s is a different number of channels at
    each end of the template and the requested width is off by up to 40%.

    Uses :func:`galspec.stellar_continuum.log_rebin`, which integrates the
    template flux over the new bins (a template is a flux density, not a set of
    point samples) and pads the limits by half an input pixel.

    Parameters
    ----------
    wave, flux : array like
        The spectrum as tabulated.  ``wave`` must be increasing.
    velscale : float, optional
        Velocity per channel [km/s].  The default, ``None``, keeps the number of
        channels: 126 km/s for ``park2022``, 107 km/s for ``boroson1992`` --
        never coarser than the red end of either original grid.
    oversample : int, optional
        Only used when ``velscale`` is None: output channels per input channel.

    Returns
    -------
    wave, flux : ndarray
        The spectrum on the new grid, with ``ln(wave)`` evenly spaced.
    """
    flux_rebin, ln_wave = log_rebin(wave, flux, velscale=velscale, oversample=oversample)
    return np.exp(ln_wave), flux_rebin


def load_iron_template(template_name='park2022'):
    """Read an Fe II template and put it on the log-even grid.

    Resampling happens here, once, at load time, so every model built on the
    template -- :class:`IronTemplate`, :class:`IronTemplate_new`,
    :class:`IronTemplate_tied` and their JAX counterparts -- broadens it in
    velocity space by construction.  Before this, the template was smoothed in
    *pixels* with a single velocity scale taken from the blue end, which applied
    a width 40% too small at the red end of ``park2022``.

    Parameters
    ----------
    template_name : {'park2022', 'boroson1992'}
        ``park2022``: Mrk 493, Park et al. (2022).  ``boroson1992``: I Zw 1,
        Boroson & Green (1992).

    Returns
    -------
    wave, flux : ndarray
        Wavelength [A], on a grid even in ``ln(wave)``, and flux normalised to
        unit maximum.
    """
    try:
        name, fmt, wave_col, flux_col = _TEMPLATE_FILES[template_name]
    except KeyError:
        raise KeyError('Cannot recognize the iron template model ({})!'.format(template_name))

    table = Table.read('{0}{1}data{1}{2}'.format(package_path, splitter, name), format=fmt)
    wave = np.asarray(table[wave_col].data, dtype=float)
    flux = np.asarray(table[flux_col].data, dtype=float)

    wave, flux = resample_log_even(wave, flux)
    return wave, flux / np.max(flux)


wave_park2022, flux_park2022 = load_iron_template('park2022')
wave_boroson1992, flux_boroson1992 = load_iron_template('boroson1992')


class IronTemplate(Fittable1DModel):
    '''
    This is a Fe template of AGN from I Zw 1 (Boroson & Green 1992).

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    amplitude : float
        Amplitude of the template, units: arbitrary.
    stddev : float
        Velocity dispersion of the AGN, units: km/s. Lower limit about 390 km/s.
    z : float
        Redshift of the AGN.
    template_name : string
        The name of the template.
        park2022 : Based on Mrk 493 from Park et al. (2022).
        boroson1992 : Based on I Zw 1 from Boroson & Green (1992).

    Returns
    -------
    flux_intp : array like
        The interpolated flux of iron emission.
    '''

    amplitude = Parameter(default=1, bounds=(0, None))
    stddev = Parameter(default=900/2.3548, bounds=(900/2.3548, None))
    z = Parameter(default=0, bounds=(0, 10))
    
    def __init__(self, amplitude=amplitude, stddev=stddev, z=z, template_name='park2022', **kwargs):
        super().__init__(amplitude=amplitude, stddev=stddev, z=z, **kwargs)
        
        if template_name == 'park2022':
            wave_temp = wave_park2022
            flux_temp = flux_park2022
            self._stddev_intr = 800 / 2.3548  # Velocity dispersion of Mrk 493.
        elif template_name == 'boroson1992':
            wave_temp = wave_boroson1992
            flux_temp = flux_boroson1992
            self._stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1.
        else:
            raise KeyError('Cannot recognize the iron template model ({})!'.format(template_name))
        
        fltr = (wave_temp > 4500) & (wave_temp < 5500)
        self._vmin, self._vmax = np.min(wave_temp[fltr]), np.max(wave_temp[fltr])
        fmax = np.max(flux_temp[fltr])
        self._wave_temp = wave_temp
        self._flux_temp = flux_temp / fmax
        #self._stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1, Mrk 493 looks similar
        # velocity per channel; the grid is even in ln(wave), so this is the same
        # everywhere (it used to be the blue end of a linear grid)
        self._vchan = (np.log(wave_temp[1]) - np.log(wave_temp[0])) * ls_km
    
    def evaluate(self, x, amplitude, stddev, z):
        """
        Gaussian model function.
        """
        if stddev < self._stddev_intr:
            flux_conv = self._flux_temp
        else:
            sig = np.sqrt(stddev**2 - self._stddev_intr**2) / self._vchan
            flux_conv = gaussian_filter1d(self._flux_temp, sig)

        f = amplitude * np.interp(x, self._wave_temp * (1 + z), flux_conv)

        return f


class IronTemplate_new(Fittable1DModel):
    '''
    This is a Fe template of AGN from I Zw 1 (Boroson & Green 1992).
    Modified version of IronTemplate with:
    - sigma instead of stddev (same meaning)
    - dv parameter for velocity shift
    - z as a fixed parameter

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    amplitude : float
        Amplitude of the template, units: arbitrary.
    sigma : float
        Velocity dispersion of the AGN, units: km/s. Lower limit about 390 km/s.
    z : float
        Redshift of the AGN (fixed parameter).
    dv : float
        Velocity shift of the template, units: km/s.
    template_name : string
        The name of the template.
        park2022 : Based on Mrk 493 from Park et al. (2022).
        boroson1992 : Based on I Zw 1 from Boroson & Green (1992).

    Returns
    -------
    flux_intp : array like
        The interpolated flux of iron emission.
    '''

    amplitude = Parameter(default=1, bounds=(0, None))
    sigma = Parameter(default=900/2.3548, bounds=(900/2.3548, None))
    z = Parameter(default=0, fixed=True)
    dv = Parameter(default=0, bounds=(-2000, 2000))

    def __init__(self, amplitude=amplitude, sigma=sigma, z=z, dv=dv, template_name='park2022', **kwargs):
        super().__init__(amplitude=amplitude, sigma=sigma, z=z, dv=dv, **kwargs)

        if template_name == 'park2022':
            wave_temp = wave_park2022
            flux_temp = flux_park2022
            self._stddev_intr = 800 / 2.3548  # Velocity dispersion of Mrk 493.
        elif template_name == 'boroson1992':
            wave_temp = wave_boroson1992
            flux_temp = flux_boroson1992
            self._stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1.
        else:
            raise KeyError('Cannot recognize the iron template model ({})!'.format(template_name))

        fltr = (wave_temp > 4500) & (wave_temp < 5500)
        self._vmin, self._vmax = np.min(wave_temp[fltr]), np.max(wave_temp[fltr])
        fmax = np.max(flux_temp[fltr])
        self._wave_temp = wave_temp
        self._flux_temp = flux_temp / fmax
        # velocity per channel; the grid is even in ln(wave), so this is the same
        # everywhere (it used to be the blue end of a linear grid)
        self._vchan = (np.log(wave_temp[1]) - np.log(wave_temp[0])) * ls_km

    def evaluate(self, x, amplitude, sigma, z, dv):
        """
        Gaussian model function with velocity shift.
        """
        if sigma < self._stddev_intr:
            flux_conv = self._flux_temp
        else:
            sig = np.sqrt(sigma**2 - self._stddev_intr**2) / self._vchan
            flux_conv = gaussian_filter1d(self._flux_temp, sig)

        # Apply both redshift (z) and velocity shift (dv) to wavelength grid
        # dv is in km/s, so dv/ls_km gives the fractional wavelength shift
        wave_shifted = self._wave_temp * (1 + z + dv / ls_km)
        f = amplitude * np.interp(x, wave_shifted, flux_conv)

        return f


class IronTemplate_tied(Fittable1DModel):
    '''
    This is a Fe template of AGN from I Zw 1 (Boroson & Green 1992) with tied convolution kernel.

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    amplitude : float
        Amplitude of the template, units: arbitrary.
    stddev : float
        Velocity dispersion of the AGN, units: km/s. Lower limit about 390 km/s.
    z : float
        Redshift of the AGN.
    template_name : string
        The name of the template.
        park2022 : Based on Mrk 493 from Park et al. (2022).
        boroson1992 : Based on I Zw 1 from Boroson & Green (1992).
    kernel : array like, optional
        Convolution kernel to use.
        If provided, will override the Gaussian convolution.

    Returns
    -------
    flux_intp : array like
        The interpolated flux of iron emission.
    '''

    amplitude = Parameter(default=1, bounds=(0, None))
    stddev = Parameter(default=910/2.3548, bounds=(910/2.3548, None))
    z = Parameter(default=0, bounds=(0, 10))
    
    def __init__(self, amplitude=amplitude, stddev=stddev, z=z, template_name='park2022', kernel=None, **kwargs):
        super().__init__(amplitude=amplitude, stddev=stddev, z=z, **kwargs)
        
        if template_name == 'park2022':
            wave_temp = wave_park2022
            flux_temp = flux_park2022
            self._stddev_intr = ls_km / 1500 # R~1500
        elif template_name == 'boroson1992':
            wave_temp = wave_boroson1992
            flux_temp = flux_boroson1992
            self._stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1.
        else:
            raise KeyError('Cannot recognize the iron template model ({})!'.format(template_name))
        
        fltr = (wave_temp > 4500) & (wave_temp < 5500)
        self._vmin, self._vmax = np.min(wave_temp[fltr]), np.max(wave_temp[fltr])
        fmax = np.max(flux_temp[fltr])
        self._wave_temp = wave_temp
        self._flux_temp = flux_temp / fmax
        #self._stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1, Mrk 493 looks similar
        ln_wave_temp = np.log(wave_temp)
        self._vchan = (ln_wave_temp[1] - ln_wave_temp[0]) * ls_km
        
        # Store the broad emission line template
        self.kernel = kernel
    
    def evaluate(self, x, amplitude, stddev, z):
        """
        Model function with either Gaussian or custom convolution kernel.
        """
        if self.kernel is not None:
            # Check if kernel is a Line_MultiGauss object and extract the template
            if hasattr(self.kernel, 'gen_template'):
                # Generate template from Line_MultiGauss object
                velc_range = np.linspace(-10000, 10000, 20000)  # Default velocity range
                kernel_template = self.kernel.gen_template(velc_range, normalized=True)
                # Normalize the kernel
                kernel = kernel_template / np.sum(kernel_template)
            else:
                # Assume kernel is already a numpy array
                kernel = np.asarray(self.kernel)
                kernel = kernel / np.sum(kernel)
            
            # Convert velocity-space kernel to wavelength-space kernel using interpolation
            from scipy.interpolate import interp1d
            from scipy.signal import convolve
            
            # Calculate velocity offsets for each wavelength point
            center_wave = self._wave_temp[len(self._wave_temp)//2]
            # exact velocity offset; on the log-even grid this is a uniform ladder
            vel_offsets = np.log(self._wave_temp / center_wave) * ls_km
            
            # Create interpolator for the kernel
            kernel_interp = interp1d(velc_range, kernel, bounds_error=False, fill_value=0)
            
            # Interpolate kernel values at velocity offsets
            wave_kernel = kernel_interp(vel_offsets)
            
            # Normalize the wavelength-space kernel
            wave_kernel = wave_kernel / np.sum(wave_kernel)
            
            # Step 1: Remove intrinsic broadening effect
            # Calculate the intrinsic sigma in pixel units
            sig_intr_pix = self._stddev_intr / self._vchan
            
            # Create a Gaussian kernel for the intrinsic broadening
            kernel_size = max(3, int(3 * sig_intr_pix))
            x_kernel = np.arange(-kernel_size, kernel_size + 1)
            gaussian_kernel = np.exp(-0.5 * (x_kernel / sig_intr_pix)**2)
            gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
            
            # Use frequency domain deconvolution for better stability
            from scipy.fft import fft, ifft
            from scipy.signal import wiener
            
            # Pad both template and kernel to avoid edge effects
            pad_size = kernel_size
            flux_padded = np.pad(self._flux_temp, (pad_size, pad_size), mode='edge')
            
            # Pad kernel to match padded template length
            kernel_padded = np.pad(gaussian_kernel, (0, len(flux_padded) - len(gaussian_kernel)), mode='constant')
            
            # Convert to frequency domain
            flux_fft = fft(flux_padded)
            kernel_fft = fft(kernel_padded)
    
            # Perform deconvolution with regularization
            # Add small constant to avoid division by zero
            epsilon = 1e-12
            flux_deconv_fft = flux_fft / (kernel_fft + epsilon)
            
            # Convert back to time domain
            flux_deconv = np.real(ifft(flux_deconv_fft))
            
            # Remove padding
            flux_deconv = flux_deconv[pad_size:pad_size + len(self._flux_temp)]
            
            # Apply Wiener filter to smooth out noise
            flux_deconv = wiener(flux_deconv, (3,))
            
            # Step 2: Apply the custom kernel (b_hb) to the deconvolved template
            # This will add the broadening effect of the custom kernel
            flux_conv = convolve(flux_deconv, wave_kernel, mode='same')
            
        else:
            # Use original Gaussian convolution
            if stddev < self._stddev_intr:
                stddev = ls_km / 1500

            sig = np.sqrt(stddev**2 - self._stddev_intr**2) / self._vchan
            flux_conv = gaussian_filter1d(self._flux_temp, sig)

        f = amplitude * np.interp(x, self._wave_temp * (1 + z), flux_conv)

        return f