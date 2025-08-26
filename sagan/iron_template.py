import numpy as np
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from .utils import splitter, package_path
from .constants import ls_km


__all__ = ['IronTemplate']


irontemp_boroson1992 = Table.read('{0}{1}data{1}irontemplate_boroson1992.ipac'.format(package_path, splitter),
                       format='ipac')
wave_boroson1992 = irontemp_boroson1992['Spectral_axis'].data
flux_boroson1992 = irontemp_boroson1992['Intensity'].data
flux_boroson1992 /= np.max(flux_boroson1992)

irontemp_park2022 = Table.read('{0}{1}data{1}irontemplate_park2022.dat'.format(package_path, splitter),
                       format='cds')
wave_park2022 = irontemp_park2022['Wave'].data
flux_park2022 = irontemp_park2022['Flux'].data
flux_park2022 /= np.max(flux_park2022)


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
    stddev = Parameter(default=910/2.3548, bounds=(910/2.3548, None))
    z = Parameter(default=0, bounds=(0, 10))
    
    def __init__(self, amplitude=amplitude, stddev=stddev, z=z, template_name='park2022', **kwargs):
        super().__init__(amplitude=amplitude, stddev=stddev, z=z, **kwargs)
        
        if template_name == 'park2022':
            wave_temp = wave_park2022
            flux_temp = flux_park2022
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
        self._stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1, Mrk 493 looks similar
        self._vchan = (wave_temp[1] - wave_temp[0]) / wave_temp[0] * ls_km  # Velocity width per channel
    
    def evaluate(self, x, amplitude, stddev, z):
        """
        Gaussian model function.
        """
        if stddev < self._stddev_intr:
            stddev = 910 / 2.3548

        sig = np.sqrt(stddev**2 - self._stddev_intr**2) / self._vchan
        flux_conv = gaussian_filter1d(self._flux_temp, sig)
        f = amplitude * np.interp(x, self._wave_temp * (1 + z), flux_conv)

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
        self._vchan = (ln_wave_temp[1]-ln_wave_temp[0])* ls_km
        #self._vchan = (wave_temp[1] - wave_temp[0]) / wave_temp[0] * ls_km  # Velocity width per channel
        
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
            vel_offsets = (self._wave_temp - center_wave) / center_wave * ls_km
            
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