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