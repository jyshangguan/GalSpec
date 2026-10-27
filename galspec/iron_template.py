import numpy as np
from functools import lru_cache
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


def load_iron_template(template_name='park2022', pad_velocity=100000.0):
    """Read an Fe II template and put it on the log-even, zero-padded grid.

    Resampling happens here, once, at load time, so every model built on the
    template -- :class:`IronTemplate`, :class:`IronTemplate_new`,
    :class:`IronTemplate_tied` and their JAX counterparts -- broadens it in
    velocity space by construction.  Before this, the template was smoothed in
    *pixels* with a single velocity scale taken from the blue end, which applied
    a width 40% too small at the red end of ``park2022``.

    The tabulated range is then extended with zeros by ``pad_velocity`` on each
    side, on the same grid.  The templates stop where their flux does not (the
    ``park2022`` table ends at 5600 A still carrying about 4% of its peak), and
    the smoothing pads by *mirroring* at the ends, so without the extension the
    model reflects that edge flux back into the spectrum and keeps a constant
    ~5% floor redward of 5600 A -- which is inside the range the iron is fitted
    over.  With the zeros the model simply fades out where the template runs out,
    and its answer at the ends agrees with a convolution that has no flux beyond
    the tabulated range.

    Parameters
    ----------
    template_name : {'park2022', 'boroson1992'}
        ``park2022``: Mrk 493, Park et al. (2022).  ``boroson1992``: I Zw 1,
        Boroson & Green (1992).
    pad_velocity : float, optional
        How far to extend each end with zero flux [km/s].  The default,
        100 000 km/s, is four sigma of the widest kernel the models are asked
        for, so the mirror padding at the ends never reaches the template's own
        flux.  ``0`` returns the tabulated range only.

    Returns
    -------
    wave, flux : ndarray
        Wavelength [A], on a grid even in ``ln(wave)``, and flux normalised to
        unit maximum.  ``flux`` is zero outside the tabulated range.

    Notes
    -----
    The read-resample-pad work is cached per ``(template_name, pad_velocity)``
    and the cached arrays are returned as copies, so building one model per
    spectrum in a large sample does not re-read the table each time and no
    caller can corrupt the cached template.  The published arrays
    (``wave_park2022``, ...) are unaffected.
    """
    wave, flux = _load_iron_template_cached(template_name, float(pad_velocity))
    return wave.copy(), flux.copy()


@lru_cache(maxsize=8)
def _load_iron_template_cached(template_name, pad_velocity):
    """:func:`load_iron_template` behind the cache: returns the shared arrays."""
    try:
        name, fmt, wave_col, flux_col = _TEMPLATE_FILES[template_name]
    except KeyError:
        raise KeyError('Cannot recognize the iron template model ({})!'.format(template_name))

    table = Table.read('{0}{1}data{1}{2}'.format(package_path, splitter, name), format=fmt)
    wave = np.asarray(table[wave_col].data, dtype=float)
    flux = np.asarray(table[flux_col].data, dtype=float)

    wave, flux = resample_log_even(wave, flux)
    wave, flux = _pad_with_zeros(wave, flux, pad_velocity)
    return wave, flux / np.max(flux)


def _pad_with_zeros(wave, flux, pad_velocity):
    """Extend a log-even spectrum with zero flux, on its own grid."""
    if pad_velocity <= 0.0:
        return wave, flux
    dln = np.log(wave[1]) - np.log(wave[0])
    count = int(np.ceil(pad_velocity / (dln * ls_km)))
    low = wave[0] * np.exp(-dln * np.arange(count, 0, -1))
    high = wave[-1] * np.exp(dln * np.arange(1, count + 1))
    return (np.concatenate([low, wave, high]),
            np.concatenate([np.zeros(count), flux, np.zeros(count)]))


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
    kernel : model, optional
        The line profile the iron is tied to, e.g.
        ``galspec.Line_MultiGauss(n_components=2)``.  Its components give the
        kernel: every component is applied to the template separately, at its own
        shift and with its own width, after subtracting the template's own width in
        quadrature (``sqrt(sigma**2 - stddev_intr**2)``) -- a convolution in velocity
        space, with no deconvolution, exactly as the JAX
        ``jax_models.IronTemplate_tied`` does.  If omitted, ``stddev`` sets a single
        Gaussian width instead.

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
        ln_wave_temp = np.log(wave_temp)
        self._vchan = (ln_wave_temp[1] - ln_wave_temp[0]) * ls_km
        
        # Store the broad emission line template
        self.kernel = kernel
    
    def _kernel_terms(self):
        """(weight, shift, width) of every component of the tied line profile.

        The kernel is a :class:`galspec.Line_MultiGauss` (or anything carrying the
        same parameters), so the components are read straight off it: component 0 is
        the core at ``dv_c``, component ``i`` sits at ``dv_c + dv_w{i}``.  The
        weights are the line's own amplitudes, normalised to sum to one -- only the
        *shape* of the line reaches the iron, so the core amplitude cancels.
        """
        kernel = self.kernel
        # ``_param_names`` is per instance (``_parameters_`` is shared by every model
        # of the class and keeps parameters of previously built ones), so it is the
        # reliable list of what this kernel actually has.
        names = list(getattr(kernel, '_param_names', None)
                     or getattr(kernel, 'param_names', ()) or ())

        def value(name, default=None):
            if name in names:
                return float(getattr(kernel, name).value)
            return default

        if value('sigma_c') is None:
            raise TypeError(
                'the kernel of IronTemplate_tied must be a line-profile model with '
                'dv_c/sigma_c/amp_w{i}/dv_w{i}/sigma_w{i} parameters (e.g. '
                'galspec.Line_MultiGauss), not {!r}: the template\'s own width has '
                'to be subtracted in quadrature from each component'
                .format(type(kernel).__name__))

        weights = [1.0]
        shifts = [value('dv_c', 0.0)]
        widths = [value('sigma_c')]
        for index in range(sum(1 for name in names if name.startswith('amp_w'))):
            weights.append(value('amp_w{}'.format(index)))
            shifts.append(value('dv_c', 0.0) + value('dv_w{}'.format(index), 0.0))
            widths.append(value('sigma_w{}'.format(index)))
        total = float(np.sum(weights))
        total = total if total > 0.0 else 1.0
        return [(w / total, shift, width)
                for w, shift, width in zip(weights, shifts, widths)]

    def _broadened_template(self, width):
        """The template broadened to a *total* dispersion ``width`` [km/s].

        The template already carries its own width, and blurring twice with sigma_1
        and sigma_2 is one blur with them in quadrature, so the extra width to apply
        is ``sqrt(width**2 - stddev_intr**2)``.  No deconvolution: an inverse filter
        amplifies whatever it divides by, and with a 2400-channel template and a
        340 km/s intrinsic width that is not a small effect.
        """
        if width <= self._stddev_intr:
            return self._flux_temp
        sig = np.sqrt(width ** 2 - self._stddev_intr ** 2) / self._vchan
        return gaussian_filter1d(self._flux_temp, sig)

    def evaluate(self, x, amplitude, stddev, z):
        """
        Model function with either Gaussian or custom convolution kernel.
        """
        x = np.asarray(x, dtype=float)
        if self.kernel is not None:
            # One broadened template per kernel component, each projected onto the
            # data grid at its own shift and added with its own weight.  Projecting
            # the components separately (rather than convolving once with a composite
            # kernel) keeps a one-component kernel identical to IronTemplate -- which
            # is what the JAX model does as well.
            flux = np.zeros(np.shape(x))
            for weight, shift, width in self._kernel_terms():
                template = self._broadened_template(width)
                wave_k = self._wave_temp * (1.0 + z + shift / ls_km)
                flux += weight * np.interp(x, wave_k, template)
            return amplitude * flux

        # -- no kernel: one Gaussian of the requested width ----------------------
        # same convention as IronTemplate/IronTemplate_new: a width narrower than
        # the template's own cannot be represented, so the template is returned as
        # it is (which is also what the JAX model does)
        if stddev < self._stddev_intr:
            flux_conv = self._flux_temp
        else:
            sig = np.sqrt(stddev**2 - self._stddev_intr**2) / self._vchan
            flux_conv = gaussian_filter1d(self._flux_temp, sig)

        return amplitude * np.interp(x, self._wave_temp * (1 + z), flux_conv)
