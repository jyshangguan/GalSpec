import numpy as np
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from .constants import ls_km
from .utils import line_wave_dict


__all__ = ['Line_Gaussian', 'Line_GaussHermite', 'Line_template', 
           'Line_MultiGauss', 'Line_MultiGauss_doublet',
           'tier_line_ratio', 
           'tier_line_sigma', 'tier_wind_dv', 'tier_abs_dv', 'find_line_peak', 
           'line_fwhm', 'extinction_ccm89', 'gen_o3doublet_gauss','gen_s2doublet_gauss', 
           'gen_o3doublet_gausshermite', 'gen_s2doublet_gausshermite',
           'fix_profile_multigauss', 'fix_profile_gausshermite', 'get_line_multigaussian']

           
class Line_Gaussian(Fittable1DModel):
    '''
    The Gaussian line profile with the sigma as the velocity.
    Parameters
    ----------
    x : array like
        Wavelength, units: arbitrary.
    amplitude : float
        The amplitude of the line profile.
    dv : float
        The velocity of the central line offset from wavec, units: km/s.
    sigma : float
        The velocity dispersion of the line profile, units: km/s.
    wavec : float
        The central wavelength of the line profile, units: same as x.
    '''

    amplitude = Parameter(default=1, bounds=(0, None))
    dv = Parameter(default=0, bounds=(-2000, 2000))
    sigma = Parameter(default=200, bounds=(20, 10000))

    wavec = Parameter(default=5000, fixed=True)

    @staticmethod
    def evaluate(x, amplitude, dv, sigma, wavec):
        """
        Gaussian model function.
        """
        v = (x - wavec) / wavec * ls_km  # convert to velocity (km/s)
        f = amplitude * np.exp(-0.5 * ((v - dv)/ sigma)**2)

        return f


class Line_GaussHermite(Fittable1DModel):
    '''
    The line profile as a fourth-order Gauss–Hermite function.
    Parameters
    ----------
    x : array like
        Wavelength, units: arbitrary.
    amplitude : float
        The amplitude of the line profile.
    dv : float
        The velocity of the central line offset from wavec, units: km/s.
    sigma : float
        The velocity dispersion of the line profile, units: km/s.
    wavec : float
        The central wavelength of the line profile, units: same as x.
    clip : bool (default: False)
        Whether to replace the negative value to 0.
    '''

    amplitude = Parameter(default=1, bounds=(0, None))
    dv = Parameter(default=0, bounds=(-2000, 2000))
    sigma = Parameter(default=200, bounds=(20, 10000))
    h3 = Parameter(default=0, bounds=(-0.4, 0.4))
    h4 = Parameter(default=0, bounds=(-0.4, 0.4))
    wavec = Parameter(default=5000, fixed=True)
    
    def __init__(self, amplitude=amplitude, dv=dv, sigma=sigma,
                 h3=h3, h4=h4, wavec=wavec, clip=True, **kwargs):
        
        self._clip = clip
        super().__init__(amplitude=amplitude, dv=dv, sigma=sigma, h3=h3, h4=h4, wavec=wavec,
                         **kwargs)

    def evaluate(self, x, amplitude, dv, sigma, h3, h4, wavec):
        '''
        GaussHermite model function.
        '''

        v = (x - wavec) / wavec * ls_km  # convert to velocity (km/s)
        w = (v - dv)/ sigma

        G = amplitude * np.exp(-0.5 * w**2)
        H3 = (2 * w**3 - 3 * w) / 3**0.5
        H4 = (4 * w**4 - 12 * w**2 + 3) / 24**0.5
        f = G * (1 + h3 * H3 + h4 * H4)

        if self._clip == 1:
            f[f < 0] = 0

        return f


class Line_template(Fittable1DModel):
    '''
    Emission model using an input template.
    
    Parameters
    ----------
    template_velc : 1d array
        Line profile velocity, units: km/s.
    template_flux : 1d array
        Line profile flux, arbitrary units.
    x : array like
        Wavelength, units: arbitrary.
    amplitude : float
        The amplitude of the line profile.
    dv : float
        The velocity of the central line offset from wavec, units: km/s.
    wavec : float
        The central wavelength of the line profile, units: same as x.
    '''
    amplitude = Parameter(default=1, bounds=(0, None))
    dv = Parameter(default=0, bounds=(-2000, 2000))

    wavec = Parameter(default=5000, fixed=True)
    
    def __init__(self, template_velc, template_flux, amplitude=amplitude, dv=dv, wavec=wavec, 
                 **kwargs):
        super().__init__(amplitude=amplitude, dv=dv, wavec=wavec, **kwargs)
        self._vmin, self._vmax = np.min(template_velc), np.max(template_velc)
        self._template_velc = template_velc
        self._template_flux = template_flux
        self._model = interp1d(template_velc, template_flux)
    
    def evaluate(self, x, amplitude, dv, wavec):
        """
        Gaussian model function.
        """
        f = np.zeros_like(x)
        v = (x - wavec) / wavec * ls_km - dv  # convert to velocity (km/s)
        
        fltr = (v >= self._vmin) & (v <= self._vmax)
        f[fltr] = amplitude * self._model(v[fltr])
        return f


class test(Fittable1DModel):
    def __init__(self, amplitude, **kwargs):
        amplitude = np.atleast_1d(amplitude)

        if len(amplitude) == 1:
            self.n_components = 1
            params = [Parameter(default=amplitude[0])]
        else:
            self.n_components = len(amplitude)
            params = [Parameter(default=amp) for amp in amplitude]

        super().__init__(*params, **kwargs)
    
    def evaluate(self, x, *parameters):
        return np.sum(parameters) * np.ones_like(x)


class Line_MultiGauss(Fittable1DModel):
    '''
    Multi-component Gaussian line profile.

    Parameters
    ----------
    x : array like
        Wavelength, units: arbitrary.
    n_components : int
        The number of Gaussian components.
    amp_c : float
        The amplitude of the core component.
    dv_c : float
        The velocity of the core component, units: km/s.
    sigma_c : float
        The velocity dispersion of the core component, units: km/s.
    wavec : float
        The central wavelength of the line profile, units: same as x.
    par_w : dict
        The parameters of the wind components.
            amp_w`i` : float
                The amplitude of the `i`th wind component, relative to the core amplitude.
            dv_w`i` : float
                The velocity of the `i`th wind component, relative to the core velocity, units: km/s.
            sigma_w`i` : float
                The velocity dispersion of the `i`th wind component, units: km/s.
    name : string
        The name of the line profile.
    **kwargs : dict
        Additional parameters like bounds, fixed, and meta.
    '''
    _param_names = ()

    def __init__(self, n_components=1, amp_c=1, dv_c=0, sigma_c=200, wavec=5000, par_w={}, name=None, **kwargs):
        '''
        Initialize the Line_MultiGauss model.
        '''
        assert isinstance(n_components, int), 'n_components must be an integer!'
        assert n_components > 0, 'n_components must be positive!'

        self.n_components = n_components

        self._param_names = ['amp_c', 'dv_c', 'sigma_c', 'wavec']
        self._parameters_['amp_c'] = Parameter(default=amp_c, bounds=(0, None))
        self._parameters_['dv_c'] = Parameter(default=dv_c, bounds=(-2000, 2000))
        self._parameters_['sigma_c'] = Parameter(default=sigma_c, bounds=(20, 10000))
        self._parameters_['wavec'] = Parameter(default=wavec, fixed=True)

        if n_components > 1:
            for loop in range(n_components - 1):
                pn_amp = f'amp_w{loop}'
                pn_dv = f'dv_w{loop}'
                pn_sigma = f'sigma_w{loop}'
                pv_amp = par_w.get(pn_amp, 0)
                pv_dv = par_w.get(pn_dv, 0)
                pv_sigma = par_w.get(pn_sigma, sigma_c)

                self._param_names.append(pn_amp)
                self._param_names.append(pn_dv)
                self._param_names.append(pn_sigma)

                self._parameters_[pn_amp] = Parameter(default=pv_amp, bounds=(0, None))
                self._parameters_[pn_dv] = Parameter(default=pv_dv, bounds=(-5000, 5000))
                self._parameters_[pn_sigma] = Parameter(default=pv_sigma, bounds=(0, 10000))

        kwargs.update(par_w)
        super().__init__(amp_c, dv_c, sigma_c, wavec, name=name, **kwargs)

    def evaluate(self, x, *params):
        '''
        Multi-component Gaussian model function.
        '''
        amp_c, dv, sigma, wavec = params[:4]

        # Calculate primary Gaussian component
        v = (x - wavec) / wavec * ls_km  # convert to velocity (km/s)
        flux_c = amp_c * np.exp(-0.5 * ((v - dv) / sigma)**2)

        if self.n_components == 1:
            return flux_c
        
        # Calculate additional Gaussian components
        n_add = self.n_components - 1
        n_pars = 3
        flux_w = sum([
            amp_c * params[4 + i * n_pars] * np.exp(-0.5 * ((v - dv - params[5 + i * n_pars]) / params[6 + i * n_pars])**2)
            for i in range(n_add)]
        )

        return flux_c + flux_w

    @property
    def param_names(self):
        '''
        Coefficient names generated based on the model's number of components.

        Subclasses should implement this to return parameter names in the
        desired format.
        '''
        return self._param_names
    
    @property
    def subcomponents(self):
        '''
        Return the individual components of the model.
        '''
        if self.n_components == 1:
            return None

        components = [Line_Gaussian(
            amplitude=self.amp_c, dv=self.dv_c, sigma=self.sigma_c, 
            wavec=self.wavec, name=f'{self.name}: core')]

        for loop in range(self.n_components - 1):
            amp_w = getattr(self, f'amp_w{loop}') * self.amp_c
            dv_w = getattr(self, f'dv_w{loop}') + self.dv_c
            sigma_w = getattr(self, f'sigma_w{loop}')
            components.append(Line_Gaussian(
                amplitude=amp_w, dv=dv_w, sigma=sigma_w, wavec=self.wavec, 
                name=f'{self.name}: wind{loop}'))
        return components


class Line_MultiGauss_doublet(Fittable1DModel):
    '''
    Line doublet with multi-Gaussian model.

    Parameters
    ----------
    x : array like
        Wavelength, units: arbitrary.
    n_components : int
        The number of Gaussian components.
    amp_c0 : float
        The amplitude of the core component of the first line.
    amp_c1 : float
        The amplitude of the core component of the second line.
    dv_c : float
        The velocity shift from the central wavelength of the core component, 
        units: km/s.
    sigma_c : float
        The velocity dispersion of the core component, units: km/s.
    wavec0 : float
        The central wavelength of the first line, units: same as x.
    wavec1 : float
        The central wavelength of the second line, units: same as x.
    par_w : dict
        The parameters of the wind components.
            amp_w`i` : float
                The amplitude of the `i`th wind component, relative to the core amplitude.
            dv_w`i` : float
                The velocity of the `i`th wind component, relative to the core velocity, units: km/s.
            sigma_w`i` : float
                The velocity dispersion of the `i`th wind component, units: km/s.
    name : string
        The name of the line profile.
    **kwargs : dict
        Additional parameters like bounds, fixed, and meta.
    '''
    _param_names = ()
    
    def __init__(self, n_components=1, amp_c0=1, amp_c1=1, dv_c=0, sigma_c=200, wavec0=5000, wavec1=5000, par_w={}, name=None, **kwargs):
        '''
        '''
        assert isinstance(n_components, int), 'n_components must be an integer!'
        assert n_components > 0, 'n_components must be positive!'

        self.n_components = n_components

        self._param_names = ['amp_c0', 'amp_c1', 'dv_c', 'sigma_c', 'wavec0', 'wavec1']
        self._parameters_['amp_c0'] = Parameter(default=amp_c0, bounds=(0, None))
        self._parameters_['amp_c1'] = Parameter(default=amp_c1, bounds=(0, None))
        self._parameters_['dv_c'] = Parameter(default=dv_c, bounds=(-2000, 2000))
        self._parameters_['sigma_c'] = Parameter(default=sigma_c, bounds=(20, 10000))
        self._parameters_['wavec0'] = Parameter(default=wavec0, fixed=True)
        self._parameters_['wavec1'] = Parameter(default=wavec1, fixed=True)

        if n_components > 1:
            for loop in range(n_components - 1):
                pn_amp = f'amp_w{loop}'
                pn_dv = f'dv_w{loop}'
                pn_sigma = f'sigma_w{loop}'
                pv_amp = par_w.get(pn_amp, 0)
                pv_dv = par_w.get(pn_dv, 0)
                pv_sigma = par_w.get(pn_sigma, sigma_c)

                self._param_names.append(pn_amp)
                self._param_names.append(pn_dv)
                self._param_names.append(pn_sigma)

                self._parameters_[pn_amp] = Parameter(default=pv_amp, bounds=(0, 1))
                self._parameters_[pn_dv] = Parameter(default=pv_dv, bounds=(-2000, 2000))
                self._parameters_[pn_sigma] = Parameter(default=pv_sigma, bounds=(0, 10000))

        kwargs.update(par_w)
        super().__init__(amp_c0, amp_c1, dv_c, sigma_c, wavec0, wavec1, name=name, **kwargs)

    def evaluate(self, x, *params):
        '''
        Doublet line of the multi-component Gaussian model function.
        '''
        amp_c0, amp_c1, dv, sigma, wavec0, wavec1 = params[:6]

        # Calculate the central flux
        v0 = (x - wavec0) / wavec0 * ls_km  # convert to velocity (km/s)
        v1 = (x - wavec1) / wavec1 * ls_km  # convert to velocity (km/s)
        flux_c =  amp_c0 * np.exp(-0.5 * ((v0 - dv) / sigma)**2) + amp_c1 * np.exp(-0.5 * ((v1 - dv) / sigma)**2)

        if self.n_components == 1:
            return flux_c

        # Calculate additional Gaussian components
        n_add = self.n_components - 1
        n_pars = 3
        flux_w = sum([
            amp_c0 * params[6 + i * n_pars] * np.exp(-0.5 * ((v0 - dv - params[7 + i * n_pars]) / params[8 + i * n_pars])**2) + \
            amp_c1 * params[6 + i * n_pars] * np.exp(-0.5 * ((v1 - dv - params[7 + i * n_pars]) / params[8 + i * n_pars])**2)
            for i in range(n_add)]
        )

        return flux_c + flux_w

    @property
    def param_names(self):
        '''
        Coefficient names generated based on the model's number of components.

        Subclasses should implement this to return parameter names in the
        desired format.
        '''
        return self._param_names

    @property
    def subcomponents(self):
        '''
        Return the individual components of the model.
        '''
        if self.n_components == 1:
            return None

        components = [Line_Gaussian(amplitude=self.amp_c0, dv=self.dv_c, sigma=self.sigma_c, wavec=self.wavec0, name=f'{self.name}: core0'), 
                      Line_Gaussian(amplitude=self.amp_c1, dv=self.dv_c, sigma=self.sigma_c, wavec=self.wavec1, name=f'{self.name}: core1')]

        for loop in range(self.n_components - 1):
            amp_w = getattr(self, f'amp_w{loop}')
            dv_w = getattr(self, f'dv_w{loop}') + self.dv_c
            sigma_w = getattr(self, f'sigma_w{loop}')
            components.append(Line_Gaussian(amplitude=amp_w*self.amp_c0, dv=dv_w, sigma=sigma_w, wavec=self.wavec0, name=f'{self.name}: wind0{loop}'))
            components.append(Line_Gaussian(amplitude=amp_w*self.amp_c1, dv=dv_w, sigma=sigma_w, wavec=self.wavec1, name=f'{self.name}: wind1{loop}'))
        return components

    def gen_template(self, v):
        '''
        Generate the template of the single line profile.

        Parameters
        ----------
        v : array like
            The velocity array, units: km/s.
        
        Returns
        -------
        flux : array like
            The flux of the line profile.
        '''
        flux =  self.amp_c0 * np.exp(-0.5 * (v / self.sigma_c)**2)

        n_add = self.n_components - 1
        for i in range(n_add):
            amp_w = getattr(self, f'amp_w{i}')
            dv_w = getattr(self, f'dv_w{i}')
            sigma_w = getattr(self, f'sigma_w{i}')
            flux += amp_w * self.amp_c0 * np.exp(-0.5 * ((v - dv_w) / sigma_w)**2)

        return flux

# Tie parameters
class tier_line_h3(object):

    def __init__(self, name_fit, name_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref

    def __repr__(self):
        return "<Set the h3 of '{0}' the same as that of '{1}'>".format(self._name_fit, self._name_ref)

    def __call__(self, model):
        return model[self._name_ref].h3.value


class tier_line_h4(object):

    def __init__(self, name_fit, name_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref

    def __repr__(self):
        return "<Set the h4 of '{0}' the same as that of '{1}'>".format(self._name_fit, self._name_ref)

    def __call__(self, model):
        return model[self._name_ref].h4.value


class tier_line_ratio(object):

    def __init__(self, name_fit, name_ref, ratio=None, ratio_names=None):

        self._name_fit = name_fit
        self._name_ref = name_ref
        self._ratio = ratio
        self._ratio_names = ratio_names

        if ((self._ratio is None) and (self._ratio_names is None)):
            raise keyError('Need to provide ratio or _ratio_names!')
        elif ((self._ratio is not None) and (self._ratio_names is not None)):
            raise keyError('Cannot set both ratio and _ratio_names!')

    def __repr__(self):

        if self._ratio is not None:
            return "<Set the amplitude of '{0}' to 1/{1} that of '{2}'>".format(self._name_fit, self._ratio, self._name_ref)
        else:
            return "<Set the amplitude of '{0}' according to '{1}' x '{2[0]}'/'{2[1]}'>".format(self._name_fit, self._name_ref, self._ratio_names)

    def __call__(self, model):

        if self._ratio is not None:
            r = 1 / self._ratio
        else:
            r = model[self._ratio_names[0]].amplitude.value / (model[self._ratio_names[1]].amplitude.value+1.0e-16)

        return model[self._name_ref].amplitude.value * r


class tier_wind_dv(object):

    def __init__(self, names_fit, names_ref):
        '''
        Tie the velocity offset of the wind components.
        Parameters
        ----------
        names_fit : list
            The names of the fitted line profile, [wind, core].
        names_ref : list
            The names of the reference line profile, [wind, core].
        '''
        self._names_fit = names_fit
        self._names_ref = names_ref

    def __repr__(self):
        return "<Set the line dv('{0[0]}')-dv('{0[1]}') = dv('{1[0]}')-dv('{1[1]}')>".format(self._names_fit, self._names_ref)

    def __call__(self, model):
        dv_ref = model[self._names_fit[1]].dv.value
        ddv_ref = model[self._names_ref[0]].dv.value - model[self._names_ref[1]].dv.value
        return dv_ref + ddv_ref


class tier_abs_dv(object):

    def __init__(self, name_fit, name_ref):
        '''
        Tie the velocity offset of the line.
        Parameters
        ----------
        name_fit : str
            The name of the component to be fitted.
        name_ref : str
            The name of the component to be tied to.
        '''
        self._name_fit = name_fit
        self._name_ref = name_ref

    def __repr__(self):
        return "<Set the velocity offset of '{0}' to that of '{1}'>".format(self._name_fit, self._name_ref)

    def __call__(self, model):
        return model[self._name_ref].dv.value


class tier_line_sigma(object):

    def __init__(self, name_fit, name_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref

    def __repr__(self):
        return "<Set the sigma of '{0}' the same as that of '{1}'>".format(self._name_fit, self._name_ref)

    def __call__(self, model):
        return model[self._name_ref].sigma.value


class extinction_ccm89(Fittable1DModel):
    '''
    The extinction model of Cardelli et al. (1989).
    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V-band wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).
    Returns
    -------
    f : array like
        The fraction of out emitting flux.
    '''
    a_v = Parameter(default=0, bounds=(0, None))
    r_v = Parameter(default=3.1, fixed=True)

    @staticmethod
    def evaluate(x, a_v, r_v):
        """
        The extinction model function (Cardelli et al. 1989).
        """
        f =10**(-0.4 * extinction.ccm89(x, a_v, r_v))
        return f


def find_line_peak(model, x0):
    '''
    Find the peak wavelength and flux of the model line profile.
    Parameters
    ----------
    model : Astropy model
        The model of the line profile. It should be all positive.
    x0 : float
        The initial guess of the wavelength.
    Returns
    -------
    w_peak, f_peak : floats
        The wavelength and flux of the peak of the line profile.
    '''
    func = lambda x: -1 * model(x)
    res = minimize(func, x0=x0)
    w_peak = res.x[0]
    try:
        f_peak = model(w_peak)[0]
    except:
        f_peak = model(w_peak)
    return w_peak, f_peak


def line_fwhm(model, x0, x1, x0_limit=None, x1_limit=None, fwhm_disp=None):
    '''
    Calculate the FWHM of the line profile.
    Parameters
    ----------
    model : Astropy model
        The model of the line profile. It should be all positive.
    x0, x1 : float
        The initial guesses of the wavelengths on the left and right sides.
    x0_limit, x1_limit (optional) : floats
        The left and right boundaries of the search.
    fwhm_disp (optional) : float
        The instrumental dispersion that should be removed from the FWHM, units
        following the wavelength.
    Returns
    -------
    fwhm : float
        The FWHM of the line, units: km/s.
    w_l, w_r : floats
        The wavelength and flux of the peak of the line profile.
    w_peak : float
        The wavelength of the line peak.
    '''
    xc = (x0 + x1) / 2
    w_peak, f_peak = find_line_peak(model, xc)
    f_half = f_peak / 2

    func = lambda x: np.abs(model(x) - f_half)

    if x0_limit is not None:
        bounds = ((x0_limit, w_peak),)
    else:
        bounds = None
    res_l = minimize(func, x0=x0, bounds=bounds)

    if x1_limit is not None:
        bounds = ((w_peak, x1_limit),)
    else:
        bounds = None
    res_r = minimize(func, x0=x1, bounds=bounds)
    w_l = res_l.x[0]
    w_r = res_r.x[0]

    fwhm_w = (w_r - w_l)
    if fwhm_disp is not None:  # Correct for instrumental dispersion
        fwhm_w = np.sqrt(fwhm_w**2 - fwhm_disp**2)

    fwhm = fwhm_w / w_peak * ls_km
    return fwhm, w_l, w_r, w_peak


# Deprecated

wave_vac_OIII_5007 = line_wave_dict['OIII_5007']
wave_vac_OIII_4959 = line_wave_dict['OIII_4959']

def gen_o3doublet_gauss(ngauss, amplitude, dv, sigma, bounds=None, amplitude_ratio=2.98, component_n=1):
    '''
    Generate the [OIII] 4959, 5007 doublet.
    Parameters
    ----------
    ngauss : int
        The number of Gaussian profile used for each of the [OIII] line.
    amplitude : float or list
        The amplitude(s) of the Gaussian profile(s).
    dv : float or list
        The velocity offset(s) of the Gaussian profile(s).
    sigma : float or list
        The velocity dispersion(s) of the Gaussian profile(s).
    bounds (optional) : dict
        The boundaries of the profile parameters, same for the two lines.
        amplitude_bounds : tuple or list of tuples
            The bounds of the amplitude.
        dv_bounds : tuple or list of tuples
            The bounds of the dv.
        sigma_bounds : tuple or list of tuples
            The bounds of the sigma.
    amplitude_ratio : float (default: 2.98; Storey & Zeippen 2000)
        The amplitude ratio of [OIII]5007 over [OIII]4959.
    Returns
    -------
    nl_o3 : astropy.modeling.CompoundModel
        The model with [OIII]4959, 5007 doublet.
    '''
    bounds = {} if bounds is None else bounds
    if bounds.get('amplitude_bounds', None) is None:
        bounds['amplitude_bounds'] = (0, None)
    if bounds.get('dv_bounds', None) is None:
        bounds['dv_bounds'] = (-2000, 2000)
    if bounds.get('sigma_bounds', None) is None:
        bounds['sigma_bounds'] = (0, 2000)
        
    if component_n == 1:
        o31 = '[OIII]5007'
        o32 = '[OIII]4959'
    else:
        o31 = '[OIII]5007_{}'.format(component_n)
        o32 = '[OIII]4959_{}'.format(component_n)
    
    nl_o31 = get_line_multigaussian(n=ngauss, wavec=wave_vac_OIII_5007, line_name=o31, amplitude=list(np.full(ngauss, amplitude)), dv=list(np.full(ngauss, dv)), sigma=list(np.full(ngauss, sigma)), **bounds)
    nl_o32 = get_line_multigaussian(n=ngauss, wavec=wave_vac_OIII_4959, line_name=o32, amplitude=list(np.full(ngauss, amplitude)), dv=list(np.full(ngauss, dv)), sigma=list(np.full(ngauss, sigma)), **bounds)

    # Tie the profiles
    nl_o3 = fix_profile_multigauss(nl_o31 + nl_o32, o31, o32)

    # Tie the amplitudes
    if ngauss == 1 and component_n == 1:
        name_o31 = '[OIII]5007'
        name_o32 = '[OIII]4959'
    elif component_n != 1:
        name_o31 = '[OIII]5007_{}'.format(component_n)
        name_o32 = '[OIII]4959_{}'.format(component_n)   
    else:
        name_o31 = '[OIII]5007: 0'
        name_o32 = '[OIII]4959: 0'

    nl_o3[name_o32].amplitude.tied = tier_line_ratio(name_o32, name_o31, ratio=amplitude_ratio)
    nl_o3[name_o32].amplitude.value = nl_o3[name_o32].amplitude.tied(nl_o3)

    # Tie the dv
    nl_o3[name_o32].dv.tied = tier_abs_dv(name_o32, name_o31)
    nl_o3[name_o32].dv.value = nl_o3[name_o32].dv.tied(nl_o3)

    return nl_o3


def gen_s2doublet_gauss(ngauss, amplitude, dv, sigma, bounds=None, component_n=1):
    '''
    Generate the [SII] 6718, 6733 doublet.
    Parameters
    ----------
    ngauss : int
        The number of Gaussian profile used for each of the [SII] line.
    amplitude : float or list
        The amplitude(s) of the Gaussian profile(s).
    dv : float or list
        The velocity offset(s) of the Gaussian profile(s).
    sigma : float or list
        The velocity dispersion(s) of the Gaussian profile(s).
    bounds (optional) : dict
        The boundaries of the profile parameters, same for the two lines.
        amplitude_bounds : tuple or list of tuples
            The bounds of the amplitude.
        dv_bounds : tuple or list of tuples
            The bounds of the dv.
        sigma_bounds : tuple or list of tuples
            The bounds of the sigma.
    Returns
    -------
    nl_s2 : astropy.modeling.CompoundModel
        The model with [SII] 6718, 6733 doublet.
    '''
    bounds = {} if bounds is None else bounds
    if bounds.get('amplitude_bounds', None) is None:
        bounds['amplitude_bounds'] = (0, None)
    if bounds.get('dv_bounds', None) is None:
        bounds['dv_bounds'] = (-2000, 2000)
    if bounds.get('sigma_bounds', None) is None:
        bounds['sigma_bounds'] = (0, 1000)
    
    if component_n == 1:
        s21 = '[SII]6718'
        s22 = '[SII]6733'
    else:
        s21 = '[SII]6718_{}'.format(component_n)
        s22 = '[SII]6733_{}'.format(component_n)
    
    nl_s21 = get_line_multigaussian(n=ngauss, wavec=wave_vac_SII_6718, line_name=s21, amplitude=list(np.full(ngauss, amplitude)), dv=list(np.full(ngauss, dv)), sigma=list(np.full(ngauss, sigma)), **bounds)
    nl_s22 = get_line_multigaussian(n=ngauss, wavec=wave_vac_SII_6733, line_name=s22, amplitude=list(np.full(ngauss, amplitude)), dv=list(np.full(ngauss, dv)), sigma=list(np.full(ngauss, sigma)), **bounds)

    # Tie the profiles
    nl_s2 = fix_profile_multigauss(nl_s21 + nl_s22, s21, s22)

    # Tie the amplitudes
    if ngauss == 1 and component_n == 1:
        name_s21 = '[SII]6718'
        name_s22 = '[SII]6733'
    elif component_n != 1:
        name_s21 = '[SII]6718_{}'.format(component_n)
        name_s22 = '[SII]6733_{}'.format(component_n)   
    else:
        name_s21 = '[SII]6718: 0'
        name_s22 = '[SII]6733: 0'

    # Tie the dv
    nl_s2[name_s22].dv.tied = tier_abs_dv(name_s22, name_s21)
    nl_s2[name_s22].dv.value = nl_s2[name_s22].dv.tied(nl_s2)

    return nl_s2


def gen_o3doublet_gausshermite(amplitude=1, dv=0, sigma=200, h3=0, h4=0, bounds=None, amplitude_ratio=2.98,
                               label=None, component_n=1):
    '''
    Generate the [OIII] 4959, 5007 doublet with Gauss-Hermite function.
    Parameters
    ----------
    amplitude : float or list
        The amplitude of the [OIII]5007 line.
    dv : float or list
        The velocity offset of the two lines.
    sigma : float or list
        The velocity dispersion of the two lines.
    bounds (optional) : dict
        The boundaries of the profile parameters, same for the two lines.
    amplitude_ratio : float (default: 2.98; Storey & Zeippen 2000)
        The amplitude ratio of [OIII]5007 over [OIII]4959.
    Returns
    -------
    nl_o3 : astropy.modeling.CompoundModel
        The model with [OIII]4959, 5007 doublet.
    '''
    bounds = {} if bounds is None else bounds
    if bounds.get('amplitude', None) is None:
        bounds['amplitude'] = (0, None)
    if bounds.get('dv', None) is None:
        bounds['dv'] = (-2000, 2000)
    if bounds.get('sigma', None) is None:
        bounds['sigma'] = (0, 1000)
    if bounds.get('h3', None) is None:
        bounds['h3'] = (-0.4, 0.4)
    if bounds.get('h4', None) is None:
        bounds['h4'] = (-0.4, 0.4)

    if label is None and component_n ==1:
        name_o31 = '[OIII]5007'
        name_o32 = '[OIII]4959'
    elif component_n != 1:
        name_o31 = '[OIII]5007_{}'.format(component_n)
        name_o32 = '[OIII]4959_{}'.format(component_n)  
    else:
        name_o31 = '[OIII]5007:{0}'.format(label)
        name_o32 = '[OIII]4959:{0}'.format(label)
    
    nl_o31 = Line_GaussHermite(amplitude=amplitude, dv=dv, sigma=sigma, h3=h3, h4=h4, wavec=wave_vac_OIII_5007, name=name_o31, bounds=bounds)
    nl_o32 = Line_GaussHermite(amplitude=amplitude, dv=dv, sigma=sigma, h3=h3, h4=h4, wavec=wave_vac_OIII_4959, name=name_o32, bounds=bounds)

    # Tie the profiles
    nl_o3 = fix_profile_gausshermite(nl_o31 + nl_o32, name_o31, name_o32)

    # Tie the amplitudes
    nl_o3[name_o32].amplitude.tied = tier_line_ratio(name_o32, name_o31, ratio=amplitude_ratio)
    nl_o3[name_o32].amplitude.value = nl_o3[name_o32].amplitude.tied(nl_o3)

    # Tie the dv
    nl_o3[name_o32].dv.tied = tier_abs_dv(name_o32, name_o31)
    nl_o3[name_o32].dv.value = nl_o3[name_o32].dv.tied(nl_o3)

    return nl_o3


def gen_s2doublet_gausshermite(amplitude=1, dv=0, sigma=200, h3=0, h4=0, bounds=None, label=None, component_n=1):
    '''
    Generate the [SII]6718, 6733 doublet with Gauss-Hermite function.

    Parameters
    ----------
    amplitude : float or list
        The amplitude of the [SII]6733 line.
    dv : float or list
        The velocity offset of the two lines.
    sigma : float or list
        The velocity dispersion of the two lines.
    bounds (optional) : dict
        The boundaries of the profile parameters, same for the two lines.

    Returns
    -------
    nl_s2 : astropy.modeling.CompoundModel
        The model with [SII]6718, 6733 doublet.
    '''
    bounds = {} if bounds is None else bounds
    if bounds.get('amplitude', None) is None:
        bounds['amplitude'] = (0, None)
    if bounds.get('dv', None) is None:
        bounds['dv'] = (-2000, 2000)
    if bounds.get('sigma', None) is None:
        bounds['sigma'] = (0, 1000)
    if bounds.get('h3', None) is None:
        bounds['h3'] = (-0.4, 0.4)
    if bounds.get('h4', None) is None:
        bounds['h4'] = (-0.4, 0.4)

    if label is None and component_n ==1:
        name_s21 = '[SII]6733'
        name_s22 = '[SII]6718'
    elif component_n != 1:
        name_s21 = '[SII]6718_{}'.format(component_n)
        name_s22 = '[SII]6733_{}'.format(component_n)  
    else:
        name_s21 = '[SII]6733:{0}'.format(label)
        name_s22 = '[SII]6718:{0}'.format(label)
    
    nl_s21 = Line_GaussHermite(amplitude=amplitude, dv=dv, sigma=sigma, h3=h3, h4=h4, wavec=wave_vac_SII_6733, name=name_s21, bounds=bounds)
    nl_s22 = Line_GaussHermite(amplitude=amplitude, dv=dv, sigma=sigma, h3=h3, h4=h4, wavec=wave_vac_SII_6718, name=name_s22, bounds=bounds)

    # Tie the profiles
    nl_s2 = fix_profile_gausshermite(nl_s21 + nl_s22, name_s21, name_s22)

    # Tie the dv
    nl_s2[name_s22].dv.tied = tier_abs_dv(name_s22, name_s21)
    nl_s2[name_s22].dv.value = nl_s2[name_s22].dv.tied(nl_s2)

    return nl_s2


def fix_profile_multigauss(model, name_ref, name_fix):
    '''
    Fix the one line profile to the other.
    Parameters
    ----------
    model : astropy.modeling.CompoundModel
        The model that consists two sets of line profiles.
    name_ref : str
        The name of the reference line.
    name_fix : str
        The name of the line to be fixed the profile.
    '''
    assert model.n_submodels > 1, 'There are not additional components to fix!'

    ncomp_ref = 0
    ncomp_fix = 0
    for n in model.submodel_names:
        if name_ref in n.split(': '):
            ncomp_ref += 1
        elif name_fix in n.split(': '):
            ncomp_fix += 1

    #print('Find {0} for {1} and {2} for {3}'.format(ncomp_ref, name_ref, ncomp_fix, name_fix))

    if ncomp_ref == 0:
        raise KeyError('The model does not consist {0}'.format(name_ref))
    elif ncomp_fix == 0:
        raise KeyError('The model does not consist {0}'.format(name_fix))
    elif ncomp_ref != ncomp_fix:
        raise KeyError('The model components does not match ({0}, {1})!'.format(ncomp_ref, ncomp_fix))

    name_ref_0 = '{0}: 0'.format(name_ref)
    name_fix_0 = '{0}: 0'.format(name_fix)

    # Fix amplitude -- all respect to the first component
    if ncomp_ref > 1:
        for n in range(ncomp_ref - 1):
            # Set the tier
            name_ref_n = '{0}: {1}'.format(name_ref, n+1)
            name_fix_n = '{0}: {1}'.format(name_fix, n+1)
            model[name_fix_n].amplitude.tied = tier_line_ratio(name_fix_n, name_ref_n, ratio_names=[name_fix_0, name_ref_0])
            # Run it
            model[name_fix_n].amplitude.value = model[name_fix_n].amplitude.tied(model)

    # Fix dv -- all respect to the first component
    if ncomp_ref > 1:
        for n in range(ncomp_ref - 1):
            # Set the tier
            names_ref_n = ['{0}: {1}'.format(name_ref, n+1), '{0}: 0'.format(name_ref)]
            names_fix_n = ['{0}: {1}'.format(name_fix, n+1), '{0}: 0'.format(name_fix)]
            model[names_fix_n[0]].dv.tied = tier_wind_dv(names_fix_n, names_ref_n)
            # Run it
            model[names_fix_n[0]].dv.value = model[names_fix_n[0]].dv.tied(model)

    # Fix sigma
    if ncomp_ref == 1:
        model[name_fix].sigma.tied = tier_line_sigma(name_fix, name_ref)
        model[name_fix].sigma.value = model[name_fix].sigma.tied(model)
    else:
        for n in range(ncomp_ref):
            # Set the tier
            name_ref_n = '{0}: {1}'.format(name_ref, n)
            name_fix_n = '{0}: {1}'.format(name_fix, n)
            model[name_fix_n].sigma.tied = tier_line_sigma(name_fix_n, name_ref_n)
            # Run it
            model[name_fix_n].sigma.value = model[name_fix_n].sigma.tied(model)

    return model


def fix_profile_gausshermite(model, name_ref, name_fix):
    '''
    Fix the one line profile to the other.
    Parameters
    ----------
    model : astropy.modeling.CompoundModel
        The model that consists at least two line profiles.
    name_ref : str
        The name of the reference line.
    name_fix : str
        The name of the line to be fixed the profile.
    '''
    assert model.n_submodels > 1, 'There are not additional components to fix!'

    # Tie dv
    model[name_fix].dv.tied = tier_abs_dv(name_fix, name_ref)
    model[name_fix].dv.value = model[name_fix].dv.tied(model)

    # Tie sigma
    model[name_fix].sigma.tied = tier_line_sigma(name_fix, name_ref)
    model[name_fix].sigma.value = model[name_fix].sigma.tied(model)

    # Tie h3
    model[name_fix].h3.tied = tier_line_h3(name_fix, name_ref)
    model[name_fix].h3.value = model[name_fix].h3.tied(model)

    # Tie h4
    model[name_fix].h4.tied = tier_line_h4(name_fix, name_ref)
    model[name_fix].h4.value = model[name_fix].h4.tied(model)

    return model


def get_line_multigaussian(n=1, wavec=5000, line_name='Line', **kwargs):
    '''
    Get a multigaussian line model.
    Parameters
    ----------
    n : int
        Number of Gaussian components.
    wavec : float
        The central wavelength of the line.
    line_name : str
        The name of the line. Each component has an additional index,
        starting from 0, e.g., "Line 0".
    amplitude (optional) : list
        The value of the line amplitude.
    dv (optional) : list
        The velocity offset of the line center, units: km/s.
    sigma (optional) : list
        The value of the line sigma.
    amplitude_bounds (optional) : tuple or list
        The bound(s) of the line amplitude.
    dv_bounds (optional) : tuple or list
        The bound(s) of the velocity offset.
    sigma_bounds (optional) : tuple or list
        The bound(s) of the line sigma.
    Returns
    -------
    model : The sum of Line_Gaussian.
    '''
    assert isinstance(n, int) & (n > 0), 'We only accept n as >0 integer!'
    parList = ['amplitude', 'dv', 'sigma']
    bouList = ['amplitude_bounds', 'dv_bounds', 'sigma_bounds']

    # Check the parameters
    for kw in kwargs:
        if kw not in parList + bouList:
            raise KeyError('{0} is not recognized!'.format(kw))

    # Generate the model
    if n > 1:
        model = Line_Gaussian(wavec=wavec, name='{0}: 0'.format(line_name), fixed=dict(wavec=True))
        for loop in range(n-1):
            model += Line_Gaussian(wavec=wavec, name='{0}: {1}'.format(line_name, loop+1), fixed=dict(wavec=True))
        model.name = '{0}'.format(line_name)
    else:
        model = Line_Gaussian(wavec=wavec, name='{0}'.format(line_name), fixed=dict(wavec=True))

    # Set the parameters
    for kw in parList:
        kv = kwargs.get(kw, None)

        if kv is not None:
            #print(kw,kv)
            assert isinstance(kv, list), 'We only accept {0} as a list!'.format(kw)
            assert len(kv) <= n, 'The length of {0} is larger than n!'.format(kw)

            if n > 1:
                for loop, v in enumerate(kv):
                    model[loop].__setattr__(kw, v)
            else:
                model.__setattr__(kw, kv[0])

    # Set the bounds of the parameters
    for kw in bouList:
        kv = kwargs.get(kw, None)
        pn, pa = kw.split('_')

        if isinstance(kv, tuple):
            assert len(kv) == 2, 'The {0} should contain 2 elements!'.format(kw)

            if n > 1:
                for loop in range(n):
                    p = model[loop].__getattribute__(pn)
                    p.__setattr__(pa, kv)
            else:
                p = model.__getattribute__(pn)
                p.__setattr__(pa, kv)

        elif isinstance(kv, list):
            assert len(kv) <= n, 'The length of {0} is larger than n!'.format(kw)

            if n > 1:
                for loop, bou in enumerate(kv):
                    p = model[loop].__getattribute__(pn)
                    p.__setattr__(pa, bou)
            else:
                p = model.__getattribute__(pn)
                p.__setattr__(pa, kv[0])

        elif kv is not None:
            raise ValueError('Cannot recognize {0} ({1})'.format(kw, kv))

    return model