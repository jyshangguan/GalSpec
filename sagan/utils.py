import os
os.environ["OMP_NUM_THREADS"] = "1"

from sys import platform
import numpy as np
import matplotlib.pyplot as plt
from .constants import ls_km
from scipy.ndimage import gaussian_filter1d
from spectres import spectres
from sfdmap2 import sfdmap
from PyAstronomy import pyasl
from copy import deepcopy
import emcee
import corner

from astropy.modeling import CompoundModel

__all__ = ['package_path', 'splitter', 'line_wave_dict', 'line_label_dict',
           'wave_to_velocity', 'velocity_to_wave', 
           'down_spectres', 'get_free_params', 'ReadSpectrum', 'MCMC_Fit']


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
    'SII_6730': 6730.810,
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
    'SII_6718': r'[S II] 6718',
    'SII_6733': r'[S II] 6733',
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
    if wavec is None:
        wavec = np.mean(wave)

    if dw is None:
        dw = np.median(np.diff(wave))

    sigma = wavec * np.sqrt(R_new**-2 - R_org**-2) / 2.3548 / dw
    flux_new = gaussian_filter1d(flux, sigma)

    if wave_new is not None:
        flux_new = spectres(wave_new, wave, flux_new, spec_errs=spec_errs, fill=fill, verbose=verbose)

    return flux_new


class MCMC_Fit:
    def __init__(self, model, wave_use, flux_use, ferr, log_prior_func=None, nwalkers=50, nsteps=6000, step_initial=0):
        self.model = deepcopy(model)
        self.wave_use = wave_use
        self.flux_use = flux_use
        self.ferr = ferr
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.step_initial = step_initial

        self.check_input_model()

        # Set bounds for parameters
        self.index_free_params()
        self.set_param_bounds()

        # Initialize parameter values for walkers
        self.theta_initial = self.get_initial_theta()
        self.ndim = len(self.theta_initial)
        self.pos = self.theta_initial + 1e-8 * np.random.randn(self.nwalkers, self.ndim)
        self.theta_best = None

        # Use the provided log_prior_func or the default self.log_prior
        if log_prior_func is None:
            self.log_prior_func = self.log_prior
        else:
            self.log_prior_func = log_prior_func
        
        print('MCMC_Fit initialized with {} free parameters.'.format(self.ndim))

    def check_convergence(self):
        """Check the convergence of the MCMC chains."""
        chain = self.sampler.get_chain()

        # Estimate autocorrelation time for each parameter
        try:
            tau = self.sampler.get_autocorr_time()
        except emcee.autocorr.AutocorrError as e:
            print("AutocorrError:", e)
            print("Chain is probably too short to reliably estimate tau.")
            tau = None

        if tau is not None:
            tau_max = np.max(tau)
            nsteps = chain.shape[0]
        
            print("tau per parameter:", tau)
            print("max tau:", tau_max)
            print("nsteps:", nsteps)
            print("nsteps / tau_max =", nsteps / tau_max)

        return chain, tau
 
    def check_input_model(self):
        '''Check the input model'''
        assert isinstance(self.model, CompoundModel), "Input model must be an instance of CompoundModel."

        for loop, m in enumerate(self.model):
            if m.name is None:
                m.name = self.model.submodel_names[loop]

    def fit(self):
        """Perform MCMC fitting."""
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, args=())

        if self.step_initial > 0:
            pos, prob, state = self.sampler.run_mcmc(self.pos, self.step_initial, progress=True)
            self.samples_initial = self.sampler.get_chain(flat=True)
            self.sampler.reset()
            self.sampler.run_mcmc(pos, self.nsteps, progress=True)
        else:
            self.sampler.run_mcmc(self.pos, self.nsteps, progress=True)
        
        flat_samples = self.sampler.get_chain(flat=True)
        return flat_samples, self.model, self.param_names
    
    def get_best_fit(self, discard=0):
        """Set the model parameters to the best fit values."""
        flat_samples = self.sampler.get_chain(flat=True, discard=discard)

        # Get best fit parameters
        self.theta_best = flat_samples[np.argmax(self.sampler.get_log_prob(flat=True, discard=discard))]
        self.model = self.set_model_params(self.theta_best)

        return self.model, self.param_names, self.theta_best

    def get_free_params(self):
        """Get the free parameters for the model."""
        free_param_names = []
        for param_name in self.model.param_names:
            param = getattr(self.model, param_name)
            if not param.fixed and not param.tied:
                free_param_names.append(param_name)
        return free_param_names

    def get_initial_theta(self):
        """Get the initial parameter values for the MCMC walkers."""
        param_values = {param_name: getattr(self.model, param_name).value for param_name in self.param_names}
        return [param_values[key] for key in self.param_names]

    def get_param_index(self, model_name, param_name):
        """Get the index of a parameter in theta."""
        full_param_name = f"{model_name}.{param_name}"
        return self.param_map[full_param_name][0]
    
    def get_param_samples(self, model_name, param_name, discard=0):
        """Get the samples of a parameter from the MCMC chains."""
        flat_samples = self.sampler.get_chain(flat=True, discard=discard)
        param_index = self.get_param_index(model_name, param_name)
        return flat_samples[:, param_index]

    def index_free_params(self):
        """Get the indices of free parameters in the model."""
        self.param_names = self.get_free_params()

        self.full_param_names = []
        for mn in self.model.submodel_names:
            submodel = self.model[mn]
            for pn in submodel.param_names:
                param = getattr(submodel, pn)
                if not param.fixed and not param.tied:
                    self.full_param_names.append(f"{mn}.{pn}")

        self.param_map = dict(zip(self.full_param_names, [(ii, pn) for ii, pn in enumerate(self.param_names)]))

    def log_likelihood(self, theta):
        """Calculate the log likelihood."""
        model = self.set_model_params(theta)
 
        model_flux = model(self.wave_use)
        return -0.5 * np.sum(((self.flux_use - model_flux) / self.ferr) ** 2)

    def log_prior(self, theta):
        """Calculate the log prior probability."""
        for i, param_name in enumerate(self.param_names):
            param = getattr(self.model, param_name)
            bound = param.bounds
            lower_bound, upper_bound = bound

            # Handle cases where bounds are None
            if lower_bound is not None and theta[i] < lower_bound:
                return -np.inf
            if upper_bound is not None and theta[i] > upper_bound:
                return -np.inf

        return 0.0

    def log_probability(self, theta):
        """Calculate the log probability."""
        # Call the log_prior_func, which is either the default or a custom function
        lp = self.log_prior_func(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def plot_chain(self, initial=False):
        """Plot the MCMC chains."""
        fig, axes = plt.subplots(len(self.param_names), figsize=(10, 2*self.ndim), sharex=True)
        if initial:
            if hasattr(self, 'samples_initial'):
                samples = self.samples_initial
            else:
                raise ValueError("Initial samples not found. Please run fit() with step_initial > 0 first.")
        else:
            samples = self.sampler.get_chain()

        for i, param_name in enumerate(self.param_names):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_ylabel(param_name)

        axes[-1].set_xlabel("Step number")
        plt.tight_layout()
        plt.show()

    def plot_corner(self, discard=0):
        """Plot the corner plot of the MCMC samples."""
        flat_samples = self.sampler.get_chain(flat=True, discard=discard)
        fig = corner.corner(flat_samples, labels=self.param_names, truths=self.theta_best)
        plt.show()

    def set_model_params(self, theta):
        """Set the model parameters from theta."""
        model = self.model

        for i, param_name in enumerate(self.param_names):
            setattr(model, param_name, theta[i])

        for param_name in model.param_names:
            param = getattr(model, param_name)
            if param.tied:
                param.value = param.tied(model)
        
        return model
    
    def set_param_bounds(self):
        """Set the bounds for the parameters."""
        for param_name in self.param_names:
            param = getattr(self.model, param_name)
            lower_bound, upper_bound = param.bounds
            if upper_bound is None:
                upper_bound = 10000
            if lower_bound is None:
                lower_bound = -100
            param.bounds = (lower_bound, upper_bound)


def get_free_params(model):
    """Get the free parameters for the model."""
    free_param_names = []
    for param_name in model.param_names:
        param = getattr(model, param_name)
        if not param.fixed and not param.tied:
            free_param_names.append(param_name)
    return free_param_names


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