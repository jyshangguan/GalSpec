import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from astropy.modeling import CompoundModel

try:
    import emcee
except ImportError:
    emcee = None

try:
    import corner
except ImportError:
    corner = None


__all__ = ['get_free_params', 'MCMC_Fit']


class MCMC_Fit:
    def __init__(self, model, wave_use, flux_use, ferr, log_prior_func=None, nwalkers=50, nsteps=6000, step_initial=0):
        """
        Initialize the MCMC_Fit class.
        """
        if emcee is None:
            raise ImportError("emcee package is not installed. Please install it to use MCMC_Fit class.")

        if corner is None:
            raise ImportError("corner package is not installed. Please install it to use MCMC_Fit class.")

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
        
        self.flat_samples = self.sampler.get_chain(flat=True)
        self.log_prob = self.sampler.get_log_prob(flat=True)
        return self.flat_samples, self.model, self.param_names
    
    def get_best_fit(self, discard=0):
        """Set the model parameters to the best fit values."""
        # Get best fit parameters
        if discard == 0:
            flat_samples = self.flat_samples
            log_prob = self.log_prob
        else:
            flat_samples = self.sampler.get_chain(flat=True, discard=discard)
            log_prob = self.sampler.get_log_prob(flat=True, discard=discard)
        self.theta_best = flat_samples[np.argmax(log_prob)]
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
        if discard == 0:
            flat_samples = self.flat_samples
        else:
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

    def load_samples(self, filename):
        """Load MCMC samples from a file."""
        data = np.load(filename)

        if not (list(self.param_names) == list(data['param_names'])):
            raise ValueError("Parameter names in the file do not match the current model.")

        self.flat_samples = data['samples']
        self.log_prob = data['log_prob']

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

    def plot_corner(self, discard=0, **kwargs):
        """Plot the corner plot of the MCMC samples."""
        if discard == 0:
            flat_samples = self.flat_samples
        else:
            flat_samples = self.sampler.get_chain(flat=True, discard=discard)
        return corner.corner(flat_samples, labels=self.param_names, truths=self.theta_best, **kwargs)

    def pos_posterior(self, nsamples=1, discard=0):
        """Get the posterior samples after discarding burn-in."""
        if discard == 0:
            samples = self.flat_samples
        else:
            samples = self.sampler.get_chain(flat=True, discard=discard)
        
        indices = np.random.choice(samples.shape[0], size=nsamples, replace=False)
        return samples[indices]

    def save_samples(self, filename):
        """Save the MCMC samples to a file."""
        np.savez(filename, samples=self.flat_samples, log_prob=self.log_prob, param_names=self.param_names)

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


