"""
Dynesty Fitting Module for SAGAN
================================

This module implements nested sampling fitting using dynesty.

Key differences from MCMC (emcee):
- Requires explicit bounds for ALL parameters (no None values)
- Uses prior transform to map unit cube to parameter space
- Provides Bayesian evidence automatically
- No burn-in period needed

Author: [Your Name]
Date: 2025-01-XX
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import warnings
from copy import deepcopy
from astropy.modeling import CompoundModel

try:
    import dynesty
    from dynesty import utils as dyfunc
except ImportError:
    dynesty = None
    dyfunc = None

try:
    import corner
except ImportError:
    corner = None


__all__ = ['Dynesty_Fit']


class Dynesty_Fit:
    """
    Nested sampling fitting using dynesty.

    Parameters
    ----------
    model : CompoundModel
        Astropy compound model to fit
    wave_use : array
        Wavelength array for fitting
    flux_use : array
        Flux array for fitting
    ferr : array
        Flux error array for fitting
    bounds_dict : dict, optional
        Custom bounds for specific parameters: {param_name: (lower, upper)}
        Overrides all default bounds
    default_bounds : dict, optional
        Default bounds for parameter types
        If None, uses built-in defaults
    sample_method : str, optional
        Sampling method for dynesty ('unif', 'rwalk', 'rslice', 'hslice', etc.)
        Default: 'rwalk'
    nlive : int, optional
        Number of live points
        Default: 500
    bound : str, optional
        Bounding method ('none', 'single', 'multi', 'balls', 'cubes')
        Default: 'multi'
    rstate : np.random.Generator, optional
        Random state for reproducibility
    """

    # Default bounds by parameter type
    DEFAULT_BOUNDS = {
        'amplitude': (1e-6, 1e6),
        'dv': (-10000, 10000),
        'sigma': (10, 20000),
        'h3': (-0.5, 0.5),
        'h4': (-0.5, 0.5),
        'logtau0': (-3, 3),
        'logtau1': (-3, 3),
        'Cf': (0, 1),
        'wavec': (0.99, 1.01),  # Will be multiplied by initial value
        'continuum': (-10, 10),
        'velscale': (10, 1000),
        'amp_c': (1e-6, 1e6),
        'dv_c': (-10000, 10000),
        'sigma_c': (10, 20000),
        'amp_w': (0, 1),  # Relative amplitude
        'dv_w': (-8000, 8000),
        'sigma_w': (0, 10000),
        'amp_c0': (1e-6, 1e6),
        'amp_c1': (1e-6, 1e6),
    }

    def __init__(self, model, wave_use, flux_use, ferr,
                 bounds_dict=None, default_bounds=None,
                 sample_method='rwalk', nlive=500, bound='multi',
                 rstate=None):

        if dynesty is None:
            raise ImportError("dynesty package is not installed. "
                            "Please install it to use Dynesty_Fit class.")

        if corner is None:
            warnings.warn("corner package is not installed. "
                         "Plotting functionality will be limited.")

        self.model = deepcopy(model)
        self.wave_use = wave_use
        self.flux_use = flux_use
        self.ferr = ferr
        self.sample_method = sample_method
        self.nlive = nlive
        self.bound = bound
        self.rstate = rstate

        # Store custom bounds
        self.bounds_dict = bounds_dict or {}
        self.default_bounds = default_bounds or self.DEFAULT_BOUNDS.copy()
        self.custom_priors = {}  # For custom prior transforms

        # Check input model
        self.check_input_model()

        # Set up parameters
        self.index_free_params()
        self.param_bounds = self.get_param_bounds()

        # Get initial parameters
        self.theta_initial = self.get_initial_theta()
        self.ndim = len(self.theta_initial)

        # Results storage
        self.results = None
        self.samples_equal_weight = None
        self.log_evidence = None
        self.log_evidence_err = None
        self.theta_best = None

        print(f'Dynesty_Fit initialized with {self.ndim} free parameters.')
        self._print_bounds_summary()

        # Check if data range exceeds amplitude bounds
        data_warnings = self._check_data_range_for_amplitude()
        if data_warnings:
            print("\nData Range Checks:")
            print("-" * 90)
            for warning in data_warnings:
                print(warning)
            print("-" * 90)
            print()

    def check_input_model(self):
        """Check the input model."""
        assert isinstance(self.model, CompoundModel), \
            "Input model must be an instance of CompoundModel."

        for loop, m in enumerate(self.model):
            if m.name is None:
                m.name = self.model.submodel_names[loop]

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

        self.param_map = dict(zip(self.full_param_names,
                                 [(ii, pn) for ii, pn in enumerate(self.param_names)]))

    def get_free_params(self):
        """Get the free parameters for the model."""
        free_param_names = []
        for param_name in self.model.param_names:
            param = getattr(self.model, param_name)
            if not param.fixed and not param.tied:
                free_param_names.append(param_name)
        return free_param_names

    def get_initial_theta(self):
        """Get the initial parameter values."""
        param_values = {param_name: getattr(self.model, param_name).value
                       for param_name in self.param_names}
        return [param_values[key] for key in self.param_names]

    def _infer_parameter_type(self, param_name, param):
        """
        Infer the type of parameter based on its name and properties.

        Returns a key for default_bounds lookup.
        """
        # Check parameter name patterns
        # FIRST: Handle polynomial coefficients (c0_0, c1_0, c2_0, etc.)
        # These must come BEFORE checking for 'c' in general
        if param_name[0] == 'c' and '_' in param_name:
            # Check if it's a polynomial coefficient (c0, c1, c2, etc. followed by underscore)
            parts = param_name.split('_')
            if parts[0][0] == 'c' and parts[0][1:].isdigit():
                return 'continuum'

        if 'amplitude' in param_name or 'amp' in param_name:
            return 'amplitude'
        elif param_name == 'dv':
            return 'dv'
        elif param_name == 'sigma':
            return 'sigma'
        elif param_name in ['h3', 'h4']:
            return param_name
        elif param_name.startswith('logtau'):
            return param_name
        elif param_name == 'Cf':
            return 'Cf'
        elif param_name == 'wavec':
            return 'wavec'
        elif 'c' in param_name and 'c0' in param_name:  # amp_c0, amp_c1
            if param_name.startswith('amp'):
                return param_name
        elif param_name.startswith('amp_c'):
            return 'amp_c'
        elif param_name.startswith('dv_c'):
            return 'dv_c'
        elif param_name.startswith('sigma_c'):
            return 'sigma_c'
        elif param_name.startswith('amp_w'):
            return 'amp_w'
        elif param_name.startswith('dv_w'):
            return 'dv_w'
        elif param_name.startswith('sigma_w'):
            return 'sigma_w'
        elif 'c' in param_name or 'c1' in param_name:
            if param_name.startswith('amp'):
                return param_name
        elif param_name.startswith('velscale'):
            return 'velscale'
        else:
            # Check bounds for clues
            lower, upper = param.bounds
            if lower is not None and upper is not None:
                if lower == 0 and upper == 1:
                    return 'Cf'  # Likely [0, 1] bounded
                elif lower >= 0 and (upper is None or upper > 100):
                    return 'amplitude'  # Likely positive
                else:
                    return 'continuum'  # Default fallback
            else:
                # No bounds information, use default
                return 'continuum'

    def _check_data_range_for_amplitude(self):
        """
        Check if data range exceeds amplitude bounds and give warning/error.

        This helps identify cases where the default bounds may be inappropriate.
        """
        data_median = np.median(self.flux_use)
        data_max = np.max(self.flux_use)
        data_min = np.min(self.flux_use)

        warnings_list = []
        for i, param_name in enumerate(self.param_names):
            # Check if this is an amplitude parameter
            if 'amplitude' in param_name or 'amp' in param_name:
                lower, upper = self.param_bounds[i]

                # For log-scale bounds, check the actual values
                if param_name in self._log_scale_params:
                    # Bounds are in log space
                    lower_actual = lower
                    upper_actual = upper
                else:
                    lower_actual = lower
                    upper_actual = upper

                # Check if data exceeds bounds
                if data_max > upper_actual:
                    warnings_list.append(
                        f"  WARNING: {param_name}: Data maximum ({data_max:.2e}) exceeds "
                        f"upper bound ({upper_actual:.2e})"
                    )
                if data_median > upper_actual:
                    warnings_list.append(
                        f"  WARNING: {param_name}: Data median ({data_median:.2e}) exceeds "
                        f"upper bound ({upper_actual:.2e})"
                    )
                if data_max < lower_actual * 10:
                    warnings_list.append(
                        f"  INFO: {param_name}: Data maximum ({data_max:.2e}) is much smaller "
                        f"than upper bound ({upper_actual:.2e}). Consider tightening bounds."
                    )

        return warnings_list

    def get_param_bounds(self):
        """
        Determine bounds for all free parameters.

        Priority order:
        1. User-specified custom bounds (bounds_dict)
        2. Model parameter bounds (if both finite and not None)
        3. Parameter type defaults

        Note: Only uses default bounds if no bound or None is provided.
        """
        bounds = []
        warnings_list = []
        self._log_scale_params = set()  # Track parameters using log-scale priors

        for i, param_name in enumerate(self.param_names):
            param = getattr(self.model, param_name)

            # Priority 1: User-specified bounds (always use these)
            if param_name in self.bounds_dict:
                bounds.append(self.bounds_dict[param_name])
                continue

            # Priority 2: Model parameter bounds
            # Only use if BOTH bounds are provided (not None) AND at least one is specified
            lower, upper = param.bounds

            # Check if model has explicit bounds set (not default None values)
            # We consider bounds as "explicitly set" if at least one is not None
            model_has_bounds = (lower is not None) or (upper is not None)

            if model_has_bounds and lower is not None and upper is not None:
                # Both bounds are explicitly set and finite
                if not (np.isinf(lower) or np.isinf(upper)):
                    bounds.append((float(lower), float(upper)))
                    continue
                else:
                    # At least one is infinite, need to use defaults
                    pass

            # Priority 3: Parameter type defaults
            # Use ONLY if model has no explicit bounds (both are None)
            param_type = self._infer_parameter_type(param_name, param)
            if param_type in self.default_bounds:
                default_lower, default_upper = self.default_bounds[param_type]

                # Special handling for wavec (make relative to initial value)
                if param_type == 'wavec':
                    initial_val = param.value
                    default_lower = initial_val * default_lower
                    default_upper = initial_val * default_upper
                elif param_type == 'amplitude' or param_type.startswith('amp'):
                    # Amplitude parameters use log-scale priors
                    # Convert to log space for uniform sampling in log
                    default_lower = np.log10(max(default_lower, 1e-10))
                    default_upper = np.log10(default_upper)
                    self._log_scale_params.add(param_name)
                    warnings_list.append(
                        f"{param_name}: Using log-scale prior [10^{default_lower:.2f}, "
                        f"10^{default_upper:.2f}]"
                    )
                else:
                    if model_has_bounds:
                        warnings_list.append(
                            f"{param_name}: Model has incomplete bounds "
                            f"(lower={lower}, upper={upper}). Using default bounds "
                            f"for type '{param_type}': [{default_lower}, {default_upper}]"
                        )
                    else:
                        warnings_list.append(
                            f"{param_name}: Using default bounds for type '{param_type}': "
                            f"[{default_lower}, {default_upper}]"
                        )

                bounds.append((default_lower, default_upper))
                continue

            # Should not reach here, but just in case
            raise ValueError(f"Cannot determine bounds for parameter '{param_name}'")

        # Print warnings if any
        if warnings_list:
            print("\nBounds Information:")
            for warning in warnings_list:
                print(warning)
            print()

        return bounds

    def _print_bounds_summary(self):
        """Print a summary of parameter bounds."""
        print("\nParameter Bounds:")
        print("-" * 90)
        for i, (param_name, (lower, upper)) in enumerate(zip(self.param_names, self.param_bounds)):
            initial_val = self.theta_initial[i]

            # For log-scale parameters, show actual values (not log space)
            if param_name in self._log_scale_params:
                lower_disp = 10 ** lower
                upper_disp = 10 ** upper
                scale_note = " (log scale)"
            else:
                lower_disp = lower
                upper_disp = upper
                scale_note = ""

            print(f"{param_name:25s} [{lower_disp:15.2e}, {upper_disp:15.2e}]{scale_note:15s} "
                  f"init={initial_val:.2e}")
        print("-" * 90)
        print()

    def prior_transform(self, u):
        """
        Transform from unit cube [0,1]^ndim to parameter space.

        This maps the uniform distribution on the unit cube to the
        uniform distribution on the parameter bounds.

        For amplitude parameters, uses log-uniform distribution (sampling log(amplitude))
        For other parameters, uses linear distribution.

        Parameters
        ----------
        u : array_like
            Unit cube coordinates, shape (ndim,) with values in [0,1]

        Returns
        -------
        theta : array_like
            Transformed parameter values
        """
        theta = np.zeros_like(u)

        for i in range(self.ndim):
            # Check if custom prior for this parameter
            param_name = self.param_names[i]
            if param_name in self.custom_priors:
                # Use custom prior transform
                theta[i] = self.custom_priors[param_name](u[i])
            elif param_name in self._log_scale_params:
                # Log-scale transformation for amplitude parameters
                # Uniform in log10(amplitude)
                log_lower, log_upper = self.param_bounds[i]
                log_theta = log_lower + (log_upper - log_lower) * u[i]
                theta[i] = 10 ** log_theta
            else:
                # Linear transformation from [0,1] to [lower, upper]
                lower, upper = self.param_bounds[i]
                theta[i] = lower + (upper - lower) * u[i]

        return theta

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

    def log_likelihood(self, theta):
        """
        Calculate the log likelihood.

        Uses Gaussian likelihood: -0.5 * sum(((flux - model) / ferr)^2)
        """
        model = self.set_model_params(theta)
        model_flux = model(self.wave_use)
        return -0.5 * np.sum(((self.flux_use - model_flux) / self.ferr) ** 2)

    def fit(self, progress=True):
        """
        Run the dynesty nested sampling.

        Parameters
        ----------
        progress : bool
            Whether to show progress bar

        Returns
        -------
        samples : ndarray
            Equal-weighted posterior samples
        model : CompoundModel
            The fitted model (not updated to best fit)
        param_names : list
            Names of fitted parameters
        """
        print("Starting dynesty nested sampling...")
        print(f"  Method: {self.sample_method}")
        print(f"  Nlive: {self.nlive}")
        print(f"  Bound: {self.bound}")

        # Initialize sampler
        sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood,
            prior_transform=self.prior_transform,
            ndim=self.ndim,
            nlive=self.nlive,
            bound=self.bound,
            sample=self.sample_method,
            rstate=self.rstate
        )

        # Run sampling
        sampler.run_nested(print_progress=progress)

        # Extract results
        self.results = sampler.results

        # Resample to get equal-weighted samples
        samples = self.results.samples
        logwt = self.results.logwt
        logz = self.results.logz

        # Normalize weights to avoid overflow
        logwt_norm = logwt - np.max(logwt)
        self.samples_equal_weight = dyfunc.resample_equal(
            samples, np.exp(logwt_norm)
        )
        self.log_evidence = logz[-1]
        self.log_evidence_err = self.results.logzerr[-1]

        print(f"\nFitting complete!")
        print(f"  Log evidence: {self.log_evidence:.2f} +/- {self.log_evidence_err:.2f}")
        print(f"  Number of samples: {len(self.samples_equal_weight)}")

        return self.samples_equal_weight, self.model, self.param_names

    def get_best_fit(self):
        """
        Get the best fit parameters (maximum likelihood).

        Updates the model with the best fit parameters.

        Returns
        -------
        model : CompoundModel
            Model updated with best fit parameters
        param_names : list
            Names of fitted parameters
        theta_best : ndarray
            Best fit parameter values
        """
        if self.results is None:
            raise RuntimeError("Must run fit() before getting best fit")

        # Find maximum likelihood sample
        idx_best = np.argmax(self.results.logl)
        self.theta_best = self.results.samples[idx_best]
        self.model = self.set_model_params(self.theta_best)

        return self.model, self.param_names, self.theta_best

    def get_quantiles(self, quantiles=[0.16, 0.5, 0.84]):
        """
        Get parameter quantiles from posterior samples.

        Parameters
        ----------
        quantiles : list
            Quantiles to compute (default: 16th, 50th, 84th percentiles)

        Returns
        -------
        q : ndarray
            Quantiles, shape (len(quantiles), ndim)
        """
        if self.samples_equal_weight is None:
            raise RuntimeError("Must run fit() before getting quantiles")

        q = np.quantile(self.samples_equal_weight, quantiles, axis=0)
        return q

    def get_evidence(self):
        """
        Get the Bayesian evidence (marginal likelihood) and its uncertainty.

        Returns
        -------
        logz : float
            Log of the Bayesian evidence
        logz_err : float
            Uncertainty in log evidence
        """
        if self.results is None:
            raise RuntimeError("Must run fit() before getting evidence")

        return self.log_evidence, self.log_evidence_err

    def plot_corner(self, **kwargs):
        """
        Plot corner plot of posterior samples.

        Requires corner package.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to corner.corner
        """
        if corner is None:
            raise ImportError("corner package is required for plotting")

        if self.samples_equal_weight is None:
            raise RuntimeError("Must run fit() before plotting")

        # Get best fit for truths
        if self.theta_best is None:
            self.get_best_fit()

        return corner.corner(
            self.samples_equal_weight,
            labels=self.param_names,
            truths=self.theta_best,
            **kwargs
        )

    def close(self):
        """Clean up resources."""
        self.results = None
        self.samples_equal_weight = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
