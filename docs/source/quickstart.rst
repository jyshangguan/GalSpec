Quick Start Guide
=================

This guide will walk you through the basic usage of SAGAN for spectral fitting.

Basic Concepts
--------------

SAGAN is built on top of **Astropy's modeling framework**, which provides:

* **Fittable1DModel**: Base class for all 1D models
* **Compound models**: Combine models using ``+`` (addition) and ``*`` (composition)
* **Fitting**: Multiple fitting algorithms (Levenberg-Marquardt, MCMC, nested sampling)
* **Parameter management**: Set bounds, fix parameters, and tie parameters together

SAGAN models are fully compatible with Astropy models and can be mixed and matched.

A Simple Example: Fitting Hα
------------------------------

Let's start with a simple example: fitting an Hα emission line with a local continuum.

Step 1: Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy.modeling.models import Linear1D
   from astropy.modeling.fitting import LevMarLSQFitter
   from sagan import Line_Gaussian

Step 2: Load or Create Spectrum Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this example, let's create synthetic data:

.. code-block:: python

   # Create wavelength array
   wave = np.linspace(6400, 6700, 1000)

   # Create true parameters
   continuum_slope = 0.0001
   continuum_intercept = 1.0
   line_center = 6563.0  # Hα in Angstroms
   line_fwhm = 10.0  # FWHM in Angstroms
   line_flux = 5.0

   # Generate true model
   continuum = Linear1D(slope=continuum_slope, intercept=continuum_intercept)
   line = Line_Gaussian(center=line_center, fwhm=line_fwhm, flux=line_flux)
   true_model = continuum + line

   # Generate noisy data
   np.random.seed(42)
   flux = true_model(wave)
   noise = np.random.randn(len(wave)) * 0.1
   flux_noisy = flux + noise

Step 3: Define the Fitting Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's define our model for fitting:

.. code-block:: python

   # Define initial model: continuum + Hα line
   continuum_init = Linear1D(slope=0.0, intercept=1.0)
   halpha_init = Line_Gaussian(
       center=6563.0,
       fwhm=15.0,  # Initial guess, will be fitted
       flux=3.0,    # Initial guess, will be fitted
       name='Halpha'
   )

   model_init = continuum_init + halpha_init

Step 4: Fit the Model
~~~~~~~~~~~~~~~~~~~~~

Use the LevMarLSQFitter for least-squares fitting:

.. code-block:: python

   fitter = LevMarLSQFitter()
   model_fit = fitter(model_init, wave, flux_noisy)

   # Print fitted parameters
   print("Fitted continuum slope:", model_fit[0].slope.value)
   print("Fitted continuum intercept:", model_fit[0].intercept.value)
   print("Fitted Hα center:", model_fit[1].center.value)
   print("Fitted Hα FWHM:", model_fit[1].fwhm.value)
   print("Fitted Hα flux:", model_fit[1].flux.value)

Step 5: Visualize the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   plt.figure(figsize=(10, 5))
   plt.plot(wave, flux_noisy, 'k-', drawstyle='steps-mid', label='Data', alpha=0.7)
   plt.plot(wave, true_model(wave), 'b--', label='True Model', linewidth=2)
   plt.plot(wave, model_fit(wave), 'r-', label='Fitted Model', linewidth=2)

   # Plot components
   plt.plot(wave, model_fit[0](wave), 'g:', label='Fitted Continuum', linewidth=1.5)
   plt.plot(wave, model_fit[1](wave), 'm:', label='Fitted Hα', linewidth=1.5)

   plt.xlabel('Wavelength (Å)')
   plt.ylabel('Flux')
   plt.legend()
   plt.tight_layout()
   plt.show()

Working with Multiple Lines
----------------------------

For multiple emission lines, simply add more line components:

.. code-block:: python

   from sagan import Line_Gaussian

   # Define multiple lines
   halpha = Line_Gaussian(center=6563, fwhm=10, flux=5.0, name='Halpha')
   hbeta = Line_Gaussian(center=4861, fwhm=10, flux=2.0, name='Hbeta')
   [OIII]_5007 = Line_Gaussian(center=5007, fwhm=10, flux=3.0, name='OIII_5007')

   # Combine with continuum
   continuum = Linear1D(slope=0.0, intercept=1.0)
   model = continuum + halpha + hbeta + [OIII]_5007

Setting Parameter Bounds
-------------------------

You can set bounds on parameters to constrain the fit:

.. code-block:: python

   line = Line_Gaussian(center=6563, fwhm=10, flux=5.0)

   # Set bounds (min, max)
   line.center.min = 6550
   line.center.max = 6580
   line.fwhm.min = 0  # FWHM must be positive
   line.fwhm.max = 100
   line.flux.min = 0  # Flux must be positive

Fixing Parameters
-----------------

To fix a parameter (prevent it from being fitted):

.. code-block:: python

   line = Line_Gaussian(center=6563, fwhm=10, flux=5.0)

   # Fix the center wavelength
   line.center.fixed = True

   # The center will not change during fitting

Tying Parameters
----------------

You can tie parameters together using expressions:

.. code-block:: python

   halpha = Line_Gaussian(center=6563, fwhm=10, flux=5.0, name='Halpha')
   hbeta = Line_Gaussian(center=4861, fwhm=10, flux=2.0, name='Hbeta')

   # Tie Hβ FWHM to Hα FWHM (they will be equal)
   hbeta.fwhm.tied = lambda model: model.Halpha.fwhm

   # Tie Hβ flux to Hα flux (theoretical Case B ratio)
   hbeta.flux.tied = lambda model: model.Halpha.flux * 0.35

Now Hβ and Hα will share the same FWHM during fitting.

Different Line Profiles
-----------------------

SAGAN provides several line profile models:

**Gaussian Profile** (most common):

.. code-block:: python

   from sagan import Line_Gaussian
   line = Line_Gaussian(center=6563, fwhm=10, flux=5.0)

**Gaussian-Hermite Profile** (for asymmetric lines):

.. code-block:: python

   from sagan import Line_GaussHermite
   line = Line_GaussHermite(
       center=6563,
       fwhm=10,
       flux=5.0,
       h3=0.0,  # Asymmetry parameter
       h4=0.0   # "Boxy" or "peaked" parameter
   )

**Multiple Gaussian Components**:

.. code-block:: python

   from sagan import Line_MultiGauss
   # Two Gaussian components for a broad line
   line = Line_MultiGauss(
       center=6563,
       fwhm=[10, 30],  # Two components with different widths
       flux=[3.0, 2.0],  # Relative fluxes
       name='Halpha'
   )

Next Steps
----------

* Learn about **continuum fitting** in the :doc:`examples/continuum_fitting` tutorial
* Explore **emission line fitting** in the :doc:`examples/emission_line_fitting` tutorial
* Read about **absorption line fitting** in the :doc:`examples/absorption_line_fitting` tutorial
* Check out the **API Reference** for detailed documentation of all models and functions
* Look at the example notebooks in the ``example/`` directory of the repository

Advanced Topics
---------------

For more advanced usage, see:

* :doc:`examples/advanced_fitting` - MCMC and nested sampling
* :doc:`api/fitting` - Fitting algorithms and utilities
* :doc:`api/continuum` - Continuum models
* :doc:`api/line_profiles` - Line profile models
