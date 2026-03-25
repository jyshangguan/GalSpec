Quick Start Guide
=================

This guide will walk you through a realistic spectral fitting example using SAGAN.
We'll analyze an SDSS spectrum of an AGN (SDSS J000605.59-092007.0) to fit both
the continuum and emission lines, and extract physical parameters.

## The Dataset: SDSS J000605.59-092007.0

We'll analyze a real spectrum from the Sloan Digital Sky Survey (SDSS) of an
active galactic nucleus (AGN). This object has:

* **Redshift**: z = 0.0699
* **Coordinates**: 00:06:05.59, -09:20:07.0
* **Features**: Strong emission lines (Hα, Hβ, [O III], [N II], [S II])
* **Host galaxy**: Visible stellar absorption features

You can download this spectrum from the SDSS database or use your own data.

## Step 1: Load and Prepare the Spectrum

First, let's load the data and apply necessary corrections:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy.io import fits
   import extinction

   # Load SDSS spectrum
   hdul = fits.open('spec-0651-52141-0434.fits')
   flux_obs = hdul[1].data['flux'].astype(float)
   loglam = hdul[1].data['loglam'].astype(float)
   wave_obs = 10**loglam  # Convert log wavelength to linear
   ivar_obs = hdul[1].data['ivar'].astype(float)
   ferr_obs = np.sqrt(1/ivar_obs)

   # Apply corrections
   zred = 0.069907  # Redshift from NED
   A_v = 0.105      # Galactic extinction from NED

   # Correct for Galactic extinction
   A_lambda = extinction.ccm89(wave_obs, A_v, r_v=3.1)
   wave_rest = wave_obs / (1 + zred)  # Convert to rest frame
   flux_dered = flux_obs * 10**(0.4 * A_lambda)
   ferr_dered = ferr_obs * 10**(0.4 * A_lambda)

   # Plot the spectrum
   plt.figure(figsize=(15, 5))
   plt.step(wave_rest, flux_dered, where='mid', color='k', alpha=0.8)
   plt.xlabel('Rest Wavelength (Å)')
   plt.ylabel('Flux (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)')
   plt.title('SDSS J000605.59-092007.0')
   plt.show()

## Step 2: Fit the Stellar Continuum

For AGN with host galaxy contamination, we need to fit both the stellar
continuum (host galaxy) and the AGN power law continuum.

We'll use stellar templates to fit the host galaxy:

.. code-block:: python

   import sys
   sys.path.append('/path/to/SAGAN')
   import sagan
   from astropy.modeling import models, fitting

   # Define continuum windows (line-free regions)
   cont_windows = [
       [3900, 4060],  # Ca II K&H region
       [4170, 4260],
       [4430, 4770],
       [5080, 5550],
       [6050, 6200],
       [6890, 7010]
   ]

   # Create weights for continuum fitting
   weights = np.zeros_like(wave_rest)
   for window in cont_windows:
       weights[(wave_rest >= window[0]) & (wave_rest <= window[1])] = 1.0

   # Define stellar templates (A, F, G, K stars)
   velscale = 20  # Velocity scale in km/s
   bounds = {'sigma': (velscale, 300)}

   star_A = sagan.StarSpectrum(
       amplitude=1.0, sigma=100, delta_z=0,
       velscale=velscale, Star_type='A', name='A Star', bounds=bounds
   )
   star_F = sagan.StarSpectrum(
       amplitude=5.0, sigma=100, delta_z=0,
       velscale=velscale, Star_type='F', name='F Star', bounds=bounds
   )
   star_G = sagan.StarSpectrum(
       amplitude=5.0, sigma=100, delta_z=0,
       velscale=velscale, Star_type='G', name='G Star', bounds=bounds
   )
   star_K = sagan.StarSpectrum(
       amplitude=3.0, sigma=100, delta_z=0,
       velscale=velscale, Star_type='K', name='K Star', bounds=bounds
   )

   # Combine stellar templates
   stars = star_A + star_F + star_G + star_K

   # Tie velocity dispersion and redshift of all stars
   for name in ['A Star', 'F Star', 'G Star']:
       stars[name].sigma.tied = sagan.tie_StarSpectrum_sigma('K Star')
       stars[name].delta_z.tied = sagan.tie_StarSpectrum_deltaz('K Star')

   # Add AGN power law continuum
   powerlaw = models.PowerLaw1D(
       amplitude=10, x_0=5000, alpha=1.7,
       name='PowerLaw', fixed={'x_0': True}
   )

   # Combine all continuum components
   continuum_model = stars + powerlaw

   # Fit the continuum
   fitter = fitting.LevMarLSQFitter()
   continuum_fit = fitter(
       continuum_model, wave_rest, flux_dered,
       weights=weights, maxiter=10000
   )

   # Plot the continuum fit
   fig, ax = plt.subplots(figsize=(15, 5))
   ax.step(wave_rest, flux_dered, where='mid', color='k', alpha=0.8, label='Data')
   ax.plot(wave_rest, continuum_fit(wave_rest), 'r-', linewidth=2, label='Continuum Fit')
   for window in cont_windows:
       ax.axvspan(window[0], window[1], color='C1', alpha=0.3)
   ax.set_xlabel('Rest Wavelength (Å)')
   ax.set_ylabel('Flux (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)')
   ax.legend()
   plt.show()

   # Get stellar velocity dispersion
   sigma_star = continuum_fit['K Star'].sigma.value
   print(f"Stellar velocity dispersion: {sigma_star:.1f} km/s")

.. note::

   **Convenience Alternative**: Instead of creating individual ``StarSpectrum`` models
   and manually tying their parameters, you can use the ``Multi_StarSpectrum`` model
   which automatically combines multiple stellar templates (A, F, G, K, M) with a
   shared velocity dispersion:

   .. code-block:: python

      # Simpler approach using Multi_StarSpectrum
      stars = sagan.Multi_StarSpectrum(
          amp_0=1.0,  # A star amplitude
          amp_1=5.0,  # F star amplitude
          amp_2=5.0,  # G star amplitude
          amp_3=3.0,  # K star amplitude
          amp_4=2.0,  # M star amplitude
          sigma=100,  # Shared velocity dispersion
          velscale=20,
          Star_types=['A', 'F', 'G', 'K', 'M'],
          bounds={'sigma': (velscale, 300)}
      )

      # All stars automatically share the same sigma parameter
      continuum_model = stars + powerlaw

## Step 3: Extract Emission Line Spectrum

Subtract the continuum to isolate the emission lines:

.. code-block:: python

   # Subtract continuum to get emission lines
   flux_lines = flux_dered - continuum_fit(wave_rest)

   plt.figure(figsize=(15, 5))
   plt.step(wave_rest, flux_lines, where='mid', color='k', alpha=0.8)
   plt.xlabel('Rest Wavelength (Å)')
   plt.ylabel('Flux (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)')
   plt.title('Emission Line Spectrum')
   plt.show()

## Step 4: Fit Emission Lines

Now let's fit the emission lines. We'll start with the Hα region:

.. code-block:: python

   # Load emission line wavelengths
   line_wave_dict = sagan.line_wave_dict
   wavec_ha = line_wave_dict['Halpha']
   wavec_nii_6583 = line_wave_dict['NII_6583']
   wavec_nii_6548 = line_wave_dict['NII_6548']

   # Focus on Hα region
   window = [6400, 6900]
   fltr = (wave_rest > window[0]) & (wave_rest < window[1])
   wave_ha = wave_rest[fltr]
   flux_ha = flux_lines[fltr]

   # Define line models
   # Broad Hα component
   bha = sagan.Line_MultiGauss(
       n_components=1,
       amp_c=4.0, dv_c=150, sigma_c=1100,
       wavec=wavec_ha, name='bHalpha'
   )

   # Narrow Hα
   nha = sagan.Line_Gaussian(
       center=wavec_ha, fwhm=100, flux=10.0,
       name='nHalpha'
   )

   # Narrow [N II] lines
   nii_6583 = sagan.Line_Gaussian(
       center=wavec_nii_6583, fwhm=100, flux=3.0,
       name='NII_6583'
   )

   nii_6548 = sagan.Line_Gaussian(
       center=wavec_nii_6548, fwhm=100, flux=1.0,
       name='NII_6548'
   )

   # Combine all lines
   model_ha = bha + nha + nii_6583 + nii_6548

   # Fit
   fitter = fitting.LevMarLSQFitter()
   model_ha_fit = fitter(model_ha, wave_ha, flux_ha, maxiter=10000)

   # Plot
   plt.figure(figsize=(15, 5))
   plt.step(wave_ha, flux_ha, where='mid', color='k', alpha=0.8, label='Data')
   plt.plot(wave_ha, model_ha_fit(wave_ha), 'r-', linewidth=2, label='Fit')
   plt.plot(wave_ha, model_ha_fit[0](wave_ha), '--', label='Broad Hα')
   plt.plot(wave_ha, model_ha_fit[1](wave_ha), '--', label='Narrow Hα')
   plt.xlabel('Rest Wavelength (Å)')
   plt.ylabel('Flux (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)')
   plt.legend()
   plt.show()

## Step 5: Calculate Physical Parameters

Now let's calculate line fluxes and physical parameters:

.. code-block:: python

   from astropy.cosmology import Planck18 as cosmo

   # Calculate line fluxes by integrating
   def integrate_line_flux(model, wave, scale=1e-17):
       flux = model(wave) * scale
       return np.trapz(flux, wave)

   # Get fluxes (in erg/s/cm^2)
   flux_ha = integrate_line_flux(model_ha_fit['bHalpha'], wave_ha)
   flux_nii = integrate_line_flux(model_ha_fit['NII_6583'], wave_ha)

   print(f"Hα flux: {flux_ha:.2e} erg/s/cm^2")
   print(f"[N II] λ6583 flux: {flux_nii:.2e} erg/s/cm^2")

   # Calculate luminosity distance
   lum_dist = cosmo.luminosity_distance(zred).to('cm').value

   # Convert to luminosity
   lum_ha = flux_ha * 4 * np.pi * lum_dist**2
   print(f"Hα luminosity: {lum_ha:.2e} erg/s")

   # Calculate continuum luminosity at 5100 Å
   lam_flam = continuum_fit['PowerLaw'].amplitude.value * 5100 * 1e-17
   nu_lnu = lam_flam * 4 * np.pi * lum_dist**2
   print(f"L_5100: {nu_lnu:.2e} erg/s")

   # Estimate black hole mass (using empirical relation)
   # From Liu et al. (2019, ApJS, 243, 21)
   fwhm_ha = model_ha_fit['bHalpha'].sigma_c.value  # Broad line width
   log_mbh = np.log10((fwhm_ha/1000)**2 * (nu_lnu/1e44)**0.533) + 6.91
   print(f"Black hole mass: 10^{log_mbh:.2f} M_sun")

Summary
-------

In this quick start, we've:

1. ✅ Loaded and preprocessed an SDSS spectrum
2. ✅ Fitted the stellar continuum and AGN power law
3. ✅ Extracted and fitted emission lines (Hα, [N II])
4. ✅ Calculated physical parameters (line fluxes, luminosities, BH mass)

This is just a basic example. SAGAN can handle much more complex cases:

* Multiple emission line components (broad + narrow)
* Fe II template fitting
* BAL (broad absorption line) fitting
* Bayesian parameter estimation with MCMC/nested sampling
* Complex line profiles (Gauss-Hermite, multiple components)

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
