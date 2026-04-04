.. SAGAN documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SAGAN: Spectral Analysis of Galaxy and Active galactic Nuclei
==============================================================

**SAGAN** is a Python package for fitting astronomical spectra, specifically designed for AGN and galaxy spectra with complex emission and absorption line features.

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://choosealicense.com/licenses/mit/
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code Style

Key Features
------------

* **Astropy Integration**: All models inherit from Astropy's modeling framework, ensuring compatibility with standard astronomical tools
* **Multiple Fitting Methods**: Support for least-squares, MCMC (emcee), and nested sampling (dynesty; under development) algorithms
* **Comprehensive Line Database**: Built-in database of UV/optical emission lines for AGN and galaxy analysis
* **Template Support**: Iron (Fe II) templates and stellar population models included
* **Flexible Model Composition**: Easily combine continuum, line, and template components
* **Professional Fitting**: Parameter constraints, tying, and bounds for complex models

Quick Start
-----------

Install SAGAN from source:

.. code-block:: bash

   git clone https://github.com/jyshangguan/SAGAN.git
   cd SAGAN
   pip install -e .

Fit a simple emission line:

.. code-block:: python

   import numpy as np
   from astropy.modeling.models import Linear1D
   from sagan import Line_Gaussian
   from astropy.modeling.fitting import LevMarLSQFitter

   # Load your spectrum data
   wave = np.linspace(4000, 7000, 1000)
   flux = np.random.randn(1000) * 0.1  # Your flux data here

   # Create a model: continuum + emission line
   continuum = Linear1D(slope=0.0, intercept=1.0)
   line = Line_Gaussian(amplitude=1.0, dv=0, sigma=200, wavec=6563, name='Halpha')
   model = continuum + line

   # Fit the model
   fitter = LevMarLSQFitter()
   fitted_model = fitter(model, wave, flux)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/continuum_fitting
   examples/emission_line_fitting
   examples/absorption_line_fitting
   examples/advanced_fitting

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/continuum
   api/line_profiles
   api/emission_lines
   api/iron_templates
   api/stellar_continuum
   api/fitting
   api/utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Links
-----

* **GitHub Repository**: https://github.com/jyshangguan/SAGAN
* **Issue Tracker**: https://github.com/jyshangguan/SAGAN/issues
* **Documentation**: https://jyshangguan.github.io/SAGAN/
