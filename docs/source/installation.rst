Installation
============

Requirements
------------

GalSpec requires Python **3.9 or later**.

Core Dependencies
~~~~~~~~~~~~~~~~~

GalSpec depends on the following packages:

* **numpy** (≥ 1.20) - Numerical computations
* **scipy** (≥ 1.7) - Scientific computing routines
* **matplotlib** (≥ 3.4) - Plotting
* **astropy** (≥ 5.0) - Astronomical algorithms and models
* **pandas** (≥ 1.3) - Data manipulation

Fitting Dependencies
~~~~~~~~~~~~~~~~~~~~

* **emcee** (≥ 3.0) - MCMC sampling
* **dynesty** (≥ 1.2) - Nested sampling
* **corner** (≥ 2.2) - Corner plots for MCMC results

Analysis Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **extinction** (≥ 0.4) - Dust extinction calculations
* **PyAstronomy** (≥ 0.18) - Astronomical tools
* **spectres** (≥ 2.0) - Spectral resampling
* **multiprocess** (≥ 0.70) - Parallel processing

Installation Methods
--------------------

From Source
~~~~~~~~~~~

GalSpec is currently not available on PyPI. Install from source:

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/yourusername/galspec.git
   cd galspec

2. Install in editable mode:

.. code-block:: bash

   pip install -e .

This installs GalSpec in "editable" mode, meaning changes to the source code
will be reflected without needing to reinstall.

With Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development, install the additional development tools:

.. code-block:: bash

   pip install -e ".[dev]"

This includes testing and development tools like pytest and Jupyter.

Verification
------------

To verify your installation, check the version:

.. code-block:: python

   import galspec
   print(f"GalSpec version: {galspec.__version__}")

You should see the version number printed (e.g., ``0.1.0``).

Run a quick test:

.. code-block:: python

   import numpy as np
   from galspec import Line_Gaussian

   # Create an Hα emission line
   halpha = Line_Gaussian(
       amplitude=5.0,   # Peak amplitude
       dv=0,           # No velocity shift (km/s)
       sigma=200,      # Velocity dispersion (km/s)
       wavec=6563,     # Hα wavelength (Angstroms)
       name='Halpha'
   )

   # Evaluate the model
   wave = np.linspace(6500, 6600, 100)
   flux = halpha(wave)

   print(f"Created line model: {halpha.name}")
   print(f"Peak flux: {np.max(flux):.3f}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error: No module named 'astropy'**

Make sure you have installed all dependencies. Try reinstalling:

.. code-block:: bash

   pip install -e .

**Data Files Not Found**

GalSpec includes template data files that should be installed automatically.
If you encounter errors about missing data files, try reinstalling:

.. code-block:: bash

   cd /path/to/galspec
   pip install -e .

**Permission Errors**

If you don't have permission to install to the system Python, use a virtual
environment (recommended):

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   cd /path/to/SAGAN
   pip install -e .

Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to learn the basics
of using GalSpec for spectral fitting.
