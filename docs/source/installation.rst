Installation
============

Requirements
------------

SAGAN requires Python **3.9 or later**.

Core Dependencies
~~~~~~~~~~~~~~~~~

SAGAN depends on the following packages:

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

Via pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~

The easiest way to install SAGAN is using pip:

.. code-block:: bash

   pip install sagan

This will install SAGAN and all required dependencies.

From Source
~~~~~~~~~~~

To install the latest development version from source:

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/jyshangguan/SAGAN.git
   cd SAGAN

2. Install in editable mode:

.. code-block:: bash

   pip install -e .

This installs SAGAN in "editable" mode, meaning changes to the source code
will be reflected without needing to reinstall.

With Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development, install the additional development tools:

.. code-block:: bash

   pip install -e ".[dev]"

This includes testing and development tools like pytest and Jupyter.

Verification
------------

To verify your installation, try importing SAGAN in Python:

.. code-block:: python

   import sagan
   print(sagan.__version__)

You should see the version number printed (e.g., ``0.1.0``).

Run a quick test:

.. code-block:: python

   from sagan import Line_Gaussian
   line = Line_Gaussian(center=6563, fwhm=10, flux=1.0)
   print(f"Created line model: {line.name}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error: No module named 'astropy'**

Make sure you have installed all dependencies. Try reinstalling:

.. code-block:: bash

   pip install -e .

**Data Files Not Found**

SAGAN includes template data files that should be installed automatically.
If you encounter errors about missing data files, try reinstalling:

.. code-block:: bash

   pip uninstall sagan
   pip install -e .

**Permission Errors**

If you don't have permission to install to the system Python, use a virtual
environment or the ``--user`` flag:

.. code-block:: bash

   pip install --user sagan

Or use a virtual environment (recommended):

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install sagan

Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to learn the basics
of using SAGAN for spectral fitting.
