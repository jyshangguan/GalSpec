"""
Python Tools for SAGAN Spectral Analysis

This package provides reusable plotting and utility functions for AGN spectral analysis
using the SAGAN (Spectral Analysis for Galaxies and AGN) package.

Modules
-------
plot_tools : Plotting functions for continuum fitting and analysis

Functions
---------
- plot_continuum_fit_diagnostic() : Visualize continuum fit with all components
- plot_continuum_subtracted_spectrum() : Visualize continuum-subtracted spectrum
- verify_continuum_subtraction() : Verify continuum subtraction quality

Usage
-----
    from python_tools.plot_tools import (
        plot_continuum_fit_diagnostic,
        plot_continuum_subtracted_spectrum,
        verify_continuum_subtraction
    )

    # After fitting continuum
    plot_continuum_fit_diagnostic(
        wave_rest, flux_rest, ferr_rest, model_cont_fit, cont_mask,
        target_name='SDSS-J0000+0000',
        filename='continuum_fit.png'
    )

    # Plot continuum-subtracted spectrum
    plot_continuum_subtracted_spectrum(
        wave_rest, flux_subtracted, ferr_rest,
        continuum_windows=[(4200, 4300), (4430, 4560), ...],
        target_name='SDSS-J0000+0000',
        filename='subtracted.png'
    )

    # Verify subtraction quality
    results = verify_continuum_subtraction(
        wave_rest, flux_subtracted, continuum_windows
    )

See Also
--------
SAGAN documentation: ../fitting_strategies/continuum_fitting.md
"""

__version__ = '1.0.0'
__author__ = 'SAGAN Development Team'

from .plot_tools import (
    plot_continuum_fit_diagnostic,
    plot_continuum_subtracted_spectrum,
    verify_continuum_subtraction
)

__all__ = [
    'plot_continuum_fit_diagnostic',
    'plot_continuum_subtracted_spectrum',
    'verify_continuum_subtraction'
]
