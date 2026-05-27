"""
Plotting Tools for GalSpec Spectral Analysis

This module provides reusable plotting functions for visualizing continuum fitting
and continuum-subtracted spectra in AGN spectral analysis.

Functions
---------
- plot_continuum_fit_diagnostic(): Visualize continuum fit with all components
- plot_continuum_subtracted_spectrum(): Visualize continuum-subtracted spectrum

Usage
-----
    from python_tools.plot_tools import plot_continuum_fit_diagnostic, plot_continuum_subtracted_spectrum

    # Plot continuum fit
    plot_continuum_fit_diagnostic(
        wave_rest, flux_rest, ferr_rest, model_cont_fit, cont_mask,
        target_name='SDSS-J0000+0000',
        filename='continuum_fit_diagnostic.png'
    )

    # Plot continuum-subtracted spectrum
    plot_continuum_subtracted_spectrum(
        wave_rest, flux_subtracted, ferr_rest,
        continuum_windows=[(4200, 4300), (4430, 4560), ...],
        target_name='SDSS-J0000+0000',
        filename='continuum_subtracted_spectrum.png'
    )
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_continuum_fit_diagnostic(wave, flux, error, model, cont_mask,
                                   target_name=None, filename=None,
                                   xlabel='Wavelength (Å, rest frame)',
                                   ylabel='Flux (1e-17 erg/cm²/s/Å)',
                                   figsize=(10, 8), dpi=150, **kwargs):
    """
    Create a diagnostic plot for continuum fitting using GalSpec's plot_fit_new().

    This function visualizes the fitted continuum model with all components,
    highlighting the continuum windows used for fitting. The plot includes:
    - Full spectrum (black step line)
    - Continuum windows highlighted (gray weight line)
    - Total continuum fit (red line)
    - Individual components (AGN power-law, stellar, iron)
    - Residuals (data - model)

    Parameters
    ----------
    wave : array-like
        Wavelength array (rest frame, in Angstroms)
    flux : array-like
        Flux array (same shape as wave)
    error : array-like
        Flux error array (same shape as wave)
    model : astropy.modeling.CompoundModel
        Fitted continuum model (typically AGN + stellar + iron)
    cont_mask : array-like, bool
        Boolean mask indicating continuum windows (True = continuum window)
    target_name : str, optional
        Target name for plot title
    filename : str, optional
        Output filename. If None, plot is displayed interactively
    xlabel : str, optional
        X-axis label (default: 'Wavelength (Å, rest frame)')
    ylabel : str, optional
        Y-axis label (default: 'Flux (1e-17 erg/cm²/s/Å)')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 8))
    dpi : int, optional
        Resolution for saved figure (default: 150)
    **kwargs : dict
        Additional keyword arguments passed to plot_fit_new()

    Returns
    -------
    ax : matplotlib.axes.Axes
        Main plot axes
    axr : matplotlib.axes.Axes
        Residual plot axes

    Examples
    --------
    >>> from sagan import plot as sagan_plot
    >>> from python_tools.plot_tools import plot_continuum_fit_diagnostic
    >>>
    >>> # After fitting continuum model
    >>> plot_continuum_fit_diagnostic(
    ...     wave_rest, flux_rest, ferr_rest, model_cont_fit, cont_mask,
    ...     target_name='SDSS-J000111.15-100155.5',
    ...     filename='continuum_fit_diagnostic.png'
    ... )

    Notes
    -----
    This function is a wrapper around sagan.plot.plot_fit_new() that:
    1. Creates a weight array highlighting continuum windows
    2. Passes all data to plot_fit_new() for standardized visualization
    3. Adds optional title and saves figure

    The plot shows:
    - Top panel: Spectrum with continuum fit and individual components
    - Bottom panel: Residuals (data - model) with error bars

    See Also
    --------
    plot_continuum_subtracted_spectrum : Plot continuum-subtracted spectrum
    sagan.plot.plot_fit_new : Underlying GalSpec plotting function
    """
    # Import GalSpec plotting module
    try:
        from sagan import plot as sagan_plot
    except ImportError:
        raise ImportError("GalSpec package not found. Please install GalSpec first.")

    # Create weight array: 1 for continuum windows, 0 elsewhere
    weight = np.zeros_like(wave, dtype=float)
    weight[cont_mask] = 1.0

    # Create plot using GalSpec's plot_fit_new()
    ax, axr = sagan_plot.plot_fit_new(
        wave,
        flux,
        model,
        weight=weight,
        error=error,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs
    )

    # Add title if target_name provided
    if target_name is not None:
        ax.set_title(f'{target_name} - Type 1 AGN Continuum Fit', fontsize=14)

    # Save or display
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return ax, axr


def plot_continuum_subtracted_spectrum(wave, flux_sub, error,
                                       continuum_windows,
                                       target_name=None,
                                       filename=None,
                                       xlabel='Wavelength (Å, rest frame)',
                                       ylabel='Flux (1e-17 erg/cm²/s/Å)',
                                       figsize=(12, 5),
                                       dpi=150,
                                       window_color='blue',
                                       window_alpha=0.1):
    """
    Plot continuum-subtracted spectrum with continuum windows highlighted.

    This function visualizes the spectrum after continuum subtraction, showing:
    - Continuum-subtracted flux (line emission/absorption)
    - Continuum windows highlighted (should be near zero)
    - Zero reference line

    This is useful for:
    - Verifying continuum subtraction quality
    - Visualizing emission lines for subsequent fitting
    - Checking for systematic offsets in continuum windows

    Parameters
    ----------
    wave : array-like
        Wavelength array (rest frame, in Angstroms)
    flux_sub : array-like
        Continuum-subtracted flux array (same shape as wave)
    error : array-like
        Flux error array (same shape as wave)
    continuum_windows : list of tuples
        List of (wmin, wmax) tuples defining continuum windows in Angstroms
        Example: [(4200, 4300), (4430, 4560), (5060, 5400), ...]
    target_name : str, optional
        Target name for plot title
    filename : str, optional
        Output filename. If None, plot is displayed interactively
    xlabel : str, optional
        X-axis label (default: 'Wavelength (Å, rest frame)')
    ylabel : str, optional
        Y-axis label (default: 'Flux (1e-17 erg/cm²/s/Å)')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 5))
    dpi : int, optional
        Resolution for saved figure (default: 150)
    window_color : str, optional
        Color for continuum window shading (default: 'blue')
    window_alpha : float, optional
        Transparency for continuum window shading (default: 0.1)

    Returns
    -------
    ax : matplotlib.axes.Axes
        Plot axes

    Examples
    --------
    >>> from python_tools.plot_tools import plot_continuum_subtracted_spectrum
    >>>
    >>> # Define continuum windows
    >>> windows = [(4200, 4300), (4430, 4560), (5060, 5400),
    ...            (5600, 5700), (6180, 6230), (6800, 7000), (7500, 8000)]
    >>>
    >>> # Plot continuum-subtracted spectrum
    >>> plot_continuum_subtracted_spectrum(
    ...     wave_rest, flux_subtracted, ferr_rest,
    ...     continuum_windows=windows,
    ...     target_name='SDSS-J000111.15-100155.5',
    ...     filename='continuum_subtracted_spectrum.png'
    ... )

    Notes
    -----
    Quality checks for continuum subtraction:
    - Continuum windows should have median flux near 0 (±0.5)
    - Emission lines should show positive flux
    - Absorption lines should show negative flux
    - No systematic slope should remain

    If continuum windows show systematic offsets, the continuum fit may
    need to be revisited (check windows, model components, or parameters).

    See Also
    --------
    plot_continuum_fit_diagnostic : Plot continuum fit with components
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot continuum-subtracted spectrum
    ax.plot(wave, flux_sub, 'k-', linewidth=0.5, alpha=0.7, label='Continuum-subtracted')

    # Highlight continuum windows
    for wmin, wmax in continuum_windows:
        # Only plot if window is within wavelength range
        if wmin >= np.min(wave) and wmax <= np.max(wave):
            ax.axvspan(wmin, wmax, color=window_color, alpha=window_alpha)

    # Zero reference line
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero')

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if target_name is not None:
        ax.set_title(f'{target_name} - Continuum-Subtracted Spectrum', fontsize=14)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or display
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return ax


def verify_continuum_subtraction(wave, flux_sub, continuum_windows,
                                  verbose=True):
    """
    Verify continuum subtraction quality by checking statistics in continuum windows.

    This function computes median and standard deviation of the continuum-subtracted
    flux in each continuum window. Good continuum subtraction should yield:
    - Median near 0 (±0.5) in all windows
    - Standard deviation consistent with measurement noise

    Parameters
    ----------
    wave : array-like
        Wavelength array (rest frame, in Angstroms)
    flux_sub : array-like
        Continuum-subtracted flux array (same shape as wave)
    continuum_windows : list of tuples
        List of (wmin, wmax) tuples defining continuum windows
    verbose : bool, optional
        If True, print statistics for each window (default: True)

    Returns
    -------
    results : list of dict
        List of dictionaries containing statistics for each window:
        [{'window': (wmin, wmax), 'median': val, 'std': val, 'n_pix': val}, ...]

    Examples
    --------
    >>> from python_tools.plot_tools import verify_continuum_subtraction
    >>>
    >>> windows = [(4200, 4300), (4430, 4560), (5060, 5400)]
    >>> results = verify_continuum_subtraction(wave_rest, flux_subtracted, windows)
    >>>
    >>> # Check if all windows have median near zero
    >>> for r in results:
    ...     if abs(r['median']) > 1.0:
    ...         print(f"Warning: Window {r['window']} has median = {r['median']:.2f}")

    Returns
    -------
    results : list of dict
        Statistics for each continuum window

    Notes
    -----
    Quality criteria:
    - |median| < 1.0: Excellent
    - |median| 1.0-2.0: Acceptable
    - |median| > 2.0: Poor - check continuum fit

    See Also
    --------
    plot_continuum_subtracted_spectrum : Visual verification
    """
    results = []

    if verbose:
        print("\nContinuum-subtracted spectrum in continuum windows:")
        print(f"{'Window':<20} {'Median':>10} {'Std':>10} {'N_pix':>7}")
        print("-" * 50)

    for i, (wmin, wmax) in enumerate(continuum_windows, 1):
        # Check if window is within wavelength range
        if wmin < np.min(wave) or wmax > np.max(wave):
            if verbose:
                print(f"Window {i:2d} ({wmin:4d}-{wmax:4d} A): OUTSIDE RANGE")
            continue

        # Extract data in this window
        mask = (wave > wmin) & (wave < wmax)
        if np.sum(mask) == 0:
            if verbose:
                print(f"Window {i:2d} ({wmin:4d}-{wmax:4d} A): NO DATA")
            continue

        median_sub = np.median(flux_sub[mask])
        std_sub = np.std(flux_sub[mask])
        n_pix = np.sum(mask)

        result = {
            'window': (wmin, wmax),
            'median': median_sub,
            'std': std_sub,
            'n_pix': n_pix
        }
        results.append(result)

        if verbose:
            print(f"Window {i:2d} ({wmin:4d}-{wmax:4d} A): "
                  f"median = {median_sub:+.3f}, std = {std_sub:.3f}")

    return results


# Module-level documentation
if __name__ == "__main__":
    """
    Example usage of plot_tools functions.
    """
    print(__doc__)
