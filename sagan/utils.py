import os
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
from .constants import ls_km
from scipy.ndimage import gaussian_filter1d
from spectres import spectres

import matplotlib as mpl
mpl.rc("xtick", direction="in", labelsize=16)
mpl.rc("ytick", direction="in", labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)

__all__ = ['package_path', 'splitter', 'line_wave_dict', 'line_label_dict',
           'wave_to_velocity', 'velocity_to_wave', 'plot_fit', 
           'down_spectres']


if platform == "linux" or platform == "linux2":  # Linux
    splitter = '/'
elif platform == "darwin":  # Mac
    splitter = '/'
elif platform == "win32":  # Windows
    splitter = '\\'

pathList = os.path.abspath(__file__).split(splitter)
package_path = splitter.join(pathList[:-1])

# Wavelengths of the typical emission lines
line_wave_dict = {
    'Halpha': 6562.819,
    'Hbeta': 4862.721,
    'Hgamma': 4341.785,
    'OIII_4959': 4960.295,
    'OIII_5007': 5008.239,
    'SII_6718': 6718.29,
    'SII_6733': 6732.68,
    'HeII_4686': 4686,
    'HeI_4471': 4471,
    'HeI_4713': 4713,
    'HeI_4922': 4922,
    'HeI_5016': 5016,
    'NI_5199': 5199,
    'NI_5201': 5201,
    'FeVI_5176': 5176
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
    'HeII_4686': r'He II 4686',
    'HeI_4471': r'He I 4471',
    'HeI_4713': r'He I 4713',
    'HeI_4922': r'He I 4922',
    'HeI_5016': r'He I 5016',
    'NI_5199': r'NI 5199',
    'NI_5201': r'NI 5201',
    'FeVI_5176': r'Fe VI 5176'
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


def plot_fit(wave, flux, model, weight=None, ax=None, axr=None, xlim=None, ylim0=None, 
             ylim1=None, xlabel=None, ylabel=None, legend_kwargs=None):
    '''
    Plot the fitting result.

    Parameters
    ----------
    wave : array like
        Wavelength.
    flux : array like
        Flux.
    model : array like
        Model.
    weight : array like
        Weight.
    ax : matplotlib.axes.Axes
        Axes of the main panel.
    axr : matplotlib.axes.Axes
        Axes of the residual panel.
    xlim : tuple
        X-axis limits.
    ylim0 : tuple
        Y-axis limits of the main axes.
    ylim1 : tuple
        Y-axis limits of the residual axes.
    xlabel : string
        X-axis label.
    ylabel : string
        Y-axis label.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes of the main panel.
    axr : matplotlib.axes.Axes
        Axes of the residual panel.
    '''
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.75])
        axr = fig.add_axes([0.05, 0.05, 0.9, 0.15])
    else:
        assert axr is not None, 'Please provide the axes of the residual panel!'
    axr.sharex(ax)
    
    ax.step(wave, flux, lw=1, color='k', label='Data')
    #ax.step(wave, weight, lw=1, color='gray', label='Weight')
    ax.plot(wave, model(wave), lw=2, color='C3', label='Total model')
    
    for loop, m in enumerate(model):
        ax.plot(wave, m(wave), lw=0.5, color=f'C{loop}', label=m.name)
    
    flux_res = flux - model(wave)
    axr.step(wave, flux_res, lw=1, color='k')
    
    ax.tick_params(labelbottom=False)
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if ylim0 is not None:
        ax.set_ylim(ylim0)
    
    if ylim1 is not None:
        axr.set_ylim(ylim1)

    if xlabel is not None:
        axr.set_xlabel(xlabel, fontsize=24)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=24)
    axr.set_ylabel(r'Res.', fontsize=24)

    ax.minorticks_on()
    axr.minorticks_on()

    if legend_kwargs is None:
        legend_kwargs = {}
    if 'loc' not in legend_kwargs:
        legend_kwargs['loc'] = 'upper left'
    if 'handlelength' not in legend_kwargs:
        legend_kwargs['handlelength'] = 1.0
    if 'columnspacing' not in legend_kwargs:
        legend_kwargs['columnspacing'] = 0.5
    if 'fontsize' not in legend_kwargs:
        legend_kwargs['fontsize'] = 16
    if 'ncol' not in legend_kwargs:
        legend_kwargs['ncol'] = model.n_submodels // 3 + 1
    ax.legend(**legend_kwargs)

    return ax, axr


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
