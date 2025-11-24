import numpy as np
import matplotlib.pyplot as plt
from .line_profile import Line_MultiGauss, Line_MultiGauss_doublet, Line_Gaussian

import matplotlib as mpl
mpl.rc("xtick", direction="in", labelsize=16)
mpl.rc("ytick", direction="in", labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)

__all__ = ['plot_fit', 'plot_fit_new']


def plot_fit_new(wave, flux, model, weight=None, error=None, ax=None, axr=None, xlim=None, ylim0=None, 
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
    
    wave_model=np.arange(wave.min(), wave.max(), np.median(np.diff(wave))/10)
    ax.step(wave, flux, lw=1, color='k', label='Data', where='mid')
    if error is not None:
        dwv=np.median(wave[1:]-wave[:-1])
        ax.errorbar(wave, flux, yerr=error, fmt='none', ecolor='gray', elinewidth=1, capsize=0)

    if weight is not None:
        axt = ax.twinx()
        axt.step(wave, weight, lw=1, color='gray', where='mid')

    ax.plot(wave_model, model(wave_model), lw=2, color='C3', label='Total model')

    if hasattr(model, 'submodel_names'):
        if 'multi' in model.submodel_names:
            m_multi = model['multi'](wave_model)
        else:
            m_multi = None
    else:
        m_multi = None

    if not isinstance(model, Line_Gaussian):
        for loop, m in enumerate(model):
            if m.name == 'multi':
                continue
            if 'tau_0' in m.param_names:
                continue
            
            if m_multi is not None:
                y = m(wave_model) * m_multi
            else:
                y = m(wave_model)

            ax.plot(wave_model, y, lw=1.5, color=f'C{loop}', label=m.name)

            if (isinstance(m, Line_MultiGauss)) | (isinstance(m, Line_MultiGauss_doublet)):
                if m.n_components > 1:
                    for ii, msub in enumerate(m.subcomponents):
                        if m_multi is not None:
                            y = msub(wave_model) * m_multi
                        else:
                            y = msub(wave_model)
                        ax.plot(wave_model, y, lw=0.5, ls='--', 
                                color=f'C{loop}')
    
    flux_res = flux - model(wave)
    axr.step(wave, flux_res, lw=1, color='k', where='mid')
    axr.axhline(0, color='k', ls='--', lw=1.5)
    if error is not None:
        axr.errorbar(wave, flux_res, yerr=error, fmt='none', ecolor='gray', elinewidth=1, capsize=0)
    
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

    if weight is not None:
        axt = ax.twinx()
        axt.step(wave, weight, lw=1, color='gray')
        axt.set_ylabel('Weight', fontsize=16)
        axt.minorticks_on()

    ax.plot(wave, model(wave), lw=2, color='C3', alpha=0.7, label='Total model')
    
    if 'multi' in model.submodel_names:
        m_multi = model['multi'](wave)
    else:
        m_multi = None

    for loop, m in enumerate(model):
        if m.name == 'multi':
            continue
        
        if m_multi is not None:
            y = m(wave) * m_multi
        else:
            y = m(wave)

        ax.plot(wave, y, lw=0.5, color=f'C{loop}', label=m.name)

        if (isinstance(m, Line_MultiGauss)) | (isinstance(m, Line_MultiGauss_doublet)):
            if m.n_components > 1:
                for ii, msub in enumerate(m.subcomponents):
                    if m_multi is not None:
                        y = msub(wave) * m_multi
                    else:
                        y = msub(wave)
                    ax.plot(wave, y, lw=0.5, ls='--', 
                            color=f'C{loop}')
    
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

