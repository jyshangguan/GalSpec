import numpy as np
import matplotlib.pyplot as plt
from .line_profile import Line_MultiGauss, Line_MultiGauss_doublet, Line_Gaussian
from .continuum import WindowedPowerLaw1D, BlackBody

import matplotlib as mpl
mpl.rc("xtick", direction="in", labelsize=16)
mpl.rc("ytick", direction="in", labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)

__all__ = ['plot_fit', 'plot_fit_new', 'reorder_legend',
           'plot_narrow_line_diagnostic', 'plot_narrow_line_template_validation']


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
             ylim1=None, xlabel=None, ylabel=None, legend_kwargs=None, plot_weight=True, 
             ignore_list=None, legend_map=None, mask_list=None):
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
    mask_list : list, optional
        A list of tuples specifying the wavelength ranges to be masked in the plot.
    
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

    if ignore_list is None:
        ignore_list = ['multi']
    else:
        if 'multi' not in ignore_list:
            ignore_list.append('multi')
    
    # Set the legend info
    if legend_map is None:
        legend_map = {}
    
    legend_dict = _map_legend(model, legend_map, ignore_list)

    # Plotting
    flux_model = model(wave)
    flux_res = flux - flux_model

    # Mask the region with weight=0
    if weight is not None:
        fltr = weight == 0
        flux_model[fltr] = np.nan
        flux_res[fltr] = np.nan

    if mask_list is not None:
        flux = flux.copy()
        for mask_range in mask_list:
            mask = (wave >= mask_range[0]) & (wave <= mask_range[1])
            flux[mask] = np.nan
            flux_res[mask] = np.nan

    ax.step(wave, flux, lw=1, color='k', label='Data')

    if (weight is not None) & plot_weight:
        axt = ax.twinx()
        axt.step(wave, weight, lw=1, color='gray')
        axt.set_ylabel('Weight', fontsize=16)
        axt.minorticks_on()

    ax.plot(wave, flux_model, lw=2, color='C3', alpha=0.7, label='Total model')
    
    if 'multi' in model.submodel_names:
        m_multi = model['multi'](wave)
    else:
        m_multi = None

    for loop, m in enumerate(model):
        if m.name in ignore_list:
            continue
        
        if m_multi is not None:
            y = m(wave) * m_multi
        else:
            y = m(wave)

        if isinstance(m, (WindowedPowerLaw1D, BlackBody)):
            fltr = (wave >= m.x_min) & (wave <= m.x_max)
            y[~fltr] = np.nan

        color, label = legend_dict[m.name]
        ax.plot(wave, y, lw=1, color=color, label=label)

        if (isinstance(m, Line_MultiGauss)) | (isinstance(m, Line_MultiGauss_doublet)):
            if m.n_components > 1:
                for ii, msub in enumerate(m.subcomponents):
                    if m_multi is not None:
                        y = msub(wave) * m_multi
                    else:
                        y = msub(wave)
                    ax.plot(wave, y, lw=0.5, ls='--', color=color)
    
    axr.step(wave, flux_res, lw=1, color='k')
    axr.axhline(0, color='k', ls='--', lw=1.0)
    
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


def _map_legend(model, legend_map, ignore_list=None):
    '''
    Map the legend names according to the legend_map dictionary.

    Parameters
    ----------
    model : array like
        Model.
    legend_map : dict
        A dictionary mapping colors and labels to model names.
    '''
    legend_dict = {}
    label_list = []
    count_colors = 0
    for loop, m in enumerate(model):
        if (ignore_list is not None) and (m.name in ignore_list):
            continue

        for (c, l), name_list in legend_map.items():
            if m.name in name_list:

                if l in label_list:
                    legend_dict[m.name] = (c, None)
                else:
                    legend_dict[m.name] = (c, l)
                    label_list.append(l)
        
        if m.name not in legend_dict:
            legend_dict[m.name] = (f'C{count_colors%10}', m.name)
            label_list.append(m.name)
            count_colors += 1

    return legend_dict


def reorder_legend(ax, order):
    '''
    Reorder the legend of a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object.
    order : list
        The desired order of legend labels.
    '''
    handles, labels = ax.get_legend_handles_labels()
    label_handle_dict = dict(zip(labels, handles))
    new_handles = []
    new_labels = []
    for label in order:
        if label in label_handle_dict:
            new_labels.append(label)
            new_handles.append(label_handle_dict[label])

    return new_handles, new_labels


def plot_narrow_line_diagnostic(wave, flux, ferr, model, title,
                               line_waves=None, filename='diagnostic.png'):
    """
    Create a 3-panel diagnostic plot for narrow line fitting.

    This is the standard plot type for intermediate fitting steps in
    narrow line template generation. It shows:
    - Panel 1: Main fit with data, model, and line positions
    - Panel 2: Raw residuals
    - Panel 3: Normalized residuals with ±3σ lines

    Parameters
    ----------
    wave : array
        Wavelength (rest frame)
    flux : array
        Flux values
    ferr : array
        Flux errors
    model : astropy Model
        Fitted model
    title : str
        Plot title
    line_waves : list, optional
        List of line wavelengths to mark (e.g., [6716.44, 6730.82] for [S II])
    filename : str, optional
        Output filename (default: 'diagnostic.png')

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : ndarray
        Array of axes objects [ax_main, ax_resid, ax_norm]

    Examples
    --------
    >>> from sagan.utils import line_wave_dict
    >>> line_waves = [line_wave_dict['SII_6716'], line_wave_dict['SII_6731']]
    >>> plot_narrow_line_diagnostic(wave_s2, flux_s2, ferr_s2, model_fit,
    ...                             '1-Component [S II] Fit',
    ...                             line_waves=line_waves,
    ...                             filename='sii_fit_1comp.png')
    """
    import numpy as np

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: Main fit
    ax = axes[0]
    ax.errorbar(wave, flux, yerr=ferr, fmt='o', markersize=3,
                color='k', capsize=2, label='Data', alpha=0.7)
    wave_plot = np.linspace(wave[0], wave[-1], 200)
    ax.plot(wave_plot, model(wave_plot), 'r-', lw=2, label='Model')

    # Mark line positions if provided
    if line_waves is not None:
        colors = ['b', 'g', 'c', 'm', 'y']
        labels = ['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5']
        for i, lw in enumerate(line_waves):
            ax.axvline(lw, color=colors[i % len(colors)],
                      linestyle='--', alpha=0.3, label=labels[i])

    ax.set_ylabel('Flux')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Raw residuals
    resid = flux - model(wave)
    ax = axes[1]
    ax.plot(wave, resid, 'r-', lw=1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)

    # Panel 3: Normalized residuals
    ax = axes[2]
    ax.plot(wave, resid/ferr, 'b-', lw=1)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axhline(3, color='r', linestyle=':', alpha=0.5)
    ax.axhline(-3, color='r', linestyle=':', alpha=0.5)
    ax.set_ylabel('Normalized Residuals')
    ax.set_xlabel('Wavelength (Å, rest frame)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_narrow_line_template_validation(wave, flux, ferr,
                                         model_gaussian, model_template,
                                         velc_temp, flux_temp,
                                         title='Template Validation',
                                         line_waves=None,
                                         filename='validation.png'):
    """
    Create a 4-panel validation plot for narrow line template quality.

    This is the standard plot type for final template validation. It shows:
    - Panel 1: Template shape in velocity space with FWHM
    - Panel 2: Original Gaussian fit
    - Panel 3: Template-based fit
    - Panel 4: Residuals comparison

    Parameters
    ----------
    wave : array
        Wavelength (rest frame)
    flux : array
        Flux values
    ferr : array
        Flux errors
    model_gaussian : astropy Model
        Original Gaussian fit (1 or 2 components)
    model_template : astropy Model
        Template-based fit
    velc_temp : array
        Template velocity array (km/s)
    flux_temp : array
        Template normalized flux
    title : str, optional
        Overall title (default: 'Template Validation')
    line_waves : list, optional
        List of line wavelengths to mark
    filename : str, optional
        Output filename (default: 'validation.png')

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : ndarray
        2x2 array of axes objects

    Examples
    --------
    >>> from sagan.utils import line_wave_dict
    >>> line_waves = [line_wave_dict['SII_6716'], line_wave_dict['SII_6731']]
    >>> plot_narrow_line_template_validation(
    ...     wave_s2, flux_s2, ferr_s2,
    ...     model_gaussian_fit, model_template_fit,
    ...     velc_temp, flux_temp,
    ...     title='[S II] Template Validation',
    ...     line_waves=line_waves,
    ...     filename='sii_template_validation.png'
    ... )
    """
    import numpy as np
    from scipy.interpolate import interp1d

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate FWHM of template
    f_interp = interp1d(velc_temp, flux_temp, kind='cubic',
                        bounds_error=False, fill_value=0)
    velc_fine = np.linspace(velc_temp[0], velc_temp[-1], 10000)
    flux_fine = f_interp(velc_fine)
    crossings = np.where(np.diff(np.sign(flux_fine - 0.5)))[0]
    if len(crossings) >= 2:
        fwhm = velc_fine[crossings[-1]] - velc_fine[crossings[0]]
    else:
        fwhm = np.nan

    # Calculate χ² values
    resid_gaussian = flux - model_gaussian(wave)
    resid_template = flux - model_template(wave)
    chi2_gaussian = np.sum((resid_gaussian / ferr)**2)
    chi2_template = np.sum((resid_template / ferr)**2)

    # Panel 1: Template shape
    ax = axes[0, 0]
    ax.plot(velc_temp, flux_temp, 'k-', lw=2)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Center')
    ax.axhline(0.5, color='b', linestyle=':', alpha=0.5, label='Half max')
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Template Shape (FWHM={fwhm:.1f} km/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Gaussian fit
    ax = axes[0, 1]
    ax.errorbar(wave, flux, yerr=ferr, fmt='o', markersize=3,
                color='k', capsize=2, label='Data', alpha=0.7)
    wave_plot = np.linspace(wave[0], wave[-1], 200)
    ax.plot(wave_plot, model_gaussian(wave_plot), 'r-', lw=2,
            label='Gaussian fit')

    if line_waves is not None:
        colors = ['b', 'g', 'c', 'm', 'y']
        for i, lw in enumerate(line_waves):
            ax.axvline(lw, color=colors[i % len(colors)],
                      linestyle='--', alpha=0.3)

    ax.set_ylabel('Flux')
    ax.set_title(f'Gaussian Fit ($\\chi^2$={chi2_gaussian:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Template fit
    ax = axes[1, 0]
    ax.errorbar(wave, flux, yerr=ferr, fmt='o', markersize=3,
                color='k', capsize=2, label='Data', alpha=0.7)
    ax.plot(wave_plot, model_template(wave_plot), 'g-', lw=2,
            label='Template fit')

    if line_waves is not None:
        colors = ['b', 'g', 'c', 'm', 'y']
        for i, lw in enumerate(line_waves):
            ax.axvline(lw, color=colors[i % len(colors)],
                      linestyle='--', alpha=0.3)

    ax.set_ylabel('Flux')
    ax.set_xlabel('Wavelength (Å, rest frame)')
    ax.set_title(f'Template Fit ($\\chi^2$={chi2_template:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Residuals comparison
    ax = axes[1, 1]
    ax.plot(wave, resid_gaussian, 'r-', lw=1, label='Gaussian resids',
            alpha=0.7)
    ax.plot(wave, resid_template, 'g--', lw=1, label='Template resids',
            alpha=0.7)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Wavelength (Å, rest frame)')
    ax.set_title(f'$\\Delta\\chi^2$ = {chi2_template - chi2_gaussian:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')

    return fig, axes