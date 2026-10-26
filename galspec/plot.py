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

__all__ = ['plot_fit', 'plot_fit_new', 'plot_posterior_hist', 'reorder_legend',
           'plot_narrow_line_diagnostic', 'plot_narrow_line_template_validation']

#: Colours cycled through for the individual model components by :func:`plot_fit`.
#:
#: matplotlib's own cycle has ten entries, which is not enough for a model with
#: twenty components: unrelated lines then share a colour, and the worst case is a
#: line taking the same red as the total model.  The first entries keep the usual
#: matplotlib order (blue, orange, green, ...) minus ``C3``, which stays reserved
#: for the total model, and the palette continues to 24 colours.
_COMPONENT_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf", "#8c8c00", "#1b9e77", "#d95f02",
    "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3",
)


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
    axr.set_ylabel(r'Res./$\sigma$' if residual_in_sigma else r'Res.', fontsize=24)

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
             ignore_list=None, legend_map=None, mask_list=None, error=None,
             residual_in_sigma=False):
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
    error : array like, optional
        Uncertainty of ``flux``.  Only used when ``residual_in_sigma`` is True.
    residual_in_sigma : bool, optional
        Draw ``(data - model) / error`` instead of the raw residual, with dashed
        guides at +/-1 sigma.  A raw residual in flux units hides how large it is
        compared with the noise, and one bright pixel can set the scale of the
        whole residual panel; in units of sigma both problems disappear.
    
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
    
    if residual_in_sigma:
        if error is None:
            raise ValueError("residual_in_sigma=True needs the `error` array")
        flux_res = flux_res / np.asarray(error, dtype=float)
        axr.step(wave, flux_res, lw=1, color='k')
        axr.axhline(0, color='k', ls='--', lw=1.0)
        for level in (-1, 1):
            axr.axhline(level, color='gray', ls=':', lw=1.0)
    else:
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
    axr.set_ylabel(r'Res./$\sigma$' if residual_in_sigma else r'Res.', fontsize=24)

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
            color = _COMPONENT_COLORS[count_colors % len(_COMPONENT_COLORS)]
            # a model may set `label` to something readable (e.g. "BH$\alpha$");
            # fall back to the machine name
            label = getattr(m, 'label', None) or m.name
            legend_dict[m.name] = (color, label)
            label_list.append(label)
            count_colors += 1

    return legend_dict


def plot_posterior_hist(fit_or_samples, params=None, labels=None, quantiles=(0.16, 0.5, 0.84),
                        bins=40, ncols=3, figsize=None, weights=None, color='C0',
                        show_bounds=True, show_stats=True, title=None, filename=None,
                        density=True, histtype='stepfilled', alpha=0.45, log_scale=None,
                        view_quantiles=(0.001, 0.999), pad=0.06, prior_text=True):
    r"""Histograms of selected one-dimensional posteriors.

    A corner plot shows every pair, which for forty parameters is unreadable; most
    of the time the marginal distribution of a handful of parameters is all that
    is wanted.  Each panel shows the histogram, the median, the 16-84 percentile
    band and, in the corner, ``median +err -err``.  When a fit object is given,
    the prior bounds are drawn as dotted vertical lines: a posterior that fills the
    prior has not been measured.

    Parameters
    ----------
    fit_or_samples : object or array like
        A fitted sampler (``JNesty_Fit`` / ``Dynesty_Fit``) or an ``(nsamples,
        ndim)`` array of equally weighted samples.
    params : sequence, optional
        Which parameters to draw: names as reported by the fitter (``Halpha.sigma_w0``
        or the flat ``sigma_w0_3``) or integer indices.  Default: every free
        parameter.
    labels : sequence of str, optional
        Panel labels; defaults to the fitter's LaTeX labels
        (:mod:`galspec.param_labels`), e.g. ``BH$\alpha$:$\sigma_{w0}$``.
    quantiles : tuple of three floats
        Lower, central and upper quantile.
    bins : int
        Histogram bins per panel.
    ncols : int
        Number of panels per row.
    figsize : tuple, optional
        Default chosen from the number of panels.
    weights : array like, optional
        Sample weights.  Defaults to ``fit.samples_equal_weight`` weights if the
        samples are not already equally weighted.
    color : str
        Face colour of the histograms.
    show_bounds : bool
        Draw the prior bounds (only possible when a fit object is given).
    show_stats : bool
        Annotate median and percentiles in each panel.
    log_scale : sequence of str, optional
        Subset of the drawn parameters to put on a logarithmic x axis.
    view_quantiles : tuple of two floats
        The x range of each panel covers this percentile range of the posterior,
        plus ``pad`` of that span.  It is taken from the *samples*, never from the
        prior: a prior ten times wider than the posterior would otherwise squeeze
        the histogram into a spike, which is exactly the panel's job to avoid.
    pad : float
        Fraction of the posterior span added on each side.
    prior_text : bool
        When a prior bound falls outside the view (the usual case), write the prior
        range in the panel corner instead of stretching the axis to reach it.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    """
    samples, names, latex, bounds, notes = _posterior_inputs(fit_or_samples)

    if params is None:
        index = list(range(samples.shape[1]))
    else:
        index = []
        for item in params:
            if isinstance(item, (int, np.integer)):
                index.append(int(item))
            elif item in names:
                index.append(names.index(item))
            else:
                matches = [i for i, n in enumerate(names)
                           if n.endswith('.' + item) or n.rsplit('.', 1)[-1] == item]
                if len(matches) != 1:
                    raise ValueError(f"parameter {item!r} is not in the fit "
                                     f"(candidates: {[n for n in names][:5]} ...)")
                index.append(matches[0])
    if labels is None:
        labels = [latex[i] if latex is not None else names[i] for i in index]
    elif len(labels) != len(index):
        raise ValueError("labels and params must have the same length")

    n_panels = len(index)
    ncols = max(1, min(ncols, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (4.1 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()

    log_scale = set(log_scale or ())
    lo_q, mid_q, hi_q = quantiles
    for panel, (column, label) in enumerate(zip(index, labels)):
        ax = flat_axes[panel]
        values = samples[:, column]
        kw = {'weights': weights, 'bins': bins, 'density': density, 'color': color,
              'alpha': alpha, 'histtype': histtype}
        if log_scale and names[column] in log_scale:
            positive = values > 0
            kw['bins'] = np.logspace(np.log10(values[positive].min()),
                                     np.log10(values.max()), bins)
            ax.set_xscale('log')
        ax.hist(values, **kw)

        lower, centre, upper = np.percentile(values, [100 * lo_q, 100 * mid_q,
                                                      100 * hi_q])
        ax.axvspan(lower, upper, color=color, alpha=0.18, lw=0)
        ax.axvline(centre, color='k', lw=1.4)

        # The view comes from the posterior.  Left to autoscaling, the prior-bound
        # lines below would drag the axis out to the prior width and turn a
        # well-measured parameter into a thin spike.
        on_log = bool(log_scale) and names[column] in log_scale
        if on_log:
            positive = values[values > 0]
            if positive.size:
                v_lo, v_hi = np.percentile(positive, [100 * view_quantiles[0],
                                                      100 * view_quantiles[1]])
                v_lo, v_hi = max(v_lo, positive.min()), max(v_hi, positive.min())
                ax.set_xlim(v_lo / (1.0 + pad), v_hi * (1.0 + pad))
        else:
            v_lo, v_hi = np.percentile(values, [100 * view_quantiles[0],
                                                100 * view_quantiles[1]])
            span = v_hi - v_lo
            if span <= 0:
                span = max(abs(v_hi), 1.0) * 1e-6
            v_lo, v_hi = v_lo - pad * span, v_hi + pad * span
            if values.min() >= 0.0 and v_lo < 0.0 < values.min():
                v_lo = 0.0
            ax.set_xlim(v_lo, v_hi)

        outside = []
        if show_bounds and bounds is not None:
            for limit in bounds[column]:
                if limit is None or not np.isfinite(limit):
                    continue
                if v_lo <= limit <= v_hi:
                    ax.axvline(limit, color='gray', ls=':', lw=1.0)
                else:
                    outside.append(limit)
        if prior_text:
            # a prior centred on another parameter is described by that centre and
            # its width, not by the box it is truncated to
            note = notes[column] if notes is not None else None
            if note:
                ax.text(0.98, 0.97, note, transform=ax.transAxes, fontsize=8.5,
                        color='gray', ha='right', va='top')
            elif bounds is not None and outside:
                limits = [limit for limit in bounds[column]
                          if limit is not None and np.isfinite(limit)]
                # the axis cannot show both the posterior and the prior: say where
                # the prior is instead of hiding it, and say that it is out of view
                span_text = ", ".join(f"{limit:.3g}" for limit in limits)
                ax.text(0.98, 0.97, f"prior [{span_text}]", transform=ax.transAxes,
                        fontsize=8.5, color='gray', ha='right', va='top')

        ax.set_xlabel(label, fontsize=15)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=12)
        if show_stats:
            ax.set_title(f"{centre:.4g} $^{{+{upper - centre:.3g}}}_{{-{centre - lower:.3g}}}$",
                         fontsize=12, pad=4)

    for ax in flat_axes[n_panels:]:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=140, bbox_inches='tight')
    return fig, axes


def _prior_bounds_for_columns(fit, flat_names):
    """Prior limits aligned with the sample columns, in physical units."""
    raw = getattr(fit, 'param_bounds', None)
    if not raw:
        return None
    if isinstance(raw, dict):
        pairs = [raw.get(name) for name in flat_names]
    else:
        pairs = list(raw)
        if len(pairs) != len(flat_names):
            return None
    # amplitudes are sampled in log10 when the default bounds are used
    log_names = set(getattr(fit, '_log_scale_params', ()) or ())
    out = []
    for name, pair in zip(flat_names, pairs):
        if pair is None:
            out.append((None, None))
            continue
        lower, upper = pair
        if name in log_names:
            lower, upper = 10.0 ** lower, 10.0 ** upper
        out.append((lower, upper))
    return out


def _posterior_inputs(fit_or_samples):
    """Normalise the input of :func:`plot_posterior_hist`.

    Returns ``(samples, names, labels, bounds, notes)``; ``notes`` is ``None``
    unless some parameters carry a prior centred on another parameter, in which
    case it holds one short description per column (or ``None``) so that the panel
    can say *what* that prior is instead of only where its box ends.
    """
    if hasattr(fit_or_samples, 'samples_equal_weight'):
        fit = fit_or_samples
        samples = np.asarray(fit.samples_equal_weight, dtype=float)
        flat = list(getattr(fit, 'param_names', []))
        names = list(getattr(fit, 'full_param_names', None) or flat)
        labels = getattr(fit, 'param_labels', None)
        bounds = _prior_bounds_for_columns(fit, flat)
        notes = _centred_prior_notes(fit, flat)
        return samples, names, labels, bounds, notes
    samples = np.asarray(fit_or_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("expected an (nsamples, ndim) array or a fit object")
    names = [f"param_{i}" for i in range(samples.shape[1])]
    return samples, names, None, None, None


def _centred_prior_notes(fit, flat_names):
    """One label per column for priors centred on another parameter, else ``None``."""
    centered = getattr(fit, 'centered_priors', None) or {}
    if not centered:
        return None
    labels = getattr(fit, 'param_labels', None)
    param_map = getattr(fit, 'param_map', None) or {}

    def spell(name):
        # ``param_map`` maps a qualified name to the column index -- either the index
        # itself (dynesty, nautilus, jax jnesty) or ``(index, name)`` (mcmc, and the
        # jnesty FITS loader), so both shapes have to work here.
        entry = param_map.get(name)
        if entry is None:
            return name
        index = entry[0] if isinstance(entry, tuple) else entry
        if labels is not None:
            return labels[index]
        return name

    notes = []
    for index in range(len(flat_names)):
        item = centered.get(index)
        if item is None:
            notes.append(None)
            continue
        prior = item[0]
        centre = spell(prior.center) if isinstance(prior.center, str) else f"{prior.center:g}"
        if prior.scale_from is not None:
            width = f"{prior.scale:g} x {spell(prior.scale_from)}"
        elif prior.relative:
            width = f"{100.0 * prior.scale:g}%"
        else:
            width = f"{prior.scale:g}"
        # a hard limit on the offset is part of the prior's definition, so it
        # belongs on the axis label with it
        nsigma = getattr(prior, 'nsigma', None)
        cap = ''
        if nsigma is not None and nsigma != 7.0:
            cap = f", $\\pm${nsigma:g}$\\sigma$"
        notes.append(f"$N$({centre}, {width}{cap})")
    return notes


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