# Plotting Functions (`galspec.plot`)

Visualization functions for spectral fitting results in GalSpec.

## Table of Contents

1. [plot_fit_new](#plot_fit_new)
2. [plot_fit](#plot_fit)

---

## plot_fit_new

Plot fitted spectrum with components and residuals.

```python
ax, axr = galspec.plot.plot_fit_new(
    wave,                   # Wavelength array (Å)
    flux,                   # Observed flux array
    model_fit,              # Fitted CompoundModel
    weight=None,            # Optional weights
    error=flux_err,         # Flux error array
    xlim=(6400, 6700),      # Optional x-axis limits
    components_to_plot=[    # Optional: specific components to plot
        'Broad Halpha',
        'nHalpha',
        'Abs. Halpha'
    ]
)
```

**Parameters**:
- `wave`: Wavelength array (Å)
- `flux`: Observed flux array
- `model_fit`: Fitted CompoundModel
- `weight`: Optional weight array
- `error`: Flux error array
- `xlim`: Optional x-axis limits as tuple (xmin, xmax)
- `components_to_plot`: Optional list of component names to plot individually

**Returns**:
- `ax`: Main plot axes
- `axr`: Residual plot axes

**Customization**:
```python
ax.set_ylabel(r'$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=16)
axr.set_xlabel('Rest Wavelength (Å)', fontsize=16)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fit_result.png', dpi=300)
```

---

## plot_fit

Alternative plotting function with more options.

```python
ax, axr = galspec.plot.plot_fit(
    wave, flux, model,
    weight=None,
    error=None,
    ax=None,                # Optional axes
    axr=None,               # Optional residual axes
    xlim=None,
    ylim0=None,             # Main panel y-limits
    ylim1=None,             # Residual panel y-limits
    xlabel=None,
    ylabel=None,
    legend_kwargs=None,
    plot_weight=True,
    ignore_list=None,       # Components to ignore
    legend_map=None,        # Custom legend mapping
    mask_list=None          # Wavelength ranges to mask
)
```

**Parameters**:
- `wave`: Wavelength array
- `flux`: Flux array
- `model`: CompoundModel to plot
- `weight`: Optional weight array
- `error`: Optional error array
- `ax`: Optional existing axes for main plot
- `axr`: Optional existing axes for residual plot
- `xlim`: X-axis limits
- `ylim0`: Y-axis limits for main panel
- `ylim1`: Y-axis limits for residual panel
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `legend_kwargs`: Dictionary of keyword arguments for legend
- `plot_weight`: Whether to plot weights
- `ignore_list`: List of component names to ignore in plot
- `legend_map`: Dictionary mapping component names to legend labels
- `mask_list`: List of wavelength ranges to mask (tuple of tuples)

**Returns**:
- `ax`: Main plot axes
- `axr`: Residual plot axes

---

**Module**: `galspec.plot`
**Source File**: `galspec/plot.py`
