# Server convolution_var.py Optimization Patch Guide

This document enables Codex on the server to reproduce, in the same codebase,
the `convolution_var.py` performance optimization that was already completed
locally.

## Objective

Optimize the variable-resolution convolution in
`galspec/convolution_var.py` so that during MCMC fitting the convolution kernel
is built only once, and every subsequent model call performs only the
precomputed weight dot product.

## Required Backup Before Execution

```bash
cp -p /path/to/GalSpec/galspec/convolution_var.py \
      /path/to/GalSpec/galspec/convolution_var.py.bak_pre_speedup
```

Replace `/path/to/GalSpec` with the actual server path.

## Target File

Target file:

```text
GalSpec/galspec/convolution_var.py
```

## Change 1: Add `_VariableKernel`

After the `ResolutionCurve` class definition and before the
`_smooth_1d_variable` function, add the following class:

```python
class _VariableKernel:
    """
    Precomputed variable-width Gaussian convolution kernels.

    The convolution weights depend only on the wavelength grid and the
    resolution curve, not on the model flux.  This class builds the per-pixel
    index/weight pairs once and then applies them to arbitrary flux vectors
    with a small Python loop.
    """

    def __init__(
        self,
        wave: np.ndarray,
        resolution_curve: ResolutionCurve,
        cfg: GaussianConv1DConfig
    ) -> None:
        wavev = np.asarray(getattr(wave, "value", wave), dtype=float)
        if wavev.ndim != 1 or len(wavev) == 0:
            raise ValueError("wave must be a non-empty 1D array")

        self.wave = wavev
        self.n = len(wavev)

        # Average wavelength spacing used by the original implementation.
        dw = float(np.median(np.diff(wavev)))
        if not np.isfinite(dw) or dw <= 0:
            raise ValueError("wave must be increasing and finite")

        sigma_x = np.asarray(resolution_curve.get_sigma_x(wavev), dtype=float)

        # Match the unit conversion performed by the original per-pixel loop.
        wave_median = float(np.median(wavev))
        if resolution_curve._wave_unit == "micron" and wave_median > 100:
            sigma_x = sigma_x * 1e4
        elif resolution_curve._wave_unit == "angstrom" and wave_median < 100:
            sigma_x = sigma_x / 1e4

        sigma_pix = sigma_x / dw
        truncate = cfg.truncate

        indices: List[np.ndarray] = []
        weights: List[np.ndarray] = []

        for i in range(self.n):
            if sigma_pix[i] <= 0.5:
                indices.append(np.array([i], dtype=int))
                weights.append(np.array([1.0]))
                continue

            kernel_width = int(truncate * sigma_pix[i])
            i_start = max(0, i - kernel_width)
            i_end = min(self.n, i + kernel_width + 1)
            idx = np.arange(i_start, i_end, dtype=int)

            w = np.exp(-0.5 * ((wavev[idx] - wavev[i]) / sigma_x[i]) ** 2)
            weight_sum = float(np.sum(w))
            if weight_sum > 0:
                w = w / weight_sum
            else:
                # Degenerate kernel: behave like the original fallback, which
                # keeps the central flux value unchanged.
                w = (idx == i).astype(float)

            indices.append(idx)
            weights.append(w)

        self.indices = tuple(indices)
        self.weights = tuple(weights)

    def apply(self, yv: np.ndarray) -> np.ndarray:
        """Convolve a flux array using the precomputed kernels."""
        if yv.shape != (self.n,):
            raise ValueError(
                f"flux has shape {yv.shape}, expected {(self.n,)}"
            )

        result = np.empty_like(yv, dtype=float)
        for i in range(self.n):
            result[i] = np.dot(self.weights[i], yv[self.indices[i]])
        return result
```

## Change 2: Replace `_smooth_1d_variable`

Replace the original function signature and body with:

```python
def _smooth_1d_variable(
    y: Any,
    wave: np.ndarray,
    resolution_curve: Optional[ResolutionCurve] = None,
    *,
    cfg: GaussianConv1DConfig,
    kernel_cache: Optional[_VariableKernel] = None
) -> Any:
    """
    Apply variable-width Gaussian convolution (simplified implementation).

    This is a basic implementation that applies local Gaussian convolution
    at each wavelength point using the wavelength-dependent sigma from the
    resolution curve.

    For production use, consider optimizations like:
    - Vectorized operations
    - Kernel caching for similar wavelength ranges
    - Numba/C acceleration

    Parameters
    ----------
    y : array-like
        Flux values to convolve
    wave : np.ndarray
        Wavelength array (must be same length as y)
    resolution_curve : ResolutionCurve
        Resolution curve object
    cfg : GaussianConv1DConfig
        Convolution configuration
    kernel_cache : _VariableKernel, optional
        Precomputed variable convolution kernel.  When provided, it avoids
        rebuilding the per-pixel kernels on every call.

    Returns
    -------
    convolved : array-like
        Convolved flux values
    """
    unit = getattr(y, 'unit', None)
    yv = np.asarray(getattr(y, 'value', y), dtype=float)

    if kernel_cache is None:
        if resolution_curve is None:
            raise ValueError(
                "resolution_curve and kernel_cache cannot both be None"
            )
        wavev = np.asarray(getattr(wave, 'value', wave), dtype=float)
        if yv.shape != wavev.shape:
            raise ValueError(
                f"y and wave must have same shape: {yv.shape} vs {wavev.shape}"
            )
        kernel_cache = _VariableKernel(wavev, resolution_curve, cfg)
    elif yv.shape != (kernel_cache.n,):
        raise ValueError(
            f"y has shape {yv.shape}, expected {(kernel_cache.n,)}"
        )

    result = kernel_cache.apply(yv)

    return result if unit is None else result * unit
```

## Change 3: Replace `_smooth_derivs_variable`

Replace the original function signature and body with:

```python
def _smooth_derivs_variable(
    derivs: Any,
    wave: np.ndarray,
    resolution_curve: Optional[ResolutionCurve] = None,
    *,
    cfg: GaussianConv1DConfig,
    kernel_cache: Optional[_VariableKernel] = None
) -> Any:
    """
    Convolve derivative vectors with variable LSF.

    For variable convolution, derivatives become more complex as the
    convolution itself varies with wavelength. This implementation provides
    a basic approach that convolves each derivative vector.

    Parameters
    ----------
    derivs : array-like
        Derivative vectors (typically for fit_deriv)
    wave : np.ndarray
        Wavelength array
    resolution_curve : ResolutionCurve
        Resolution curve object
    cfg : GaussianConv1DConfig
        Convolution configuration
    kernel_cache : _VariableKernel, optional
        Precomputed variable convolution kernel.

    Returns
    -------
    convolved_derivs : array-like
        Convolved derivative vectors
    """
    if kernel_cache is None:
        if resolution_curve is None:
            raise ValueError(
                "resolution_curve and kernel_cache cannot both be None"
            )
        wavev = np.asarray(getattr(wave, 'value', wave), dtype=float)
        kernel_cache = _VariableKernel(wavev, resolution_curve, cfg)

    if isinstance(derivs, (list, tuple)):
        out = [
            _smooth_1d_variable(
                d, wave, resolution_curve, cfg=cfg, kernel_cache=kernel_cache
            )
            for d in derivs
        ]
        return type(derivs)(out)

    d = np.asarray(derivs)
    if d.ndim == 1:
        return _smooth_1d_variable(
            d, wave, resolution_curve, cfg=cfg, kernel_cache=kernel_cache
        )
    if d.ndim == 2:
        # Convolve each parameter's derivative
        result = np.zeros_like(d)
        for i in range(d.shape[0]):
            result[i] = _smooth_1d_variable(
                d[i], wave, resolution_curve, cfg=cfg, kernel_cache=kernel_cache
            )
        return result

    raise ValueError(f"Unsupported fit_deriv shape: {d.shape}")
```

## Change 4: Build and Reuse the Cache in `convolve_lsf_var`

After this check:

```python
    # Check if already convolved
    if getattr(model, _TAG_VAR, None):
        return model
```

Add:

```python
    # Build the variable convolution kernel once.  The kernel depends only on
    # the wavelength grid and resolution curve, so it can be reused by every
    # subsequent __call__/fit_deriv invocation (important for MCMC).
    _kernel_cache = _VariableKernel(_wavec_ref, _rc_ref, cfg)
```

Then change the convolution call in `__call__` to:

```python
        result = _smooth_1d_variable(
            y_to_convolve,
            wave_to_use,
            _rc_ref,
            cfg=cfg,
            kernel_cache=_kernel_cache,
        )
```

Change the call in `fit_deriv` to:

```python
            return _smooth_derivs_variable(
                d0,
                _wavec_ref,
                _rc_ref,
                cfg=cfg,
                kernel_cache=_kernel_cache,
            )
```

## Change 5: Update `refresh_variable_convolved_submodels_inplace`

Before rewrapping the model in the refresh function, add:

```python
            _kernel_cache = _VariableKernel(_wavec_ref, _rc_ref, cfg)
```

Then update the convolution calls in its internal `__call__` and `fit_deriv` to
pass `kernel_cache=_kernel_cache`, in the same way as Change 4.

## Mandatory Verification After Modification

1. Syntax check:

```bash
python -c "compile(open('GalSpec/galspec/convolution_var.py').read(), 'GalSpec/galspec/convolution_var.py', 'exec')"
```

2. Numerical consistency: compare the old backup module and the new module for
   `_smooth_1d_variable`, `_smooth_derivs_variable`, and wrapped-model outputs.
   The maximum absolute error should be on the order of `1e-16`.

3. Performance regression: use the 13704 real data or a grid of the same size
   to test a single `log_probability` call. Confirm that the runtime drops
   significantly; the expected single convolution-wrapper call should drop from
   about `2.3 ms` to about `0.06 ms`.

4. Run the full `13704_prism_fit.py` fit once with a smaller `nsteps`. After
   confirming the results are normal, restore the original number of steps.

## Expected Results

| Metric | Before | After |
| --- | --- | --- |
| Single convolution-wrapper call | ~2.34 ms | ~0.057 ms |
| Single full `log_probability` | ~2.53 ms | ~0.258 ms |
| 500000 likelihood evaluations | ~21 min | ~2.2 min |
