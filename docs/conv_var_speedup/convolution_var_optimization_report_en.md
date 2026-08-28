# convolution_var.py Performance Optimization Report

## 1. Background

`galspec.convolution_var.convolve_lsf_var` uses point-by-point variable-width
Gaussian convolution. During MCMC fitting, each `log_probability` evaluation
invokes the convolution-wrapped model once, so even with only a few dozen
spectral grid points, hundreds of thousands of likelihood evaluations make the
convolution cost non-negligible.

This optimization targets the real 13704 PRISM spectral fitting path. Its goal
is to build the variable-resolution convolution kernel only once and then
perform only dot products for each subsequent likelihood evaluation.

## 2. Modified Files

- Modified file:
  [convolution_var.py](../../galspec/convolution_var.py)
- Original backup:
  [convolution_var.py.bak_pre_speedup](../../galspec/convolution_var.py.bak_pre_speedup)

## 3. Changes

### 3.1 Added `_VariableKernel`

Added `_VariableKernel` after the `ResolutionCurve` class. It:

1. Computes `sigma_x` for every pixel once from the wavelength grid and the
   resolution curve.
2. Precomputes the convolution-window indices `indices` and normalized weights
   `weights` for every output pixel.
3. Provides `apply(flux)`, so subsequent calls only perform
   `weights @ flux[indices]`.

The convolution kernel depends only on the wavelength grid, the resolution
curve, and `GaussianConv1DConfig`; it does not depend on model parameters.
Therefore it can be safely reused during MCMC.

### 3.2 `_smooth_1d_variable`

- Adds an optional parameter
  `kernel_cache: Optional[_VariableKernel] = None`.
- If `kernel_cache` is provided, directly calls `kernel_cache.apply(yv)`.
- If not provided, builds a `_VariableKernel` once and uses it, preserving
  backward compatibility with the old API.
- Keeps the original unit-handling logic.

### 3.3 `_smooth_derivs_variable`

- Adds the same optional `kernel_cache` parameter.
- The list, 1D, and 2D derivative paths all reuse the same kernel cache,
  avoiding kernel rebuilds for each derivative component.

### 3.4 `convolve_lsf_var`

- Builds `_kernel_cache` once when wrapping the model.
- Passes this cache to both `__call__` and `fit_deriv`.

### 3.5 `refresh_variable_convolved_submodels_inplace`

- Builds `_kernel_cache` again when reapplying the wrapper, ensuring that
  refreshed models still use the cached path.

## 4. Numerical Consistency Verification

Comparison using the 13704 real spectral grid and constructed models:

| Comparison | Maximum absolute error |
| --- | --- |
| Old `_smooth_1d_variable` vs new fallback | ~2.2e-16 |
| Old `_smooth_1d_variable` vs new cached | ~2.2e-16 |
| Old `_smooth_derivs_variable` vs new cached | ~1.1e-16 |
| Old wrapped model vs new wrapped model (same grid) | ~1.1e-16 |
| Old wrapped model vs new wrapped model (interpolated grid) | ~2.2e-16 |
| Output before and after `refresh_variable_convolved_submodels_inplace` | 0.0 |

All errors are within floating-point rounding range, so the algorithm semantics
are unchanged.

## 5. Performance Results

13704 dataset size: 78 wavelength points; MCMC configuration:
50 walkers x 10000 steps = 500000 likelihood evaluations.

| Metric | Before | After | Speedup |
| --- | --- | --- | --- |
| Single convolution-wrapper call | ~2.34 ms | ~0.057 ms | ~41x |
| Single full `log_probability` | ~2.53 ms | ~0.258 ms | ~9.8x |
| 500000 likelihood evaluations | ~1264 s (~21 min) | ~129 s (~2.2 min) | ~9.8x |

The remaining `log_probability` cost is dominated by Astropy `CompoundModel`
parameter setup and model evaluation, not by the variable convolution itself.

## 6. Regression Recommendations

1. First run the full `13704_prism_fit.py` workflow with a smaller `nsteps`.
2. Confirm that the convolved spectral shapes match previous results.
3. Then restore the full `nsteps` and run the production fit.
