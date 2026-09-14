# 服务器 convolution_var.py 优化修改指南

本文件用于让服务器上的 Codex 在相同代码库中复现本地已经完成的
`convolution_var.py` 性能优化。

## 任务目标

优化 `galspec/convolution_var.py` 中的可变分辨率卷积，使 MCMC 拟合期间卷积核
只构建一次，后续每次模型调用仅执行预计算权重点积。

## 执行前必须备份

```bash
cp -p /path/to/GalSpec/galspec/convolution_var.py \
      /path/to/GalSpec/galspec/convolution_var.py.bak_pre_speedup
```

请根据服务器实际路径替换 `/path/to/GalSpec`。

## 修改文件

目标文件：

```text
GalSpec/galspec/convolution_var.py
```

## 修改点 1：新增 `_VariableKernel`

在 `ResolutionCurve` 类定义之后、`_smooth_1d_variable` 函数之前，新增以下类：

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

## 修改点 2：替换 `_smooth_1d_variable`

将原函数签名和函数体替换为：

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

## 修改点 3：替换 `_smooth_derivs_variable`

将原函数签名和函数体替换为：

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

## 修改点 4：在 `convolve_lsf_var` 中构建并复用缓存

在以下检查之后：

```python
    # Check if already convolved
    if getattr(model, _TAG_VAR, None):
        return model
```

新增：

```python
    # Build the variable convolution kernel once.  The kernel depends only on
    # the wavelength grid and resolution curve, so it can be reused by every
    # subsequent __call__/fit_deriv invocation (important for MCMC).
    _kernel_cache = _VariableKernel(_wavec_ref, _rc_ref, cfg)
```

然后将 `__call__` 中的卷积调用改为：

```python
        result = _smooth_1d_variable(
            y_to_convolve,
            wave_to_use,
            _rc_ref,
            cfg=cfg,
            kernel_cache=_kernel_cache,
        )
```

将 `fit_deriv` 中的调用改为：

```python
            return _smooth_derivs_variable(
                d0,
                _wavec_ref,
                _rc_ref,
                cfg=cfg,
                kernel_cache=_kernel_cache,
            )
```

## 修改点 5：同步修改 `refresh_variable_convolved_submodels_inplace`

在 refresh 函数中重新包装模型前，同样新增：

```python
            _kernel_cache = _VariableKernel(_wavec_ref, _rc_ref, cfg)
```

并将其内部 `__call__` 和 `fit_deriv` 的卷积调用改为传入
`kernel_cache=_kernel_cache`，方式与修改点 4 相同。

## 修改后必须验证

1. 语法检查：

```bash
python -c "compile(open('GalSpec/galspec/convolution_var.py').read(), 'GalSpec/galspec/convolution_var.py', 'exec')"
```

2. 数值一致性：用旧备份模块和新模块对比
   `_smooth_1d_variable`、`_smooth_derivs_variable` 以及包装后模型的输出，
   最大绝对误差应处于 `1e-16` 量级。

3. 性能回归：用 13704 真实数据或相同规模网格测试单次 `log_probability`，
   确认耗时比修改前明显下降；预期单次卷积包装调用从约 `2.3 ms` 降至约
   `0.06 ms`。

4. 用较小 `nsteps` 跑通一次 `13704_prism_fit.py` 完整拟合，确认结果正常后
   再恢复原步数。

## 预期效果

| 指标 | 修改前 | 修改后 |
| --- | --- | --- |
| 单次卷积包装调用 | ~2.34 ms | ~0.057 ms |
| 单次完整 `log_probability` | ~2.53 ms | ~0.258 ms |
| 500000 次似然估算 | ~21 min | ~2.2 min |

