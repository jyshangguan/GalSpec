# convolution_var.py 性能优化修改报告

## 1. 背景

`galspec.convolution_var.convolve_lsf_var` 使用逐点变宽度高斯卷积。在 MCMC
拟合中，每次 `log_probability` 都会调用一次被卷积包装的模型，因此即使光谱
网格只有几十个点，几十万次似然计算也会把卷积成本放大到不可忽略的程度。

本次优化针对 13704 PRISM 光谱拟合的真实路径，目标是让可变分辨率卷积的核
只构建一次，之后每次似然仅做点积。

## 2. 修改文件

- 修改文件：
  [convolution_var.py](../../galspec/convolution_var.py)
- 原始备份：
  [convolution_var.py.bak_pre_speedup](../../galspec/convolution_var.py.bak_pre_speedup)

## 3. 修改内容

### 3.1 新增 `_VariableKernel`

在 `ResolutionCurve` 类之后新增 `_VariableKernel`，用于：

1. 从波长网格和分辨率曲线一次性计算每个像素的 `sigma_x`。
2. 对每个输出像素预计算卷积窗口下标 `indices` 和归一化权重 `weights`。
3. 提供 `apply(flux)`，后续调用只做 `weights @ flux[indices]`。

卷积核只依赖波长网格、分辨率曲线和 `GaussianConv1DConfig`，不依赖模型参数，
因此 MCMC 期间可以安全复用。

### 3.2 `_smooth_1d_variable`

- 增加可选参数 `kernel_cache: Optional[_VariableKernel] = None`。
- 如果传入了 `kernel_cache`，直接调用 `kernel_cache.apply(yv)`。
- 如果没有传入，则自动构建一次 `_VariableKernel` 再使用，保持旧 API 兼容。
- 保留原有单位处理逻辑。

### 3.3 `_smooth_derivs_variable`

- 同样增加可选参数 `kernel_cache`。
- 列表、1D、2D 导数路径都复用同一个核缓存，避免每个导数分量重复建核。

### 3.4 `convolve_lsf_var`

- 在包装模型时构建一次 `_kernel_cache`。
- `__call__` 和 `fit_deriv` 均传入该缓存。

### 3.5 `refresh_variable_convolved_submodels_inplace`

- 重新应用包装时同样构建 `_kernel_cache`，保证 refresh 后仍走缓存路径。

## 4. 数值一致性验证

使用 13704 真实光谱网格及构造模型进行对比：

| 对比项 | 最大绝对误差 |
| --- | --- |
| 旧 `_smooth_1d_variable` vs 新 fallback | ~2.2e-16 |
| 旧 `_smooth_1d_variable` vs 新 cached | ~2.2e-16 |
| 旧 `_smooth_derivs_variable` vs 新 cached | ~1.1e-16 |
| 旧包装模型 vs 新包装模型（同网格） | ~1.1e-16 |
| 旧包装模型 vs 新包装模型（插值网格） | ~2.2e-16 |
| `refresh_variable_convolved_submodels_inplace` 前后输出 | 0.0 |

误差均在浮点舍入范围内，算法语义未改变。

## 5. 性能结果

13704 数据规模：78 个波长点；MCMC 配置：50 walkers × 10000 steps =
500000 次似然。

| 指标 | 修改前 | 修改后 | 加速比 |
| --- | --- | --- | --- |
| 单次卷积包装调用 | ~2.34 ms | ~0.057 ms | ~41x |
| 单次完整 `log_probability` | ~2.53 ms | ~0.258 ms | ~9.8x |
| 500000 次似然估算 | ~1264 s（约 21 min） | ~129 s（约 2.2 min） | ~9.8x |

实际 `log_probability` 的剩余耗时主要是 Astropy `CompoundModel` 的参数设置与模型
评估开销，而不是可变卷积本身。

## 6. 回归建议

1. 先使用较小的 `nsteps` 跑通 `13704_prism_fit.py` 完整流程。
2. 确认卷积后谱型与之前结果一致。
3. 再恢复为完整 `nsteps` 运行。

