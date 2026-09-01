# Dynesty development

This folder now contains one maintained real-data demo and a focused regression
test. Historical Claude Code experiments and reports are retained in
`bak/2026-08-31_pre_parallel_rewrite/`.

Run from the GalSpec repository with the environment that has the project
dependencies installed:

```bash
/Users/shangguan/Softwares/miniforge3/envs/norm/bin/python develop/dev_dynesty/test_dynesty.py
/Users/shangguan/Softwares/miniforge3/envs/norm/bin/python develop/dev_dynesty/demo_ir07546.py --cores 4
```

The demo uses the same IRAS 07546+3928 FITS spectrum as the earlier work, fits
the 4900–5100 Å [O III] region, runs identical serial and parallel nested-sampling
fits, and writes `demo_results.json` plus `demo_ir07546_fit.png`. By default it
interpolates the native spectrum by 8× and scales the errors by √8, preserving
approximately the same total likelihood weight while making the timing comparison
representative of a more expensive/full-spectrum fit. Use `--oversample 1` to fit
only native pixels; that tiny workload may not benefit materially from processes.

Parallel sampling is enabled with `n_processes=N`. GalSpec creates and closes a
`multiprocess.Pool`, which supports Astropy models containing locally defined tie
functions. Set `OMP_NUM_THREADS=1` (the demo does this) to avoid each worker also
starting a BLAS thread. Parallelism helps when each likelihood evaluation is
expensive; for very small models, process overhead can outweigh the benefit.

`demo_ir07546_full_spectrum.py` reproduces the notebook's additive 4200–6800 Å
model without the multiplicative polynomial. It keeps all line components, ties,
and the 5770–6000 Å mask. By default it displays dynesty's progress bar and runs
until the evidence stopping criterion is reached. `--maxiter` is only an optional
safety limit for short diagnostic runs.
