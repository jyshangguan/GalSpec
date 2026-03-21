# SAGAN Function Reference

Complete reference documentation for all SAGAN functions, classes, and modules.

## Module Organization

This reference is organized by the Python modules in the `sagan` package:

1. **[Line Profile Models](line_profile_models.md)** (`sagan.line_profile`)
   - Gaussian, Gauss-Hermite, Multi-Gauss line profiles
   - Template-based and absorption lines

2. **[Continuum Models](continuum_models.md)** (`sagan.continuum`)
   - Power law, black body, Balmer continuum
   - Extinction models

3. **[Iron Templates](iron_templates.md)** (`sagan.iron_template`)
   - Iron emission templates
   - Velocity-shifted and tied convolution versions

4. **[Stellar Continuum](stellar_continuum.md)** (`sagan.stellar_continuum`)
   - Stellar spectrum templates
   - Multi-component stellar populations

5. **[Convolution Functions](convolution.md)** (`sagan.convolution`, `sagan.convolution_var`)
   - LSF convolution for constant and variable resolution

6. **[Fitting Classes](fitting_classes.md)** (`sagan.mcmc_fit`, `sagan.dynesty_fit`)
   - MCMC fitting with emcee
   - Nested sampling with dynesty

7. **[Parameter Tying Functions](parameter_tying.md)** (various modules)
   - Link parameters across components
   - Absorption, MultiGauss, template, and stellar tying

8. **[Plotting Functions](plotting.md)** (`sagan.plot`)
   - Plot fitted spectra with components and residuals

9. **[Utility Functions](utils.md)** (`sagan.utils`)
   - Line dictionaries, velocity conversions
   - BIC calculation, spectrum reading
   - Resolution degradation

10. **[Measurement Methods](measurement.md)** (`sagan.measure_method`)
    - Line FWHM measurements

11. **[I/O Functions](io.md)** (`sagan.mcmc_fit_io`, `sagan.dynesty_fit_io`)
    - Save and load fitting results

## Related Documentation

- **[Spectral Fitting Guide](../sagan_spectral_fitting.md)** - General workflow and data preparation
- **[Fitting Strategies](../fitting_strategies/)** - Specific strategies for different astronomical objects

---

**Version**: 1.0
**Last Updated**: 2025-03-20
