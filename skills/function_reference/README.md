# GalSpec Function Reference

Complete reference documentation for all GalSpec functions, classes, and modules.

## Module Organization

This reference is organized by the Python modules in the `galspec` package:

1. **[Line Profile Models](line_profile_models.md)** (`galspec.line_profile`)
   - Gaussian, Gauss-Hermite, Multi-Gauss line profiles
   - Template-based and absorption lines

2. **[Continuum Models](continuum_models.md)** (`galspec.continuum`)
   - Power law, black body, Balmer continuum
   - Extinction models

3. **[Iron Templates](iron_templates.md)** (`galspec.iron_template`)
   - Iron emission templates
   - Velocity-shifted and tied convolution versions

4. **[Stellar Continuum](stellar_continuum.md)** (`galspec.stellar_continuum`)
   - Stellar spectrum templates
   - Multi-component stellar populations

5. **[Convolution Functions](convolution.md)** (`galspec.convolution`, `galspec.convolution_var`)
   - LSF convolution for constant and variable resolution

6. **[Fitting Classes](fitting_classes.md)** (`galspec.mcmc_fit`, `galspec.dynesty_fit`)
   - MCMC fitting with emcee
   - Nested sampling with dynesty

7. **[Parameter Tying Functions](parameter_tying.md)** (various modules)
   - Link parameters across components
   - Absorption, MultiGauss, template, and stellar tying

8. **[Plotting Functions](plotting.md)** (`galspec.plot`)
   - Plot fitted spectra with components and residuals

9. **[Utility Functions](utils.md)** (`galspec.utils`)
   - Line dictionaries, velocity conversions
   - BIC calculation, spectrum reading
   - Resolution degradation

10. **[Measurement Methods](measurement.md)** (`galspec.measure_method`)
    - Line FWHM measurements

11. **[I/O Functions](io.md)** (`galspec.mcmc_fit_io`, `galspec.dynesty_fit_io`)
    - Save and load fitting results

## Related Documentation

- **[Spectral Fitting Guide](../galspec_spectral_fitting.md)** - General workflow and data preparation
- **[Fitting Strategies](../fitting_strategies/)** - Specific strategies for different astronomical objects

---

**Version**: 1.0
**Last Updated**: 2025-03-20
