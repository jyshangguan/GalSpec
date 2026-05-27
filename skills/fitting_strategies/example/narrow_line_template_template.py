#!/usr/bin/env python
"""
Narrow Line Template Generation Template

This script is a template for generating narrow line templates from emission
lines. It demonstrates the standard workflow and plotting format recommended
by GalSpec.

Instructions:
1. Copy this script to your working directory
2. Modify the sections marked with "TODO" to fit your data
3. Run the script step by step to ensure quality at each stage
4. All plots will be saved automatically with standardized format

Author: GalSpec Community
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
from astropy.io import fits
import sys

# TODO: Modify path to point to your GalSpec installation
sys.path.insert(0, '/path/to/GalSpec')

import galspec
from galspec.utils import line_wave_dict
from galspec.continuum import WindowedPowerLaw1D
from galspec.plot import plot_narrow_line_diagnostic, plot_narrow_line_template_validation

# ========================================
# USER CONFIGURATION SECTION
# ========================================

# TODO: Set your target name and files
TARGET_NAME = 'YOUR_TARGET_NAME'  # e.g., 'SDSS-J000111.15-100155.5'
FITS_FILE = 'path/to/your/spectrum.fits'  # Path to your spectrum file
OUTPUT_PREFIX = 'narrow_template'  # Prefix for output files

# TODO: Set redshift and MW extinction
# These should be obtained from your spectrum header or catalog
REDSHIFT = 0.0  # TODO: Set your redshift
MW_EBV = 0.0    # TODO: Set Milky Way E(B-V)

# TODO: Select which line to use for template
# Options: 'SII' or 'OIII'
TEMPLATE_LINE = 'SII'  # 'SII' or 'OIII'

# ========================================
# 1. LOAD SPECTRUM
# ========================================
print('='*70)
print(f'Loading spectrum: {TARGET_NAME}')
print('='*70)

# TODO: Adapt this section to your data format
# This example shows how to load SDSS-style FITS files
# Modify if your data has a different format

if TEMPLATE_LINE == 'SII':
    # [S II] doublet wavelengths
    LINE1_REST = line_wave_dict['SII_6716']
    LINE2_REST = line_wave_dict['SII_6731']
    REGION_MIN = 6700  # TODO: Adjust for your spectrum
    REGION_MAX = 6745  # TODO: Adjust for your spectrum
elif TEMPLATE_LINE == 'OIII':
    # [O III] doublet wavelengths
    LINE1_REST = line_wave_dict['OIII_4959']
    LINE2_REST = line_wave_dict['OIII_5007']
    REGION_MIN = 4940  # TODO: Adjust for your spectrum
    REGION_MAX = 5020  # TODO: Adjust for your spectrum
else:
    raise ValueError(f"Unknown template line: {TEMPLATE_LINE}")

print(f'\nTemplate line: {TEMPLATE_LINE}')
print(f'  {LINE1_REST:.2f} Å')
print(f'  {LINE2_REST:.2f} Å')
print(f'  Fitting region: {REGION_MIN} - {REGION_MAX} Å (rest frame)')

# TODO: Load your spectrum data here
# Example for SDSS:
# hdul = fits.open(FITS_FILE)
# data = hdul['COADD'].data
# loglam = data['loglam']
# flux_obs = data['flux']
# ivar = data['ivar']
#
# # Mask bad values
# mask = ivar > 0
# loglam = loglam[mask]
# flux_obs = flux_obs[mask]
# ivar = ivar[mask]
#
# # Convert to linear wavelength
# wave_obs = 10**loglam
# ferr_obs = 1.0 / np.sqrt(ivar)

# Placeholder - replace with actual data loading
print('\nTODO: Load your spectrum data here')
print('  Set: wave_obs, flux_obs, ferr_obs')
print('  Convert to rest frame: wave_rest = wave_obs / (1 + z)')

# TODO: Apply MW extinction correction if needed
# from extinction import ccm89
# a_v = MW_EBV * 3.1
# r_v = 3.1
# flux_dered = flux_obs * 10**(0.4 * ccm89(wave_obs, a_v, r_v))
# ferr_dered = ferr_obs * 10**(0.4 * ccm89(wave_obs, a_v, r_v))

# TODO: Convert to rest frame
# wave_rest = wave_obs / (1 + REDSHIFT)

# Placeholder - replace with actual data
wave_rest = np.linspace(REGION_MIN, REGION_MAX, 100)  # TODO
flux_rest = np.random.randn(100) * 0.1 + 1.0  # TODO
ferr_rest = np.ones(100) * 0.1  # TODO

# Select region
region_mask = (wave_rest > REGION_MIN) & (wave_rest < REGION_MAX)
wave = wave_rest[region_mask]
flux = flux_rest[region_mask]
ferr = ferr_rest[region_mask]

print(f'\nRegion selected:')
print(f'  Wavelength range: {wave.min():.1f} - {wave.max():.1f} Å')
print(f'  Number of data points: {len(wave)}')

# ========================================
# 2. INITIAL CONTINUUM ESTIMATE
# ========================================
print('\n' + '='*70)
print('Estimating local continuum')
print('='*70)

# TODO: Estimate continuum from your data
# This is a rough initial guess - the fit will refine it
# Look at the continuum level on both sides of the lines

CONT_AMPLITUDE = 5.0  # TODO: Set based on your continuum level
CONT_X0 = (REGION_MIN + REGION_MAX) / 2  # Center of fitting region
CONT_ALPHA = 0.0  # Power-law index (0 = flat)

print(f'\nInitial continuum parameters:')
print(f'  Amplitude: {CONT_AMPLITUDE:.2f}')
print(f'  Reference wavelength: {CONT_X0:.1f} Å')
print(f'  Power-law index: {CONT_ALPHA:.2f}')

# ========================================
# 3. STEP 1: FIT WITH 1 COMPONENT
# ========================================
print('\n' + '='*70)
print(f'Step 1: Fit {TEMPLATE_LINE} with 1 Gaussian component')
print('='*70)

# Build model
cont_1 = WindowedPowerLaw1D(
    amplitude=CONT_AMPLITUDE,
    x_0=CONT_X0,
    alpha=CONT_ALPHA,
    x_min=REGION_MIN,
    x_max=REGION_MAX,
    name=f'Cont_{TEMPLATE_LINE}_1'
)

doublet_1 = galspec.Line_MultiGauss_doublet(
    n_components=1,
    amp_c0=5.0,      # TODO: Set based on your line strength
    amp_c1=4.0,      # TODO: Set based on your line strength
    dv_c=0.0,
    sigma_c=100.0,   # Initial guess for width (km/s)
    wavec0=LINE1_REST,
    wavec1=LINE2_REST,
    name=f'{TEMPLATE_LINE}_doublet_1'
)

model_init_1 = cont_1 + doublet_1
fitter = fitting.LevMarLSQFitter()
model_fit_1 = fitter(model_init_1, wave, flux,
                     weights=1/ferr**2, maxiter=10000)

# Calculate statistics
chi2_1 = np.sum(((flux - model_fit_1(wave)) / ferr)**2)
dof_1 = len(wave) - len(model_fit_1.parameters)
bic_1 = chi2_1 + len(model_fit_1.parameters) * np.log(len(wave))

print(f'\n1-component fit results:')
print(f'  Parameters: {len(model_fit_1.parameters)}')
print(f'  χ² = {chi2_1:.1f}, DOF = {dof_1}')
print(f'  χ²/DOF = {chi2_1/dof_1:.2f}')
print(f'  BIC = {bic_1:.1f}')
print(f'\nFitted parameters:')
print(f'  amp_c0 = {model_fit_1.amp_c0_1.value:.4f}')
print(f'  amp_c1 = {model_fit_1.amp_c1_1.value:.4f}')
print(f'  sigma_c = {model_fit_1.sigma_c_1.value:.2f} km/s')
print(f'  dv_c = {model_fit_1.dv_c_1.value:.2f} km/s')

# Create diagnostic plot (Type A)
fig, axes = plot_narrow_line_diagnostic(
    wave, flux, ferr, model_fit_1,
    title=f'Step 1: 1-Component {TEMPLATE_LINE} Fit',
    line_waves=[LINE1_REST, LINE2_REST],
    filename=f'{OUTPUT_PREFIX}_fit_1comp.png'
)
plt.close()

print(f'\n✓ Diagnostic plot saved to: {OUTPUT_PREFIX}_fit_1comp.png')

# ========================================
# 4. STEP 2: ADD SECOND COMPONENT (IF NEEDED)
# ========================================
print('\n' + '='*70)
print(f'Step 2: Fit {TEMPLATE_LINE} with 2 Gaussian components')
print('='*70)

user_input = input('\nDo you want to try a 2-component fit? (y/n): ')

if user_input.lower() == 'y':
    # Build 2-component model using best-fit from 1-component as initial guess
    cont_2 = WindowedPowerLaw1D(
        amplitude=model_fit_1.amplitude_0.value,
        x_0=model_fit_1.x_0_0.value,
        alpha=model_fit_1.alpha_0.value,
        x_min=REGION_MIN,
        x_max=REGION_MAX,
        name=f'Cont_{TEMPLATE_LINE}_2'
    )

    doublet_2 = galspec.Line_MultiGauss_doublet(
        n_components=2,
        amp_c0=model_fit_1.amp_c0_1.value,
        amp_c1=model_fit_1.amp_c1_1.value,
        dv_c=model_fit_1.dv_c_1.value,
        sigma_c=model_fit_1.sigma_c_1.value,
        amp_w0=0.5,        # TODO: Adjust wing amplitude initial guess
        dv_w0=-200.0,      # TODO: Adjust wing velocity initial guess
        sigma_w0=150.0,    # TODO: Adjust wing width initial guess
        wavec0=LINE1_REST,
        wavec1=LINE2_REST,
        name=f'{TEMPLATE_LINE}_doublet_2'
    )

    model_init_2 = cont_2 + doublet_2
    model_fit_2 = fitter(model_init_2, wave, flux,
                         weights=1/ferr**2, maxiter=10000)

    # Calculate statistics
    chi2_2 = np.sum(((flux - model_fit_2(wave)) / ferr)**2)
    dof_2 = len(wave) - len(model_fit_2.parameters)
    bic_2 = chi2_2 + len(model_fit_2.parameters) * np.log(len(wave))

    print(f'\n2-component fit results:')
    print(f'  Parameters: {len(model_fit_2.parameters)}')
    print(f'  χ² = {chi2_2:.1f}, DOF = {dof_2}')
    print(f'  χ²/DOF = {chi2_2/dof_2:.2f}')
    print(f'  BIC = {bic_2:.1f}')

    # Compare BIC
    print(f'\nBIC comparison:')
    print(f'  ΔBIC = {bic_2 - bic_1:.1f}')
    if bic_2 < bic_1:
        print(f'  ✓ 2-component model is preferred (ΔBIC < 0)')
        model_final = model_fit_2
        n_components = 2
    else:
        print(f'  ✓ 1-component model is preferred (ΔBIC > 0)')
        model_final = model_fit_1
        n_components = 1

    # Create diagnostic plot for 2-component
    fig, axes = plot_narrow_line_diagnostic(
        wave, flux, ferr, model_fit_2,
        title=f'Step 2: 2-Component {TEMPLATE_LINE} Fit',
        line_waves=[LINE1_REST, LINE2_REST],
        filename=f'{OUTPUT_PREFIX}_fit_2comp.png'
    )
    plt.close()
    print(f'\n✓ Diagnostic plot saved to: {OUTPUT_PREFIX}_fit_2comp.png')

else:
    print('\nUsing 1-component model')
    model_final = model_fit_1
    n_components = 1

# ========================================
# 5. GENERATE SINGLE-LINE TEMPLATE
# ========================================
print('\n' + '='*70)
print('Generating SINGLE-LINE narrow template')
print('='*70)

# Get the doublet component from the final model
doublet_fit = model_final[1]  # Second component is the doublet

# Create velocity array
velc_temp = np.linspace(-800, 800, 2000)  # km/s

# Generate template using gen_template() method
# This extracts a single-line profile shape from the doublet model
flux_temp = doublet_fit.gen_template(velc_temp, normalized=True)

print(f'\nTemplate properties:')
print(f'  Velocity range: {velc_temp[0]:.0f} to {velc_temp[-1]:.0f} km/s')
peak_idx = np.argmax(flux_temp)
peak_vel = velc_temp[peak_idx]
print(f'  Peak at velocity: {peak_vel:.2f} km/s')

# Quality check: Peak should be at v ≈ 0
if abs(peak_vel) < 10:
    print(f'  ✓ Peak correctly centered near 0 km/s')
else:
    print(f'  ✗ WARNING: Peak at {peak_vel:.2f} km/s (should be ~0)')

# Measure FWHM
from scipy.interpolate import interp1d
f_interp = interp1d(velc_temp, flux_temp, kind='cubic',
                    bounds_error=False, fill_value=0)
velc_fine = np.linspace(velc_temp[0], velc_temp[-1], 10000)
flux_fine = f_interp(velc_fine)
crossings = np.where(np.diff(np.sign(flux_fine - 0.5)))[0]
if len(crossings) >= 2:
    fwhm_temp = velc_fine[crossings[-1]] - velc_fine[crossings[0]]
    print(f'  Template FWHM: {fwhm_temp:.1f} km/s')

    # Quality check: FWHM should be reasonable
    if fwhm_temp < 800:
        print(f'  ✓ FWHM is reasonable')
    else:
        print(f'  ✗ WARNING: FWHM is large - check for broad line contamination')

# ========================================
# 6. VERIFICATION: FIT WITH TEMPLATE
# ========================================
print('\n' + '='*70)
print(f'Verification: Fitting {TEMPLATE_LINE} doublet with template')
print('='*70)

# Get velocity shift from the fit
dv_measured = model_final.dv_c_1.value

# Build continuum
cont_verify = WindowedPowerLaw1D(
    amplitude=model_final.amplitude_0.value,
    x_0=model_final.x_0_0.value,
    alpha=model_final.alpha_0.value,
    x_min=REGION_MIN,
    x_max=REGION_MAX,
    name='Cont'
)

# Create template lines for both doublet members
line1_temp = galspec.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=model_final.amp_c0_1.value,
    dv=dv_measured,
    wavec=LINE1_REST,
    name=f'{TEMPLATE_LINE}_1'
)

line2_temp = galspec.Line_template(
    template_velc=velc_temp,
    template_flux=flux_temp,
    amplitude=model_final.amp_c1_1.value,
    dv=dv_measured,
    wavec=LINE2_REST,
    name=f'{TEMPLATE_LINE}_2'
)

# Combine and fit
model_template_init = cont_verify + line1_temp + line2_temp
model_template_fit = fitter(model_template_init, wave, flux,
                            weights=1/ferr**2, maxiter=10000)

print(f'\nTemplate fit results:')
print(f'  {TEMPLATE_LINE} {LINE1_REST:.0f} amplitude: {model_template_fit.amplitude_1.value:.4f}')
print(f'  {TEMPLATE_LINE} {LINE2_REST:.0f} amplitude: {model_template_fit.amplitude_2.value:.4f}')
print(f'  dv: {model_template_fit.dv_1.value:.2f} km/s')

# Calculate line ratio
ratio = model_template_fit.amplitude_1.value / model_template_fit.amplitude_2.value
print(f'  Line ratio: {ratio:.3f}')

# Compare χ²
resid_original = flux - model_final(wave)
resid_template = flux - model_template_fit(wave)
chi2_original = np.sum((resid_original / ferr)**2)
chi2_template = np.sum((resid_template / ferr)**2)

print(f'\nχ² comparison:')
print(f'  Original {n_components}-component Gaussian: χ² = {chi2_original:.1f}')
print(f'  Template-based fit: χ² = {chi2_template:.1f}')
print(f'  Difference: {chi2_template - chi2_original:.1f}')

# Quality check: Δχ² should be small
if abs(chi2_template - chi2_original) < 20:
    print(f'  ✓ Template fit matches original (Δχ² < 20)')
else:
    print(f'  ✗ WARNING: Large Δχ² - check template generation')

# ========================================
# 7. CREATE VALIDATION PLOT (Type B)
# ========================================
print('\n' + '='*70)
print('Creating validation plot')
print('='*70)

fig, axes = plot_narrow_line_template_validation(
    wave, flux, ferr,
    model_final,  # Original Gaussian fit
    model_template_fit,  # Template-based fit
    velc_temp, flux_temp,
    title=f'{TEMPLATE_LINE} Template Validation',
    line_waves=[LINE1_REST, LINE2_REST],
    filename=f'{OUTPUT_PREFIX}_validation.png'
)
plt.close()

print(f'\n✓ Validation plot saved to: {OUTPUT_PREFIX}_validation.png')

# ========================================
# 8. SAVE TEMPLATE
# ========================================
print('\n' + '='*70)
print('Saving narrow line template')
print('='*70)

# Check template quality before saving
quality_ok = True
issues = []

if abs(peak_vel) >= 10:
    quality_ok = False
    issues.append(f'Peak not centered (at {peak_vel:.1f} km/s)')

if fwhm_temp >= 800:
    quality_ok = False
    issues.append(f'FWHM too large ({fwhm_temp:.1f} km/s)')

if abs(chi2_template - chi2_original) >= 20:
    quality_ok = False
    issues.append(f'Large Δχ² ({chi2_template - chi2_original:.1f})')

if quality_ok:
    print('✓ SUCCESS: Template quality checks passed!')
    print('')

    # Save as ASCII text
    np.savetxt(f'{TARGET_NAME}_{OUTPUT_PREFIX}.txt',
               np.column_stack([velc_temp, flux_temp]),
               header='velocity_kms normalized_flux')

    print('Template saved:')
    print(f'  - {TARGET_NAME}_{OUTPUT_PREFIX}.txt')
    print('')
    print('Template properties:')
    print(f'  - Single-line profile shape')
    print(f'  - FWHM: {fwhm_temp:.1f} km/s')
    print(f'  - Core σ: {model_final.sigma_c_1.value:.1f} km/s')
    if n_components == 2:
        print(f'  - Wing component: amp={model_final.amp_w0_1.value:.3f}, '
              f'dv={model_final.dv_w0_1.value:.1f} km/s, '
              f'σ={model_final.sigma_w0_1.value:.1f} km/s')
    print(f'  - Centered at dv=0')
    print('')
    print('Usage example:')
    print('  from galspec.utils import line_wave_dict')
    print('  nha = galspec.Line_template(')
    print('      template_velc=velc_temp,')
    print('      template_flux=flux_temp,')
    print('      amplitude=50.0,  # Adjust for your line')
    print('      dv=10,           # Initial velocity guess')
    print('      wavec=line_wave_dict["Halpha"])')

else:
    print('✗ WARNING: Template quality checks failed!')
    print('\nIssues found:')
    for issue in issues:
        print(f'  - {issue}')
    print('\nPlease check your fit and try again:')
    print('  1. Examine the diagnostic plots')
    print('  2. Check for broad line contamination')
    print('  3. Verify continuum subtraction')
    print('  4. Try adjusting fitting region')

print('\n' + '='*70)
print('Template generation workflow complete!')
print('='*70)
print('\nOutput files:')
print(f'  - {OUTPUT_PREFIX}_fit_1comp.png (1-component diagnostic)')
if user_input.lower() == 'y':
    print(f'  - {OUTPUT_PREFIX}_fit_2comp.png (2-component diagnostic)')
print(f'  - {OUTPUT_PREFIX}_validation.png (template validation)')
if quality_ok:
    print(f'  - {TARGET_NAME}_{OUTPUT_PREFIX}.txt (template data)')
print('='*70)
