import numpy as np
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
import astropy.units as u
from .constants import ls_km

__all__ = ['WindowedPowerLaw1D', 'BlackBody', 'BalmerPseudoContinuum']


class WindowedPowerLaw1D(Fittable1DModel):
    """Power law that is zero outside [x_min, x_max]."""
    amplitude = Parameter(default=1.0)
    x_0 = Parameter(default=4500.0, fixed=True)
    alpha = Parameter(default=-1.0)
    x_min = Parameter(default=4000.0, fixed=True)
    x_max = Parameter(default=5000.0, fixed=True)

    def evaluate(self, x, amplitude, x_0, alpha, x_min, x_max):
        """Power law that is zero outside [x_min, x_max]."""
        y = amplitude * (x / x_0)**-alpha
        mask = (x >= x_min) & (x <= x_max)
        return np.where(mask, y, 0.0)
    

class BlackBody(Fittable1DModel):
    """Black body radiation model."""
    temperature = Parameter(default=5000.0)  # in Kelvin
    scale = Parameter(default=1.0)
    x_min = Parameter(default=-np.inf, fixed=True)
    x_max = Parameter(default=np.inf, fixed=True)

    def evaluate(self, x, temperature, scale, x_min, x_max):
        """Evaluate the black body radiation at wavelength x."""
        bb_lam = _planck_B_lambda(x * u.AA.to(u.cm), temperature)
        bb_lam /= np.max([bb_lam.max(), 1e-16])  # Normalize for convenience
        mask = (x >= x_min) & (x <= x_max)
        bb_lam = np.where(mask, bb_lam, 0.0)
        return scale * bb_lam


class BalmerPseudoContinuum(Fittable1DModel):
    """
    Balmer pseudo-continuum model following Kovačević et al. (2013,
    arXiv:1311.6653), combining:
      - Balmer continuum (λ <= λ_BE) from Grandi (1982), with Te, τ_BE
      - Balmer emission lines (n -> 2, n = 3..400) as Gaussians with
        common width and shift
      - Balmer continuum intensity at λ_BE derived from the sum of
        high-order Balmer lines at λ_BE (not a free parameter)

    Input
    -----
    wave : array_like
        Wavelength in Angstrom.

    Parameters
    ----------
    i_ref : float
        Peak intensity of the reference Balmer line (Hβ, n_ref = 4).
        This sets the absolute scale of *all* Balmer lines and,
        through Eq. (5–7) in Kovačević+2013, the Balmer continuum.

    sigma : float
        Doppler width parameter sigma [km/s] used in the Gaussian:
            V = (λ - λ_i_shift) / λ_i_shift * c
            G_i(λ) = I_i * exp( - (V / sigma)**2 )
        (This matches their Eq. (6) form without the 1/2.)

    dv : float
        Velocity shift of all Balmer lines:
            λ_i_shift = λ_i * (1 + dv / ls_km)
    
    te : float
        Electron temperature [K] used both in the Planck function
        and in the relative line intensities. Fixed by default to
        15000 K.

    tau_be : float
        Optical depth at the Balmer edge (λ_BE). Fixed by default to 1.

    lambda_be : float
        Balmer edge wavelength [Angstrom]. Fixed by default to 3646 Å.

    Notes
    -----
    - Balmer series wavelengths (n -> 2) are computed using the Rydberg
      formula:
          1 / λ_n = R * (1/2^2 - 1/n^2)
      with R in cm^-1, λ in Angstrom.

    - Relative intensities follow the thermodynamic approximation of
      Kovačević+2013 (Eq. 4):
          I_n / I_ref ≈ exp[ - (E_n - E_ref) / (k_B * T_e) ]
      where E_n are the hydrogen level energies (proportional to -1/n^2),
      and I_ref is the intensity of the reference Balmer line (Hβ, n=4).

    - High-order Balmer lines used to normalize the continuum are
      those with n >= 6 (as in their Eq. 5–7). Lines with n >= 3 are
      included in the λ > λ_BE pseudo-continuum.

    - Piecewise definition (their Eq. 5–7, 199–215):
        F(λ) =
          sum_{n=3}^{400} G_n(λ)                     for λ > λ_BE
          F_BaC * B_λ(T_e) [1 - exp(-τ_λ)]          for λ <= λ_BE

      where F_BaC is derived from the sum of high-order Balmer lines
      at λ_BE, and B_λ is the Planck function.
    """

    inputs = ('wave',)
    outputs = ('flux',)

    # Main free parameters: reference line amplitude, width, shift
    i_ref = Parameter(default=1.0)
    sigma = Parameter(default=1000.0)     # Å, typical broad-line Doppler width scale
    dv = Parameter(default=0.0)            # velocity shift [km/s]

    # Physical parameters (fixed by default, but can be freed if desired)
    te = Parameter(default=15000.0, fixed=True)      # K
    tau_be = Parameter(default=1.0, fixed=True)      # optical depth at Balmer edge
    lambda_be = Parameter(default=3646.0, fixed=True)  # Å

    # --- class-level constants / caches ---
    # Balmer upper-level range: n -> 2, n = 3..400
    _n_min_all = 3
    _n_max_all = 400
    _n_min_edge = 6        # for the Balmer-edge sum (high-order lines)
    _n_ref = 4             # Hβ as reference Balmer line

    # Structural caches
    _n_cache = None
    _lambda_cache = None

    @classmethod
    def _precompute_level_data(cls):
        """
        Precompute:
          - n_all: array of upper levels n (3..400) for Balmer series
          - lambda_n: wavelengths [Å] for transitions n -> 2
          - E_n: upper-level energies [eV], approx. E_n = 13.6 (1 - 1/n^2)
        """
        n_all = np.arange(cls._n_min_all, cls._n_max_all + 1, dtype=int)

        # Balmer wavelengths from Rydberg formula: 1/λ = R (1/2^2 - 1/n^2)
        R_cm = 1.0973731568160e5  # cm^-1
        inv_lambda_cm = R_cm * (1.0 / 4.0 - 1.0 / (n_all.astype(float)**2))
        lambda_cm = 1.0 / inv_lambda_cm
        lambda_ang = lambda_cm * 1e8   # cm -> Å

        # Approximate upper-level energies [eV] measured from ground:
        # E_n = 13.6 eV * (1/2^2 - 1/n^2)
        E_n = 13.6 * (0.25 - 1.0 / (n_all.astype(float)**2))

        return n_all, lambda_ang, E_n

    @classmethod
    def _get_level_data(cls):
        if cls._n_cache is None:
            cls._n_cache, cls._lambda_cache, cls._E_cache = cls._precompute_level_data()
        return cls._n_cache, cls._lambda_cache, cls._E_cache

    @staticmethod
    def evaluate(wave, i_ref, sigma, dv, te, tau_be, lambda_be):
        wave = np.asanyarray(wave, dtype=float)

        # --- Get Balmer level data (n, λ_n, ΔE_n) ---
        n_all, lambda_n, E_n = BalmerPseudoContinuum._get_level_data()

        # Precompute masks for "all Balmer lines" and "high-order lines"
        # (these are index masks over n_all)
        mask_high = n_all >= BalmerPseudoContinuum._n_min_edge

        # --- Relative line intensities via thermodynamic approximation (Eq. 4) ---
        n_ref = BalmerPseudoContinuum._n_ref
        k_B_eV = 8.617333262e-5  # eV K^-1

        # Energy of reference upper level
        idx_ref = np.where(n_all == n_ref)[0]
        if len(idx_ref) == 0:
            raise RuntimeError("Reference level n_ref not in n_all.")
        idx_ref = idx_ref[0]
        E_ref = E_n[idx_ref]

        # I_n / I_ref ≈ exp( - (E_n - E_ref) / (k_B * te) )
        dE = E_n - E_ref
        rel_int = np.exp(-dE / (k_B_eV * te))

        # Ensure exactly 1 at n_ref (avoid rounding issues)
        rel_int /= rel_int[idx_ref]

        # Peak intensities I_n
        I_n = i_ref * rel_int   # shape (N_lines,)

        # --- Shifted line centers ---
        lambda_n_shift = lambda_n * (1.0 + dv / ls_km)  # Å
        lambda_be_shift = lambda_be * (1.0 + dv / ls_km)  # Å

        # --- Evaluate all Balmer Gaussians at wave (λ > λ_BE part) ---
        # G_i(λ) = I_i * exp( - ((λ - λ_i_shift) / wd)^2 )
        # Broadcast wave over lines
        dw = lambda_n_shift * sigma / ls_km
        diff = (wave[None, :] - lambda_n_shift[:, None]) / dw[:, None]
        gauss_all = I_n[:, None] / (dw[:, None] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * diff**2)   # shape (N_lines, N_wave)

        # Sum of ALL Balmer lines for each λ (for λ > λ_BE)
        sum_lines_wave = gauss_all.sum(axis=0)        # shape (N_wave,)

        # --- Compute sum of high-order lines at Balmer edge (λ_BE) ---
        dw_edge = lambda_n_shift * sigma / ls_km
        diff_edge = (lambda_be_shift - lambda_n_shift) / dw_edge
        gauss_edge = I_n  / (dw_edge * np.sqrt(2 * np.pi)) * np.exp(-0.5 * diff_edge**2)      # shape (N_lines,)

        # Only high-order lines (n >= 6) contribute to the Balmer-edge sum
        sum_edge_high = gauss_edge[mask_high].sum()

        # Guard against pathologies
        if sum_edge_high <= 0.0:
            # Degenerate case: no meaningful Balmer continuum
            # -> return only the lines
            # (this avoids division by zero)
            flux = np.where(wave > lambda_be_shift, sum_lines_wave, 0.0)
            return flux

        # --- Planck function and optical depth for Balmer continuum ---
        # Convert Angstrom to cm
        lam_cm = wave * 1e-8
        lam_be_cm = lambda_be_shift * 1e-8

        B_wave = _planck_B_lambda(lam_cm, te)
        B_be = _planck_B_lambda(lam_be_cm, te)

        # τ_λ = τ_BE * (λ_BE / λ)^3 (Grandi 1982 / Kovačević+2013)
        tau_lambda = tau_be * (lambda_be_shift / wave)**3
        tau_be_val = tau_be  # τ at λ_BE

        # Balmer continuum normalization at edge (Eq. 7)
        # F_BaC = [ Σ_high G_i(λ_BE) ] / [ B_λ_BE (1 - e^{-τ_BE}) ]
        denom_edge = B_be * (1.0 - np.exp(-tau_be_val))
        FBaC = sum_edge_high / denom_edge

        # Balmer continuum at each λ (λ <= λ_BE)
        BC_wave = FBaC * B_wave * (1.0 - np.exp(-tau_lambda))

        # --- Final piecewise flux ---
        flux = np.zeros_like(wave, dtype=float)

        # λ > λ_BE: only Balmer lines
        mask_red = wave > lambda_be_shift
        flux[mask_red] = sum_lines_wave[mask_red]

        # λ <= λ_BE: Balmer continuum
        mask_blue = ~mask_red
        flux[mask_blue] = BC_wave[mask_blue]

        return flux


def _planck_B_lambda(lam_cm, T):
    """
    Planck function B_lambda(lam, T) in cgs units:
        B_lambda = 2 h c^2 / [ lam^5 (exp(hc / (lam kT)) - 1) ]
    lam_cm : wavelength in cm
    T      : temperature in K
    """
    h = 6.62607015e-27      # erg * s
    c = 2.99792458e10       # cm / s
    k_B = 1.380649e-16      # erg / K

    x = h * c / (lam_cm * k_B * T)
    x = np.clip(x, 1e-3, 700.0)  # avoid overflow
    ex = np.exp(x)
    return 2.0 * h * c**2 / (lam_cm**5 * (ex - 1.0))