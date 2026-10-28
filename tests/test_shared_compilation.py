"""Tests for the opt-in shared compilation path of ``JNesty_Fit``.

``share_compilation=True`` hands the spectrum to the compiled likelihood as an
*argument* instead of closing over it, and reuses the compiled program for the
next fit of the same model structure.  Two properties matter and are checked
here:

* the numbers do not change -- the likelihood and the prior transform of a
  shared fit agree with a legacy fit of the same model and data, and a short
  run gives the same evidence;
* the program is really reused -- the registry stays at two entries while the
  data change, which is what saves the compilation for the rest of a sample.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from galspec import jax_models as jxm
from galspec.jnesty_fit import (JNesty_Fit, _jnesty_context_supported,
                               clear_shared_compilation_cache,
                               shared_compilation_cache_size)
from galspec.priors import LogUniform, Normal, Uniform

VELC = np.linspace(-1500.0, 1500.0, 201)
TEMPLATE = np.exp(-0.5 * (VELC / 400.0) ** 2)
WAVE = np.linspace(4200.0, 6800.0, 300)
ERROR = np.full(WAVE.size, 0.05)


def synthetic_flux(seed: int, level: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level + 0.02 * rng.normal(size=WAVE.size)


def small_model(n_lines: int = 3):
    """A compound model with a fixed continuum shape and ``n_lines`` templates."""
    model = jxm.PowerLaw1D(amplitude=0.55, x_0=5500.0, alpha=1.0,
                           fixed={"x_0": True}, name="Continuum")
    for index in range(n_lines):
        model = model + jxm.Line_template(
            template_velc=VELC, template_flux=TEMPLATE, amplitude=0.5, dv=0.0,
            wavec=4500.0 + 200.0 * index, name=f"line{index}")
    return model


def make_fit(flux, shared: bool, model=None, **kwargs) -> JNesty_Fit:
    return JNesty_Fit(model if model is not None else small_model(), WAVE, flux,
                      ERROR, nlive=50, share_compilation=shared, **kwargs)


# --------------------------------------------------------------------------------------
# the numbers do not change
# --------------------------------------------------------------------------------------
def test_shared_agrees_with_legacy_where_it_matters():
    flux = synthetic_flux(1)
    legacy = make_fit(flux, shared=False)
    shared = make_fit(flux, shared=True)
    assert legacy.param_names == shared.param_names
    assert legacy.ndim == shared.ndim == len(legacy.param_names)

    theta = np.asarray(shared.model.get_free_params(), dtype=float)
    assert shared.log_likelihood(theta) == pytest.approx(
        legacy.log_likelihood(theta), rel=1e-13)

    u = np.linspace(0.02, 0.98, shared.ndim)
    assert np.allclose(shared.prior_transform(u), legacy.prior_transform(u),
                       rtol=0.0, atol=1e-12)

    # the batch path is the same map, not a different one
    batch = np.tile(theta, (3, 1)) + np.linspace(0.0, 0.01, 3)[:, None]
    assert np.allclose(shared.log_likelihood_batch(batch),
                       legacy.log_likelihood_batch(batch), rtol=1e-13, atol=1e-9)


def test_a_shared_fit_runs_and_gives_the_same_evidence():
    clear_shared_compilation_cache()
    flux = synthetic_flux(2)
    outcomes = {}
    for shared in (False, True):
        fit = make_fit(flux, shared=shared)
        fit.warmup()
        fit.fit(progress=False, dlogz=2.0, max_iterations=2000)
        outcomes[shared] = (fit.log_evidence, fit.n_iterations)
    assert outcomes[True][0] == pytest.approx(outcomes[False][0], abs=0.5)
    assert outcomes[True][1] > 0


# --------------------------------------------------------------------------------------
# the program is reused
# --------------------------------------------------------------------------------------
def test_one_compiled_program_serves_every_source():
    clear_shared_compilation_cache()
    fits, legacies = [], []
    for seed in (1, 2, 3):
        flux = synthetic_flux(seed, level=1.0 + 0.1 * seed)
        fit = make_fit(flux, shared=True)
        fit.warmup()
        fits.append(fit)
        legacies.append(make_fit(flux, shared=False))

    # one likelihood and one prior transform, no matter how many sources
    assert shared_compilation_cache_size() == 2

    theta = np.asarray(fits[0].model.get_free_params(), dtype=float)
    likelihoods = []
    for fit, legacy in zip(fits, legacies):
        # each fit still sees *its own* spectrum
        value = fit.log_likelihood(theta)
        assert value == pytest.approx(legacy.log_likelihood(theta), rel=1e-13)
        likelihoods.append(value)
    assert len({round(float(value), 6) for value in likelihoods}) == 3


def test_a_different_model_structure_compiles_again():
    clear_shared_compilation_cache()
    three = make_fit(synthetic_flux(1), shared=True, model=small_model(3))
    three.warmup()
    assert shared_compilation_cache_size() == 2

    same = make_fit(synthetic_flux(2), shared=True, model=small_model(3))
    same.warmup()
    assert shared_compilation_cache_size() == 2

    different = make_fit(synthetic_flux(3), shared=True, model=small_model(4))
    different.warmup()
    assert shared_compilation_cache_size() == 4


def test_a_model_that_differs_only_in_a_constant_compiles_again():
    """A profile (or a fixed value) is a constant *inside* the program.

    JAX prints a large array constant in abbreviated form, so the traced text of
    two models that differ only in that array is byte for byte the same.  Hashing
    the text alone would hand the second model the first one's kernel -- a line
    fitted with someone else's profile, which no diagnostic downstream would show
    -- so the key hashes the constants too.  The cache entries and the numbers
    both say it: two programs, each one's likelihood equal to its own legacy fit.
    """
    clear_shared_compilation_cache()
    narrow = jxm.Line_template(template_velc=VELC,
                               template_flux=np.exp(-0.5 * (VELC / 100.0) ** 2),
                               amplitude=1.0, dv=0.0, wavec=5000.0)
    wide = jxm.Line_template(template_velc=VELC, template_flux=TEMPLATE,
                             amplitude=1.0, dv=0.0, wavec=5000.0)
    flux = synthetic_flux(1)
    first = JNesty_Fit(narrow, WAVE, flux, ERROR, nlive=20, share_compilation=True)
    first.warmup()
    second = JNesty_Fit(wide, WAVE, flux, ERROR, nlive=20, share_compilation=True)
    second.warmup()
    # two likelihood programs (one per profile) and one shared prior transform
    assert shared_compilation_cache_size() == 3

    theta = np.asarray(first.model.get_free_params(), dtype=float)
    for fit, model in ((first, narrow), (second, wide)):
        legacy = JNesty_Fit(model, WAVE, flux, ERROR, nlive=20,
                            share_compilation=False)
        assert fit.log_likelihood(theta) == pytest.approx(
            legacy.log_likelihood(theta), rel=1e-13)
    assert first.log_likelihood(theta) != pytest.approx(second.log_likelihood(theta))


def test_clear_reports_what_it_freed():
    clear_shared_compilation_cache()
    make_fit(synthetic_flux(1), shared=True).warmup()
    assert clear_shared_compilation_cache() == 2
    assert shared_compilation_cache_size() == 0
    assert clear_shared_compilation_cache() == 0


def test_legacy_fits_do_not_enter_the_registry():
    clear_shared_compilation_cache()
    make_fit(synthetic_flux(1), shared=False).warmup()
    assert shared_compilation_cache_size() == 0


# --------------------------------------------------------------------------------------
# priors: the shared path must carry them, including the one-prior-per-parameter box
# --------------------------------------------------------------------------------------
def _model_with_priors(n_lines=2):
    """The same model twice, with the same priors, for a shared/legacy comparison."""
    model = small_model(n_lines)
    for index in range(n_lines):
        line = model[f"line{index}"]
        line.amplitude.prior = LogUniform(0.01, 5.0)
        line.dv.prior = Uniform(-200.0, 200.0)
    model["line1"].amplitude.prior = Normal(mean="line0.amplitude", sigma=0.5,
                                            nsigma=3)
    return model


def test_priors_are_shared_and_still_centred():
    clear_shared_compilation_cache()
    flux = synthetic_flux(1)

    shared = JNesty_Fit(_model_with_priors(), WAVE, flux, ERROR, nlive=50,
                        share_compilation=True)
    shared.warmup()
    legacy = JNesty_Fit(_model_with_priors(), WAVE, flux, ERROR, nlive=50,
                        share_compilation=False)

    theta = np.asarray(shared.model.get_free_params(), dtype=float)
    assert shared.log_likelihood(theta) == pytest.approx(
        legacy.log_likelihood(theta), rel=1e-13)
    u = np.linspace(0.05, 0.95, shared.ndim)
    # the centred prior reads its centre from another parameter, so the shared
    # and legacy transforms agree to rounding, not to the bit
    assert np.allclose(shared.prior_transform(u), legacy.prior_transform(u),
                       rtol=0.0, atol=1e-10)
    assert shared_compilation_cache_size() == 2


def test_a_log_space_centred_prior_falls_back_to_a_per_fit_prior():
    """A centred prior in log units cannot be traced; it warns and keeps working."""
    clear_shared_compilation_cache()
    model = small_model(2)
    model["line0"].amplitude.prior = Normal(mean="line1.amplitude", sigma=0.3,
                                            log=True)
    flux = synthetic_flux(1)
    fit = JNesty_Fit(model, WAVE, flux, ERROR, nlive=50, share_compilation=True)
    with pytest.warns(RuntimeWarning, match="cannot be shared between fits"):
        fit.warmup()
    assert fit.log_likelihood(np.asarray(model.get_free_params(), dtype=float)) < 0
    assert shared_compilation_cache_size() < 2      # the prior did not get cached


# --------------------------------------------------------------------------------------
# the optional on-disk cache, in a fresh process (it is global jax config)
# --------------------------------------------------------------------------------------
def test_persistent_cache_flags_are_set(tmp_path):
    code = (
        "import jax\n"
        "from galspec.jnesty_fit import enable_persistent_compilation_cache\n"
        f"enable_persistent_compilation_cache({str(tmp_path)!r})\n"
        "print(jax.config.jax_compilation_cache_dir)\n"
        "print(bool(jax.config.jax_enable_compilation_cache))\n"
        "print(jax.config.jax_persistent_cache_min_compile_time_secs)\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**dict(), "JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin",
             "MPLCONFIGDIR": str(tmp_path)})
    assert finished.returncode == 0, finished.stderr
    lines = finished.stdout.strip().splitlines()
    assert lines[0] == str(tmp_path)
    assert lines[1] == "True"
    assert float(lines[2]) == 0.0


def test_jnesty_accepts_a_context():
    """The pipeline's speed-up relies on JNesty taking the data as an argument."""
    assert _jnesty_context_supported() is True
    import inspect

    import jnesty

    assert "context" in inspect.signature(jnesty.NestedSampler.__init__).parameters