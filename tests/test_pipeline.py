"""Tests for the configuration-driven fitting pipeline (``galspec.pipeline``).

The pipeline is a consumer of the models and of ``JNesty_Fit``, so these tests
concentrate on what the pipeline itself adds: reading a TOML file strictly,
finding the three arrays in a FITS file, turning the ``[model]`` section into the
model a script would have written, and running a batch that isolates failures.

The one fit test uses a deliberately small ``nlive``/``dlogz`` so it takes a few
seconds; it checks the contract (files, report, posterior, resume) rather than
the science, which ``test_jax_parity.py`` covers for the models.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import replace
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from galspec.pipeline import (ConfigError, PipelineError, build_model, load_config,
                              read_spectrum, resolve_sources, run_pipeline,
                              source_namespace)
from galspec.pipeline.__main__ import main as pipeline_main

DATA = Path(__file__).resolve().parents[1] / "example" / "data"
MOCK = DATA / "mock_two_line_spectrum.fits"
# the small mock examples live in develop/dev_config; the configuration that ships
# in example/ (fit_iras07546_config.toml) is covered by test_the_shipped_example_config
REPO = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO / "develop" / "dev_config" / "mock_two_line.toml"
SHIPPED_CONFIG = REPO / "example" / "fit_iras07546_config.toml"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def write(tmp_path: Path, text: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def make_spectrum(tmp_path: Path, name: str = "spec.fits", n: int = 200,
                  wave=(4000.0, 4200.0), z: float | None = None,
                  columns=("WAVELENGTH", "FLUX", "ERROR")) -> Path:
    """A three-array FITS file in a binary table (the layout the pipeline wants)."""
    wave_col, flux_col, error_col = columns
    w = np.linspace(*wave, n)
    flux = 1.0 + 0.1 * np.sin(w / 10.0)
    error = np.full(n, 0.05)
    columns = [fits.Column(name=wave_col, format="D", array=w, unit="Angstrom"),
               fits.Column(name=flux_col, format="D", array=flux),
               fits.Column(name=error_col, format="D", array=error)]
    header = fits.Header()
    if z is not None:
        header["REDSHIFT"] = z
    path = tmp_path / name
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(columns, header=header)]).writeto(path)
    return path


MINIMAL_MODEL = """
[model]
[[model.component]]
type = "Linear1D"
name = "Continuum"

[model.component.bounds]
slope = [-1e-3, 1e-3]
intercept = [0.0, 3.0]
"""


def config_text(spectrum: Path, extra: str = "", model: str = MINIMAL_MODEL,
                data: str = "") -> str:
    # z_default and extinction are stated here so the tests do not depend on the
    # "you did not say" warnings; the tests that check those warnings leave them out
    return f"""
[data]
file = "{spectrum}"
z_default = 0.0
extinction = false
{data}

[fit]
nlive = 50
dlogz = 2.0
max_iterations = 2000

[output]
fit_plot = false
corner = true

{model}
{extra}
"""


# --------------------------------------------------------------------------------------
# schema: the TOML file is validated before anything expensive happens
# --------------------------------------------------------------------------------------
def test_example_config_loads_and_resolves_paths():
    config = load_config(EXAMPLE_CONFIG)
    assert config.data.source == "file"
    assert Path(config.data.file).is_file()
    assert Path(config.data.file).name == "mock_two_line_spectrum.fits"
    assert config.model.require_priors is True
    assert [component.kind for component in config.model.components] == [
        "Linear1D", "Line_Gaussian", "Line_Gaussian"]
    assert config.fit.share_compilation is True
    # output_dir is resolved next to the config file, not the cwd
    assert Path(config.run.output_dir) == EXAMPLE_CONFIG.parent / "out" / "mock_two_line"
    assert len(config.digest) == 64


def test_unknown_key_is_rejected(tmp_path):
    path = write(tmp_path, """
        [data]
        file = "x.fits"
        wavelenght = "WAVE"
        [model]
        [[model.component]]
        type = "Linear1D"
    """)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "[data]" in str(excinfo.value) and "wavelenght" in str(excinfo.value)
    assert "wave_key" in str(excinfo.value)          # the suggestion is actionable


def test_missing_section_and_source_selection_are_rejected(tmp_path):
    no_model = write(tmp_path, """
        [data]
        file = "x.fits"
    """, "a.toml")
    with pytest.raises(ConfigError, match=r"missing required section \[model\]"):
        load_config(no_model)

    two_sources = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        glob = "*.fits"
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "b.toml")
    with pytest.raises(ConfigError, match="exactly one of file, files, glob"):
        load_config(two_sources)


def test_expression_language_is_restricted(tmp_path):
    bad_token = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Linear1D"
        slope = "3*peak_flx"
    """, "a.toml")
    with pytest.raises(ConfigError, match="unknown token 'peak_flx'"):
        load_config(bad_token)

    call = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Linear1D"
        slope = "__import__('os').system('echo hi')"
    """, "b.toml")
    with pytest.raises(ConfigError, match="only numbers, the available tokens"):
        load_config(call)


def test_token_expressions_evaluate_per_source():
    namespace = source_namespace(np.array([0.0, 1.0, 4.0, -2.0]), redshift=0.5)
    assert namespace["peak_flux"] == 4.0
    assert namespace["median_flux"] == 0.5
    assert namespace["n_pixels"] == 4.0
    assert namespace["z"] == 0.5
    from galspec.pipeline.schema import TokenValue

    assert TokenValue("3*peak_flux").evaluate(namespace) == 12.0
    assert TokenValue("0.5*(max_flux + min_flux)").evaluate(namespace) == 1.0
    with pytest.raises(ConfigError, match="divides by zero"):
        TokenValue("1/(max_flux - max_flux)").evaluate(namespace)


# --------------------------------------------------------------------------------------
# loader: finding the three arrays
# --------------------------------------------------------------------------------------
def test_the_shipped_example_config():
    """The IRAS configuration in example/ is the one users run: it must load."""
    config = load_config(SHIPPED_CONFIG)
    assert Path(config.data.file).is_file()
    assert len(config.model.components) == 17
    assert [component.kind for component in config.model.components][:3] == [
        "PowerLaw1D", "IronTemplate_tied", "Line_template"]
    assert config.fit.nlive == 400 and config.fit.dlogz == 0.01
    assert config.fit.max_iterations == 200_000
    assert Path(config.run.output_dir) == SHIPPED_CONFIG.parent / "out" / "iras07546"
    # the narrow-line profile travels inside the data file, and every templated
    # component reads it: no second file has to travel with the sample
    assert config.data.narrow_template_key == "NARROW"
    for component in config.model.components:
        assert "template_file" not in component.values, component.name
    source = resolve_sources(config)[0]
    spectrum = read_spectrum(config, source)
    assert spectrum.n_pixels == 56956
    assert spectrum.z_source == "header:REDSHIFT"
    assert spectrum.e_bv_source == "header:EBV"
    velocity, profile = spectrum.narrow_template
    reference = np.loadtxt(SHIPPED_CONFIG.parent / "data"
                           / "iras07546_narrow_template.txt", comments="#")
    assert np.array_equal(velocity, reference[:, 0])
    assert np.array_equal(profile, reference[:, 1])
    built = build_model(config, source_namespace(
        spectrum.flux, spectrum.wave, spectrum.redshift,
        config.data.continuum_percentile), spectrum)
    assert len(built.free_parameters) == 27
    # the reference fit's per-component priors are the pipeline's defaults here,
    # except the four the example script tunes itself
    assert len(built.default_priors) == 23
    assert {line.split(":")[0] for line in built.default_priors} >= {
        "n_halpha.amplitude", "b_halpha.sigma_c", "b_halpha.dv_w0", "OIII.amp_c0"}
    assert {line.split(":")[0] for line in built.default_priors}.isdisjoint({
        "Continuum.amplitude", "FeII.amplitude", "b_halpha.dv_c", "OIII.dv_c"})


def test_reading_the_example_spectrum():
    config = load_config(EXAMPLE_CONFIG)
    source = resolve_sources(config)[0]
    assert source.source_id == "mock_two_line_spectrum"
    spectrum = read_spectrum(config, source)
    assert spectrum.n_pixels == 2000
    assert spectrum.wave.min() == pytest.approx(4000.0)
    assert spectrum.wave_unit == "Angstrom"
    assert spectrum.z_source == "z_default"
    assert spectrum.provenance["wavelength"] == "HDU 1 (WAVELENGTH)"


def test_extinction_is_removed_from_flux_and_error(tmp_path):
    from extinction import ccm89, remove

    e_bv = 0.35
    path = make_spectrum(tmp_path, "extinguished.fits", n=60)
    with fits.open(path, mode="update") as hdul:
        hdul[1].header["EBV"] = e_bv
    config = load_config(write(tmp_path, f"""
        [data]
        file = "{path}"
        z_default = 0.0
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])

    with fits.open(path) as hdul:
        columns = hdul[1].data
        wave = np.asarray(columns["WAVELENGTH"], dtype=float)
        raw_flux = np.asarray(columns["FLUX"], dtype=float)
        raw_error = np.asarray(columns["ERROR"], dtype=float)
    factor = 10.0 ** (0.4 * ccm89(wave, e_bv, r_v=3.1, unit="aa"))
    assert np.allclose(spectrum.flux, remove(ccm89(wave, e_bv, r_v=3.1, unit="aa"),
                                             raw_flux), rtol=0.0, atol=1e-13)
    # the error is corrected by the same factor: dereddening is multiplicative
    assert np.allclose(spectrum.error, raw_error * factor, rtol=1e-12)
    assert spectrum.e_bv == pytest.approx(e_bv)
    assert spectrum.e_bv_source == "header:EBV"
    assert spectrum.summary()["e_bv"] == pytest.approx(e_bv)


def test_a_v_is_converted_to_a_colour_excess(tmp_path):
    path = make_spectrum(tmp_path, "av.fits", n=40)
    with fits.open(path, mode="update") as hdul:
        hdul[1].header["A_V"] = 0.93          # 0.3 with R_V = 3.1
    config = load_config(write(tmp_path, f"""
        [data]
        file = "{path}"
        z_default = 0.0
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert spectrum.e_bv == pytest.approx(0.3, rel=1e-6)
    assert "A_V / R_V" in spectrum.e_bv_source


def test_missing_extinction_warns_and_assumes_none(tmp_path):
    path = make_spectrum(tmp_path, "plain.fits", n=40)
    config = load_config(write(tmp_path, f"""
        [data]
        file = "{path}"
        z_default = 0.0
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "a.toml"))
    with pytest.warns(RuntimeWarning, match="no Galactic colour excess"):
        spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert spectrum.e_bv == 0.0 and spectrum.e_bv_source == "assumed"


def test_extinction_can_be_declared_absent(tmp_path):
    path = make_spectrum(tmp_path, "known_clean.fits", n=40)
    with fits.open(path, mode="update") as hdul:
        hdul[1].header["EBV"] = 0.5          # present in the file, declared away below
    config = load_config(write(tmp_path, f"""
        [data]
        file = "{path}"
        z_default = 0.0
        extinction = false
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "a.toml"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)     # must not warn
        spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert spectrum.e_bv == 0.0 and spectrum.e_bv_source == "none"


def test_an_undeclared_redshift_warns_but_a_declared_one_does_not(tmp_path):
    path = make_spectrum(tmp_path, "noz.fits", n=40)
    body = f"""
        [data]
        file = "{path}"
        extinction = false
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """
    undeclared = load_config(write(tmp_path, body, "a.toml"))
    with pytest.warns(RuntimeWarning, match="no redshift"):
        read_spectrum(undeclared, resolve_sources(undeclared)[0])

    declared = load_config(write(tmp_path, body.replace('file = ', 'z_default = 0.0\nfile = '),
                                "b.toml"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        spectrum = read_spectrum(declared, resolve_sources(declared)[0])
    assert spectrum.z_source == "z_default"


def test_wavelength_unit_is_converted_and_flux_unit_is_recorded(tmp_path):
    w = np.linspace(400.0, 500.0, 50)                 # nm on purpose
    columns = [fits.Column(name="WAVE", format="D", array=w, unit="nm"),
               fits.Column(name="FLUX", format="D", array=np.ones(50),
                           unit="erg / (Angstrom cm2 s)"),
               fits.Column(name="ERR", format="D", array=np.full(50, 0.1))]
    path = tmp_path / "nm.fits"
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(columns)]).writeto(path)
    config = load_config(write(tmp_path, f"""
        [data]
        file = "{path}"
        wave_key = "WAVE"
        flux_key = "FLUX"
        error_key = "ERR"
        z_default = 0.0
        extinction = false
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert spectrum.wave_unit == "nm"
    assert spectrum.wave.min() == pytest.approx(4000.0)          # nm -> Angstrom
    assert spectrum.flux_unit == "erg / (Angstrom cm2 s)"        # recorded, not converted


def test_a_missing_error_array_is_an_error_not_a_guess(tmp_path):
    path = tmp_path / "noerror.fits"
    columns = [fits.Column(name="WAVELENGTH", format="D", array=np.linspace(4000, 4100, 20)),
               fits.Column(name="FLUX", format="D", array=np.ones(20))]
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(columns)]).writeto(path)
    config = load_config(write(tmp_path, config_text(path)))
    with pytest.raises(PipelineError) as excinfo:
        read_spectrum(config, resolve_sources(config)[0])
    message = str(excinfo.value)
    assert "error" in message and "WAVELENGTH" in message
    assert "FLUX" in message                        # says what it did find


def test_wrong_lengths_and_non_length_units_are_rejected(tmp_path):
    # the wavelength axis in an image HDU, the flux and error as table columns:
    # a bintable's columns always share a length, so this is the only way the
    # three arrays can disagree
    mismatch = tmp_path / "mismatch.fits"
    columns = [fits.Column(name="FLUX", format="D", array=np.ones(19)),
               fits.Column(name="ERROR", format="D", array=np.ones(19))]
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.linspace(4000, 4100, 20), name="WAVELENGTH"),
                  fits.BinTableHDU.from_columns(columns)]).writeto(mismatch)
    config = load_config(write(tmp_path, config_text(mismatch), name="c.toml"))
    with pytest.raises(PipelineError, match="different lengths"):
        read_spectrum(config, resolve_sources(config)[0])

    velocity = make_spectrum(tmp_path, "velocity.fits")
    fits.open(velocity).close()
    config = load_config(write(tmp_path, f"""
        [data]
        file = "{velocity}"
        wave_unit = "km / s"
        z_default = 0.0
        extinction = false
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "d.toml"))
    with pytest.raises(PipelineError, match="not a length"):
        read_spectrum(config, resolve_sources(config)[0])


def test_redshift_from_header_catalog_and_default(tmp_path):
    header_z = make_spectrum(tmp_path, "z.fits", z=0.25)
    config = load_config(write(tmp_path, config_text(header_z), "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert spectrum.redshift == pytest.approx(0.25)
    assert spectrum.z_source == "header:REDSHIFT"
    # the grid is de-shifted: 4000-4200 observed at z=0.25 -> 3200-3360 rest
    assert spectrum.wave.min() == pytest.approx(4000.0 / 1.25)

    plain = make_spectrum(tmp_path, "plain.fits")
    catalog = tmp_path / "sample.csv"
    catalog.write_text("id,file,z\nsrc1,plain.fits,0.1\n", encoding="utf-8")
    config = load_config(write(tmp_path, f"""
        [data]
        catalog = "{catalog}"
        catalog_z_column = "z"
        extinction = false
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "b.toml"))
    sources = resolve_sources(config)
    assert [source.source_id for source in sources] == ["src1"]
    spectrum = read_spectrum(config, sources[0])
    assert spectrum.redshift == pytest.approx(0.1)
    assert spectrum.z_source == "catalog"


def test_masking_trimming_and_min_pixels(tmp_path):
    path = make_spectrum(tmp_path, "cut.fits", n=400, wave=(4000.0, 4400.0))
    config = load_config(write(tmp_path, config_text(path, data="""
        wave_min = 4100.0
        wave_max = 4300.0
        mask = [[4150.0, 4180.0]]
        min_pixels = 10
    """), "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert spectrum.wave.min() >= 4100.0 - 1e-9
    assert spectrum.wave.max() <= 4300.0 + 1e-9
    assert not np.any((spectrum.wave >= 4150.0) & (spectrum.wave <= 4180.0))

    too_few = load_config(write(tmp_path, config_text(path, data="min_pixels = 10000"),
                                "b.toml"))
    with pytest.raises(PipelineError, match="fewer than"):
        read_spectrum(too_few, resolve_sources(too_few)[0])


def test_glob_and_files_selection(tmp_path):
    make_spectrum(tmp_path, "a1.fits", wave=(4000.0, 4100.0))
    make_spectrum(tmp_path, "a2.fits", wave=(4000.0, 4100.0))
    globbed = load_config(write(tmp_path, f"""
        [data]
        glob = "a*.fits"
        extinction = false
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "a.toml"))
    assert [source.source_id for source in resolve_sources(globbed)] == ["a1", "a2"]

    listed = load_config(write(tmp_path, f"""
        [data]
        files = ["{tmp_path / 'a1.fits'}", "{tmp_path / 'a2.fits'}"]
        extinction = false
        min_pixels = 10
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "b.toml"))
    assert len(resolve_sources(listed)) == 2


def test_empty_glob_and_missing_catalog_column_are_reported(tmp_path):
    config = load_config(write(tmp_path, f"""
        [data]
        glob = "nothing-*.fits"
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "a.toml"))
    with pytest.raises(PipelineError, match="glob matched no files"):
        resolve_sources(config)

    catalog = tmp_path / "bad.csv"
    catalog.write_text("identifier,path\nx,y\n", encoding="utf-8")
    config = load_config(write(tmp_path, f"""
        [data]
        catalog = "{catalog}"
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "b.toml"))
    with pytest.raises(PipelineError, match="no column 'file'"):
        resolve_sources(config)


# --------------------------------------------------------------------------------------
# builder: the TOML becomes the model a script would have written
# --------------------------------------------------------------------------------------
def param(model, qualified: str):
    """``"Submodel.parameter"`` -> the ``JaxParameter`` handle (compound or leaf)."""
    if hasattr(model, "n_submodels"):
        label, _, base = qualified.rpartition(".")
        return getattr(model[label], base)
    return getattr(model, qualified.rpartition(".")[2])


def test_example_model_structure_priors_and_ties():
    config = load_config(EXAMPLE_CONFIG)
    namespace = {"peak_flux": 2.0, "median_flux": 0.1, "continuum_level": 0.1,
                 "rms": 0.3, "min_flux": -0.5, "max_flux": 2.0, "n_pixels": 10.0,
                 "z": 0.0}
    built = build_model(config, namespace)
    model = built.model
    assert built.components == ("Continuum", "Line1", "Line2")
    assert list(model.free_param_names) == [
        "slope_0", "intercept_0", "amplitude_1", "dv_1", "sigma_1", "wavec_1",
        "amplitude_2", "sigma_2"]

    # expressions are resolved per source: 0.01*peak_flux with peak_flux = 2.0
    assert param(model, "Line1.amplitude").value == pytest.approx(0.5 * 2.0)
    assert model.spec("amplitude_1").bounds == (pytest.approx(0.02),
                                                pytest.approx(6.0))
    # ties are pure data after the build (they serialise without Python code)
    tie = param(model, "Line2.dv").tied
    assert tie.is_copy and tie.submodel == "Line1" and tie.parameter == "dv"
    # priors keep their kind
    from galspec.priors import LogUniform, Normal, Uniform

    assert isinstance(param(model, "Line1.amplitude").prior, LogUniform)
    assert isinstance(param(model, "Line1.dv").prior, Normal)
    assert isinstance(param(model, "Line1.sigma").prior, Uniform)
    assert param(model, "Line1.wavec").prior is not None
    assert model.get_prior("Line2.amplitude") is not None


def test_ties_are_evaluated_when_the_model_is_called():
    config = load_config(EXAMPLE_CONFIG)
    namespace = source_namespace(np.array([1.0]))
    built = build_model(config, namespace)
    model = built.model
    theta = np.asarray(model.get_free_params(), dtype=float)
    flux = np.asarray(model.flux(np.array([4400.0, 4600.0]), theta))
    assert np.all(np.isfinite(flux))
    # changing the reference dv must move both lines
    index = list(model.free_param_names).index("dv_1")
    shifted = theta.copy()
    shifted[index] += 100.0
    moved = np.asarray(model.flux(np.array([4400.0, 4600.0]), shifted))
    assert not np.allclose(flux, moved)


def test_free_parameter_checks_are_early_and_specific(tmp_path):
    """A range the pipeline cannot guess has to be written down."""
    no_bounds = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Linear1D"
    """, "a.toml")
    config = load_config(no_bounds)
    with pytest.raises(PipelineError, match=r"no finite sampling range: "
                                            r"Linear1D\.slope, Linear1D\.intercept"):
        build_model(config, source_namespace(np.array([1.0])))

    needs_prior = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        require_priors = true
        [[model.component]]
        type = "Linear1D"
        name = "Continuum"
        [model.component.bounds]
        slope = [-1.0, 1.0]
        intercept = [-1.0, 1.0]
    """, "b.toml")
    config = load_config(needs_prior)
    with pytest.raises(PipelineError, match="have no prior written in the file: "
                                            "Continuum.slope"):
        build_model(config, source_namespace(np.array([1.0])))


def test_a_line_component_must_say_which_line_it_is(tmp_path):
    """``wavec`` defaults to 5000 A, which is never the line the user means."""
    path = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_Gaussian"
        amplitude = 1.0
    """, "a.toml")
    config = load_config(path)
    with pytest.raises(PipelineError, match="wavec has no value"):
        build_model(config, source_namespace(np.array([1.0])))

    # a bound pins the window the sampler draws from, so it counts as saying so
    bounded = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_Gaussian"
        amplitude = 1.0
        [model.component.bounds]
        wavec = [6550.0, 6570.0]
    """, "b.toml")
    built = build_model(load_config(bounded), source_namespace(np.array([1.0])))
    assert param(built.model, "wavec").bounds == (6550.0, 6570.0)


def test_prior_on_a_fixed_parameter_is_rejected(tmp_path):
    path = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_Gaussian"
        wavec = 6563.0
        amplitude = 1.0
        [model.component.bounds]
        amplitude = [0.1, 2.0]
        dv = [-100.0, 100.0]
        sigma = [50.0, 500.0]
        [model.component.priors.wavec]
        kind = "uniform"
        lower = 6540.0
        upper = 6580.0
    """, "a.toml")
    config = load_config(path)
    with pytest.raises(PipelineError, match="cannot put a prior on"):
        build_model(config, source_namespace(np.array([1.0])))


def test_tie_to_a_missing_parameter_is_rejected(tmp_path):
    path = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_Gaussian"
        name = "A"
        amplitude = 1.0
        fixed = {{ wavec = false }}
        [model.component.bounds]
        amplitude = [0.1, 2.0]
        dv = [-100.0, 100.0]
        sigma = [50.0, 500.0]
        wavec = [6500.0, 6600.0]
        [[model.component]]
        type = "Line_Gaussian"
        name = "B"
        wavec = 6563.0
        amplitude = 1.0
        [model.component.bounds]
        amplitude = [0.1, 2.0]
        [model.component.ties]
        dv = "A.dvv"
    """, "a.toml")
    config = load_config(path)
    with pytest.raises(PipelineError, match="'A.dvv' is not a parameter"):
        build_model(config, source_namespace(np.array([1.0])))


def test_tie_ratio_conventions(tmp_path):
    """``doublet_ratio`` divides by the ratio, as the GalSpec tie classes do."""
    path = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_MultiGauss_doublet"
        name = "OIII"
        wavec0 = "@OIII_5007"
        wavec1 = "@OIII_4959"
        amp_c0 = 2.0

        [model.component.bounds]
        amp_c0 = [0.1, 10.0]
        dv_c = [-100.0, 100.0]
        sigma_c = [50.0, 500.0]

        [model.component.priors.amp_c0]
        kind = "uniform"
        lower = 0.1
        upper = 10.0

        [model.component.priors.dv_c]
        kind = "uniform"
        lower = -100.0
        upper = 100.0

        [model.component.priors.sigma_c]
        kind = "uniform"
        lower = 50.0
        upper = 500.0

        [model.component.ties]
        amp_c1 = {{ kind = "doublet_ratio", ref = "OIII", ratio = 2.98 }}
    """, "a.toml")
    config = load_config(path)
    built = build_model(config, source_namespace(np.array([1.0])))
    tie = param(built.model, "OIII.amp_c1").tied
    assert tie.submodel == "OIII" and tie.parameter == "amp_c0"
    assert tie.scale == pytest.approx(1.0 / 2.98)
    assert tie.offset == 0.0


def test_named_line_tokens_resolve_to_rest_wavelengths(tmp_path):
    path = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_Gaussian"
        wavec = "@Halpha"
        amplitude = 1.0
        [model.component.bounds]
        amplitude = [0.1, 2.0]
        dv = [-100.0, 100.0]
        sigma = [50.0, 500.0]
    """, "a.toml")
    config = load_config(path)
    built = build_model(config, source_namespace(np.array([1.0])))
    assert param(built.model, "wavec").value == pytest.approx(6562.819)


def test_template_file_feeds_a_template_model(tmp_path):
    template = tmp_path / "template.txt"
    velc = np.linspace(-1000.0, 1000.0, 101)
    np.savetxt(template, np.column_stack([velc, np.exp(-0.5 * (velc / 200.0) ** 2)]))
    path = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_template"
        template_file = "{template}"
        wavec = 6560.0
        amplitude = 1.0
        [model.component.bounds]
        amplitude = [0.1, 2.0]
        dv = [-100.0, 100.0]
        [model.component.priors.amplitude]
        kind = "uniform"
        lower = 0.1
        upper = 2.0
        [model.component.priors.dv]
        kind = "uniform"
        lower = -100.0
        upper = 100.0
    """, "a.toml")
    config = load_config(path)
    built = build_model(config, source_namespace(np.array([1.0])))
    model = built.model
    assert np.allclose(np.asarray(model._template_velc), velc)
    line = np.asarray(model.flux(np.linspace(6540, 6580, 50),
                                 np.asarray(model.get_free_params())))
    assert line.max() == pytest.approx(1.0, abs=0.01)     # unit-peak template
    assert line.min() < 0.01                              # wings fall to zero


def test_a_configuration_can_omit_every_value_and_prior(tmp_path):
    """Only the file and the model are required: a fit runs on the defaults."""
    spectrum = make_spectrum(tmp_path, wave=(6500.0, 6630.0))
    path = write(tmp_path, f"""
        [data]
        file = "{spectrum}"
        z_default = 0.0
        extinction = false
        [model]
        [[model.component]]
        type = "Line_Gaussian"
        wavec = "@Halpha"
    """, "a.toml")
    config = load_config(path)
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    built = build_model(config, source_namespace(spectrum.flux), spectrum)
    model = built.model
    assert list(built.free_parameters) == ["amplitude", "dv", "sigma"]
    assert [type(model.get_prior(name)).__name__ for name in built.free_parameters] \
        == ["LogUniform", "Uniform", "Uniform"]
    # the amplitude interval is the spectrum's, the rest are the physical ones
    amplitude = model.get_prior("amplitude")
    assert amplitude.lower == pytest.approx(1e-3)
    assert amplitude.upper == pytest.approx(3.0 * np.max(spectrum.flux))
    assert model.get_prior("dv").lower == pytest.approx(-3000.0)
    assert model.get_prior("sigma").upper == pytest.approx(8000.0)
    # the values the model starts from are its own defaults, not a guess by us
    assert param(model, "sigma").value == pytest.approx(200.0)


def test_the_default_amplitude_prior_scales_to_a_faint_spectrum(tmp_path):
    """``1e-3`` is a number the data has to justify: keep the interval non-empty."""
    spectrum = make_spectrum(tmp_path, wave=(6500.0, 6630.0), n=60)
    with fits.open(spectrum, mode="update") as hdul:
        for column in ("FLUX", "ERROR"):
            hdul[1].data[column] = hdul[1].data[column] * 1e-15
    path = config_text(spectrum, model="""
        [model]
        [[model.component]]
        type = "Line_Gaussian"
        wavec = "@Halpha"
    """)
    config = load_config(write(tmp_path, path, "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    built = build_model(config, source_namespace(spectrum.flux), spectrum)
    prior = built.model.get_prior("amplitude")
    assert prior.lower < 1e-3                       # an interval, not an inversion
    assert prior.upper == pytest.approx(3.0 * np.max(spectrum.flux))
    assert "scaled to the spectrum" in built.default_priors[0]


def _spectrum_with_template(tmp_path, name="spec.fits", velocity=None, unit="km/s",
                            columns=("VELOCITY", "TFLUX"), extname="NARROW"):
    """A spectrum whose FITS file also carries the narrow-line profile."""
    path = make_spectrum(tmp_path, name=name, wave=(6500.0, 6630.0))
    if velocity is None:
        velocity = np.linspace(-500.0, 500.0, 201)
    profile = np.exp(-0.5 * (velocity / 100.0) ** 2)
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name=columns[0], format="D", array=velocity, unit=unit),
         fits.Column(name=columns[1], format="D", array=profile)],
        name=extname)
    with fits.open(path, mode="append") as hdul:
        hdul.append(table)
    return path, np.asarray(velocity, dtype=float), profile


TEMPLATE_MODEL = """
[model]
[[model.component]]
type = "Line_template"
wavec = "@Halpha"
"""


def test_narrow_template_key_reads_the_profile_from_the_spectrum(tmp_path):
    path, velocity, profile = _spectrum_with_template(tmp_path)
    config = load_config(write(tmp_path, config_text(
        path, data='narrow_template_key = "NARROW"', model=TEMPLATE_MODEL), "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert spectrum.summary()["narrow_template"] == {
        "points": velocity.size,
        "velocity_range": [pytest.approx(velocity.min()), pytest.approx(velocity.max())]}
    built = build_model(config, source_namespace(spectrum.flux), spectrum)
    model = built.model
    assert np.allclose(np.asarray(model._template_velc), velocity)
    assert np.allclose(np.asarray(model._template_flux), profile)
    # template models need no prior of their own: the default amplitude carries them
    assert [line.split(":")[0] for line in built.default_priors] == [
        "Line_template.amplitude", "Line_template.dv"]


def test_the_template_is_sorted_and_its_velocity_unit_is_converted(tmp_path):
    velocity = np.linspace(-500.0, 500.0, 101)[::-1]          # descending on purpose
    path, _, profile = _spectrum_with_template(
        tmp_path, velocity=velocity, unit="m/s", columns=("V", "F"))
    config = load_config(write(tmp_path, config_text(
        path, data='narrow_template_key = "NARROW"', model=TEMPLATE_MODEL), "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    assert np.all(np.diff(spectrum.narrow_template[0]) > 0)
    assert spectrum.narrow_template[0].min() == pytest.approx(-0.5)   # m/s -> km/s
    built = build_model(config, source_namespace(spectrum.flux), spectrum)
    # the columns are read by position, so their names do not matter
    assert np.allclose(np.asarray(built.model._template_flux), profile[::-1])


def test_template_file_wins_over_the_spectrum_profile(tmp_path):
    path, _, _ = _spectrum_with_template(tmp_path)
    template = tmp_path / "own.txt"
    velc = np.linspace(-300.0, 300.0, 61)
    np.savetxt(template, np.column_stack([velc, np.exp(-0.5 * (velc / 60.0) ** 2)]))
    config = load_config(write(tmp_path, config_text(path, data='narrow_template_key '
                                                     '= "NARROW"', model=f"""
        [model]
        [[model.component]]
        type = "Line_template"
        template_file = "{template}"
        wavec = "@Halpha"
    """), "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    built = build_model(config, source_namespace(spectrum.flux), spectrum)
    assert np.allclose(np.asarray(built.model._template_velc), velc)


def test_a_missing_template_extension_is_reported(tmp_path):
    path = make_spectrum(tmp_path, wave=(6500.0, 6630.0))
    config = load_config(write(tmp_path, config_text(
        path, data='narrow_template_key = "NARROW"', model=TEMPLATE_MODEL), "a.toml"))
    with pytest.raises(PipelineError, match="names no HDU"):
        read_spectrum(config, resolve_sources(config)[0])


def test_a_template_component_without_a_profile_is_reported(tmp_path):
    path = make_spectrum(tmp_path, wave=(6500.0, 6630.0))
    config = load_config(write(tmp_path, config_text(path, model=TEMPLATE_MODEL), "a.toml"))
    spectrum = read_spectrum(config, resolve_sources(config)[0])
    with pytest.raises(PipelineError, match="narrow_template_key"):
        build_model(config, source_namespace(spectrum.flux), spectrum)


def test_unknown_component_kind_lists_the_known_ones(tmp_path):
    path = write(tmp_path, f"""
        [data]
        file = "{MOCK}"
        [model]
        [[model.component]]
        type = "Line_Multigaus"
    """, "a.toml")
    config = load_config(path)
    with pytest.raises(PipelineError, match="Line_MultiGauss"):
        build_model(config, source_namespace(np.array([1.0])))


# --------------------------------------------------------------------------------------
# runner: one fit, the files it leaves, and a batch that survives a bad source
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    """Fit the small synthetic spectrum once; several tests read the result."""
    directory = tmp_path_factory.mktemp("run")
    spectrum = make_spectrum(directory, "spec.fits", n=120, wave=(4000.0, 4100.0))
    path = write(directory, config_text(
        spectrum,
        model="""
[model]
[[model.component]]
type = "Linear1D"
name = "Continuum"
[model.component.bounds]
slope = [-1e-3, 1e-3]
intercept = [0.0, 3.0]
"""), "config.toml")
    config = load_config(path)
    report = run_pipeline(config, log=lambda message: None)
    return config, report


def test_run_writes_the_documented_files(finished_run):
    config, report = finished_run
    assert report["counts"] == {"total": 1, "ok": 1, "failed": 0, "skipped": 0}
    entry = report["sources"][0]
    assert entry["status"] == "ok"
    assert entry["n_iterations"] > 0
    assert entry["model"]["free_parameters"] == 2
    assert entry["spectrum"]["n_pixels"] == 120
    stem = Path(config.run.output_dir) / "spec" / "spec"
    for suffix in ("_fit.fits", "_posterior.npz", "_corner.png"):
        assert Path(f"{stem}{suffix}").is_file()
    assert Path(report["report_file"]).is_file()
    assert json.loads(Path(report["report_file"]).read_text())["counts"]["ok"] == 1

    # the posterior is readable with plain numpy, no pickling
    with np.load(f"{stem}_posterior.npz", allow_pickle=False) as posterior:
        assert posterior["samples_equal_weight"].shape == (posterior["samples"].shape[0], 2)
        # a single-component model keeps the bare names; a compound adds the
        # astropy ``_index`` suffix (that is the model's own convention)
        assert list(posterior["param_names"]) == ["slope", "intercept"]
        assert float(posterior["log_evidence"]) == pytest.approx(entry["log_evidence"])

    # the fit file carries the provenance the pipeline added
    with fits.open(f"{stem}_fit.fits") as hdul:
        header = hdul[0].header
        assert header["OBJECT"] == "spec"
        assert header["CONFIGID"] == config.digest[:16]
        assert header["NDATA"] == 120
        assert "LOGZ" in header and "THETA_BEST" in [hdu.name for hdu in hdul]


def test_resume_skips_finished_sources_and_force_refits(finished_run):
    config, _ = finished_run
    again = run_pipeline(config, log=lambda message: None)
    assert again["counts"]["skipped"] == 1
    assert again["sources"][0]["reason"] == "outputs already exist"

    forced = run_pipeline(config, force=True, log=lambda message: None)
    assert forced["counts"]["ok"] == 1


def test_one_bad_source_does_not_stop_the_batch(tmp_path):
    good = make_spectrum(tmp_path, "good.fits", n=100)
    broken = tmp_path / "broken.fits"
    columns = [fits.Column(name="WAVELENGTH", format="D", array=np.linspace(4000, 4100, 10)),
               fits.Column(name="FLUX", format="D", array=np.ones(10))]
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(columns)]).writeto(broken)
    config = load_config(write(tmp_path, f"""
        [data]
        files = ["{good}", "{broken}"]
        z_default = 0.0
        extinction = false
        min_pixels = 10
        [fit]
        nlive = 50
        dlogz = 2.0
        max_iterations = 2000
        [output]
        fit_plot = false
        [model]
        [[model.component]]
        type = "Linear1D"
        name = "Continuum"
        [model.component.bounds]
        slope = [-1e-3, 1e-3]
        intercept = [0.0, 3.0]
    """, "config.toml"))
    report = run_pipeline(config, log=lambda message: None)
    assert report["counts"]["ok"] == 1 and report["counts"]["failed"] == 1
    statuses = {entry["source_id"]: entry for entry in report["sources"]}
    assert statuses["good"]["status"] == "ok"
    assert statuses["broken"]["status"] == "failed"
    assert "error" in statuses["broken"]["error"].lower()


def test_cli_dry_run_and_list_sources(capsys):
    assert pipeline_main([str(EXAMPLE_CONFIG), "--list-sources"]) == 0
    captured = capsys.readouterr()
    assert "mock_two_line_spectrum" in captured.out
    assert "1 source(s)" in captured.out

    assert pipeline_main([str(EXAMPLE_CONFIG), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "2000 pixels" in captured.out
    assert "nothing was fitted" in captured.out


def test_template_writes_a_configuration_that_works(tmp_path, capsys):
    """``--template`` must hand out a file that loads, reads and builds as written."""
    path = tmp_path / "starter.toml"
    assert pipeline_main(["--template", str(path)]) == 0
    assert f"wrote {path}" in capsys.readouterr().out

    text = path.read_text()
    for section in ("[run]", "[data]", "[fit]", "[output]", "[model]",
                    "[[model.component]]", "[model.component.ties]",
                    "[model.component.priors.amplitude]"):
        assert section in text, section
    # it is a *working* file, not a sheet of placeholders: this one names a real
    # spectrum, so the first --dry-run succeeds before the user edits anything
    config = load_config(path)
    assert Path(config.data.file).is_file()
    assert Path(config.run.output_dir) == tmp_path / "out"
    sources = resolve_sources(config)
    assert len(sources) == 1
    import warnings as warnings_module

    with warnings_module.catch_warnings():          # z and extinction are declared,
        warnings_module.simplefilter("error")       # so reading must be silent
        spectrum = read_spectrum(config, sources[0])
    assert spectrum.n_pixels == 2000
    assert spectrum.z_source == "z_default"
    assert spectrum.e_bv_source == "none"
    built = build_model(config, source_namespace(
        spectrum.flux, spectrum.wave, spectrum.redshift,
        config.data.continuum_percentile))
    assert built.components == ("Continuum", "Line1", "Line2")
    assert len(built.free_parameters) == 8

    assert pipeline_main([str(path), "--dry-run"]) == 0
    assert "2000 pixels" in capsys.readouterr().out


def test_template_refuses_to_overwrite_and_can_be_printed(tmp_path, capsys):
    path = tmp_path / "starter.toml"
    assert pipeline_main(["--template", str(path)]) == 0
    before = path.read_text()

    assert pipeline_main(["--template", str(path)]) == 2       # do not eat a real file
    assert "already exists" in capsys.readouterr().err
    assert path.read_text() == before

    assert pipeline_main(["--template", str(path), "--force"]) == 0
    assert path.read_text() == before
    capsys.readouterr()                                        # drop the write messages

    assert pipeline_main(["--template", "-"]) == 0              # to stdout, nothing written
    printed = capsys.readouterr().out
    assert printed.startswith("# GalSpec pipeline configuration")
    assert "[model]" in printed


def test_a_configuration_is_required(capsys):
    assert pipeline_main([]) == 2
    captured = capsys.readouterr()
    assert "a configuration file is required" in captured.err
    assert "usage:" in captured.err


def test_the_progress_bar_is_automatic_and_can_be_overridden(monkeypatch):
    """A bar in front of a person, none in a log; [run] progress forces either way."""
    from galspec.pipeline.runner import wants_progress

    config = load_config(EXAMPLE_CONFIG)
    assert config.run.progress is None                     # the default: decide at run time

    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert wants_progress(config) is True                  # a terminal gets a bar
    assert wants_progress(config, workers=4) is False      # ... unless it is a pool

    monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
    assert wants_progress(config) is False                 # a redirected log does not

    forced = replace(config, run=replace(config.run, progress=True))
    assert wants_progress(forced) is True                  # ... unless the file says so
    assert wants_progress(forced, workers=4) is False

    muted = replace(config, run=replace(config.run, progress=False))
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert wants_progress(muted) is False


def test_cli_quiet_also_silences_the_progress_bar(monkeypatch):
    from galspec.pipeline import __main__ as cli

    seen = {}

    def fake_run_pipeline(config, **kwargs):
        seen["progress"] = config.run.progress
        return {"counts": {"total": 0, "ok": 0, "failed": 0, "skipped": 0},
                "sources": []}

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    assert cli.main([str(EXAMPLE_CONFIG)]) == 0
    assert seen["progress"] is None                        # the config decides
    assert cli.main([str(EXAMPLE_CONFIG), "--quiet"]) == 0
    assert seen["progress"] is False
    # the flag pair is gone: --quiet is the only verbosity switch
    assert "--progress" not in cli.build_parser().format_help()


def test_a_quiet_fit_says_why_it_is_quiet(capsys):
    """A nested-sampling run prints nothing for minutes; the pipeline says so."""
    from galspec.pipeline.runner import run_pipeline as real_run_pipeline

    config = load_config(EXAMPLE_CONFIG)
    messages = []
    real_run_pipeline(config, force=False, log=messages.append)
    assert any("quiet until it finishes" in message for message in messages)


def test_cli_reports_a_config_error(tmp_path, capsys):
    path = write(tmp_path, "[data]\nfile = 'x.fits'\n")
    assert pipeline_main([str(path)]) == 2
    assert "missing required section [model]" in capsys.readouterr().err