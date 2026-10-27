"""Tests for the figure-level conventions the worked example relies on.

Two things regressed once and are easy to regress again, so they are pinned here:

* a model with twenty components must not reuse colours -- matplotlib's default
  cycle has ten, and the collision put the total model and a fitted line in the
  same red, exactly in the panels meant to show the decomposition;
* the residual panel must be readable: drawn in flux units one bright pixel sets
  the scale of the whole panel, so ``residual_in_sigma=True`` divides by the
  error and labels the axis accordingly.

A third convention is pinned here: ``plot_fit_jax`` must break a
``Line_MultiGauss`` / ``Line_MultiGauss_doublet`` into its Gaussians, because a JAX
leaf is not an instance of the astropy class ``plot_fit`` tests against.

Run with::

    python -m pytest tests/test_plot.py -v
    python tests/test_plot.py            # standalone, no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from astropy.modeling import models as amodels                        # noqa: E402
from matplotlib.colors import to_rgba                                 # noqa: E402

import galspec                                                        # noqa: E402


def _many_component_model(n_lines: int = 12):
    """Continuum plus ``n_lines`` Gaussians: more components than tab10 colours."""
    model = amodels.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=1.0, name="Continuum")
    for index in range(n_lines):
        centre = 4000.0 + 150.0 * index
        model += amodels.Gaussian1D(amplitude=0.1, mean=centre, stddev=3.0,
                                    name=f"line{index}")
    return model


def test_components_get_distinct_colours():
    """Twenty components must not share a colour, and none may take the total model's."""
    model = _many_component_model()
    figure = plt.figure()
    ax = figure.add_axes([0.1, 0.2, 0.8, 0.7])
    axr = figure.add_axes([0.1, 0.05, 0.8, 0.12])
    wave = np.linspace(3900.0, 5900.0, 2000)
    galspec.plot.plot_fit(wave, np.ones_like(wave) * 1.2, model, ax=ax, axr=axr,
                          legend_kwargs={"ncol": 2, "fontsize": 7})
    figure.canvas.draw()

    handles, labels = ax.get_legend_handles_labels()
    colours = {label: to_rgba(handle.get_color()) for handle, label in zip(handles, labels)}
    components = {label: colour for label, colour in colours.items()
                  if label not in ("Data", "Total model")}
    total = colours["Total model"]

    assert len(components) == model.n_submodels
    assert len(set(components.values())) == len(components), \
        "two components share a colour"
    assert total not in components.values(), \
        "a component is drawn in the same colour as the total model"
    plt.close(figure)
    print(f"  {len(components)} components, {len(set(components.values()))} colours, "
          "none equal to the total model")


def test_residual_panel_in_sigma():
    """``residual_in_sigma`` divides by the error and says so on the axis."""
    wave = np.linspace(6400.0, 6700.0, 600)
    flux = np.full_like(wave, 1.0)
    error = np.full_like(wave, 0.1)
    # plot_fit walks submodels, so the model has to be a compound one
    model = (amodels.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=0.0, name="Continuum")
             + amodels.Gaussian1D(amplitude=0.0, mean=6500.0, stddev=5.0, name="empty"))

    figure = plt.figure()
    ax = figure.add_axes([0.1, 0.2, 0.8, 0.7])
    axr = figure.add_axes([0.1, 0.05, 0.8, 0.12])
    galspec.plot.plot_fit(wave, flux, model, ax=ax, axr=axr, error=error,
                          residual_in_sigma=True, legend_kwargs={"fontsize": 8})
    figure.canvas.draw()

    residual = np.asarray(axr.lines[0].get_ydata(), dtype=float)
    # flux - model = 0 everywhere, so the panel must be flat at zero in sigma too
    assert np.allclose(residual, 0.0, atol=1e-12)
    assert "sigma" in axr.get_ylabel() or "\\sigma" in axr.get_ylabel()
    plt.close(figure)

    # a residual of one sigma must show up as 1.0, not as 0.1
    flux = np.where(np.arange(wave.size) == 10, 1.1, 1.0)
    figure = plt.figure()
    ax = figure.add_axes([0.1, 0.2, 0.8, 0.7])
    axr = figure.add_axes([0.1, 0.05, 0.8, 0.12])
    galspec.plot.plot_fit(wave, flux, model, ax=ax, axr=axr, error=error,
                          residual_in_sigma=True, legend_kwargs={"fontsize": 8})
    figure.canvas.draw()
    residual = np.asarray(axr.lines[0].get_ydata(), dtype=float)
    assert np.isclose(np.nanmax(residual), 1.0, atol=1e-9)
    plt.close(figure)
    print("  residual panel is scaled in sigma")


def test_residual_in_sigma_needs_errors():
    """Forgetting the error array must fail loudly rather than silently mislabel."""
    wave = np.linspace(6400.0, 6700.0, 100)
    model = (amodels.PowerLaw1D(amplitude=1.0, x_0=5500.0, alpha=0.0, name="Continuum")
             + amodels.Gaussian1D(amplitude=0.0, mean=6500.0, stddev=5.0, name="empty"))
    figure = plt.figure()
    ax = figure.add_axes([0.1, 0.2, 0.8, 0.7])
    axr = figure.add_axes([0.1, 0.05, 0.8, 0.12])
    try:
        galspec.plot.plot_fit(wave, np.ones_like(wave), model, ax=ax, axr=axr,
                              residual_in_sigma=True)
    except ValueError as exc:
        assert "error" in str(exc)
    else:                                                        # pragma: no cover
        raise AssertionError("residual_in_sigma without `error` should raise")
    finally:
        plt.close(figure)
    print("  missing error array raises ValueError")


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        try:
            print(f"[run ] {name}")
            test()
            print(f"[ ok ] {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            import traceback
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
    print()
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


def _jax_doublet(n_components=3):
    from galspec import jax_models as jxm
    return jxm.Line_MultiGauss_doublet(
        n_components=n_components, amp_c0=2.0, amp_c1=0.67, dv_c=40.0, sigma_c=250.0,
        wavec0=5007.0, wavec1=4959.0, amp_w0=0.3, dv_w0=-900.0, sigma_w0=1200.0,
        amp_w1=0.05, dv_w1=1500.0, sigma_w1=3000.0, name="OIII")


def test_jax_subcomponents_match_the_astropy_model():
    """The JAX property must be the astropy property: same count, same curves."""
    from galspec import jax_models as jxm
    from galspec.line_profile import (Line_MultiGauss as CpuMultiGauss,
                                      Line_MultiGauss_doublet as CpuDoublet)
    wave = np.linspace(4900.0, 5100.0, 4000)

    single = dict(n_components=3, amp_c=2.0, dv_c=40.0, sigma_c=250.0, wavec=5007.0,
                  amp_w0=0.3, dv_w0=-900.0, sigma_w0=1200.0,
                  amp_w1=0.05, dv_w1=1500.0, sigma_w1=3000.0, name="OIII")
    jax_single, cpu_single = jxm.Line_MultiGauss(**single), CpuMultiGauss(**single)
    jax_parts, cpu_parts = jax_single.subcomponents, cpu_single.subcomponents
    assert len(jax_parts) == len(cpu_parts) == 3
    for a, b in zip(jax_parts, cpu_parts):
        assert np.allclose(np.asarray(a(wave)), b(wave), rtol=1e-12, atol=1e-14), a.name
    total = np.asarray(jax_single(wave))
    assert np.allclose(sum(np.asarray(p(wave)) for p in jax_parts), total,
                       rtol=1e-12, atol=1e-13), "the parts must add up to the model"

    doublet = dict(n_components=3, amp_c0=2.0, amp_c1=0.67, dv_c=40.0, sigma_c=250.0,
                   wavec0=5007.0, wavec1=4959.0, amp_w0=0.3, dv_w0=-900.0,
                   sigma_w0=1200.0, amp_w1=0.05, dv_w1=1500.0, sigma_w1=3000.0,
                   name="OIII")
    jax_double, cpu_double = jxm.Line_MultiGauss_doublet(**doublet), CpuDoublet(**doublet)
    jax_parts, cpu_parts = jax_double.subcomponents, cpu_double.subcomponents
    assert len(jax_parts) == len(cpu_parts) == 6, "two cores, then two curves per wind"
    for a, b in zip(jax_parts, cpu_parts):
        assert np.allclose(np.asarray(a(wave)), b(wave), rtol=1e-12, atol=1e-14), a.name

    # a single-component model has nothing to break down, exactly as in astropy
    assert jxm.Line_MultiGauss(n_components=1, wavec=5007.0).subcomponents is None
    assert jxm.Line_MultiGauss_doublet(n_components=1, wavec0=5007.0,
                                       wavec1=4959.0).subcomponents is None
    print("  JAX subcomponents match the astropy ones, and add up to the model")


def test_plot_fit_jax_draws_the_subcomponents():
    """One dashed curve per Gaussian, in the colour of the component they belong to."""
    from galspec import jax_models as jxm
    wave = np.linspace(4900.0, 5100.0, 2000)
    model = (jxm.PowerLaw1D(amplitude=1.0, x_0=5000.0, alpha=0.0, fixed={"x_0": True},
                            name="Continuum")
             + _jax_doublet())
    flux = np.asarray(model(wave))
    error = np.full_like(flux, 0.05)

    figure = plt.figure()
    ax = figure.add_axes([0.1, 0.2, 0.8, 0.7])
    axr = figure.add_axes([0.1, 0.05, 0.8, 0.12])
    galspec.plot.plot_fit_jax(wave, flux, model, ax=ax, axr=axr, error=error,
                              residual_in_sigma=True,
                              legend_kwargs={"fontsize": 8})
    figure.canvas.draw()

    dashed = [line for line in ax.lines if line.get_linestyle() == '--']
    assert len(dashed) == 6, f"expected six subcomponent curves, got {len(dashed)}"
    # same colour as the component above them, and not the total model's
    solid = {to_rgba(line.get_color()) for line in ax.lines
             if line.get_linestyle() != '--' and line.get_label() == 'OIII'}
    assert solid, "the [O III] component is missing from the panel"
    assert all(to_rgba(line.get_color()) in solid for line in dashed)

    # the panels agree with the data: a perfect fit has a flat residual
    residual = np.asarray(axr.lines[0].get_ydata(), dtype=float)
    assert np.allclose(residual, 0.0, atol=1e-9)

    # the drawn total is the model, and the drawn parts add up to it
    total = [line for line in ax.lines if line.get_label() == 'Total model'][0]
    assert np.allclose(np.asarray(total.get_ydata(), dtype=float), flux, rtol=1e-12)
    total_sub = sum(np.asarray(line.get_ydata(), dtype=float) for line in dashed)
    oiii = [line for line in ax.lines if line.get_label() == 'OIII'][0]
    assert np.allclose(total_sub, np.asarray(oiii.get_ydata(), dtype=float),
                       rtol=1e-10, atol=1e-12)
    plt.close(figure)
    print("  plot_fit_jax draws six dashed subcomponents under [O III]")


def test_plot_fit_jax_reuses_tied_values():
    """A tied parameter must be drawn at its fitted value, not its default.

    A JAX leaf stores its own copy of a parameter, so a tied one holds its
    construction default until something writes the tie back.  ``plot_fit`` walked
    into exactly that once (a tied [N II] width drawn at 700 km/s instead of
    [O III]'s 298), so the stale value is put back here on purpose and the test
    requires the figure to be right anyway.
    """
    from galspec import jax_models as jxm
    from galspec.line_profile import Line_MultiGauss_doublet as CpuDoublet

    core = jxm.Line_MultiGauss(n_components=2, amp_c=1.0, dv_c=0.0, sigma_c=300.0,
                               wavec=5007.0, amp_w0=0.4, dv_w0=0.0, sigma_w0=2000.0,
                               name="OIII")
    tied = jxm.Line_MultiGauss_doublet(n_components=2, amp_c0=0.3, amp_c1=0.1,
                                       dv_c=300.0, sigma_c=900.0,
                                       wavec0=6584.0, wavec1=6548.0,
                                       amp_w0=0.4, dv_w0=0.0, sigma_w0=1500.0,
                                       name="NII")
    tied.amp_w0.tied = jxm.tie.tie_MultiGauss_amp_w0("OIII")
    model = core + tied

    # what the tie would be ignored: NII's stored wind amplitude back to 0.5
    tied._values["amp_w0"] = 0.5

    # the same doublet, drawn from the value the tie should produce
    expected = CpuDoublet(n_components=2, amp_c0=0.3, amp_c1=0.1, dv_c=300.0,
                          sigma_c=900.0, wavec0=6584.0, wavec1=6548.0,
                          amp_w0=0.4, dv_w0=0.0, sigma_w0=1500.0, name="NII")

    wave = np.linspace(6500.0, 6650.0, 1200)
    flux = np.asarray(model(wave))
    figure = plt.figure()
    ax = figure.add_axes([0.1, 0.2, 0.8, 0.7])
    axr = figure.add_axes([0.1, 0.05, 0.8, 0.12])
    galspec.plot.plot_fit_jax(wave, flux, model, ax=ax, axr=axr,
                              legend_kwargs={"fontsize": 8})
    figure.canvas.draw()

    drawn = [line for line in ax.lines if line.get_label() == "NII"][0]
    assert np.allclose(np.asarray(drawn.get_ydata(), dtype=float), expected(wave),
                       rtol=1e-10, atol=1e-13), "drew the stale, untied value"
    assert np.isclose(tied._values["amp_w0"], 0.4), "the tie was not written back"
    plt.close(figure)
    print("  plot_fit_jax draws tied components at their fitted values")


if __name__ == "__main__":
    raise SystemExit(main())