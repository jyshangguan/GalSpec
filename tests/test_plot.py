"""Tests for the figure-level conventions the worked example relies on.

Two things regressed once and are easy to regress again, so they are pinned here:

* a model with twenty components must not reuse colours -- matplotlib's default
  cycle has ten, and the collision put the total model and a fitted line in the
  same red, exactly in the panels meant to show the decomposition;
* the residual panel must be readable: drawn in flux units one bright pixel sets
  the scale of the whole panel, so ``residual_in_sigma=True`` divides by the
  error and labels the axis accordingly.

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


if __name__ == "__main__":
    raise SystemExit(main())