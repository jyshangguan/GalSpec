"""The worked-example notebook builds the same model as the module the scripts import.

``example/fit_iras07546_jax.ipynb`` writes the model out literally -- seventeen
component constructors, then the ties, then the priors -- so that a reader can copy a
cell and change it.  ``example/iras07546_model.py`` keeps ``build_model()`` and
``set_priors()`` because the three example scripts and the pipeline parity tests import
them.  Two copies of the same model is exactly the kind of thing that drifts silently:
one number changes on one side and no test notices.

So this file executes the notebook's model cells and compares the result with the
module's, through the model serializer -- which records everything that matters about a
model: the submodel tree, every constructor argument, the template arrays, which
parameters are fixed, which are tied and *to what*, and the prior on each parameter.

Run with::

    python -m pytest tests/test_example_notebook.py -v
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example"
NOTEBOOK = EXAMPLE / "fit_iras07546_jax.ipynb"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLE))

#: The cells that define the model.  Everything else in the notebook either reads the
#: data, runs the sampler or draws -- all of which the module does not do.
MODEL_CELLS = ("setup", "settings", "data",
               "model_continuum", "model_iron", "model_broad", "model_narrow",
               "model_sum", "ties_narrow", "ties_broad", "ties_iron", "priors")


def notebook_cells(path: Path = NOTEBOOK) -> dict[str, str]:
    """``{marker: source}`` for every ``# %% marker`` cell of a notebook."""
    notebook = json.loads(path.read_text())
    cells = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        first = source.splitlines()[0] if source.strip() else ""
        if first.startswith("# %%"):
            cells[first[4:].strip()] = source
    return cells


@contextmanager
def in_example_directory():
    """The notebook assumes it is run from ``example/`` (``HERE = Path.cwd()``)."""
    previous = Path.cwd()
    import os
    os.chdir(EXAMPLE)
    try:
        yield
    finally:
        os.chdir(previous)


@pytest.fixture(scope="module")
def notebook_namespace():
    """The notebook's own namespace, after its model and priors cells have run."""
    import matplotlib
    matplotlib.use("Agg")

    cells = notebook_cells()
    missing = [name for name in MODEL_CELLS if name not in cells]
    assert not missing, f"the notebook lost these # %% cells: {missing}"

    namespace = {"__name__": "__main__"}
    with in_example_directory():
        for name in MODEL_CELLS:
            exec(compile(cells[name], f"<nb:{name}>", "exec"), namespace)
    return namespace


def test_the_notebook_has_no_wrapper_functions():
    """The point of the notebook is that there is nothing to look up.

    A ``def`` in there would be a wrapper, and the reader would have to jump to the
    body to see what the fit actually is.  The model cells are the constraint this
    test exists for; the rest of the notebook is checked too, because a helper added
    later would undo the same thing.
    """
    for marker, source in notebook_cells().items():
        definitions = [line.strip() for line in source.splitlines()
                       if line.strip().startswith("def ")]
        assert not definitions, f"cell {marker!r} defines {definitions}"


def test_the_notebook_model_is_the_module_model(notebook_namespace):
    """Same submodels, same arguments, same ties, same priors -- checked structurally.

    The serializer is the right comparison because a tie is stored as a *reference*:
    ``equals`` would only notice that two parameters hold the same number today, while
    the serialized tree says ``b_hbeta.sigma_c`` is tied to ``b_halpha.sigma_c``.
    """
    from galspec.jax_models.serialization import serialize_jax_model_tree
    from iras07546_model import build_model, set_priors

    notebook_model = notebook_namespace["model"]
    module_model = set_priors(build_model(), notebook_namespace["flux"])

    notebook_tree, notebook_arrays = serialize_jax_model_tree(notebook_model)
    module_tree, module_arrays = serialize_jax_model_tree(module_model)

    assert notebook_tree == module_tree
    assert sorted(notebook_arrays) == sorted(module_arrays)
    for key in notebook_arrays:
        assert np.array_equal(np.asarray(notebook_arrays[key]),
                              np.asarray(module_arrays[key])), key


def test_the_notebook_and_the_module_agree_on_every_number(notebook_namespace):
    """The tie graph, the free parameters and the prior bounds, bit for bit."""
    from iras07546_model import build_model, set_priors

    notebook_model = notebook_namespace["model"]
    module_model = set_priors(build_model(), notebook_namespace["flux"])

    assert list(notebook_model.submodel_names) == list(module_model.submodel_names)
    assert list(notebook_model.free_param_names) == list(module_model.free_param_names)

    # one value per free parameter, pushed through the prior: equal samples mean equal
    # distributions, which is what a prior comparison has to be
    for name in notebook_model.free_param_names:
        mine = notebook_model.get_prior(name)
        theirs = module_model.get_prior(name)
        assert mine is not None, f"{name} has no prior in the notebook"
        assert theirs is not None, f"{name} has no prior in the module"
        for u in (0.0, 0.13, 0.5, 0.77, 1.0):
            assert float(mine(u)) == pytest.approx(float(theirs(u)), rel=1e-12), \
                f"{name} prior differs at u = {u}"


def test_the_notebook_fits_the_same_arrays_as_the_scripts(notebook_namespace):
    """``(wave, flux, error)`` must be *identical* to ``load_spectrum()``, not close.

    The error is measured from the spectrum, and the recipe reads neighbouring pixels:
    applying it before the wavelength cut instead of after moves about 4 000 pixels at
    the window edges by up to 8%, which is worth tens of nats of evidence. So the order
    of the operations in that cell is part of the model, and this is what pins it.
    """
    from iras07546_model import load_spectrum

    wave, flux, error = notebook_namespace["wave"], notebook_namespace["flux"], \
        notebook_namespace["error"]
    wave_module, flux_module, error_module = load_spectrum()

    assert wave.size == flux.size == error.size == 56_956
    assert np.array_equal(wave, wave_module)
    assert np.array_equal(flux, flux_module)
    assert np.array_equal(error, error_module)
    assert np.all(error > 0.0)
    # the flux is positive where it matters, and in the units the doc quotes
    assert 4.0 < flux.max() < 5.0