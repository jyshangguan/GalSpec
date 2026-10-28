# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import json
import os
import sys

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GalSpec'
copyright = '2025, GalSpec Developers'
author = 'GalSpec Developers'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'nbsphinx',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = 'GalSpec Documentation'
html_logo = '_static/logo.jpg'  # Logo for sidebar/header
# html_favicon = '_static/favicon.ico'  # Uncomment if you have a favicon

# Theme options
html_theme_options = {
    'show_prev_next': False,
    'repository_url': 'https://github.com/jyshangguan/GalSpec',
    'repository_branch': 'main',
    'path_to_docs': 'docs',
    'use_edit_page_button': True,
    'use_repository_button': True,
    'use_issues_button': True,
    'use_download_button': True,
    'home_page_in_toc': False,
    'extra_footer': '',
    'show_navbar_depth': 2,
    'navigation_with_keys': False,
}

# Add any paths that contain custom static files (such as style sheets)
html_css_files = [
    'custom.css',
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'emcee': ('https://emcee.readthedocs.io/en/stable/', None),
    'dynesty': ('https://dynesty.readthedocs.io/en/stable/', None),
}

# nbsphinx settings
nbsphinx_execute = 'never'  # Set to 'always' to execute notebooks during build
nbsphinx_allow_errors = False
nbsphinx_require_js_path = ''

# -- Master document ---------------------------------------------------------

master_doc = 'index'

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'galspecdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'paper',
    'pointsize': '10pt',
}

latex_documents = [
    (master_doc, 'GalSpec.tex', 'GalSpec Documentation',
     'GalSpec Developers', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'galspec', 'GalSpec Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'GalSpec', 'GalSpec Documentation',
     author, 'GalSpec', 'Spectral Analysis of Galaxy and Active galactic Nuclei.',
     'Miscellaneous'),
]


# -- A notebook cell as a code block ---------------------------------------------------
# Every code cell of ``example/fit_iras07546_jax.ipynb`` begins with a ``# %% name``
# marker.  The walkthrough page quotes those cells, so the code a reader copies out of
# the documentation is the code the notebook runs -- there is no second copy to drift::
#
#     .. nbcode:: ../../example/fit_iras07546_jax.ipynb
#        :cell: model_broad

_CELL_CACHE = {}


def _notebook_cells(path):
    """``{marker: source}`` for every marked code cell of a notebook."""
    path = os.path.abspath(path)
    if path not in _CELL_CACHE:
        with open(path, "r", encoding="utf-8") as handle:
            notebook = json.load(handle)
        cells = {}
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell["source"])
            first = source.splitlines()[0] if source.strip() else ""
            if first.startswith("# %%"):
                cells[first[4:].strip()] = source
        _CELL_CACHE[path] = cells
    return _CELL_CACHE[path]


class NotebookCellDirective(SphinxDirective):
    """Insert one ``# %% name`` cell of a Jupyter notebook, highlighted as Python."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec = {"cell": str}

    def run(self):
        notebook = self.arguments[0]
        marker = self.options.get("cell")
        if not marker:
            raise self.error("nbcode needs a :cell: option naming the cell's marker")
        cells = _notebook_cells(os.path.join(self.env.srcdir, notebook))
        if marker not in cells:
            known = ", ".join(sorted(cells))
            raise self.error(f"no %%%% cell {marker!r} in {notebook}; it has: {known}")
        block = nodes.literal_block(cells[marker], cells[marker])
        block["language"] = "python"
        self.set_source_info(block)
        return [block]


def setup(app):
    app.add_directive("nbcode", NotebookCellDirective)
    return {"version": "1.0", "parallel_read_safe": True}
