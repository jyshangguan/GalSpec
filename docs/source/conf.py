# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SAGAN'
copyright = '2025, SAGAN Developers'
author = 'SAGAN Developers'
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
html_title = 'SAGAN Documentation'
# html_logo = '../_static/logo.png'  # Uncomment if you have a logo
# html_favicon = '../_static/favicon.ico'  # Uncomment if you have a favicon

# Theme options
html_theme_options = {
    'show_prev_next': False,
    'repository_url': 'https://github.com/jyshangguan/SAGAN',
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

htmlhelp_basename = 'SAGANdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'paper',
    'pointsize': '10pt',
}

latex_documents = [
    (master_doc, 'SAGAN.tex', 'SAGAN Documentation',
     'SAGAN Developers', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'sagan', 'SAGAN Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'SAGAN', 'SAGAN Documentation',
     author, 'SAGAN', 'Spectral Analysis of Galaxy and Active galactic Nuclei.',
     'Miscellaneous'),
]
