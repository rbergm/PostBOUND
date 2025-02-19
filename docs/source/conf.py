# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'PostBOUND'
copyright = '2025, Database Group @ TU Dresden'
author = 'Database Group @ TU Dresden'
release = '0.12.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'
html_theme_options = {
    'body_min_width': 0,
    'body_max_width': 'none',
    'sidebarwidth': '30%'
}

# -- Options for API documentation -------------------------------------------------

autodoc_default_options = {
    'member_order': 'bysource',
    'special-members': '__init__'
}
autodoc_member_order = 'bysource'
