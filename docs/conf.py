# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyTransport3.0'
copyright = '2025, Andrea Costantini, David Mulryne, John W. Ronayne'
author = 'Andrea Costantini, David Mulryne, John W. Ronayne'
release = '3.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',   # Automatically generate docs from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'breathe',              # Bridge between Sphinx and Doxygen
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Breathe Configuration -------------------------------------------------
# This tells Breathe where to find the Doxygen XML output.
breathe_projects = {
    "PyTransport3.0": "xml/"
}
breathe_default_project = "PyTransport3.0"
