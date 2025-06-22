# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# This line is crucial: it adds the root of your project to the Python path,
# allowing Sphinx to import your 'PyTransport' package.
sys.path.insert(0, os.path.abspath('../../')) # Adjust this path if 'docs' is not directly under the project root
sys.path.insert(0, os.path.abspath('../PyTransport/')) # Add PyTransport directory as well for direct imports
sys.path.insert(0, os.path.abspath('../PyTransport/PyTransPy/')) # Add PyTransPy

# -- Project information -----------------------------------------------------

project = 'PyTransport'
copyright = '2025, Andrea Costantini, David Mulryne, John W. Ronayne'
author = 'Andrea Costantini, David Mulryne, John W. Ronayne'

# The full version, including alpha/beta/rc tags
release = '3.0' # Assuming PyTransport3.0 as the version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',      # Automatically pulls documentation from docstrings
    'sphinx.ext.napoleon',     # Supports NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Adds links to highlighted source code
    'sphinx.ext.autosummary',  # Generates summary tables for API items
    'sphinx.ext.mathjax',      # Renders LaTeX math
    'sphinx.ext.intersphinx',  # Links to other Sphinx documentations (e.g., NumPy)
    'breathe',                 # <--- NEW: For integrating Doxygen C++ documentation
]

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
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


# Breathe Configuration
# This tells Breathe where to find the Doxygen XML output.
# The path is relative to the directory where conf.py resides.
breathe_projects = {
    "PyTransportCpp": os.path.abspath(os.path.join(os.path.dirname(__file__), "_build/doxygen_xml"))
}
# Set the default Doxygen project to use, so you don't have to specify :project: in every directive.
breathe_default_project = "PyTransportCpp"


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' # Recommended theme for clean, responsive design

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    # Add other libraries your project uses if you want to link to their docs
}

# Autosummary settings
autosummary_generate = True
