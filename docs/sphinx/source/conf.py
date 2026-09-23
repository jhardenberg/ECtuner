# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../..')) 

project = 'ECtuner'
copyright = '2026, Jost von Hardenberg, Federico Fabiano, Marianna Albanese'
author = 'Jost von Hardenberg, Federico Fabiano, Marianna Albanese'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',        
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',        
    'myst_parser',                
    'sphinx_autodoc_typehints'  
]  

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = 'bysource'
autoclass_content = 'both' 

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

napoleon_include_init_with_doc = True 
napoleon_use_param = True
napoleon_use_rtype = True

# -- Typehints configuration -------------------------------------------------
typehints_document_rtype = True
typehints_use_signature_return = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None)
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static'] # Decommenta se hai un logo o CSS custom

# -- Mock Imports ------------------------------------------------------------
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'pandas',
    'xarray',
    'matplotlib',
    'cartopy',
    'smmregrid',
    'ecmean',
    'tabulate',
    'ruamel.yaml'
]