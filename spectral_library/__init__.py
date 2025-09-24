"""
SpectralLibrary - A Python package for managing ice and mineral spectra laboratory data.

This package provides tools for:
- Loading spectral data from various sources and formats
- Normalizing and processing spectra
- Interactive visualization and filtering
- Storing and retrieving combined spectral libraries

The package is designed specifically for interpreting telescope observations
of icy moon surfaces by providing a common framework for laboratory spectra.
"""

from .core import LaboratorySpectrum
from .loaders import load_library_from_json, load_data, save_library_to_pickle
from .processing import normalize, convert_to_albedo
from .visualization import SpectralPlotter, create_plot
from .filters import SpectralFilter
from .database import SpectralDatabase

__version__ = "0.1.0"
__author__ = "Ryleigh Davis"

__all__ = [
    "LaboratorySpectrum",
    "load_library_from_json",
    "load_data",
    "save_library_to_pickle",
    "normalize",
    "convert_to_albedo",
    "SpectralPlotter",
    "create_plot",
    "SpectralFilter",
    "SpectralDatabase"
]