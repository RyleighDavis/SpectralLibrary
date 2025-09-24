"""
Core data structures and classes for the spectral library.
"""
import json
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional


@dataclass_json
@dataclass
class LaboratorySpectrum:
    """Dataclass for storing laboratory spectrum data and associated metadata."""
    species: str
    wavelength: np.ndarray  # wavelength in microns
    spectrum: np.ndarray
    spectral_units: str  # one of 'abs reflectance', 'rel reflectance'
    temperature: str  # e.g. room, or T in K
    atmosphere: str  # e.g. ambient, vacuum, dry air
    category: str  # one of 'mineral', 'organic', 'ice'
    mineral_type: str  # e.g. 'silicate', 'carbonate', 'sulfate'
    mineral_subtype: str  # e.g. 'pyroxene', 'phyllosilicate'
    formula: str  # chemical formula of the sample
    grain_size: str  # grain size of the sample
    source: str  # spectral library source: e.g. USGS, RELAB, PSF, etc.
    sample_id: str  # unique identifier for the sample, from original source
    sample_location: str  # where sample was acquired from (e.g. field site, meteorite, etc.)
    facility: str  # facility where spectrum was measured
    spectrometer: str  # type of spectrometer used to measure the spectrum
    sample_description: str  # description of the sample
    file_name: str  # original file name of the spectrum

    def __init__(self, species: str, wavelength: np.ndarray, spectrum: np.ndarray,
                 spectral_units: str, source: str, **kwargs):
        self.species = species
        assert len(wavelength) == len(spectrum), "Wavelength and spectrum must have same length"
        self.wavelength = wavelength
        self.spectrum = spectrum
        self.spectral_units = spectral_units
        self.source = source

        # Set optional attributes with defaults
        self.temperature = kwargs.get('temperature', '')
        self.atmosphere = kwargs.get('atmosphere', '')
        self.category = kwargs.get('category', '')
        self.mineral_type = kwargs.get('mineral_type', '')
        self.mineral_subtype = kwargs.get('mineral_subtype', '')
        self.formula = kwargs.get('formula', '')
        self.grain_size = kwargs.get('grain_size', '')
        self.sample_id = kwargs.get('sample_id', '')
        self.sample_location = kwargs.get('sample_location', '')
        self.facility = kwargs.get('facility', '')
        self.spectrometer = kwargs.get('spectrometer', '')
        self.sample_description = kwargs.get('sample_description', '')
        self.file_name = kwargs.get('file_name', '')

        # Set any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def save_json(self, filepath: str, prefix: str = ''):
        """Save the spectrum to a JSON file."""
        filename = f"{prefix}{self.species}_{self.sample_id}.json".replace(' ', '').replace('/', '_')
        with open(f"{filepath}/{filename}", 'w') as f:
            json.dump(self.to_json(), f)

    def get_wavelength_range(self) -> tuple:
        """Get the wavelength range of this spectrum."""
        return (float(np.min(self.wavelength)), float(np.max(self.wavelength)))

    def interpolate_to_wavelengths(self, new_wavelengths: np.ndarray) -> 'LaboratorySpectrum':
        """Interpolate spectrum to new wavelength grid."""
        from scipy import interpolate

        # Create interpolation function
        interp_func = interpolate.interp1d(
            self.wavelength, self.spectrum,
            kind='linear', bounds_error=False, fill_value=np.nan
        )

        # Interpolate to new wavelengths
        new_spectrum = interp_func(new_wavelengths)

        # Create new LaboratorySpectrum object
        return LaboratorySpectrum(
            species=self.species,
            wavelength=new_wavelengths,
            spectrum=new_spectrum,
            spectral_units=self.spectral_units,
            source=self.source,
            temperature=self.temperature,
            atmosphere=self.atmosphere,
            category=self.category,
            mineral_type=self.mineral_type,
            mineral_subtype=self.mineral_subtype,
            formula=self.formula,
            grain_size=self.grain_size,
            sample_id=self.sample_id,
            sample_location=self.sample_location,
            facility=self.facility,
            spectrometer=self.spectrometer,
            sample_description=self.sample_description,
            file_name=self.file_name
        )