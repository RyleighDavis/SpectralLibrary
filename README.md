# SpectralLibrary

A Python package for managing ice and mineral laboratory spectra relevant for icy moon surfaces.

## Installation

### From PyPI (when published)
```bash
pip install spectral-library
```

### Development Installation
```bash
git clone https://github.com/ryleighdavis/spectral-library.git
cd spectral-library
pip install -e .
```

## Quick Start

### Basic Usage
```python
from spectral_library import LaboratorySpectrum, load_data, SpectralPlotter

# Load spectral data
data = load_data("your_spectra.pkl")

# Create interactive plots
plotter = SpectralPlotter()
fig = plotter.plot_spectra(data[:5])
fig.show()
```

### Database Management
```bash
# Check database info
spectral-db info /path/to/database/

# Create subsets for projects
spectral-db create-subset /path/to/database/ europa_ice.pkl --category ice --temperature "77K"

# Load new spectral libraries
spectral-db load-usgs /path/to/usgs/files/ /path/to/database/
```

### Interactive Spectral Filtering
```bash
# Compatible with original filter_library.py usage
spectral-filter your_library.pkl --spectra all --reference-file reference.txt --save-mode
```

## Features

- **Unified Data Model**: `LaboratorySpectrum` class for consistent data handling
- **Efficient Database**: SQLite + HDF5 storage for fast queries of large libraries
- **Multiple Loaders**: Support for JSON, pickle, text, and CSV formats
- **Interactive Tools**: Browse, filter, and select spectra with web interface
- **Processing Functions**: Normalize, interpolate, and analyze spectral data
- **Command Line Tools**: Database management and filtering from terminal

## Core Components

### Data Structures
```python
from spectral_library import LaboratorySpectrum
import numpy as np

# Create spectrum object
spectrum = LaboratorySpectrum(
    species="Water Ice",
    wavelength=np.linspace(1.0, 5.0, 100),
    spectrum=np.random.random(100),
    spectral_units="rel reflectance",
    source="Laboratory",
    temperature="77K",
    category="ice"
)
```

### Database Operations
```python
from spectral_library import SpectralDatabase

# Open database
db = SpectralDatabase("database_path/")

# Query spectra
ice_spectra = db.query(category="ice", temperature="77K")
spectra_objects = db.get_spectra(ice_spectra[:10])

# Export subsets
df = db.export_to_dataframe(ice_spectra)
```

### Processing and Visualization
```python
from spectral_library import normalize, SpectralPlotter, SpectralFilter

# Normalize spectra
normalized = normalize(wavelength, spectrum, 2.0, 2.1)

# Create plots
plotter = SpectralPlotter()
fig = plotter.plot_spectra(spectra_objects)

# Interactive filtering
filter_tool = SpectralFilter(dataframe)
filter_tool.interactive_browser()
```

## Command Line Interface

### Database Management (`spectral-db`)
```bash
# Show database statistics
spectral-db info database/

# Create filtered subsets
spectral-db create-subset database/ output.pkl --category mineral --source RELAB

# Import new libraries
spectral-db load-usgs usgs_files/ database/
spectral-db load-psf psf_files/ database/
```

### Interactive Filtering (`spectral-filter`)
```bash
# View and filter spectra (compatible with original filter_library.py)
spectral-filter library.pkl --spectra all --save-mode
spectral-filter library.pkl --spectra "spectrum1" "spectrum2" --reference-file ref.txt
```

## Typical Workflow

### 1. Install Package
```bash
pip install spectral-library
```

### 2. Set Up Database (one-time)
```bash
# If migrating from JSON files
spectral-db migrate json_directory/ database/

# Add additional libraries
spectral-db load-usgs usgs_files/ database/
```

### 3. Create Project Subsets
```bash
# Europa ice analysis
spectral-db create-subset database/ europa_ice.pkl --category ice --temperature "77K"

# Mars mineral analysis
spectral-db create-subset database/ mars_minerals.pkl --category mineral --mineral_type silicate
```

### 4. Use in Analysis
```python
# In your analysis script/notebook
from spectral_library import load_data, SpectralPlotter

# Load project-specific data
data = load_data("europa_ice.pkl")

# Analyze and visualize
plotter = SpectralPlotter()
fig = plotter.plot_spectra(data[:10])
fig.show()
```

## Supported Data Sources

- **RELAB**: Brown University spectral library
- **USGS**: US Geological Survey spectral library
- **PSF**: Planetary Science Foundation
- **Custom**: JSON, CSV, and text file formats

## Requirements

- Python 3.8+
- numpy, pandas, plotly, dash
- h5py (for efficient database storage)
- scipy, astropy, matplotlib
- dataclasses-json, beautifulsoup4, tqdm

## Development

### Local Installation
```bash
git clone https://github.com/ryleighdavis/spectral-library.git
cd spectral-library
pip install -e .[dev]
```

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black spectral_library/
```

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{spectral_library,
  title = {SpectralLibrary: A Python package for managing ice and mineral spectra laboratory data},
  author = {Davis, Ryleigh},
  year = {2024},
  url = {https://github.com/ryleighdavis/spectral-library}
}
```