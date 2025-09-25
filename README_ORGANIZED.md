# SpectralLibrary - Project Organization Guide

## Recommended Project Structure

```
project_directory/
├── data/
│   ├── database/               # Main spectral database (SQLite + HDF5)
│   ├── subsets/                # Project-specific data subsets
│   │   ├── ice_spectra.pkl     # Ice spectral data
│   │   ├── mineral_spectra.pkl # Mineral spectral data
│   │   └── custom_subset.pkl   # Custom filtered datasets
│   └── backup/                 # Backup files
├── analysis/                   # Analysis scripts and notebooks
├── results/                    # Output files and plots
└── docs/                       # Project documentation
```

## Quick Start

### Using the Database
```python
from spectral_library import SpectralDatabase

# Open main database
db = SpectralDatabase("data/database/")

# Query for specific types
ice_spectra = db.query(category="ice")
print(f"Found {len(ice_spectra)} ice spectra")

# Get spectra objects
spectra = db.get_spectra(ice_spectra[:10])
```

### Working with Data Subsets
```python
# Create and save a subset from query results
mir_specs = db.query(wavelength_range=(5, 12))  # only spectra that have data in the 5-12 um range
lib = db.export_to_dataframe(mir_specs)

# Save the subset to data/subsets directory
lib.to_pickle("data/subsets/mir_5_12um.pkl")

# Load data subsets for analysis
from spectral_library import load_data

# Load specific datasets for your project
data = load_data("ice_spectra.pkl")
```

### Creating New Data Subsets
```bash
# Create custom subsets using the CLI
spectral-db create-subset data/database/ data/subsets/europa_ice.pkl --category ice --temperature "77K"
```

## Database Management

### Check Database Status
```bash
# Get information about your database
spectral-db info data/database/
```

### Add New Spectral Libraries
```bash
# Import data from various sources
spectral-db load-usgs /path/to/usgs/files/ data/database/
spectral-db load-psf /path/to/psf/files/ data/database/
```

### Create Custom Subsets
```bash
# Ice spectra for Europa analysis
spectral-db create-subset data/database/ data/subsets/europa_ice_77K.pkl --category ice --temperature "77K"

# Phyllosilicate minerals
spectral-db create-subset data/database/ data/subsets/phyllosilicates.pkl --mineral-type "*phyllo*"

# RELAB meteorite data
spectral-db create-subset data/database/ data/subsets/relab_meteorites.pkl --source "RELAB" --category meteorite
```

## Usage as Installed Package

Installing and using SpectralLibrary:

```python
# Install once
pip install spectral-library

# Use anywhere
from spectral_library import SpectralDatabase, load_data

# Copy data subsets to your project directory as needed
# Load and analyze in your scripts
data = load_data("data/subsets/europa_ice_77K.pkl")
```

## Common Data Subset Types

Typical subsets you might create:
- **Ice spectra**: Low-temperature ice measurements for outer solar system studies
- **Mineral spectra**: Rock and mineral samples for terrestrial planet analysis
- **Meteorite spectra**: Meteorite and cosmic dust measurements
- **Custom filtered sets**: Spectra matching specific criteria for targeted analysis

## Data Management Best Practices

- **Main database**: Central repository for all spectral data (SQLite + HDF5)
- **Project subsets**: Smaller, focused datasets for specific analyses
- **Version control**: Track analysis scripts and documentation, exclude large data files
- **Backups**: Regular backups of database and important subsets

Example backup: `tar -czf spectral_backup_$(date +%Y%m%d).tar.gz data/`

## Key Features

- **Scalable database**: Efficiently handles large spectral libraries
- **Flexible subsets**: Create custom datasets for specific research questions
- **Cross-platform**: Works on any system with Python 3.8+
- **Portable data**: Data subsets can be shared and used across different projects
- **Standard formats**: Compatible with common scientific Python tools