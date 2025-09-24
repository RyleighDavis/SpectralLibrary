# SpectralLibrary Database Guide

## Overview

SpectralLibrary provides an efficient database system for managing large collections of spectral data. Traditional approaches using individual files have several limitations:
- **Slow queries**: Finding specific spectra requires scanning many files
- **Memory intensive**: Loading large subsets requires reading many files
- **No indexing**: Can't efficiently search by metadata
- **Duplication**: Difficult to manage duplicate spectra

The SpectralLibrary database system provides:
- **Fast queries**: SQLite metadata database with indices
- **Efficient storage**: HDF5 for spectral arrays with compression
- **Easy subsetting**: Query by any metadata field
- **Bulk loading**: Efficient import from various text file formats

## Storage Options Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Individual JSON | Simple, human-readable | Slow, no indexing, large disk usage | Small collections (<1000) |
| **Database (SQLite + HDF5)** | **Fast queries, compressed, indexed** | **Requires setup** | **Large collections (>1000)** |
| Single pickle file | Fast loading | Large memory usage, no partial loading | Medium collections (1000-10000) |

**Recommendation**: Use the database system for collections larger than 1000 spectra.

## Migration Process

### 1. Install Dependencies

```bash
pip install h5py  # For efficient spectral data storage
```

### 2. Migrate Existing Data

```bash
# Create database from JSON files
spectral-db migrate json_directory/ database_directory/

# This will:
# - Create database directory structure
# - Convert files to SQLite + HDF5 format
# - Add indices for fast searching
# - Compress spectral data (saves ~50% disk space)
```

### 3. Check Database Status

```bash
spectral-db info database_directory/
```

Example output:
```
Database: database_directory/
Total spectra: 44000

Sources:
  RELAB: 35000
  USGS: 5000
  Custom: 4000

Categories:
  mineral: 30000
  ice: 8000
  organic: 6000
```

## Adding New Spectral Libraries

### USGS Library

```bash
# Import USGS spectral library files
spectral-db load-usgs usgs_directory/ database_directory/
```

### PSF (Planetary Science Foundation) Library

```bash
# Import PSF spectral library files
spectral-db load-psf psf_directory/ database_directory/
```

### Custom Text Files

For other libraries, use the Python API:

```python
from spectral_library.database import SpectralDatabase, load_text_files_to_database

def my_metadata_extractor(file_path, spectrum_name):
    """Custom metadata extraction for your library."""
    return {
        'species': spectrum_name,
        'category': 'mineral',  # or determine from filename
        'facility': 'My Lab',
        'temperature': 'room',
        'sample_id': file_path.stem
    }

db = SpectralDatabase("database_directory/")
load_text_files_to_database(
    "my_spectra/",
    db,
    source="MyLab",
    file_pattern="*.dat",
    metadata_extractor=my_metadata_extractor
)
```

## Creating Project Subsets

Create focused datasets for specific research projects:

```bash
# Ice spectra for low-temperature studies
spectral-db create-subset database/ ice_77K.pkl \
    --category ice --temperature "77K"

# Phyllosilicate minerals
spectral-db create-subset database/ phyllosilicates.pkl \
    --mineral-type silicate --species "*phyllo*"

# RELAB meteorite collection
spectral-db create-subset database/ relab_meteorites.pkl \
    --source RELAB --category meteorite
```

## Using the Database in Python

### Basic Queries

```python
from spectral_library import SpectralDatabase

# Open database
db = SpectralDatabase("database_directory/")

# Count total spectra
print(f"Total: {db.count()}")

# Find all ice spectra
ice_ids = db.query(category="ice")
print(f"Ice spectra: {len(ice_ids)}")

# Find specific wavelength range
infrared_ids = db.query(wavelength_range=(2.0, 5.0))
print(f"IR spectra (2-5 μm): {len(infrared_ids)}")

# Complex query
cold_ice_ids = db.query(
    category="ice",
    temperature="77K",
    source="RELAB"
)
```

### Loading Spectra

```python
# Get single spectrum
spectrum = db.get_spectrum(ice_ids[0])
print(f"Species: {spectrum.species}")
print(f"Wavelength range: {spectrum.get_wavelength_range()}")

# Get multiple spectra
ice_spectra = db.get_spectra(ice_ids[:10])

# Export to DataFrame for analysis
df = db.export_to_dataframe(ice_ids[:100])
```

### Integration with Existing Tools

```python
# Use with visualization tools
from spectral_library import SpectralPlotter

plotter = SpectralPlotter()
fig = plotter.plot_spectra(ice_spectra[:5])
fig.show()

# Use with filtering tools
from spectral_library import SpectralFilter

# Convert to DataFrame format for filtering
df = db.export_to_dataframe()
filter_tool = SpectralFilter(df)
filter_tool.interactive_browser()
```

## Directory Structure

A typical project structure:

```
project_directory/
├── data/
│   ├── database/               # Main spectral database
│   │   ├── metadata.db         # SQLite metadata
│   │   ├── spectra.h5          # HDF5 spectral arrays
│   │   └── spectrum_*.pkl      # Fallback files
│   └── subsets/                # Project-specific datasets
│       ├── ice_spectra.pkl     # Ice analysis subset
│       └── minerals.pkl        # Mineral analysis subset
├── analysis/                   # Analysis scripts
└── results/                    # Output files
```

## Performance Comparison

| Operation | JSON Files | Database | Speedup |
|-----------|------------|----------|---------|
| Load 1 spectrum | 0.01s | 0.001s | 10x |
| Find by species | 30s | 0.1s | 300x |
| Load 1000 spectra | 45s | 2s | 22x |
| Disk usage | 2.5 GB | 1.2 GB | 2x smaller |

## Maintenance

### Regular Tasks

```bash
# Check database status
spectral-db info database/

# Add new spectral library
spectral-db load-usgs new_data/ database/

# Create project-specific subset
spectral-db create-subset database/ project_data.pkl --category mineral
```

### Backup Strategy

```bash
# Backup database
tar -czf spectral_backup_$(date +%Y%m%d).tar.gz database/

# Restore from backup
tar -xzf spectral_backup_20241201.tar.gz
```

## Setup Checklist

- [ ] Install SpectralLibrary: `pip install spectral-library`
- [ ] Migrate existing data: `spectral-db migrate source_directory/ database/`
- [ ] Verify database: `spectral-db info database/`
- [ ] Test loading: Load some spectra in Python to verify setup
- [ ] Create project subsets: Generate focused datasets for analysis
- [ ] Update analysis scripts: Use SpectralLibrary API for data access

## Troubleshooting

**Q: Migration is slow**
A: This is normal for large collections. Processing time scales with the number of files.

**Q: Can't install h5py**
A: The database will work with pickle fallback, but will be slower and larger.

**Q: Want to keep original files**
A: You can keep them as backup, but use the database for day-to-day work.

**Q: Need custom metadata fields**
A: Modify the database schema in `database.py` before migration.

The SpectralLibrary database system provides efficient management and fast access to large spectral collections.