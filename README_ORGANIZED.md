# SpectralLibrary - Organized Structure

## Directory Structure

```
SpectralLibrary/
- data/                           # All data files
  - database/                     # Main spectral database (SQLite + HDF5)
  - backup/                       # Backup files
    - original_json_archive.tar.gz  # Original JSON files (compressed)
  - subsets/                      # Data subsets for different use cases
    - RELAB_salts_forSam.pkl      # Europa salt spectra
    - RELAB_meteorites.pkl        # Meteorite spectra
    - matches_filtered_df.pkl     # Filtered spectral matches
    - [other .pkl files]          # Additional data subsets
- scripts/                        # Management scripts
  - organize_database.py          # Database management
  - migrate_existing_data.py      # Data migration tools
- exports/                        # General export files
- examples/                       # Usage examples
- spectral_library/               # Python package source
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
# Load pre-made data subsets
from spectral_library import load_data

# Work from any directory - just copy the files you need
data = load_data("RELAB_salts_forSam.pkl")
```

### Creating New Data Subsets
```bash
cd scripts/
python organize_database.py create-subset ../data/database/ ../data/subsets/europa_ice.pkl --category ice --temperature "77K"
```

## Database Management

### Check database status
```bash
cd scripts/
python organize_database.py info ../data/database/
```

### Add new spectral library
```bash
python organize_database.py load-usgs /path/to/usgs/files/ ../data/database/
```

### Create custom subsets
```bash
# Ice spectra for Europa analysis
python organize_database.py create-subset ../data/database/ ../data/subsets/europa_ice_77K.pkl --category ice --temperature "77K"

# Phyllosilicate minerals
python organize_database.py create-subset ../data/database/ ../data/subsets/phyllosilicates.pkl --mineral_subtype "*phyllo*"

# All RELAB meteorites
python organize_database.py create-subset ../data/database/ ../data/subsets/relab_meteorites.pkl --source "RELAB" --category meteorite
```

## Usage as Installed Package

When SpectralLibrary is installed on different computers:

```python
# Install once
pip install spectral-library

# Use anywhere
from spectral_library import SpectralDatabase, load_data

# Copy relevant .pkl files to your project directory
# Then load and analyze
data = load_data("europa_ice_77K.pkl")
```

## Available Data Subsets

Current subsets in `data/subsets/`:
- `RELAB_salts_forSam.pkl` - Salt spectra from RELAB for Europa surface analysis
- `RELAB_meteorites.pkl` - Meteorite spectra collection
- `RELAB_phyllosilicates.pkl` - Phyllosilicate mineral spectra
- `matches_filtered_df.pkl` - Pre-filtered spectral matches
- `converted_*` files - Database-migrated versions

## Backup Strategy

- **Main database**: `data/database/` (SQLite + HDF5, ~1-2 GB)
- **Original JSON backup**: `data/backup/original_json_archive.tar.gz` (~500 MB compressed)
- **Data subsets**: Individual `.pkl` files in `data/subsets/`

For full backup: `tar -czf spectral_backup_$(date +%Y%m%d).tar.gz data/`

## Notes

- The main database contains all ~40,000+ spectra from your original JSON files
- Data subsets are curated collections for specific research purposes
- Original JSON files are preserved in compressed backup
- The package can be installed and used on any computer
- Data subsets are portable - copy the .pkl files you need to any project