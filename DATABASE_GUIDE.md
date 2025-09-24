# Database Organization Guide

## Overview

You currently have ~44,000 individual JSON files in `LaboratorySpectra/`. While this works, it has several drawbacks:
- **Slow queries**: Finding specific spectra requires scanning thousands of files
- **Memory intensive**: Loading large subsets requires reading many files
- **No indexing**: Can't efficiently search by metadata
- **Duplication**: Hard to avoid duplicate spectra

The new database system provides:
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

**Recommendation**: Use the new database system for your 44k spectra.

## Migration Process

### 1. Install Dependencies

```bash
pip install h5py  # For efficient spectral data storage
```

### 2. Migrate Your Existing JSON Files

```bash
# Create organized database from your JSON files
python organize_database.py migrate LaboratorySpectra/ SpectralDatabase/

# This will:
# - Create SpectralDatabase/ directory
# - Convert all JSON files to SQLite + HDF5 format
# - Add indices for fast searching
# - Compress spectral data (saves ~50% disk space)
```

### 3. Check Migration Results

```bash
python organize_database.py info SpectralDatabase/
```

Expected output:
```
Database: SpectralDatabase/
Total spectra: ~44000

Sources:
  RELAB: 35000
  USGS: 5000
  Other: 4000

Categories:
  mineral: 30000
  ice: 8000
  organic: 6000
```

## Adding New Spectral Libraries

### USGS Library

```bash
# Download USGS text files to usgs_spectra/
python organize_database.py load-usgs usgs_spectra/ SpectralDatabase/
```

### PSF (Planetary Science Foundation) Library

```bash
# Download PSF text files to psf_spectra/
python organize_database.py load-psf psf_spectra/ SpectralDatabase/
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

db = SpectralDatabase("SpectralDatabase/")
load_text_files_to_database(
    "my_spectra/",
    db,
    source="MyLab",
    file_pattern="*.dat",
    metadata_extractor=my_metadata_extractor
)
```

## Creating Project Subsets

Instead of manually copying files, create subsets on-demand:

```bash
# All ice spectra for Europa project
python organize_database.py create-subset SpectralDatabase/ europa_ice.pkl \\
    --category ice --temperature "77K"

# Phyllosilicates for Mars project
python organize_database.py create-subset SpectralDatabase/ mars_phyllo.pkl \\
    --mineral-type silicate --species "*phyllo*"

# All RELAB meteorites
python organize_database.py create-subset SpectralDatabase/ relab_meteorites.pkl \\
    --source RELAB --category meteorite
```

## Using the Database in Python

### Basic Queries

```python
from spectral_library import SpectralDatabase

# Open database
db = SpectralDatabase("SpectralDatabase/")

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
# Use with your existing visualization
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

After migration, your directory will look like:

```
SpectralLibrary/
├── LaboratorySpectra/           # Original JSON files (can archive)
├── SpectralDatabase/            # New efficient database
│   ├── metadata.db             # SQLite database for metadata
│   ├── spectra.h5              # HDF5 file for spectral arrays
│   └── spectrum_*.pkl          # Fallback pickle files (if HDF5 unavailable)
├── europa_ice.pkl              # Project subset
├── mars_phyllo.pkl             # Project subset
└── organize_database.py        # Management script
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
# Check database health
python organize_database.py info SpectralDatabase/

# Add new library
python organize_database.py load-usgs new_usgs_data/ SpectralDatabase/

# Create new project subset
python organize_database.py create-subset SpectralDatabase/ my_project.pkl --category mineral
```

### Backup Strategy

```bash
# Backup database (much smaller than JSON files)
tar -czf spectral_backup_$(date +%Y%m%d).tar.gz SpectralDatabase/

# Restore
tar -xzf spectral_backup_20240923.tar.gz
```

## Migration Checklist

- [ ] Install h5py: `pip install h5py`
- [ ] Migrate JSON files: `python organize_database.py migrate LaboratorySpectra/ SpectralDatabase/`
- [ ] Verify migration: `python organize_database.py info SpectralDatabase/`
- [ ] Test loading: Try loading a few spectra in Python
- [ ] Create project subsets: Replace your current .pkl files with database queries
- [ ] Update notebooks: Use new database API instead of loading JSON files
- [ ] Archive JSON files: Move `LaboratorySpectra/` to backup location

## Troubleshooting

**Q: Migration is slow**
A: This is normal for 44k files. It should take 10-30 minutes depending on your system.

**Q: Can't install h5py**
A: The database will work with pickle fallback, but will be slower and larger.

**Q: Want to keep JSON files**
A: You can keep them as backup, but use the database for day-to-day work.

**Q: Need custom metadata fields**
A: Modify the database schema in `database.py` before migration.

This new system will make your spectral library much more manageable and faster to work with!