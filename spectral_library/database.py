"""
Database management for large-scale spectral libraries.

This module provides efficient storage and retrieval of spectral data,
designed to handle thousands of spectra from multiple sources.
"""

import os
import json
import sqlite3
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterator
import logging
from tqdm import tqdm

from .core import LaboratorySpectrum
from .loaders import load_data

logger = logging.getLogger(__name__)


class SpectralDatabase:
    """
    Efficient database for storing and querying large numbers of spectra.

    Uses SQLite for metadata and HDF5/parquet for spectral data arrays.
    This is much more efficient than individual JSON files.
    """

    def __init__(self, db_path: str):
        """
        Initialize spectral database.

        Parameters:
        -----------
        db_path : str
            Path to database directory
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        # Database files
        self.metadata_db = self.db_path / "metadata.db"
        self.spectra_file = self.db_path / "spectra.h5"

        # Initialize database
        self._init_metadata_db()

    def _init_metadata_db(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spectra_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                source TEXT NOT NULL,
                sample_id TEXT,
                spectral_units TEXT,
                temperature TEXT,
                atmosphere TEXT,
                category TEXT,
                mineral_type TEXT,
                mineral_subtype TEXT,
                formula TEXT,
                grain_size TEXT,
                sample_location TEXT,
                facility TEXT,
                spectrometer TEXT,
                sample_description TEXT,
                file_name TEXT,
                wavelength_min REAL,
                wavelength_max REAL,
                n_points INTEGER,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(species, source, sample_id, file_name)
            )
        ''')

        # Create indices for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON spectra_metadata(species)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON spectra_metadata(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON spectra_metadata(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mineral_type ON spectra_metadata(mineral_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_wavelength ON spectra_metadata(wavelength_min, wavelength_max)')

        conn.commit()
        conn.close()

    def add_spectrum(self, spectrum: LaboratorySpectrum) -> int:
        """
        Add a single spectrum to the database.

        Parameters:
        -----------
        spectrum : LaboratorySpectrum
            Spectrum to add

        Returns:
        --------
        int
            Database ID of added spectrum
        """
        # Add metadata
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO spectra_metadata (
                    species, source, sample_id, spectral_units, temperature,
                    atmosphere, category, mineral_type, mineral_subtype,
                    formula, grain_size, sample_location, facility,
                    spectrometer, sample_description, file_name,
                    wavelength_min, wavelength_max, n_points
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                spectrum.species, spectrum.source, spectrum.sample_id,
                spectrum.spectral_units, spectrum.temperature, spectrum.atmosphere,
                spectrum.category, spectrum.mineral_type, spectrum.mineral_subtype,
                spectrum.formula, spectrum.grain_size, spectrum.sample_location,
                spectrum.facility, spectrum.spectrometer, spectrum.sample_description,
                spectrum.file_name, float(np.min(spectrum.wavelength)),
                float(np.max(spectrum.wavelength)), len(spectrum.wavelength)
            ))

            spectrum_id = cursor.lastrowid
            conn.commit()

        except sqlite3.IntegrityError:
            # Spectrum already exists, get its ID
            cursor.execute('''
                SELECT id FROM spectra_metadata
                WHERE species=? AND source=? AND sample_id=? AND file_name=?
            ''', (spectrum.species, spectrum.source, spectrum.sample_id, spectrum.file_name))
            spectrum_id = cursor.fetchone()[0]

        finally:
            conn.close()

        # Store spectral data in HDF5
        self._store_spectral_data(spectrum_id, spectrum.wavelength, spectrum.spectrum)

        return spectrum_id

    def _store_spectral_data(self, spectrum_id: int, wavelength: np.ndarray, spectrum: np.ndarray):
        """Store wavelength and spectrum arrays in HDF5 file."""
        try:
            import h5py

            with h5py.File(self.spectra_file, 'a') as f:
                # Create group for this spectrum
                grp = f.create_group(f'spectrum_{spectrum_id}') if f'spectrum_{spectrum_id}' not in f else f[f'spectrum_{spectrum_id}']

                # Store arrays
                if 'wavelength' in grp:
                    del grp['wavelength']
                if 'spectrum' in grp:
                    del grp['spectrum']

                grp.create_dataset('wavelength', data=wavelength, compression='gzip')
                grp.create_dataset('spectrum', data=spectrum, compression='gzip')

        except ImportError:
            logger.warning("h5py not available, falling back to pickle storage")
            # Fallback to pickle if HDF5 not available
            pickle_file = self.db_path / f"spectrum_{spectrum_id}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump({'wavelength': wavelength, 'spectrum': spectrum}, f)

    def get_spectrum(self, spectrum_id: int) -> Optional[LaboratorySpectrum]:
        """
        Retrieve a spectrum by ID.

        Parameters:
        -----------
        spectrum_id : int
            Database ID of spectrum

        Returns:
        --------
        Optional[LaboratorySpectrum]
            Retrieved spectrum or None if not found
        """
        # Get metadata
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM spectra_metadata WHERE id=?', (spectrum_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Get spectral data
        wavelength, spectrum = self._load_spectral_data(spectrum_id)

        if wavelength is None or spectrum is None:
            return None

        # Create LaboratorySpectrum object
        return LaboratorySpectrum(
            species=row[1], wavelength=wavelength, spectrum=spectrum,
            spectral_units=row[4], source=row[2],
            sample_id=row[3], temperature=row[5], atmosphere=row[6],
            category=row[7], mineral_type=row[8], mineral_subtype=row[9],
            formula=row[10], grain_size=row[11], sample_location=row[12],
            facility=row[13], spectrometer=row[14], sample_description=row[15],
            file_name=row[16]
        )

    def _load_spectral_data(self, spectrum_id: int):
        """Load wavelength and spectrum arrays."""
        try:
            import h5py

            with h5py.File(self.spectra_file, 'r') as f:
                if f'spectrum_{spectrum_id}' in f:
                    grp = f[f'spectrum_{spectrum_id}']
                    wavelength = grp['wavelength'][:]
                    spectrum = grp['spectrum'][:]
                    return wavelength, spectrum

        except (ImportError, KeyError, OSError):
            # Fallback to pickle
            pickle_file = self.db_path / f"spectrum_{spectrum_id}.pkl"
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    return data['wavelength'], data['spectrum']

        return None, None

    def query(self, **criteria) -> List[int]:
        """
        Query database for spectra matching criteria.

        Parameters:
        -----------
        **criteria : dict
            Query criteria. Supports multiple query types:

            1. **Exact matches** (text/numeric fields):
               - species='ice'
               - category='mineral'
               - wavelength_max=1000

            2. **Text wildcards** (use * for wildcard):
               - species='ice*'      # matches 'ice', 'ice_crystal', etc.
               - category='*mineral' # matches 'silicate_mineral', etc.

            3. **Numeric comparisons** (use double underscore syntax):
               - wavelength_max__gt=5      # wavelength_max > 5
               - wavelength_min__lt=1000   # wavelength_min < 1000
               - n_points__gte=500         # n_points >= 500
               - wavelength_max__lte=2000  # wavelength_max <= 2000
               - n_points__ne=1024         # n_points != 1024

            4. **Wavelength range** (overlaps with spectrum range):
               - wavelength_range=(400, 800)  # spectra covering 400-800 range

            **Available comparison operators:**
            - __gt: greater than (>)
            - __lt: less than (<)
            - __gte: greater than or equal (>=)
            - __lte: less than or equal (<=)
            - __ne: not equal (!=)

            **Examples:**
            ```python
            # Find ice spectra with wavelength_max > 1000
            db.query(species='ice', wavelength_max__gt=1000)

            # Find all mineral spectra with < 500 data points
            db.query(category='*mineral', n_points__lt=500)

            # Find spectra covering visible range with high resolution
            db.query(wavelength_range=(400, 700), n_points__gte=1000)
            ```

        Returns:
        --------
        List[int]
            List of spectrum IDs matching criteria
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        # Build query
        where_clauses = []
        params = []

        for key, value in criteria.items():
            # Check for comparison operators
            if '__' in key:
                field, operator = key.rsplit('__', 1)

                # Validate field is numeric
                if field not in ['wavelength_min', 'wavelength_max', 'n_points']:
                    raise ValueError(f"Comparison operators only supported for numeric fields: {field}")

                # Map operators to SQL
                op_map = {
                    'gt': '>',
                    'lt': '<',
                    'gte': '>=',
                    'lte': '<=',
                    'ne': '!='
                }

                if operator not in op_map:
                    raise ValueError(f"Unsupported operator: {operator}. Use: {list(op_map.keys())}")

                where_clauses.append(f"{field} {op_map[operator]} ?")
                params.append(value)

            elif key in ['wavelength_min', 'wavelength_max', 'n_points']:
                # Numeric equality comparisons
                where_clauses.append(f"{key} = ?")
                params.append(value)
            elif key == 'wavelength_range':
                # Special case for wavelength range
                min_wl, max_wl = value
                where_clauses.append("wavelength_min <= ? AND wavelength_max >= ?")
                params.extend([max_wl, min_wl])
            else:
                # Text comparisons (with wildcards)
                if '*' in str(value):
                    where_clauses.append(f"{key} LIKE ?")
                    params.append(str(value).replace('*', '%'))
                else:
                    where_clauses.append(f"{key} = ?")
                    params.append(value)

        query = "SELECT id FROM spectra_metadata"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        cursor.execute(query, params)
        results = [row[0] for row in cursor.fetchall()]

        conn.close()
        return results

    def get_spectra(self, ids: List[int]) -> List[LaboratorySpectrum]:
        """Get multiple spectra by IDs."""
        return [self.get_spectrum(id_) for id_ in ids if self.get_spectrum(id_) is not None]

    def count(self) -> int:
        """Get total number of spectra in database."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM spectra_metadata')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_unique_values(self, column: str) -> List[str]:
        """Get unique values in a metadata column."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute(f'SELECT DISTINCT {column} FROM spectra_metadata WHERE {column} IS NOT NULL ORDER BY {column}')
        values = [row[0] for row in cursor.fetchall()]
        conn.close()
        return values

    def export_to_dataframe(self, ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Export spectra to pandas DataFrame format for compatibility.

        Parameters:
        -----------
        ids : Optional[List[int]]
            Specific spectrum IDs to export, or None for all

        Returns:
        --------
        pd.DataFrame
            DataFrame with spectral data
        """
        conn = sqlite3.connect(self.metadata_db)

        if ids:
            placeholders = ','.join('?' * len(ids))
            query = f"SELECT * FROM spectra_metadata WHERE id IN ({placeholders})"
            df_meta = pd.read_sql_query(query, conn, params=ids)
        else:
            df_meta = pd.read_sql_query("SELECT * FROM spectra_metadata", conn)

        conn.close()

        # Add spectral data
        wavelengths = []
        spectra = []

        for spectrum_id in tqdm(df_meta['id'], desc="Loading spectral data"):
            wl, sp = self._load_spectral_data(spectrum_id)
            wavelengths.append(wl)
            spectra.append(sp)

        df_meta['wavelength'] = wavelengths
        df_meta['spectrum'] = spectra

        return df_meta

    def migrate_from_json_directory(self, json_dir: str, batch_size: int = 1000):
        """
        Migrate from individual JSON files to database.

        Parameters:
        -----------
        json_dir : str
            Directory containing JSON files
        batch_size : int
            Number of files to process at once
        """
        json_files = list(Path(json_dir).glob("*.json"))
        total_files = len(json_files)

        print(f"Found {total_files} JSON files to migrate...")

        # Track statistics
        successful = 0
        skipped = 0
        errors = 0
        initial_count = self.count()

        for i in tqdm(range(0, total_files, batch_size), desc="Migrating files"):
            batch_files = json_files[i:i+batch_size]

            for json_file in batch_files:
                try:
                    with open(json_file, 'r') as f:
                        content = f.read().strip()

                    # Handle double-encoded JSON (JSON string inside JSON file)
                    try:
                        # First, load as JSON (might be a string)
                        data = json.loads(content)
                        # If it's a string, parse it again
                        if isinstance(data, str):
                            data = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in {json_file}")
                        errors += 1
                        continue

                    # Manually create LaboratorySpectrum from parsed data
                    spectrum = self._create_spectrum_from_dict(data)
                    if spectrum is not None:
                        self.add_spectrum(spectrum)
                        successful += 1
                    else:
                        logger.warning(f"Skipped invalid spectrum in {json_file}")
                        skipped += 1

                except Exception as e:
                    logger.warning(f"Error migrating {json_file}: {e}")
                    errors += 1
                    continue

        final_count = self.count()
        added_count = final_count - initial_count

        print(f"\nMigration complete!")
        print(f"  Total files processed: {total_files}")
        print(f"  Successfully migrated: {successful}")
        print(f"  Skipped (invalid data): {skipped}")
        print(f"  Errors: {errors}")
        print(f"  Database now contains: {final_count} spectra")
        print(f"  New spectra added: {added_count}")

    def _create_spectrum_from_dict(self, data: dict) -> Optional[LaboratorySpectrum]:
        """Create LaboratorySpectrum from dictionary, handling numpy array conversion."""
        # Validate required fields
        wavelength_data = data.get('wavelength')
        spectrum_data = data.get('spectrum')

        if wavelength_data is None or spectrum_data is None:
            logger.warning(f"Missing wavelength or spectrum data: wavelength={wavelength_data is not None}, spectrum={spectrum_data is not None}")
            return None

        if not wavelength_data or not spectrum_data:
            logger.warning(f"Empty wavelength or spectrum data: wavelength={len(wavelength_data) if wavelength_data else 0}, spectrum={len(spectrum_data) if spectrum_data else 0}")
            return None

        try:
            # Convert lists to numpy arrays
            wavelength = np.array(wavelength_data)
            spectrum = np.array(spectrum_data)

            # Check if arrays have same length
            if len(wavelength) != len(spectrum):
                logger.warning(f"Wavelength and spectrum length mismatch: {len(wavelength)} vs {len(spectrum)}")
                return None

            # Check for valid data (no all NaN or all zero)
            if np.all(np.isnan(wavelength)) or np.all(np.isnan(spectrum)):
                logger.warning("All wavelength or spectrum values are NaN")
                return None

        except Exception as e:
            logger.warning(f"Error converting arrays: {e}")
            return None

        # Helper function to safely get string values
        def safe_get(key, default=''):
            value = data.get(key, default)
            if value is None:
                return default
            return str(value).strip()

        # Create spectrum with all available fields
        try:
            return LaboratorySpectrum(
                species=safe_get('species', 'Unknown'),
                wavelength=wavelength,
                spectrum=spectrum,
                spectral_units=safe_get('spectral_units', 'reflectance'),
                source=safe_get('source', 'Unknown'),
                temperature=safe_get('temperature'),
                atmosphere=safe_get('atmosphere'),
                category=safe_get('category'),
                mineral_type=safe_get('mineral_type'),
                mineral_subtype=safe_get('mineral_subtype'),
                formula=safe_get('formula'),
                grain_size=safe_get('grain_size'),
                sample_id=safe_get('sample_id'),
                sample_location=safe_get('sample_location'),
                facility=safe_get('facility'),
                spectrometer=safe_get('spectrometer'),
                sample_description=safe_get('sample_description'),
                file_name=safe_get('file_name')
            )
        except Exception as e:
            logger.warning(f"Error creating LaboratorySpectrum: {e}")
            return None


def load_text_files_to_database(text_dir: str, db: SpectralDatabase,
                               source: str, file_pattern: str = "*.txt",
                               metadata_extractor: Optional[callable] = None):
    """
    Load text files from spectral libraries (USGS, PSF, etc.) into database.

    Parameters:
    -----------
    text_dir : str
        Directory containing text files
    db : SpectralDatabase
        Database to add spectra to
    source : str
        Source library name (e.g., 'USGS', 'PSF')
    file_pattern : str
        File pattern to match
    metadata_extractor : Optional[callable]
        Function to extract metadata from filename/content
    """
    text_files = list(Path(text_dir).glob(file_pattern))

    print(f"Loading {len(text_files)} text files from {source}...")

    for text_file in tqdm(text_files):
        try:
            # Load spectral data
            df = load_data(str(text_file))

            # Extract wavelength and spectrum
            wavelength = df.iloc[:, 0].values
            spectrum_cols = df.columns[1:]  # All columns except first (wavelength)

            for spec_col in spectrum_cols:
                spectrum = df[spec_col].values

                # Extract metadata
                if metadata_extractor:
                    metadata = metadata_extractor(text_file, spec_col)
                else:
                    metadata = _default_metadata_extractor(text_file, spec_col)

                # Create LaboratorySpectrum
                lab_spectrum = LaboratorySpectrum(
                    species=metadata.get('species', spec_col),
                    wavelength=wavelength,
                    spectrum=spectrum,
                    spectral_units=metadata.get('spectral_units', 'reflectance'),
                    source=source,
                    sample_id=metadata.get('sample_id', text_file.stem),
                    file_name=text_file.name,
                    **{k: v for k, v in metadata.items()
                      if k not in ['species', 'spectral_units', 'sample_id']}
                )

                # Add to database
                db.add_spectrum(lab_spectrum)

        except Exception as e:
            logger.warning(f"Error loading {text_file}: {e}")
            continue

    print(f"Loaded spectra from {source}. Database now contains {db.count()} spectra.")


def _default_metadata_extractor(file_path: Path, spectrum_name: str) -> Dict[str, str]:
    """Default metadata extractor for text files."""
    return {
        'species': spectrum_name,
        'category': 'unknown',
        'temperature': 'room',
        'atmosphere': 'ambient',
        'sample_id': f"{file_path.stem}_{spectrum_name}",
        'sample_description': f"Loaded from {file_path.name}"
    }