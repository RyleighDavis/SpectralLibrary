"""
Functions for loading spectral data from various sources and formats.
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
import sys
from typing import Union, List, Optional
from .core import LaboratorySpectrum


def load_library_from_json(speclibpath: str) -> pd.DataFrame:
    """
    Load spectral library from a directory of JSON files.

    Parameters:
    -----------
    speclibpath : str
        Path to directory containing JSON files

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all loaded spectra
    """
    # List to hold data from all JSON files
    data_list = []

    # Iterate over all files in the specified directory
    for filename in os.listdir(speclibpath):
        if filename.endswith('.json'):
            file_path = os.path.join(speclibpath, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                data_list.append(json.loads(data))

    # Create a DataFrame from the list of data
    df = pd.DataFrame(data_list)
    return df


def load_data(file_path: str) -> Union[pd.DataFrame, List[LaboratorySpectrum]]:
    """
    Load data from CSV, TXT, or PKL file.

    Parameters:
    -----------
    file_path : str
        Path to the file to load

    Returns:
    --------
    Union[pd.DataFrame, List[LaboratorySpectrum]]
        Loaded data in appropriate format
    """
    if '.pkl' in file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    elif '.txt' in file_path:
        try:
            # Handle text files (assumes first column is wavelength, second is spectrum)
            # Read the file manually to handle header lines that start with # properly
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Extract column names from header
            header_line = None
            for line in lines:
                if line.strip().startswith('#'):
                    header_line = line.strip().lstrip('#').strip()
                    break

            # Parse column names from header line if found
            column_names = None
            if header_line:
                # Try to extract column names, handle multiple spaces
                try:
                    column_names = [name.strip() for name in header_line.split() if name.strip()]
                    if len(column_names) >= 2:
                        # First column is wavelength, second is spectrum name
                        wavelength_name = column_names[0]
                        spectrum_name = column_names[1]
                        print(f"Found column names: {column_names}")
                    else:
                        wavelength_name = 'wavelength'
                        spectrum_name = 'spectrum'
                        print(f"Header line didn't have enough columns, using default names")
                except:
                    wavelength_name = 'wavelength'
                    spectrum_name = 'spectrum'
            else:
                wavelength_name = 'wavelength'
                spectrum_name = 'spectrum'

            # Read numeric data
            data = []
            max_cols = 0

            # First pass to determine how many columns we have
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                max_cols = max(max_cols, len(parts))

            print(f"Detected {max_cols} columns in the data file")

            # Create column names if needed
            if column_names is None or len(column_names) < max_cols:
                # First column is wavelength, rest are spectra
                if max_cols >= 2:
                    if column_names and len(column_names) >= 1:
                        wavelength_name = column_names[0]
                    else:
                        wavelength_name = 'wavelength'

                    # Generate names for additional spectrum columns
                    all_column_names = [wavelength_name]
                    for i in range(1, max_cols):
                        if column_names and i < len(column_names):
                            all_column_names.append(column_names[i])
                        else:
                            all_column_names.append(f'spectrum_{i}')
                else:
                    all_column_names = ['wavelength']
            else:
                all_column_names = column_names[:max_cols]

            print(f"Using column names: {all_column_names}")

            # Second pass to read the data with all columns
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        # Convert all available values to float
                        row_data = [float(val) for val in parts[:max_cols]]
                        # Pad with NaN if needed
                        while len(row_data) < max_cols:
                            row_data.append(float('nan'))
                        data.append(row_data)
                except Exception as e:
                    print(f"Skipping invalid line: {line}, error: {e}")

            # Create DataFrame with all column names
            df = pd.DataFrame(data, columns=all_column_names)
            print(f"Loaded text file with columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            sys.exit(1)

    elif '.csv' in file_path:
        try:
            # Try to detect if it's a CSV or other delimiter
            return pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {e}")
            sys.exit(1)
    else:
        print(f"Unsupported file type: {file_path}")
        sys.exit(1)


def save_library_to_pickle(data: Union[pd.DataFrame, List[LaboratorySpectrum]],
                          filepath: str) -> None:
    """
    Save spectral library data to pickle file.

    Parameters:
    -----------
    data : Union[pd.DataFrame, List[LaboratorySpectrum]]
        Data to save
    filepath : str
        Output file path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {filepath}")


def load_relab_data(file_path: str, category: str = 'mineral') -> List[LaboratorySpectrum]:
    """
    Load RELAB spectral data from pickle file and convert to LaboratorySpectrum objects.

    Parameters:
    -----------
    file_path : str
        Path to RELAB pickle file
    category : str
        Category of spectra ('mineral', 'ice', etc.)

    Returns:
    --------
    List[LaboratorySpectrum]
        List of LaboratorySpectrum objects
    """
    # Load the data
    df = load_data(file_path)

    spectra = []
    for _, row in df.iterrows():
        try:
            spectrum = LaboratorySpectrum(
                species=row.get('species', 'Unknown'),
                wavelength=np.array(row['wavelength']),
                spectrum=np.array(row['spectrum']),
                spectral_units=row.get('spectral_units', 'rel reflectance'),
                source='RELAB',
                temperature=row.get('temperature', ''),
                atmosphere=row.get('atmosphere', ''),
                category=category,
                mineral_type=row.get('mineral_type', ''),
                mineral_subtype=row.get('mineral_subtype', ''),
                sample_id=row.get('sample_id', ''),
                grain_size=row.get('grain_size', ''),
                facility='RELAB',
                sample_description=row.get('sample_description', '')
            )
            spectra.append(spectrum)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    return spectra