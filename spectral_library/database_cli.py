"""
Database management CLI for SpectralLibrary.

This provides command-line tools for managing spectral databases.
"""

import argparse
import sys
from pathlib import Path

from .database import SpectralDatabase, load_text_files_to_database


def migrate_json_directory(json_dir: str, db_path: str):
    """Migrate JSON directory to database."""
    print(f"Migrating JSON files from {json_dir} to database at {db_path}")

    db = SpectralDatabase(db_path)
    db.migrate_from_json_directory(json_dir)

    print(f"Migration complete! Database contains {db.count()} spectra")


def load_usgs_library(usgs_dir: str, db_path: str):
    """Load USGS spectral library text files."""
    def usgs_metadata_extractor(file_path, spectrum_name):
        """Extract metadata from USGS file names and content."""
        filename = file_path.stem.lower()

        metadata = {
            'species': spectrum_name,
            'category': 'mineral',  # Most USGS are minerals
            'temperature': 'room',
            'atmosphere': 'ambient',
            'facility': 'USGS',
            'sample_id': file_path.stem,
            'sample_description': f"USGS spectral library sample from {file_path.name}"
        }

        # Try to infer mineral type from filename
        if any(term in filename for term in ['olivine', 'pyroxene', 'feldspar']):
            metadata['mineral_type'] = 'silicate'
        elif any(term in filename for term in ['calcite', 'dolomite']):
            metadata['mineral_type'] = 'carbonate'
        elif any(term in filename for term in ['gypsum', 'alunite']):
            metadata['mineral_type'] = 'sulfate'
        elif any(term in filename for term in ['hematite', 'magnetite', 'goethite']):
            metadata['mineral_type'] = 'oxide'

        return metadata

    db = SpectralDatabase(db_path)
    load_text_files_to_database(usgs_dir, db, 'USGS', "*.txt", usgs_metadata_extractor)


def load_psf_library(psf_dir: str, db_path: str):
    """Load PSF spectral library text files."""
    def psf_metadata_extractor(file_path, spectrum_name):
        """Extract metadata from PSF files."""
        filename = file_path.stem.lower()

        metadata = {
            'species': spectrum_name,
            'temperature': '77K',  # PSF often focuses on low-temp ices
            'atmosphere': 'vacuum',
            'facility': 'PSF',
            'sample_id': file_path.stem,
            'sample_description': f"PSF spectral library sample from {file_path.name}"
        }

        # Infer category from filename
        if any(term in filename for term in ['ice', 'h2o', 'co2', 'ch4', 'n2']):
            metadata['category'] = 'ice'
        elif any(term in filename for term in ['tholin', 'organic']):
            metadata['category'] = 'organic'
        else:
            metadata['category'] = 'mineral'

        return metadata

    db = SpectralDatabase(db_path)
    load_text_files_to_database(psf_dir, db, 'PSF', "*.txt", psf_metadata_extractor)


def create_subset(db_path: str, output_file: str, **criteria):
    """Create a subset for a specific project."""
    db = SpectralDatabase(db_path)

    print(f"Querying database with criteria: {criteria}")
    ids = db.query(**criteria)

    if not ids:
        print("No spectra match the criteria")
        return

    print(f"Found {len(ids)} matching spectra")

    # Export to DataFrame format for compatibility
    df = db.export_to_dataframe(ids)

    # Save as pickle
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)

    print(f"Saved {len(df)} spectra to {output_file}")


def database_info(db_path: str):
    """Print information about the database."""
    try:
        db = SpectralDatabase(db_path)
    except Exception as e:
        print(f"Error opening database '{db_path}': {e}")
        print("Make sure the database path is correct and the database exists.")
        sys.exit(1)

    print(f"Database: {db_path}")
    print(f"Total spectra: {db.count()}")
    print()

    # Show breakdown by source
    sources = db.get_unique_values('source')
    print("Sources:")
    for source in sources:
        count = len(db.query(source=source))
        print(f"  {source}: {count}")
    print()

    # Show breakdown by category
    categories = db.get_unique_values('category')
    print("Categories:")
    for category in categories:
        count = len(db.query(category=category))
        print(f"  {category}: {count}")
    print()

    # Show some example species
    species = db.get_unique_values('species')[:10]
    print(f"Example species (first 10 of {len(db.get_unique_values('species'))}):")
    for spec in species:
        print(f"  {spec}")


def main():
    """Main database management CLI."""
    parser = argparse.ArgumentParser(
        description="SpectralLibrary database management",
        prog="spectral-db"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Migration command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate JSON files to database')
    migrate_parser.add_argument('json_dir', help='Directory containing JSON files')
    migrate_parser.add_argument('db_path', help='Path for new database')

    # USGS loader
    usgs_parser = subparsers.add_parser('load-usgs', help='Load USGS text files')
    usgs_parser.add_argument('usgs_dir', help='Directory containing USGS text files')
    usgs_parser.add_argument('db_path', help='Database path')

    # PSF loader
    psf_parser = subparsers.add_parser('load-psf', help='Load PSF text files')
    psf_parser.add_argument('psf_dir', help='Directory containing PSF text files')
    psf_parser.add_argument('db_path', help='Database path')

    # Subset creation
    subset_parser = subparsers.add_parser('create-subset', help='Create project subset')
    subset_parser.add_argument('db_path', help='Database path')
    subset_parser.add_argument('output_file', help='Output pickle file')
    subset_parser.add_argument('--category', help='Filter by category')
    subset_parser.add_argument('--source', help='Filter by source')
    subset_parser.add_argument('--species', help='Filter by species (wildcards with *)')
    subset_parser.add_argument('--mineral-type', help='Filter by mineral type')
    subset_parser.add_argument('--temperature', help='Filter by temperature')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show database information')
    info_parser.add_argument('db_path', help='Database path')

    args = parser.parse_args()

    if args.command == 'migrate':
        migrate_json_directory(args.json_dir, args.db_path)

    elif args.command == 'load-usgs':
        load_usgs_library(args.usgs_dir, args.db_path)

    elif args.command == 'load-psf':
        load_psf_library(args.psf_dir, args.db_path)

    elif args.command == 'create-subset':
        criteria = {}
        if args.category:
            criteria['category'] = args.category
        if args.source:
            criteria['source'] = args.source
        if args.species:
            criteria['species'] = args.species
        if hasattr(args, 'mineral_type') and args.mineral_type:
            criteria['mineral_type'] = args.mineral_type
        if args.temperature:
            criteria['temperature'] = args.temperature

        create_subset(args.db_path, args.output_file, **criteria)

    elif args.command == 'info':
        database_info(args.db_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()