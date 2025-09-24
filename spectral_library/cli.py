"""
Command line interface for the spectral library package.

This provides compatibility with the original filter_library.py script.
"""
import argparse
import sys
from typing import List

from .loaders import load_data
from .visualization import create_plot
from .filters import SpectralFilter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Spectral Viewer with Normalization Options')

    parser.add_argument('lib_path', type=str, help='Path to the spectral library CSV file')

    parser.add_argument('--spectra', nargs='+', required=True,
                        help='Names of spectra to display (column names from lib file or "all" for all spectra)')

    parser.add_argument('--reference-file', type=str,
                        help='Path to a text file containing reference spectra to always display in black. '
                             'First column should be wavelengths, other columns are spectra, '
                             'with column headers for names.')

    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable normalization')

    parser.add_argument('--norm-range', nargs=2, type=float, default=[2.5, 2.55],
                        help='Min and max wavelength range for normalization (default: 2.5 2.55)')

    parser.add_argument('--x-range', nargs=2, type=float, default=[1, 5],
                        help='Min and max x-axis range (default: 1 5)')

    parser.add_argument('--save-mode', action='store_true',
                        help='Enable interactive save mode to create a filtered library')

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Load the spectral library
    try:
        lib = load_data(args.lib_path)
    except Exception as e:
        print(f"Error loading library file '{args.lib_path}': {e}")
        print("\nTip: Make sure the file path is correct and the file exists.")
        sys.exit(1)

    # Load reference spectra if provided
    reference_df = None
    if args.reference_file:
        try:
            reference_df = load_data(args.reference_file)
            print(f"Loaded reference spectra from {args.reference_file}")
            print(f"Reference columns: {reference_df.columns.tolist()}")
            print(f"Reference file shape: {reference_df.shape}")

            # Make sure we have valid columns
            if len(reference_df.columns) < 2:
                print("Warning: Reference file has less than 2 columns. Expected at least wavelength and one spectrum column.")

            # Explicitly list all spectral columns
            spectral_columns = reference_df.columns[1:].tolist()
            print(f"Spectral columns that will be plotted: {spectral_columns}")

        except Exception as e:
            print(f"Error loading reference file: {e}")
            reference_df = None

    # If save mode is enabled, enter interactive mode
    if args.save_mode:
        # Check if library has required structure for save mode
        if not ('wavelength' in lib.columns and 'spectrum' in lib.columns):
            print("Error: Save mode requires 'wavelength' and 'spectrum' columns in the library")
            print(f"Available columns: {lib.columns.tolist()}")
            sys.exit(1)

        filter_tool = SpectralFilter(lib, normalize_range=tuple(args.norm_range))
        filter_tool.interactive_browser(
            reference_spectra=reference_df,
            normalize_enabled=not args.no_normalize,
            x_range=tuple(args.x_range)
        )
        return

    # Create and show plot
    fig = create_plot(
        lib,
        args.spectra,
        reference_df,
        not args.no_normalize,
        args.norm_range[0],
        args.norm_range[1],
        args.x_range
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    main()