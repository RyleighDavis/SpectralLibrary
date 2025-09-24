"""
Basic usage examples for the SpectralLibrary package.

This example demonstrates how to:
1. Create LaboratorySpectrum objects
2. Load data from files
3. Create visualizations
4. Use filtering tools
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_library import (
    LaboratorySpectrum,
    SpectralPlotter,
    SpectralFilter,
    load_data,
    save_library_to_pickle
)


def create_synthetic_spectra():
    """Create some synthetic spectral data for demonstration."""
    print("Creating synthetic spectral data...")

    # Define wavelength range
    wavelengths = np.linspace(1.0, 5.0, 200)

    spectra = []

    # Water ice spectrum (high reflectance, absorption around 1.5, 2.0 μm)
    water_ice_refl = 0.8 + 0.1 * np.random.normal(0, 0.05, len(wavelengths))
    # Add absorption features
    water_ice_refl *= (1 - 0.3 * np.exp(-((wavelengths - 1.5) / 0.1)**2))  # 1.5 μm
    water_ice_refl *= (1 - 0.4 * np.exp(-((wavelengths - 2.0) / 0.15)**2))  # 2.0 μm
    water_ice_refl = np.clip(water_ice_refl, 0, 1)

    water_ice = LaboratorySpectrum(
        species="Water Ice",
        wavelength=wavelengths,
        spectrum=water_ice_refl,
        spectral_units="rel reflectance",
        source="Synthetic Lab",
        temperature="77K",
        atmosphere="vacuum",
        category="ice",
        sample_id="H2O_001",
        grain_size="10-50 μm"
    )
    spectra.append(water_ice)

    # CO2 ice spectrum (moderate reflectance, absorption around 2.35 μm)
    co2_ice_refl = 0.6 + 0.1 * np.random.normal(0, 0.05, len(wavelengths))
    co2_ice_refl *= (1 - 0.5 * np.exp(-((wavelengths - 2.35) / 0.08)**2))  # 2.35 μm
    co2_ice_refl = np.clip(co2_ice_refl, 0, 1)

    co2_ice = LaboratorySpectrum(
        species="CO2 Ice",
        wavelength=wavelengths,
        spectrum=co2_ice_refl,
        spectral_units="rel reflectance",
        source="Synthetic Lab",
        temperature="77K",
        atmosphere="vacuum",
        category="ice",
        sample_id="CO2_001",
        grain_size="20-100 μm"
    )
    spectra.append(co2_ice)

    # Olivine spectrum (lower reflectance, absorption around 1.0 μm)
    olivine_refl = 0.3 + 0.1 * np.random.normal(0, 0.05, len(wavelengths))
    olivine_refl *= (1 - 0.3 * np.exp(-((wavelengths - 1.0) / 0.2)**2))  # 1.0 μm
    olivine_refl = np.clip(olivine_refl, 0, 1)

    olivine = LaboratorySpectrum(
        species="Olivine",
        wavelength=wavelengths,
        spectrum=olivine_refl,
        spectral_units="rel reflectance",
        source="Synthetic Lab",
        temperature="room",
        atmosphere="ambient",
        category="mineral",
        mineral_type="silicate",
        mineral_subtype="olivine",
        sample_id="OLV_001",
        grain_size="45-90 μm"
    )
    spectra.append(olivine)

    print(f"Created {len(spectra)} synthetic spectra")
    return spectra


def demonstrate_visualization(spectra):
    """Demonstrate visualization capabilities."""
    print("\\nDemonstrating visualization...")

    # Create plotter
    plotter = SpectralPlotter(normalize_range=(2.0, 2.1))

    # Plot without normalization
    fig1 = plotter.plot_spectra(
        spectra,
        normalize_enabled=False,
        title="Raw Spectra"
    )
    fig1.write_html("raw_spectra.html")
    print("Saved raw_spectra.html")

    # Plot with normalization
    fig2 = plotter.plot_spectra(
        spectra,
        normalize_enabled=True,
        title="Normalized Spectra (2.0-2.1 μm)"
    )
    fig2.write_html("normalized_spectra.html")
    print("Saved normalized_spectra.html")


def demonstrate_filtering():
    """Demonstrate filtering capabilities using synthetic data."""
    print("\\nDemonstrating filtering...")

    # Create more synthetic data for filtering demo
    wavelengths = np.linspace(1.0, 5.0, 100)

    # Create DataFrame-style data for filtering
    import pandas as pd

    data = []
    for i in range(20):
        # Random spectrum
        spectrum = 0.3 + 0.3 * np.random.random(len(wavelengths))

        # Random metadata
        categories = ['ice', 'mineral', 'organic']
        species_options = {
            'ice': ['Water Ice', 'CO2 Ice', 'Methane Ice'],
            'mineral': ['Olivine', 'Pyroxene', 'Feldspar'],
            'organic': ['Tholin', 'Kerogen', 'Asphaltite']
        }

        category = np.random.choice(categories)
        species = np.random.choice(species_options[category])

        row = {
            'wavelength': wavelengths,
            'spectrum': spectrum,
            'species': species,
            'category': category,
            'temperature': np.random.choice(['77K', 'room', '150K']),
            'atmosphere': np.random.choice(['vacuum', 'ambient', 'dry air']),
            'sample_id': f"{category}_{i:03d}"
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Create filter tool
    filter_tool = SpectralFilter(df)

    # Demonstrate various filtering options
    print(f"Total spectra: {len(df)}")

    # Filter by category
    ice_spectra = filter_tool.filter_by_criteria(category='ice')
    print(f"Ice spectra: {len(ice_spectra)}")

    # Filter by multiple criteria
    cold_ice = filter_tool.filter_by_criteria(category='ice', temperature='77K')
    print(f"Cold ice spectra: {len(cold_ice)}")

    # Filter by wavelength coverage
    full_coverage = filter_tool.filter_by_wavelength_range(1.2, 4.8)
    print(f"Spectra with full wavelength coverage: {len(full_coverage)}")

    # Get unique values for building interfaces
    categories = filter_tool.get_unique_values('category')
    print(f"Available categories: {categories}")

    # Save a filtered selection
    if len(ice_spectra) > 0:
        filter_tool.selected_spectra = ice_spectra
        filter_tool.save_selected("example_ice_spectra.pkl")
        print("Saved ice spectra selection to example_ice_spectra.pkl")


def demonstrate_data_loading():
    """Demonstrate loading data from files."""
    print("\\nDemonstrating data loading...")

    # Save some synthetic data first
    spectra = create_synthetic_spectra()
    save_library_to_pickle(spectra, "example_library.pkl")
    print("Saved example library to example_library.pkl")

    # Load it back
    loaded_data = load_data("example_library.pkl")
    print(f"Loaded {len(loaded_data)} spectra from file")

    # Print information about loaded spectra
    for i, spectrum in enumerate(loaded_data):
        if hasattr(spectrum, 'species'):
            print(f"  {i+1}. {spectrum.species} ({spectrum.category})")


def main():
    """Run all examples."""
    print("SpectralLibrary Package Examples")
    print("=" * 40)

    try:
        # Create synthetic data
        spectra = create_synthetic_spectra()

        # Demonstrate visualization
        demonstrate_visualization(spectra)

        # Demonstrate filtering
        demonstrate_filtering()

        # Demonstrate data loading
        demonstrate_data_loading()

        print("\\n" + "=" * 40)
        print("All examples completed successfully!")
        print("\\nGenerated files:")
        print("- raw_spectra.html")
        print("- normalized_spectra.html")
        print("- example_library.pkl")
        print("- example_ice_spectra.pkl")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have installed the package and all dependencies:")
        print("  pip install -e .")
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()