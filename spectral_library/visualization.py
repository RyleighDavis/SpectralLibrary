"""
Functions for visualizing and plotting spectral data.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Union, Dict, Any
import warnings

from .core import LaboratorySpectrum
from .processing import normalize


class SpectralPlotter:
    """Class for creating interactive spectral plots."""

    def __init__(self, normalize_range: Optional[tuple] = None):
        """
        Initialize the plotter.

        Parameters:
        -----------
        normalize_range : Optional[tuple]
            Wavelength range for normalization (min, max)
        """
        self.normalize_range = normalize_range or (2.5, 2.55)
        self.colors = px.colors.qualitative.Set1

    def plot_spectra(self, spectra: Union[List[LaboratorySpectrum], pd.DataFrame],
                    reference_spectra: Optional[Union[List[LaboratorySpectrum], pd.DataFrame]] = None,
                    normalize_enabled: bool = True,
                    x_range: Optional[tuple] = None,
                    title: str = "Spectral Comparison") -> go.Figure:
        """
        Create an interactive plot of multiple spectra.

        Parameters:
        -----------
        spectra : Union[List[LaboratorySpectrum], pd.DataFrame]
            Spectra to plot
        reference_spectra : Optional[Union[List[LaboratorySpectrum], pd.DataFrame]]
            Reference spectra to plot in black
        normalize_enabled : bool
            Whether to normalize spectra
        x_range : Optional[tuple]
            X-axis range (min, max)
        title : str
            Plot title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        fig = go.Figure()

        # Plot reference spectra first (in black/gray)
        if reference_spectra is not None:
            self._add_reference_spectra(fig, reference_spectra, normalize_enabled)

        # Plot main spectra
        if isinstance(spectra, list):
            self._add_laboratory_spectra(fig, spectra, normalize_enabled)
        elif isinstance(spectra, pd.DataFrame):
            self._add_dataframe_spectra(fig, spectra, normalize_enabled)

        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title='Wavelength (μm)',
            yaxis_title='Normalized Reflectance' if normalize_enabled else 'Reflectance',
            legend_title='Spectra',
            template='plotly_white',
            showlegend=True
        )

        # Set x-axis range if provided
        if x_range:
            fig.update_xaxes(range=x_range)

        return fig

    def _add_laboratory_spectra(self, fig: go.Figure, spectra: List[LaboratorySpectrum],
                               normalize_enabled: bool):
        """Add LaboratorySpectrum objects to plot."""
        for i, spectrum in enumerate(spectra):
            wavelength = spectrum.wavelength
            spec_data = spectrum.spectrum

            if normalize_enabled:
                spec_data = normalize(wavelength, spec_data,
                                    self.normalize_range[0], self.normalize_range[1])
                label = f"{spectrum.species} (normalized)"
            else:
                label = spectrum.species

            color = self.colors[i % len(self.colors)]

            fig.add_trace(go.Scatter(
                x=wavelength,
                y=spec_data,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{spectrum.species}</b><br>"
                    f"Source: {spectrum.source}<br>"
                    f"Wavelength: %{{x:.3f}} μm<br>"
                    f"Reflectance: %{{y:.4f}}<br>"
                    f"<extra></extra>"
                )
            ))

    def _add_dataframe_spectra(self, fig: go.Figure, df: pd.DataFrame,
                              normalize_enabled: bool):
        """Add DataFrame spectra to plot."""
        if 'wavelength' not in df.columns:
            raise ValueError("DataFrame must contain 'wavelength' column")

        # Get spectral columns (all except wavelength and metadata)
        metadata_cols = ['species', 'source', 'sample_id', 'temperature',
                        'atmosphere', 'category', 'mineral_type', 'mineral_subtype']
        spec_cols = [col for col in df.columns
                    if col not in ['wavelength'] + metadata_cols]

        for i, col in enumerate(spec_cols):
            wavelength = df['wavelength'].values
            spec_data = df[col].values

            if normalize_enabled:
                spec_data = normalize(wavelength, spec_data,
                                    self.normalize_range[0], self.normalize_range[1])
                label = f"{col} (normalized)"
            else:
                label = col

            color = self.colors[i % len(self.colors)]

            fig.add_trace(go.Scatter(
                x=wavelength,
                y=spec_data,
                mode='lines',
                name=label,
                line=dict(color=color, width=2)
            ))

    def _add_reference_spectra(self, fig: go.Figure,
                              reference_data: Union[List[LaboratorySpectrum], pd.DataFrame],
                              normalize_enabled: bool):
        """Add reference spectra in black/gray."""
        colors = ['black', 'gray', 'darkgray', 'dimgray']

        if isinstance(reference_data, list):
            for i, spectrum in enumerate(reference_data):
                wavelength = spectrum.wavelength
                spec_data = spectrum.spectrum

                if normalize_enabled:
                    spec_data = normalize(wavelength, spec_data,
                                        self.normalize_range[0], self.normalize_range[1])
                    label = f"{spectrum.species} (reference, normalized)"
                else:
                    label = f"{spectrum.species} (reference)"

                color = colors[i % len(colors)]

                fig.add_trace(go.Scatter(
                    x=wavelength,
                    y=spec_data,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2, dash='solid'),
                    legendgroup='references'
                ))

        elif isinstance(reference_data, pd.DataFrame):
            wavelength_col = reference_data.columns[0]  # First column is wavelength
            ref_wavelength = reference_data[wavelength_col].values

            # Plot each spectrum column
            ref_columns = [col for col in reference_data.columns if col != wavelength_col]

            for i, col in enumerate(ref_columns):
                spec_data = reference_data[col].values

                if normalize_enabled:
                    spec_data = normalize(ref_wavelength, spec_data,
                                        self.normalize_range[0], self.normalize_range[1])
                    label = f"{col} (reference, normalized)"
                else:
                    label = f"{col} (reference)"

                color = colors[i % len(colors)]

                fig.add_trace(go.Scatter(
                    x=ref_wavelength,
                    y=spec_data,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2, dash='solid'),
                    legendgroup='references'
                ))


def create_plot(lib: pd.DataFrame, spectra_names: List[str],
               reference_df: Optional[pd.DataFrame] = None,
               normalize_enabled: bool = True,
               norm_min: float = 2.5, norm_max: float = 2.55,
               x_range: tuple = (1, 5)) -> go.Figure:
    """
    Create a plotly figure with the specified spectra and reference spectra.

    This function maintains compatibility with the original filter_library.py interface.

    Parameters:
    -----------
    lib : pd.DataFrame
        Spectral library DataFrame
    spectra_names : List[str]
        Names of spectra to display
    reference_df : Optional[pd.DataFrame]
        Reference spectra DataFrame
    normalize_enabled : bool
        Whether to normalize spectra
    norm_min : float
        Minimum wavelength for normalization
    norm_max : float
        Maximum wavelength for normalization
    x_range : tuple
        X-axis range

    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    # Check if any of the spectra names are not in the dataframe
    for name in spectra_names:
        if name != "all" and name not in lib.columns:
            warnings.warn(f"'{name}' not found in library. Available columns: {lib.columns.tolist()}")

    # If 'all' is specified, plot all spectra
    if "all" in spectra_names:
        spectra_to_plot = [col for col in lib.columns if col != 'wavelength']
    else:
        spectra_to_plot = [name for name in spectra_names if name in lib.columns]

    if 'wavelength' not in lib.columns:
        raise ValueError("'wavelength' column not found in library")

    wavelength = lib['wavelength']

    # First add reference spectra in black if provided
    if reference_df is not None:
        wavelength_col = reference_df.columns[0]  # First column is wavelength
        ref_wavelength = reference_df[wavelength_col]

        # Add each reference spectrum with different line styles for visibility
        line_styles = ['solid', 'solid', 'solid', 'solid']
        colors = ['black', 'gray', 'darkgray', 'dimgray']

        # Skip the wavelength column
        ref_columns = [col for col in reference_df.columns if col != wavelength_col]

        for i, col in enumerate(ref_columns):
            # Get line style and color (cycle through options if more columns than styles)
            line_style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]

            try:
                ref_spectrum = reference_df[col]

                if normalize_enabled:
                    y_values = normalize(ref_wavelength, ref_spectrum, norm_min, norm_max)
                    label = f"{col} (reference, normalized)"
                else:
                    y_values = ref_spectrum
                    label = f"{col} (reference)"

                fig.add_trace(go.Scatter(
                    x=ref_wavelength,
                    y=y_values,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2, dash=line_style),
                    legendgroup='references'
                ))

            except Exception as e:
                warnings.warn(f"Error plotting reference spectrum {col}: {e}")

    # Then add requested spectra with colors
    for name in spectra_to_plot:
        spectrum = lib[name]

        if normalize_enabled:
            y_values = normalize(wavelength, spectrum, norm_min, norm_max)
            label = f"{name} (normalized)"
        else:
            y_values = spectrum
            label = name

        fig.add_trace(go.Scatter(x=wavelength, y=y_values, mode='lines', name=label))

    # Configure layout
    fig.update_layout(
        title='Spectral Comparison',
        xaxis_title='Wavelength',
        yaxis_title='Relative Reflectance' if normalize_enabled else 'Reflectance',
        legend_title='Spectra',
        template='plotly_white'
    )

    # Set x-axis range
    fig.update_xaxes(range=x_range)

    return fig