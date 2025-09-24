"""
Interactive filtering and selection tools for spectral data.
"""
import pandas as pd
import numpy as np
import pickle
from typing import List, Optional, Dict, Any, Callable, Union
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output, State
import webbrowser
from threading import Timer

from .core import LaboratorySpectrum
from .processing import normalize
from .visualization import create_plot


class SpectralFilter:
    """Class for interactive filtering and selection of spectral data."""

    def __init__(self, library: pd.DataFrame, normalize_range: tuple = (2.5, 2.55)):
        """
        Initialize the spectral filter.

        Parameters:
        -----------
        library : pd.DataFrame
            Spectral library DataFrame
        normalize_range : tuple
            Wavelength range for normalization
        """
        self.library = library
        self.normalize_range = normalize_range
        self.selected_spectra = pd.DataFrame(columns=library.columns)
        self.current_index = 0

    def filter_by_criteria(self, **criteria) -> pd.DataFrame:
        """
        Filter library by various criteria.

        Parameters:
        -----------
        **criteria : dict
            Filter criteria (e.g., species='ice', mineral_type='silicate')

        Returns:
        --------
        pd.DataFrame
            Filtered library
        """
        filtered_lib = self.library.copy()

        for key, value in criteria.items():
            if key in filtered_lib.columns:
                if isinstance(value, list):
                    filtered_lib = filtered_lib[filtered_lib[key].isin(value)]
                elif isinstance(value, str) and '*' in value:
                    # Simple wildcard matching
                    pattern = value.replace('*', '.*')
                    filtered_lib = filtered_lib[filtered_lib[key].str.contains(pattern, regex=True, na=False)]
                else:
                    filtered_lib = filtered_lib[filtered_lib[key] == value]

        return filtered_lib

    def filter_by_wavelength_range(self, min_wavelength: float, max_wavelength: float) -> pd.DataFrame:
        """
        Filter spectra by wavelength coverage.

        Parameters:
        -----------
        min_wavelength : float
            Minimum required wavelength
        max_wavelength : float
            Maximum required wavelength

        Returns:
        --------
        pd.DataFrame
            Filtered library
        """
        def has_coverage(row):
            if 'wavelength' in row and isinstance(row['wavelength'], (list, np.ndarray)):
                wavelengths = np.array(row['wavelength'])
                return (np.min(wavelengths) <= min_wavelength and
                       np.max(wavelengths) >= max_wavelength)
            return False

        mask = self.library.apply(has_coverage, axis=1)
        return self.library[mask]

    def filter_by_spectral_range(self, wavelength_range: tuple,
                                min_reflectance: float = None,
                                max_reflectance: float = None) -> pd.DataFrame:
        """
        Filter spectra by reflectance values in a specific wavelength range.

        Parameters:
        -----------
        wavelength_range : tuple
            Wavelength range to examine (min, max)
        min_reflectance : float, optional
            Minimum reflectance in the range
        max_reflectance : float, optional
            Maximum reflectance in the range

        Returns:
        --------
        pd.DataFrame
            Filtered library
        """
        def check_reflectance(row):
            if 'wavelength' in row and 'spectrum' in row:
                wavelengths = np.array(row['wavelength'])
                spectrum = np.array(row['spectrum'])

                # Find indices in wavelength range
                mask = ((wavelengths >= wavelength_range[0]) &
                       (wavelengths <= wavelength_range[1]))

                if np.any(mask):
                    range_reflectance = spectrum[mask]
                    mean_reflectance = np.nanmean(range_reflectance)

                    if min_reflectance is not None and mean_reflectance < min_reflectance:
                        return False
                    if max_reflectance is not None and mean_reflectance > max_reflectance:
                        return False
                    return True
            return False

        mask = self.library.apply(check_reflectance, axis=1)
        return self.library[mask]

    def get_unique_values(self, column: str) -> List[str]:
        """
        Get unique values in a column for building filter interfaces.

        Parameters:
        -----------
        column : str
            Column name

        Returns:
        --------
        List[str]
            Unique values in the column
        """
        if column in self.library.columns:
            return sorted(self.library[column].dropna().unique().tolist())
        return []

    def save_selected(self, output_path: str, deduplicate: bool = True):
        """
        Save selected spectra to file.

        Parameters:
        -----------
        output_path : str
            Output file path
        deduplicate : bool
            Whether to remove duplicates
        """
        if len(self.selected_spectra) == 0:
            print("No spectra selected to save")
            return

        data_to_save = self.selected_spectra.copy()

        if deduplicate:
            # Define columns for deduplication
            dedup_columns = ['species', 'spectral_units', 'temperature', 'atmosphere',
                           'category', 'mineral_type', 'mineral_subtype', 'sample_id']
            available_columns = [col for col in dedup_columns if col in data_to_save.columns]

            if available_columns:
                data_to_save = data_to_save.drop_duplicates(subset=available_columns)

        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"Saved {len(data_to_save)} spectra to {output_path}")

    def interactive_browser(self, reference_spectra: Optional[pd.DataFrame] = None,
                           normalize_enabled: bool = True, x_range: tuple = (1, 5),
                           port: int = 8050):
        """
        Launch interactive browser for spectral selection.

        Parameters:
        -----------
        reference_spectra : Optional[pd.DataFrame]
            Reference spectra to display
        normalize_enabled : bool
            Whether to normalize spectra
        x_range : tuple
            X-axis range for plots
        port : int
            Port for Dash server
        """
        if not ('wavelength' in self.library.columns and 'spectrum' in self.library.columns):
            print("Error: Library must contain 'wavelength' and 'spectrum' columns for interactive browser")
            return

        # Create Dash app
        app = Dash(__name__)

        app.layout = html.Div([
            html.H1("Spectral Library Browser"),
            html.Div([
                html.Div(id='spectrum-info', style={'marginBottom': '10px', 'fontWeight': 'bold'}),
                html.Div(id='saved-info', style={'marginBottom': '10px'}),
                html.Div(id='save-status', style={'marginBottom': '10px', 'color': 'green', 'fontWeight': 'bold'}),
            ]),
            dcc.Graph(id='spectrum-plot', style={'height': '70vh'}),
            html.Div([
                html.Button("Previous", id='prev-button', style={'margin': '10px'}),
                html.Button("Save Current", id='save-button', style={'margin': '10px'}),
                html.Button("Next", id='next-button', style={'margin': '10px'}),
                html.Button("Save to File", id='save-file-button',
                          style={'margin': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            html.Div([
                html.Div([
                    html.Label("Scale Spectrum"),
                    dcc.Input(id='scale-input', type='number', min=0, max=100.0,
                             step=0.05, value=1.0, style={'width': '100px'}),
                ], style={'margin': '10px'}),
                html.Div([
                    html.Label("Offset Spectrum"),
                    dcc.Input(id='offset-input', type='number', min=-100.0, max=100.0,
                             step=0.05, value=0.0, style={'width': '100px'}),
                ], style={'margin': '10px'}),
                html.Button("Apply", id='apply-button', style={'margin': '10px'}),
                html.Button("Reset", id='reset-button',
                          style={'margin': '10px', 'backgroundColor': '#f0f0f0'}),
            ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
            dcc.Store(id='current-index', data=0),
            dcc.Store(id='saved-data', data={'indices': [], 'count': 0}),
            dcc.Store(id='save-status-store', data=''),
            dcc.Store(id='spectrum-adjustments', data={'scale': 1.0, 'offset': 0.0}),
        ])

        @app.callback(
            [Output('spectrum-plot', 'figure'),
             Output('spectrum-info', 'children'),
             Output('saved-info', 'children'),
             Output('save-status', 'children')],
            [Input('current-index', 'data'),
             Input('saved-data', 'data'),
             Input('save-status-store', 'data'),
             Input('spectrum-adjustments', 'data')]
        )
        def update_plot(current_index, saved_data, save_status, adjustments):
            return self._update_plot_callback(current_index, saved_data, save_status,
                                            adjustments, reference_spectra,
                                            normalize_enabled, x_range)

        @app.callback(
            Output('current-index', 'data'),
            [Input('prev-button', 'n_clicks'),
             Input('next-button', 'n_clicks')],
            [State('current-index', 'data')]
        )
        def update_index(prev_clicks, next_clicks, current_index):
            from dash import ctx as dash_ctx
            if not dash_ctx.triggered_id:
                return current_index

            button_id = dash_ctx.triggered_id

            if button_id == 'prev-button' and current_index > 0:
                return current_index - 1
            elif button_id == 'next-button' and current_index < len(self.library) - 1:
                return current_index + 1

            return current_index

        @app.callback(
            [Output('saved-data', 'data'),
             Output('save-status-store', 'data')],
            [Input('save-button', 'n_clicks'),
             Input('save-file-button', 'n_clicks')],
            [State('current-index', 'data'),
             State('saved-data', 'data')]
        )
        def handle_save_actions(save_clicks, save_file_clicks, current_index, saved_data):
            from dash import ctx as dash_ctx
            if not dash_ctx.triggered_id:
                return saved_data, ""

            button_id = dash_ctx.triggered_id

            if button_id == 'save-button':
                indices = saved_data['indices']
                if current_index not in indices:
                    indices.append(current_index)
                    saved_data['count'] = len(indices)
                    saved_data['indices'] = indices
                    # Save the row to selected_spectra
                    self.selected_spectra = pd.concat([
                        self.selected_spectra,
                        pd.DataFrame([self.library.iloc[current_index]])
                    ], ignore_index=False)
                    return saved_data, "Spectrum saved"
                return saved_data, "Spectrum already saved"

            elif button_id == 'save-file-button':
                if len(self.selected_spectra) > 0:
                    self.save_selected('saved_spectra.pkl')
                    return saved_data, f"Saved {len(self.selected_spectra)} spectra to file"
                else:
                    return saved_data, "No spectra to save"

            return saved_data, ""

        # Open browser automatically
        def open_browser():
            webbrowser.open_new(f"http://127.0.0.1:{port}/")

        Timer(1, open_browser).start()

        # Run the app
        app.run(debug=False, port=port)

    def _update_plot_callback(self, current_index, saved_data, save_status, adjustments,
                             reference_spectra, normalize_enabled, x_range):
        """Update plot callback for Dash app."""
        fig = go.Figure()

        # Define x-range for plotting and y-limit calculation
        x_min, x_max = x_range[0], x_range[1]
        all_y_values = []

        # Add reference spectra if provided
        if reference_spectra is not None:
            wavelength_col = reference_spectra.columns[0]
            ref_wavelength = reference_spectra[wavelength_col]
            ref_wavelength_np = np.array(ref_wavelength)

            colors = ['black', 'gray', 'darkgray', 'dimgray']
            ref_columns = [col for col in reference_spectra.columns if col != wavelength_col]

            for i, col in enumerate(ref_columns):
                color = colors[i % len(colors)]
                try:
                    ref_spectrum = reference_spectra[col]
                    if normalize_enabled:
                        y_values = normalize(ref_wavelength_np, ref_spectrum,
                                           self.normalize_range[0], self.normalize_range[1])
                        label = f"{col} (normalized)"
                    else:
                        y_values = ref_spectrum
                        label = f"{col}"

                    y_values_np = np.array(y_values)
                    fig.add_trace(go.Scatter(
                        x=ref_wavelength_np,
                        y=y_values_np,
                        mode='lines',
                        name=label,
                        line=dict(color=color, width=2, dash='solid')
                    ))

                    # Collect y values for range calculation
                    in_range_indices = np.where((ref_wavelength_np >= x_min) &
                                               (ref_wavelength_np <= x_max))
                    if len(in_range_indices[0]) > 0:
                        all_y_values.extend(y_values_np[in_range_indices].tolist())

                except Exception as e:
                    print(f"Error plotting reference spectrum {col}: {e}")

        # Add current spectrum
        try:
            x_data = self.library.iloc[current_index]['wavelength']
            y_data = self.library.iloc[current_index]['spectrum']

            if isinstance(x_data, list):
                x_data = np.array(x_data)
            if isinstance(y_data, list):
                y_data = np.array(y_data)

            if normalize_enabled:
                y_data = normalize(x_data, y_data, self.normalize_range[0], self.normalize_range[1])

            # Apply scaling and offset
            scale = adjustments.get('scale', 1.0)
            offset = adjustments.get('offset', 0.0)
            y_data = y_data * scale + offset

            name = f"Row {self.library.iloc[current_index].name} (Scale: {scale:.2f}, Offset: {offset:.2f})"
            color = 'red' if current_index in saved_data['indices'] else 'blue'

            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))

            # Collect y values for range calculation
            in_range_indices = np.where((x_data >= x_min) & (x_data <= x_max))
            if len(in_range_indices[0]) > 0:
                all_y_values.extend(y_data[in_range_indices].tolist())

            # Create title with metadata
            title = f"Row {self.library.iloc[current_index].name}"
            if 'species' in self.library.columns:
                title += f": {self.library.iloc[current_index]['species']}"

            # Calculate y-axis limits
            if all_y_values:
                y_min = max(0, min(all_y_values) * 0.95)
                y_max = max(all_y_values) * 1.05
                fig.update_layout(yaxis=dict(range=[y_min, y_max]))

            fig.update_layout(
                title=title,
                xaxis_title='Wavelength (μm)',
                yaxis_title='Normalized Reflectance' if normalize_enabled else 'Reflectance',
                xaxis=dict(range=[x_min, x_max])
            )

            spectrum_info = f"Displaying row {current_index} of {len(self.library) - 1}"
            saved_info = f"Total saved: {saved_data['count']} spectra"

            return fig, spectrum_info, saved_info, save_status

        except Exception as e:
            print(f"Error plotting spectrum: {e}")
            return go.Figure(), f"Error: {str(e)}", "Cannot display data", ""