import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import pickle
import os

def normalize(wavelength, spectrum, range_min, range_max):
    """
    Normalize the spectrum within a specific wavelength range.

    Parameters:
    wavelength (array-like): Array of wavelength values.
    spectrum (array-like): Array of spectrum values.
    range_min (float): Minimum wavelength of the normalization range.
    range_max (float): Maximum wavelength of the normalization range.

    Returns:
    normalized_spectrum (array-like): Normalized spectrum.
    """
    # Convert to numpy arrays if not already
    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)

    # Find indices within the specified range
    indices = np.where((wavelength >= range_min) & (wavelength <= range_max))
    
    if len(indices[0]) == 0:
        print(f"Warning: No data points found in range {range_min} to {range_max}")
        return spectrum
    
    # Calculate the mean value of the spectrum within the specified range
    mean_value = np.nanmean(spectrum[indices])
    
    if mean_value == 0 or np.isnan(mean_value):
        print("Warning: Mean value is zero or NaN. Cannot normalize.")
        return spectrum
    
    # Normalize the spectrum
    normalized_spectrum = spectrum / mean_value

    return normalized_spectrum

def load_data(file_path):
    """Load data from CSV, TXT, or PKL file"""
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

def parse_args():
    """Parse command line arguments"""
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

def create_plot(lib, spectra_names, reference_df, normalize_enabled, norm_min, norm_max, x_range):
    """Create the plotly figure with the specified spectra and reference spectra"""
    fig = go.Figure()
    
    # Check if any of the spectra names are not in the dataframe
    for name in spectra_names:
        if name != "all" and name not in lib.columns:
            print(f"Warning: '{name}' not found in library. Available columns: {lib.columns.tolist()}")
    
    # If 'all' is specified, plot all spectra
    if "all" in spectra_names:
        spectra_to_plot = [col for col in lib.columns if col != 'wavelength']
    else:
        spectra_to_plot = [name for name in spectra_names if name in lib.columns]
    
    if 'wavelength' not in lib.columns:
        print("Error: 'wavelength' column not found in library")
        sys.exit(1)
    
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
        print(f"Reference spectra to plot in create_plot: {ref_columns}")
        
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
                
                print(f"Successfully plotted reference spectrum in create_plot: {col}")
                
            except Exception as e:
                print(f"Error plotting reference spectrum {col} in create_plot: {e}")
    
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

def save_mode_with_buttons(lib, reference_df, normalize_enabled, norm_min, norm_max, x_range=[1, 5]):
    """Interactive mode to browse through spectra and save selected ones using buttons"""
    import plotly.io as pio
    from dash import Dash, html, dcc, callback, Input, Output, State
    import webbrowser
    from threading import Timer
    
    # Initialize the saved DataFrame with the same structure as lib
    savedf = pd.DataFrame(columns=lib.columns)
    saved_indices = []
    
    # Create a Dash application
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Spectral Viewer - Save Mode"),
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
            html.Button("Save to File", id='save-file-button', style={'margin': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
        ], style={'display': 'flex', 'justifyContent': 'center'}),
        html.Div([
            html.Div([
                html.Label("Scale Spectrum"),
                dcc.Input(
                    id='scale-input',
                    type='number',
                    min=0,
                    max=100.0,
                    step=0.05,
                    value=1.0,
                    style={'width': '100px'}
                ),
            ], style={'margin': '10px'}),
            html.Div([
                html.Label("Offset Spectrum"),
                dcc.Input(
                    id='offset-input',
                    type='number',
                    min=-100.0,
                    max=100.0,
                    step=0.05,
                    value=0.0,
                    style={'width': '100px'}
                ),
            ], style={'margin': '10px'}),
            html.Button("Apply", id='apply-button', style={'margin': '10px'}),
            html.Button("Reset", id='reset-button', style={'margin': '10px', 'backgroundColor': '#f0f0f0'}),
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
        fig = go.Figure()
        import numpy as np
        
        # Define x-range for plotting and y-limit calculation
        x_min, x_max = x_range[0], x_range[1]  # Use the x-range passed from command line
        all_y_values = []  # To collect all y values within x-range for y-axis limits
        
        # First add reference spectra in black if provided
        if reference_df is not None:
            wavelength_col = reference_df.columns[0]  # First column is wavelength
            ref_wavelength = reference_df[wavelength_col]
            
            # Convert wavelength to numpy array if needed
            ref_wavelength_np = np.array(ref_wavelength) if isinstance(ref_wavelength, list) else ref_wavelength
            
            # Print debug info
            print(f"Reference DataFrame columns: {reference_df.columns.tolist()}")
            
            # Add each reference spectrum in black with different line styles for visibility
            line_styles = ['solid', 'solid', 'solid', 'solid']
            colors = ['black', 'gray', 'darkgray', 'dimgray']
            
            # Skip the wavelength column
            ref_columns = [col for col in reference_df.columns if col != wavelength_col]
            print(f"Reference spectra to plot: {ref_columns}")
            
            for i, col in enumerate(ref_columns):
                # Get line style and color (cycle through options if more columns than styles)
                line_style = line_styles[i % len(line_styles)]
                color = colors[i % len(colors)]
                
                try:
                    # Get spectrum data
                    ref_spectrum = reference_df[col]
                    
                    # Normalize if needed
                    if normalize_enabled:
                        y_values = normalize(ref_wavelength_np, ref_spectrum, norm_min, norm_max)
                        label = f"{col} (normalized)"
                    else:
                        y_values = ref_spectrum
                        label = f"{col}"
                    
                    # Convert to numpy array if needed
                    y_values_np = np.array(y_values) if isinstance(y_values, list) else y_values
                    
                    # Add to plot
                    fig.add_trace(go.Scatter(
                        x=ref_wavelength_np, 
                        y=y_values_np, 
                        mode='lines', 
                        name=label,
                        line=dict(color=color, width=2, dash=line_style)
                    ))
                    
                    # Collect y values within x-range for y-limit calculation
                    in_range_indices = np.where((ref_wavelength_np >= x_min) & (ref_wavelength_np <= x_max))
                    if len(in_range_indices[0]) > 0:
                        all_y_values.extend(y_values_np[in_range_indices].tolist())
                        
                    print(f"Successfully plotted reference spectrum: {col}")
                    
                except Exception as e:
                    print(f"Error plotting reference spectrum {col}: {e}")
        
        # Add the current spectrum
        if 'wavelength' in lib.columns and 'spectrum' in lib.columns:
            try:
                x_data = lib.iloc[current_index]['wavelength']
                y_data = lib.iloc[current_index]['spectrum']
                
                # Convert to numpy arrays if they're lists
                if isinstance(x_data, list):
                    x_data = np.array(x_data)
                if isinstance(y_data, list):
                    y_data = np.array(y_data)
                
                if normalize_enabled:
                    y_data = normalize(x_data, y_data, norm_min, norm_max)
                
                # Apply scaling and offset to y_data
                scale = adjustments.get('scale', 1.0)
                offset = adjustments.get('offset', 0.0)
                y_data = y_data * scale + offset
                
                name = f"Row {lib.iloc[current_index].name} (Scale: {scale:.2f}, Offset: {offset:.2f})"

                color = 'red' if current_index in saved_data['indices'] else 'blue'
                
                fig.add_trace(go.Scatter(
                    x=x_data, 
                    y=y_data, 
                    mode='lines', 
                    name=name,
                    line=dict(color=color, width=2)
                ))
                
                # Collect y values within x-range for y-limit calculation
                in_range_indices = np.where((x_data >= x_min) & (x_data <= x_max))
                if len(in_range_indices[0]) > 0:
                    all_y_values.extend(y_data[in_range_indices].tolist())
                
                # Update layout with available metadata
                title = f"Row {lib.iloc[current_index].name}"
                if 'species' in lib.columns:
                    title += f": {lib.iloc[current_index]['species']}"
                if 'mineral_type' in lib.columns and not pd.isna(lib.iloc[current_index]['mineral_type']):
                    title += f", {lib.iloc[current_index]['mineral_type']}"
                if 'mineral_subtype' in lib.columns and not pd.isna(lib.iloc[current_index]['mineral_subtype']):
                    title += f", {lib.iloc[current_index]['mineral_subtype']}"
                
                # Calculate y-axis limits based on collected values
                if all_y_values:
                    y_min = max(0, min(all_y_values) * 0.95)  # Add 5% padding, but don't go below 0
                    y_max = max(all_y_values) * 1.05  # Add 5% padding
                    
                    fig.update_layout(
                        title=title, 
                        xaxis_title='Wavelength (µm)', 
                        yaxis_title='Normalized Reflectance' if normalize_enabled else 'Reflectance',
                        xaxis=dict(range=[x_min, x_max]),
                        yaxis=dict(range=[y_min, y_max])
                    )
                else:
                    fig.update_layout(
                        title=title, 
                        xaxis_title='Wavelength (µm)', 
                        yaxis_title='Normalized Reflectance' if normalize_enabled else 'Reflectance',
                        xaxis=dict(range=[x_min, x_max])
                    )
                
                spectrum_info = f"Displaying row {current_index} of {len(lib) - 1}"
                saved_info = f"Total saved: {saved_data['count']} spectra"
                
                return fig, spectrum_info, saved_info, save_status
            
            except Exception as e:
                print(f"Error plotting spectrum: {e}")
                fig.update_layout(
                    title="Error plotting spectrum", 
                    xaxis_title='Wavelength (µm)', 
                    yaxis_title='Reflectance',
                    annotations=[dict(
                        text=f"Error: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )]
                )
                return fig, f"Error plotting spectrum: {str(e)}", "Cannot display data", ""
        else:
            print("Error: Library must contain 'wavelength' and 'spectrum' columns for save mode")
            return fig, "Error: Missing required columns", "Error: Cannot proceed", ""
    
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
        elif button_id == 'next-button' and current_index < len(lib) - 1:
            return current_index + 1
        
        return current_index
    
    @app.callback(
        [Output('scale-input', 'value'),
         Output('offset-input', 'value'),
         Output('spectrum-adjustments', 'data')],
        [Input('current-index', 'data'),
         Input('reset-button', 'n_clicks'),
         Input('apply-button', 'n_clicks')],
        [State('scale-input', 'value'),
         State('offset-input', 'value')]
    )
    def handle_adjustments(current_index, reset_clicks, apply_clicks, scale_value, offset_value):
        from dash import ctx as dash_ctx
        if not dash_ctx.triggered_id:
            return 1.0, 0.0, {'scale': 1.0, 'offset': 0.0}
        
        trigger_id = dash_ctx.triggered_id
        
        # Reset case (navigation or reset button)
        if trigger_id in ['current-index', 'reset-button']:
            return 1.0, 0.0, {'scale': 1.0, 'offset': 0.0}
        
        # Apply button case
        elif trigger_id == 'apply-button':
            # Validate inputs
            try:
                scale = float(scale_value) if scale_value is not None else 1.0
                offset = float(offset_value) if offset_value is not None else 0.0
                
                # Clamp values to allowed ranges
                scale = max(0.1, min(100.0, scale))
                offset = max(-100.0, min(100.0, offset))
            except:
                # In case of any conversion error, use default values
                scale = 1.0
                offset = 0.0
                
            return scale_value, offset_value, {'scale': scale, 'offset': offset}
            
        # Default case
        return 1.0, 0.0, {'scale': 1.0, 'offset': 0.0}
    
    @app.callback(
        [Output('saved-data', 'data'),
         Output('save-status-store', 'data')],
        [Input('save-button', 'n_clicks'),
         Input('save-file-button', 'n_clicks')],
        [State('current-index', 'data'),
         State('saved-data', 'data'),
         State('save-status-store', 'data')]
    )
    def handle_save_actions(save_clicks, save_file_clicks, current_index, saved_data, save_status):
        from dash import ctx as dash_ctx
        if not dash_ctx.triggered_id:
            return saved_data, save_status
        
        button_id = dash_ctx.triggered_id
        
        if button_id == 'save-button':
            indices = saved_data['indices']
            if current_index not in indices:
                indices.append(current_index)
                saved_data['count'] = len(indices)
                saved_data['indices'] = indices
                # Save the row to our dataframe (this happens in memory)
                nonlocal savedf
                savedf = pd.concat([savedf, pd.DataFrame([lib.iloc[current_index]])], ignore_index=False)
                print(f"Saved row {current_index}. Total saved: {len(savedf)}")
                return saved_data, "Spectrum saved"
            return saved_data, "Spectrum already saved"
        
        elif button_id == 'save-file-button':
            if len(savedf) > 0:
                # Save to file but don't exit
                save_to_file(savedf)
                return saved_data, f"Saved {len(savedf)} spectra to file"
            else:
                return saved_data, "No spectra to save"
        
        return saved_data, save_status
    
    def save_to_file(savedf):
        if len(savedf) > 0:
            # Deduplicate if needed
            if all(col in savedf.columns for col in ['species', 'spectral_units', 'temperature', 'atmosphere', 
                                                  'category', 'mineral_type', 'mineral_subtype', 'sample_id']):
                savedf_final = savedf.drop_duplicates(subset=['species', 'spectral_units', 'temperature',
                                                           'atmosphere', 'category', 'mineral_type', 
                                                           'mineral_subtype', 'sample_id'])
            else:
                savedf_final = savedf
                
            output_file = 'saved_spectra.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(savedf_final, f)
            print(f"Saved {len(savedf_final)} spectra to {output_file}")
        else:
            print("No spectra saved yet")
    
    # Open browser automatically
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")
    
    Timer(1, open_browser).start()
    
    # Run the app
    app.run(debug=False)

def handle_specialized_lib(lib):
    """Handles the case when lib has a specific structure with wavelength and spectrum per row"""
    # Check if lib has the right structure for the original script's format
    necessary_columns = ['wavelength', 'spectrum', 'species']
    
    # Check if lib has rows with wavelength and spectrum as arrays
    if all(col in lib.columns for col in necessary_columns):
        # This is the format from the original script, where each row has a wavelength and spectrum array
        return lib
    else:
        print("Warning: Library does not have the expected structure for save mode.")
        print("Expected columns: wavelength, spectrum, species")
        print(f"Found columns: {lib.columns.tolist()}")
        return lib

def main():
    args = parse_args()
    
    # Load the spectral library
    lib = load_data(args.lib_path)
    
    # Load reference spectra if provided
    reference_df = None
    if args.reference_file:
        try:
            reference_df = load_data(args.reference_file)
            print(f"Loaded reference spectra from {args.reference_file}")
            print(f"Reference columns: {reference_df.columns.tolist()}")
            print(f"Reference file shape: {reference_df.shape}")
            print(f"First few rows of reference data:")
            print(reference_df.head())
            
            # Make sure we have valid columns (at least wavelength and one spectrum)
            if len(reference_df.columns) < 2:
                print("Warning: Reference file has less than 2 columns. Expected at least wavelength and one spectrum column.")
                
            # Explicitly list all spectral columns
            spectral_columns = reference_df.columns[1:].tolist()
            print(f"Spectral columns that will be plotted: {spectral_columns}")
            
        except Exception as e:
            print(f"Error loading reference file: {e}")
            reference_df = None
    
    # If save mode is enabled, enter interactive mode with buttons
    if args.save_mode:
        # Handle specialized library format if needed
        lib = handle_specialized_lib(lib)
        save_mode_with_buttons(lib, reference_df, not args.no_normalize, args.norm_range[0], args.norm_range[1], args.x_range)
        return
    
    # Create plot
    fig = create_plot(
        lib, 
        args.spectra, 
        reference_df,
        not args.no_normalize, 
        args.norm_range[0], 
        args.norm_range[1],
        args.x_range
    )

if __name__ == "__main__":
    main()

    # Example usage: python filter_library.py RELAB_salts_forSam.pkl --reference-file proteus.txt --norm-range 2.5 2.55 --x-range 1 5 --save-mode --spectra all