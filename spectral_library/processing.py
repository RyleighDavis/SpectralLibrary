"""
Functions for processing and manipulating spectral data.
"""
import numpy as np
from scipy import interpolate
from typing import Union, Tuple, Optional
import warnings


def normalize(wavelength: np.ndarray, spectrum: np.ndarray,
              range_min: float, range_max: float) -> np.ndarray:
    """
    Normalize the spectrum within a specific wavelength range.

    Parameters:
    -----------
    wavelength : np.ndarray
        Array of wavelength values
    spectrum : np.ndarray
        Array of spectrum values
    range_min : float
        Minimum wavelength of the normalization range
    range_max : float
        Maximum wavelength of the normalization range

    Returns:
    --------
    np.ndarray
        Normalized spectrum
    """
    # Convert to numpy arrays if not already
    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)

    # Find indices within the specified range
    indices = np.where((wavelength >= range_min) & (wavelength <= range_max))

    if len(indices[0]) == 0:
        warnings.warn(f"No data points found in range {range_min} to {range_max}")
        return spectrum

    # Calculate the mean value of the spectrum within the specified range
    mean_value = np.nanmean(spectrum[indices])

    if mean_value == 0 or np.isnan(mean_value):
        warnings.warn("Mean value is zero or NaN. Cannot normalize.")
        return spectrum

    # Normalize the spectrum
    normalized_spectrum = spectrum / mean_value

    return normalized_spectrum


def convert_to_albedo(satellite_radius: float, flux: np.ndarray, waves: np.ndarray,
                     soldist: float = 29.794, jwstdist: float = 29.895,
                     sun_spectrum: Optional[dict] = None) -> np.ndarray:
    """
    Convert flux measurements to albedo.

    Parameters:
    -----------
    satellite_radius : float
        Radius of the satellite
    flux : np.ndarray
        Flux measurements
    waves : np.ndarray
        Wavelengths
    soldist : float
        Solar distance
    jwstdist : float
        JWST distance
    sun_spectrum : Optional[dict]
        Solar spectrum dictionary with 'wv' and 'flux' keys

    Returns:
    --------
    np.ndarray
        Albedo values
    """
    if sun_spectrum is None:
        warnings.warn("Solar spectrum not provided, using placeholder calculation")
        # Placeholder - in real implementation would need actual solar spectrum
        prefactor = 1.0
        solar_flux = np.ones_like(waves)
    else:
        prefactor = 1.0  # This would need to be calculated based on your specific setup
        solar_flux = np.interp(waves, sun_spectrum['wv']/10000, sun_spectrum['flux'])

    albedo = (prefactor * flux * 1e-6 / solar_flux) / \
             (1**2 / (soldist**2) * (satellite_radius / (1.5 * 10**8))**2 / jwstdist**2)

    return albedo


def interpolate_spectrum(wavelength: np.ndarray, spectrum: np.ndarray,
                        new_wavelengths: np.ndarray,
                        kind: str = 'linear') -> np.ndarray:
    """
    Interpolate spectrum to new wavelength grid.

    Parameters:
    -----------
    wavelength : np.ndarray
        Original wavelength array
    spectrum : np.ndarray
        Original spectrum array
    new_wavelengths : np.ndarray
        New wavelength grid
    kind : str
        Interpolation method ('linear', 'cubic', etc.)

    Returns:
    --------
    np.ndarray
        Interpolated spectrum
    """
    # Create interpolation function
    interp_func = interpolate.interp1d(
        wavelength, spectrum,
        kind=kind, bounds_error=False, fill_value=np.nan
    )

    # Interpolate to new wavelengths
    return interp_func(new_wavelengths)


def resample_spectrum_to_resolution(wavelength: np.ndarray, spectrum: np.ndarray,
                                   target_resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample spectrum to a target spectral resolution.

    Parameters:
    -----------
    wavelength : np.ndarray
        Original wavelength array
    spectrum : np.ndarray
        Original spectrum array
    target_resolution : float
        Target spectral resolution (R = λ/Δλ)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        New wavelength and spectrum arrays
    """
    # Calculate wavelength spacing for target resolution
    # R = λ/Δλ, so Δλ = λ/R
    lambda_min = np.min(wavelength)
    lambda_max = np.max(wavelength)

    # Create new wavelength grid
    new_wavelengths = []
    current_lambda = lambda_min

    while current_lambda <= lambda_max:
        new_wavelengths.append(current_lambda)
        delta_lambda = current_lambda / target_resolution
        current_lambda += delta_lambda

    new_wavelengths = np.array(new_wavelengths)

    # Interpolate spectrum to new grid
    new_spectrum = interpolate_spectrum(wavelength, spectrum, new_wavelengths)

    return new_wavelengths, new_spectrum


def smooth_spectrum(spectrum: np.ndarray, window_size: int = 5,
                   method: str = 'gaussian') -> np.ndarray:
    """
    Smooth spectrum using various methods.

    Parameters:
    -----------
    spectrum : np.ndarray
        Input spectrum
    window_size : int
        Size of smoothing window
    method : str
        Smoothing method ('gaussian', 'boxcar', 'savgol')

    Returns:
    --------
    np.ndarray
        Smoothed spectrum
    """
    from scipy import ndimage
    from scipy.signal import savgol_filter

    if method == 'gaussian':
        # Gaussian smoothing
        sigma = window_size / 3.0  # Convert window size to sigma
        return ndimage.gaussian_filter1d(spectrum, sigma)

    elif method == 'boxcar':
        # Simple boxcar averaging
        return ndimage.uniform_filter1d(spectrum, size=window_size)

    elif method == 'savgol':
        # Savitzky-Golay filter
        if window_size % 2 == 0:
            window_size += 1  # Must be odd
        return savgol_filter(spectrum, window_size, 3)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def remove_continuum(wavelength: np.ndarray, spectrum: np.ndarray,
                    continuum_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Remove continuum from spectrum.

    Parameters:
    -----------
    wavelength : np.ndarray
        Wavelength array
    spectrum : np.ndarray
        Spectrum array
    continuum_points : Optional[np.ndarray]
        Wavelength points to use for continuum fit

    Returns:
    --------
    np.ndarray
        Continuum-removed spectrum
    """
    if continuum_points is None:
        # Use endpoints and some middle points
        n_points = min(10, len(wavelength) // 10)
        indices = np.linspace(0, len(wavelength)-1, n_points, dtype=int)
        continuum_points = wavelength[indices]

    # Find closest points in wavelength array
    continuum_indices = []
    for point in continuum_points:
        idx = np.argmin(np.abs(wavelength - point))
        continuum_indices.append(idx)

    continuum_indices = np.array(continuum_indices)

    # Fit continuum (linear interpolation between points)
    continuum_spectrum = np.interp(wavelength,
                                  wavelength[continuum_indices],
                                  spectrum[continuum_indices])

    # Remove continuum
    return spectrum / continuum_spectrum


def calculate_band_depth(wavelength: np.ndarray, spectrum: np.ndarray,
                        band_center: float, continuum_left: float,
                        continuum_right: float) -> float:
    """
    Calculate absorption band depth.

    Parameters:
    -----------
    wavelength : np.ndarray
        Wavelength array
    spectrum : np.ndarray
        Spectrum array
    band_center : float
        Center wavelength of absorption band
    continuum_left : float
        Left continuum point wavelength
    continuum_right : float
        Right continuum point wavelength

    Returns:
    --------
    float
        Band depth (0 to 1, where 1 is complete absorption)
    """
    # Find indices for band center and continuum points
    center_idx = np.argmin(np.abs(wavelength - band_center))
    left_idx = np.argmin(np.abs(wavelength - continuum_left))
    right_idx = np.argmin(np.abs(wavelength - continuum_right))

    # Linear continuum between left and right points
    continuum_value = np.interp(band_center,
                               [wavelength[left_idx], wavelength[right_idx]],
                               [spectrum[left_idx], spectrum[right_idx]])

    # Calculate band depth
    band_depth = 1.0 - (spectrum[center_idx] / continuum_value)

    return max(0.0, band_depth)  # Ensure non-negative