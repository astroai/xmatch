"""
Utilities for astronomy-specific operations used across the xmatch package.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

logger = logging.getLogger(__name__)


def validate_coordinates(df: pd.DataFrame, ra_col: str, dec_col: str):
    """
    Validate that RA/Dec coordinates are within proper bounds and not NaN.

    Args:
        df: DataFrame containing the coordinates
        ra_col: Right Ascension column name
        dec_col: Declination column name

    Raises:
        ValueError: If columns are missing, contain NaNs, or are out of bounds.
    """
    if ra_col not in df.columns:
        raise ValueError(f"RA column '{ra_col}' not found in DataFrame.")
    if dec_col not in df.columns:
        raise ValueError(f"Dec column '{dec_col}' not found in DataFrame.")

    # Check for missing values
    if df[ra_col].isnull().any():
        nan_count = df[ra_col].isnull().sum()
        raise ValueError(f"Found {nan_count} NaN value(s) in RA column '{ra_col}'.")
    if df[dec_col].isnull().any():
        nan_count = df[dec_col].isnull().sum()
        raise ValueError(f"Found {nan_count} NaN value(s) in Dec column '{dec_col}'.")

    # Check RA/Dec bounds (allowing for floating point inaccuracies)
    epsilon = 1e-9
    invalid_ra = (df[ra_col] < 0 - epsilon) | (df[ra_col] > 360 + epsilon)
    invalid_dec = (df[dec_col] < -90 - epsilon) | (df[dec_col] > 90 + epsilon)

    if invalid_ra.any():
        count = invalid_ra.sum()
        # Optional: Log first few invalid values?
        raise ValueError(f"Found {count} invalid RA values outside [0, 360] range in column '{ra_col}'.")

    if invalid_dec.any():
        count = invalid_dec.sum()
        raise ValueError(f"Found {count} invalid Dec values outside [-90, 90] range in column '{dec_col}'.")

    # If validation passes, implicitly returns None


def create_skycoord(
    df: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    frame: str = "icrs",
    unit: Tuple[u.Unit, u.Unit] = (u.deg, u.deg),
) -> SkyCoord:
    """
    Create a SkyCoord object from DataFrame columns after validation.

    Args:
        df: DataFrame containing the coordinates
        ra_col: Right Ascension column name
        dec_col: Declination column name
        frame: Coordinate frame (default: 'icrs')
        unit: Units for RA and Dec (default: degrees)

    Returns:
        SkyCoord object

    Raises:
        ValueError: If coordinate validation fails.
    """
    # Validate first; this will raise ValueError if invalid
    validate_coordinates(df, ra_col, dec_col)

    try:
        return SkyCoord(ra=df[ra_col].values * unit[0], dec=df[dec_col].values * unit[1], frame=frame)
    except Exception as e:
        # Catch potential errors during SkyCoord creation itself
        raise ValueError(f"Failed to create SkyCoord object from columns '{ra_col}', '{dec_col}': {e}") from e


def calculate_separation(coords1: SkyCoord, coords2: SkyCoord) -> np.ndarray:
    """
    Calculate separation between two sets of coordinates.

    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates (must have same length as coords1)

    Returns:
        Array of separations in arcseconds
    """
    if len(coords1) != len(coords2):
        raise ValueError(f"Coordinate sets must have same length: {len(coords1)} vs {len(coords2)}")

    sep = coords1.separation(coords2)
    return sep.to(u.arcsec).value


def crossmatch_coords(
    coords1: SkyCoord, coords2: SkyCoord, max_sep: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a coordinate crossmatch between two sets of coordinates.

    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates
        max_sep: Maximum separation in arcseconds

    Returns:
        Tuple of (idx, sep) arrays: indices into coords2 matching coords1
        and separation values in arcseconds
    """
    if len(coords1) == 0 or len(coords2) == 0:
        logger.warning("One or both coordinate sets are empty, returning empty match.")
        return np.array([], dtype=int), np.array([], dtype=float)

    idx, d2d, _ = coords1.match_to_catalog_sky(coords2)
    sep_arcsec = d2d.to(u.arcsec).value

    # Create mask for matches within max_sep
    mask = sep_arcsec <= max_sep

    return idx[mask], sep_arcsec[mask]


def _propagate_coords_df(
    df: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    pm_ra_col: Optional[str],
    pm_dec_col: Optional[str],
    epoch_col: Optional[str],
    target_epoch: float,
) -> pd.DataFrame:
    """
    Core logic to propagate coordinates in a DataFrame to a target epoch.

    Assumes pm_ra is proper motion in RA * cos(Dec) and pm_dec is proper motion in Dec.
    Proper motion units are assumed to be mas/yr.
    Epoch unit is assumed to be Julian year.
    Assumes required columns (ra, dec, pm_ra, pm_dec, epoch) are present in the DataFrame.

    Args:
        df: DataFrame containing source data.
        ra_col: Name of the RA column (degrees).
        dec_col: Name of the Dec column (degrees).
        pm_ra_col: Name of the proper motion in RA*cos(Dec) column (mas/yr).
        pm_dec_col: Name of the proper motion in Dec column (mas/yr).
        epoch_col: Name of the column containing the source epoch (Julian year).
        target_epoch: The target epoch (Julian year) to propagate to.

    Returns:
        DataFrame with new 'ra_propagated' and 'dec_propagated' columns.
        If propagation cannot be performed, these columns will contain the original values.

    Raises:
        ValueError: If base RA/Dec columns are missing or invalid, or propagation fails.
    """
    # Basic validation done here - caller should ensure columns exist
    validate_coordinates(df, ra_col, dec_col) # Raises ValueError if invalid

    # Check if all required columns for propagation exist in the DataFrame
    # The calling function should ideally ensure this, but double-check here.
    required_prop_cols_in_df = {pm_ra_col, pm_dec_col, epoch_col}
    missing_in_df = required_prop_cols_in_df - set(df.columns)
    if missing_in_df:
        logger.warning(
            f"Cannot perform epoch propagation. Columns missing in DataFrame: {missing_in_df}. "
            f"Using original coordinates from '{ra_col}', '{dec_col}'."
        )
        # Fallback: return original coordinates renamed
        df_out = df.copy()
        df_out["ra_propagated"] = df_out[ra_col]
        df_out["dec_propagated"] = df_out[dec_col]
        return df_out

    logger.info(f"Attempting epoch propagation from column '{epoch_col}' to {target_epoch}...")

    # Handle potential NaNs only in the propagation-specific columns
    # Make a copy to avoid modifying the original DataFrame during filling
    cols_for_prop = [ra_col, dec_col, pm_ra_col, pm_dec_col, epoch_col]
    # Ensure no None values sneak into the list of columns to select
    cols_for_prop = [c for c in cols_for_prop if c is not None]
    try:
        df_filled = df[cols_for_prop].copy()
    except KeyError as e:
         # This should be caught by the missing_in_df check above, but handle defensively
         logger.error(f"Internal error: Columns expected for propagation missing: {e}")
         raise ValueError(f"Columns expected for propagation missing: {e}") from e

    nan_mask_pm_ra = df_filled[pm_ra_col].isnull()
    nan_mask_pm_dec = df_filled[pm_dec_col].isnull()
    nan_mask_epoch = df_filled[epoch_col].isnull()

    # Fill NaNs with 0 for proper motion (no motion)
    df_filled.loc[nan_mask_pm_ra, pm_ra_col] = 0.0
    df_filled.loc[nan_mask_pm_dec, pm_dec_col] = 0.0
    # Fill NaN epochs with target_epoch (no propagation time difference)
    df_filled.loc[nan_mask_epoch, epoch_col] = target_epoch

    if nan_mask_pm_ra.any() or nan_mask_pm_dec.any() or nan_mask_epoch.any():
        count = (nan_mask_pm_ra | nan_mask_pm_dec | nan_mask_epoch).sum()
        logger.warning(f"Filled NaN values in {count} rows for PM/Epoch columns before propagation.")

    # Create SkyCoord object with proper motion and epoch
    try:
        coords = SkyCoord(
            ra=df_filled[ra_col].values * u.deg,
            dec=df_filled[dec_col].values * u.deg,
            pm_ra_cosdec=df_filled[pm_ra_col].values * u.mas / u.yr,
            pm_dec=df_filled[pm_dec_col].values * u.mas / u.yr,
            obstime=Time(df_filled[epoch_col].values, format="jyear", scale="tdb"), # Use TDB for consistency?
            frame="icrs",
        )

        # Define target epoch
        target_obstime = Time(target_epoch, format="jyear", scale="tdb")

        # Propagate coordinates
        coords_propagated = coords.apply_space_motion(new_obstime=target_obstime)

        # Add propagated coordinates back to the original DataFrame (make a copy first)
        df_out = df.copy()
        df_out["ra_propagated"] = coords_propagated.ra.deg
        df_out["dec_propagated"] = coords_propagated.dec.deg

        # Minimal validation on propagated coordinates (check for NaNs)
        if df_out["ra_propagated"].isnull().any() or df_out["dec_propagated"].isnull().any():
             nan_count = df_out["ra_propagated"].isnull().sum() + df_out["dec_propagated"].isnull().sum()
             logger.error(f"Epoch propagation resulted in {nan_count} NaN coordinates. Check input data/units.")
             # Fallback: return original coordinates renamed
             df_out["ra_propagated"] = df_out[ra_col]
             df_out["dec_propagated"] = df_out[dec_col]

        logger.info("Epoch propagation applied successfully.")
        return df_out

    except Exception as e:
        logger.error(f"Error during coordinate propagation: {e}", exc_info=True)
        logger.error("Falling back to original coordinates due to propagation error.")
        # Fallback: return original coordinates renamed
        df_out = df.copy()
        df_out["ra_propagated"] = df_out[ra_col]
        df_out["dec_propagated"] = df_out[dec_col]
        # Raise ValueError instead of returning fallback silently? Yes, let caller handle fallback.
        raise ValueError(f"Error during coordinate propagation: {e}") from e


def apply_epoch_propagation(
    df: pd.DataFrame,
    target_epoch: float,
    config: Dict[str, Any],
    catalogue_name: str = "catalogue", # Optional name for logging
) -> pd.DataFrame:
    """Applies epoch propagation based on catalogue configuration.

    Extracts necessary column names and the catalogue's reference epoch from
    the config dictionary. Adds the reference epoch as a column to the
    DataFrame if the specified epoch_col doesn't already exist. Then calls
    the core propagation logic (_propagate_coords_df).

    Args:
        df: The input DataFrame.
        target_epoch: The target epoch (Julian year) to propagate to.
        config: The configuration dictionary for this catalogue, expected to
                contain keys like 'ra_column', 'dec_column', 'pm_ra_column',
                'pm_dec_column', 'epoch_column', 'epoch'.
        catalogue_name: Optional name of the catalogue for logging purposes.

    Returns:
        DataFrame with 'ra_propagated' and 'dec_propagated' columns. If
        propagation cannot be performed (missing config/columns) or fails,
        it returns the original DataFrame with these columns added,
        containing the original coordinate values.

    Raises:
        ValueError: If essential configuration is missing or propagation fails.
                   (Caller should handle this and potentially raise CrossMatchError).
    """
    ra_col = config.get("ra_column")
    dec_col = config.get("dec_column")
    pm_ra_col = config.get("pm_ra_column")
    pm_dec_col = config.get("pm_dec_column")
    epoch_col = config.get("epoch_column") # The column containing the source epoch
    current_epoch = config.get("epoch") # The reference epoch from YAML

    # Check 1: Are essential config keys present?
    required_config_keys = [ra_col, dec_col, pm_ra_col, pm_dec_col, epoch_col, current_epoch]
    if not all(required_config_keys):
        missing_keys_str = []
        if not ra_col: missing_keys_str.append("'ra_column'")
        if not dec_col: missing_keys_str.append("'dec_column'")
        if not pm_ra_col: missing_keys_str.append("'pm_ra_column'")
        if not pm_dec_col: missing_keys_str.append("'pm_dec_column'")
        if not epoch_col: missing_keys_str.append("'epoch_column'")
        if not current_epoch: missing_keys_str.append("'epoch'")

        logger.warning(
            f"Catalogue '{catalogue_name}' lacks required configuration "
            f"({', '.join(missing_keys_str)}) for propagation. Using original coordinates."
        )
        # Fallback: return original coordinates renamed
        df_out = df.copy()
        if ra_col and ra_col in df_out.columns: df_out["ra_propagated"] = df_out[ra_col]
        else: df_out["ra_propagated"] = np.nan # Or should error earlier if ra_col missing? validate_coords will fail.
        if dec_col and dec_col in df_out.columns: df_out["dec_propagated"] = df_out[dec_col]
        else: df_out["dec_propagated"] = np.nan
        return df_out

    # Create a working copy to potentially add epoch column
    df_work = df.copy()

    # Check 2: Add the fixed epoch from YAML as a column if epoch_col doesn't exist in df
    # This ensures the _propagate_coords_df function finds the epoch column.
    if epoch_col not in df_work.columns:
        if current_epoch is None:
             # Should have been caught above, but defensive check
             raise ValueError(f"Config specified epoch_col '{epoch_col}' but it's missing from DataFrame and config has no 'epoch' value to add.")
        logger.debug(f"Adding fixed epoch column '{epoch_col}' = {current_epoch} from config for propagation.")
        df_work[epoch_col] = current_epoch

    # Check 3: Do necessary PM columns exist in the DataFrame?
    # _propagate_coords_df also checks this, but checking earlier avoids unnecessary work.
    required_df_cols = {pm_ra_col, pm_dec_col, epoch_col}
    missing_in_df = required_df_cols - set(df_work.columns)
    if missing_in_df:
        logger.warning(
            f"Cannot perform epoch propagation for '{catalogue_name}'. Columns missing in DataFrame: {missing_in_df}. "
            f"Using original coordinates from '{ra_col}', '{dec_col}'."
        )
        # Fallback: return original coordinates renamed
        df_out = df_work # Use df_work as it might have the added epoch_col now
        df_out["ra_propagated"] = df_out[ra_col]
        df_out["dec_propagated"] = df_out[dec_col]
        return df_out


    logger.info(
        f"Applying epoch propagation for '{catalogue_name}' to target epoch {target_epoch}"
    )
    try:
        # Call the core propagation function
        propagated_df = _propagate_coords_df(
            df=df_work, # Pass the potentially modified DataFrame
            ra_col=ra_col,
            dec_col=dec_col,
            pm_ra_col=pm_ra_col,
            pm_dec_col=pm_dec_col,
            epoch_col=epoch_col,
            target_epoch=target_epoch,
        )
        return propagated_df
    except ValueError as e:
         # Log the error here, but raise it for the caller (CrossMatch) to handle
         logger.error(f"Error applying epoch propagation for '{catalogue_name}': {e}", exc_info=True)
         raise # Re-raise the ValueError


def find_coord_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to automatically identify RA/Dec columns in a DataFrame.

    Args:
        df: DataFrame to search

    Returns:
        Tuple of (ra_column, dec_column) or (None, None) if not found
    """
    # Common patterns for RA/Dec columns
    ra_patterns = ["ra", "alpha", "raj2000", "ra_deg", "ra_icrs", "rightascension"]
    dec_patterns = ["dec", "delta", "dej2000", "dec_deg", "dec_icrs", "declination"]

    # Convert all column names to lowercase for case-insensitive matching
    cols_lower = {col.lower(): col for col in df.columns}

    ra_col = None
    dec_col = None

    # Try exact lowercase matches first
    for pattern in ra_patterns:
        if pattern in cols_lower:
            ra_col = cols_lower[pattern]
            break
    for pattern in dec_patterns:
         if pattern in cols_lower:
            dec_col = cols_lower[pattern]
            break

    # If both found via exact match, return
    if ra_col and dec_col:
        logger.info(f"Auto-identified coordinate columns (exact match): {ra_col}, {dec_col}")
        return ra_col, dec_col

    # If not found exactly, try other common variations or patterns if needed
    # (Current logic is simple exact match on common names)
    # Consider adding more sophisticated pattern matching if required.

    logger.warning("Could not automatically identify RA/Dec columns. Manual specification might be required.")
    return None, None
