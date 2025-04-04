"""
Utilities for astronomy-specific operations used across the xmatch package.
"""
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def validate_coordinates(df: pd.DataFrame, ra_col: str, dec_col: str) -> bool:
    """
    Validate that RA/Dec coordinates are within proper bounds.
    
    Args:
        df: DataFrame containing the coordinates
        ra_col: Right Ascension column name
        dec_col: Declination column name
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    if ra_col not in df.columns or dec_col not in df.columns:
        logger.error(f"Coordinate columns {ra_col}/{dec_col} not found in DataFrame")
        return False
    
    # Check for missing values
    if df[ra_col].isna().any() or df[dec_col].isna().any():
        logger.warning(f"Missing values found in coordinate columns {ra_col}/{dec_col}")
        return False
    
    # Check RA/Dec bounds
    invalid_ra = (df[ra_col] < 0) | (df[ra_col] > 360)
    invalid_dec = (df[dec_col] < -90) | (df[dec_col] > 90)
    
    if invalid_ra.any():
        count = invalid_ra.sum()
        logger.error(f"Found {count} invalid RA values outside [0, 360] range")
        return False
    
    if invalid_dec.any():
        count = invalid_dec.sum()
        logger.error(f"Found {count} invalid Dec values outside [-90, 90] range")
        return False
    
    return True

def create_skycoord(df: pd.DataFrame, 
                   ra_col: str, 
                   dec_col: str, 
                   frame: str = 'icrs',
                   unit: Tuple[u.Unit, u.Unit] = (u.deg, u.deg)) -> SkyCoord:
    """
    Create a SkyCoord object from DataFrame columns.
    
    Args:
        df: DataFrame containing the coordinates
        ra_col: Right Ascension column name
        dec_col: Declination column name
        frame: Coordinate frame (default: 'icrs')
        unit: Units for RA and Dec (default: degrees)
        
    Returns:
        SkyCoord object
    """
    if not validate_coordinates(df, ra_col, dec_col):
        raise ValueError(f"Invalid coordinates in {ra_col}/{dec_col}")
    
    return SkyCoord(ra=df[ra_col].values * unit[0], 
                    dec=df[dec_col].values * unit[1],
                    frame=frame)

def calculate_separation(coords1: SkyCoord, 
                        coords2: SkyCoord) -> np.ndarray:
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

def crossmatch_coords(coords1: SkyCoord, 
                     coords2: SkyCoord, 
                     max_sep: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
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
    idx, d2d, _ = coords1.match_to_catalog_sky(coords2)
    sep_arcsec = d2d.to(u.arcsec).value
    
    # Create mask for matches within max_sep
    mask = sep_arcsec <= max_sep
    
    return idx[mask], sep_arcsec[mask]

def find_coord_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to automatically identify RA/Dec columns in a DataFrame.
    
    Args:
        df: DataFrame to search
        
    Returns:
        Tuple of (ra_column, dec_column) or (None, None) if not found
    """
    # Common patterns for RA/Dec columns
    ra_patterns = ['ra', 'alpha', 'raj2000', 'ra_deg', 'ra_icrs', 'rightascension']
    dec_patterns = ['dec', 'delta', 'dej2000', 'dec_deg', 'dec_icrs', 'declination']
    
    # Convert all column names to lowercase for case-insensitive matching
    cols_lower = {col.lower(): col for col in df.columns}
    
    # Try exact matches first
    for ra_pattern in ra_patterns:
        if ra_pattern in cols_lower:
            ra_col = cols_lower[ra_pattern]
            # Look for matching Dec column
            for dec_pattern in dec_patterns:
                if dec_pattern in cols_lower:
                    dec_col = cols_lower[dec_pattern]
                    logger.info(f"Found coordinate columns: {ra_col}, {dec_col}")
                    return ra_col, dec_col
    
    # Try partial matches
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ra_patterns):
            ra_col = col
            # Look for matching Dec column
            for col2 in df.columns:
                col2_lower = col2.lower()
                if any(pattern in col2_lower for pattern in dec_patterns):
                    logger.info(f"Found coordinate columns: {ra_col}, {col2}")
                    return ra_col, col2
    
    logger.warning("Could not automatically identify RA/Dec columns")
    return None, None
