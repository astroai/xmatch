"""
Utilities for spatial chunking of astronomical queries.

This module provides functions for dividing large spatial regions into smaller chunks,
which can be processed in parallel or sequentially for more efficient cross-matching.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _generate_grid_chunks(
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    box_size_deg: float,
    max_chunks: int = 64,
) -> List[Dict[str, Any]]:
    """Generate a grid of spatial chunks covering the search area.

    Args:
        center_ra: RA of the search area center (degrees)
        center_dec: Dec of the search area center (degrees)
        radius_deg: Radius of the search area (degrees)
        box_size_deg: Size of each chunk (degrees)
        max_chunks: Maximum number of chunks to create

    Returns:
        List of dictionaries with chunk definitions
    """
    # Validate inputs
    if radius_deg <= 0 or box_size_deg <= 0:
        raise ValueError("radius_deg and box_size_deg must be positive")

    # Ensure box_size is reasonable
    if box_size_deg > 2 * radius_deg:
        logger.warning(
            f"Box size ({box_size_deg}°) is larger than search diameter, using a single chunk"
        )
        return [
            {
                "id": "single_chunk",
                "center_ra": center_ra,
                "center_dec": center_dec,
                "width_deg": 2 * radius_deg,
                "height_deg": 2 * radius_deg,
                "ra_min": (center_ra - radius_deg) % 360,
                "ra_max": (center_ra + radius_deg) % 360,
                "dec_min": max(center_dec - radius_deg, -90),
                "dec_max": min(center_dec + radius_deg, 90),
            }
        ]

    # Calculate the number of chunks needed in each dimension
    num_chunks_1d = math.ceil(2 * radius_deg / box_size_deg)

    # Check if the number of chunks exceeds max_chunks
    total_chunks = num_chunks_1d * num_chunks_1d
    if total_chunks > max_chunks:
        # Adjust box size to limit the number of chunks
        adjusted_chunks_1d = math.ceil(math.sqrt(max_chunks))
        box_size_deg = 2 * radius_deg / adjusted_chunks_1d
        num_chunks_1d = adjusted_chunks_1d
        logger.info(f"Adjusted chunk size to {box_size_deg:.4f}° to limit to {max_chunks} chunks")

    # Create a grid of chunks
    chunks = []

    # Calculate the starting coordinates
    dec_start = center_dec - radius_deg
    dec_end = center_dec + radius_deg
    ra_start = center_ra - radius_deg
    ra_end = center_ra + radius_deg

    # Create chunks
    dec_vals = np.linspace(dec_start, dec_end, num_chunks_1d + 1)
    ra_vals = np.linspace(ra_start, ra_end, num_chunks_1d + 1)

    for i in range(num_chunks_1d):
        for j in range(num_chunks_1d):
            chunk_dec_min = dec_vals[i]
            chunk_dec_max = dec_vals[i + 1]
            chunk_ra_min = ra_vals[j] % 360
            chunk_ra_max = ra_vals[j + 1] % 360

            # Calculate the center of this chunk
            chunk_ra_center = ((chunk_ra_min + chunk_ra_max) / 2) % 360
            chunk_dec_center = (chunk_dec_min + chunk_dec_max) / 2

            # Calculate the width and height
            ra_width = min((chunk_ra_max - chunk_ra_min) % 360, (chunk_ra_min - chunk_ra_max) % 360)
            dec_height = chunk_dec_max - chunk_dec_min

            # Check if this chunk is within the search radius
            if _is_box_within_circle(
                chunk_ra_center,
                chunk_dec_center,
                ra_width / 2,
                dec_height / 2,
                center_ra,
                center_dec,
                radius_deg,
            ):
                # Create the chunk
                chunk = {
                    "id": f"chunk_{i}_{j}",
                    "center_ra": chunk_ra_center,
                    "center_dec": chunk_dec_center,
                    "width_deg": ra_width,
                    "height_deg": dec_height,
                    "ra_min": chunk_ra_min,
                    "ra_max": chunk_ra_max,
                    "dec_min": chunk_dec_min,
                    "dec_max": chunk_dec_max,
                }
                chunks.append(chunk)

    logger.info(f"Generated {len(chunks)} spatial chunks for matching")
    return chunks


def _is_box_within_circle(
    box_ra: float,
    box_dec: float,
    box_ra_half_width: float,
    box_dec_half_height: float,
    circle_ra: float,
    circle_dec: float,
    circle_radius: float,
) -> bool:
    """Check if a box overlaps with a circle.

    Args:
        box_ra/box_dec: Center of the box (degrees)
        box_ra_half_width/box_dec_half_height: Half-width/height of the box (degrees)
        circle_ra/circle_dec: Center of the circle (degrees)
        circle_radius: Radius of the circle (degrees)

    Returns:
        True if the box and circle overlap
    """
    # Calculate the corners of the box
    corners = [
        (box_ra - box_ra_half_width, box_dec - box_dec_half_height),
        (box_ra - box_ra_half_width, box_dec + box_dec_half_height),
        (box_ra + box_ra_half_width, box_dec - box_dec_half_height),
        (box_ra + box_ra_half_width, box_dec + box_dec_half_height),
    ]

    # Check if any corner is within the circle
    for corner_ra, corner_dec in corners:
        dist = _angular_distance(corner_ra, corner_dec, circle_ra, circle_dec)
        if dist <= circle_radius:
            return True

    # Check if the circle center is within the box
    if (
        box_ra - box_ra_half_width <= circle_ra <= box_ra + box_ra_half_width
        and box_dec - box_dec_half_height <= circle_dec <= box_dec + box_dec_half_height
    ):
        return True

    # Check if the circle intersects any edge of the box
    # This is a simplified check
    min_dist_to_edge = min(
        abs(circle_dec - (box_dec - box_dec_half_height)),
        abs(circle_dec - (box_dec + box_dec_half_height)),
        abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width)),
    )

    return min_dist_to_edge <= circle_radius


def _angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Calculate angular distance between two sky positions.

    Args:
        ra1, dec1: First sky position (degrees)
        ra2, dec2: Second sky position (degrees)

    Returns:
        Angular distance (degrees)
    """
    # Convert to radians
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)

    # Calculate distance using haversine formula
    dlon = ra2_rad - ra1_rad
    dlat = dec2_rad - dec1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Convert back to degrees
    return math.degrees(c)


def _angular_distance_1d(angle_diff: float, half_width: float) -> float:
    """Calculate minimum angular distance between an angle and an interval.

    Args:
        angle_diff: Difference between angles (degrees)
        half_width: Half-width of the interval (degrees)

    Returns:
        Minimum angular distance (degrees)
    """
    return max(0, abs(angle_diff) - half_width)


def estimate_optimal_chunk_size(
    search_radius_deg: float,
    estimated_source_density: Optional[float] = None,
    target_sources_per_chunk: int = 100000,
) -> float:
    """Estimate optimal chunk size based on search radius and source density.

    Args:
        search_radius_deg: Radius of the search area (degrees)
        estimated_source_density: Estimated source density (sources per square degree)
        target_sources_per_chunk: Target number of sources per chunk

    Returns:
        Estimated optimal chunk size (degrees)
    """
    # If no source density provided, use a default value
    if estimated_source_density is None:
        # Default to a moderate source density (e.g., typical for Gaia)
        estimated_source_density = 10000.0  # sources per sq. deg
        logger.info(
            f"Using default estimated source density: {estimated_source_density} sources/deg²"
        )

    # Calculate the area of the search region
    search_area = math.pi * search_radius_deg**2

    # Estimate the total number of sources
    total_sources = search_area * estimated_source_density

    # Calculate the number of chunks needed
    num_chunks = max(1, total_sources / target_sources_per_chunk)

    # Calculate the area per chunk
    area_per_chunk = search_area / num_chunks

    # Calculate the linear size of each chunk (assuming square chunks)
    chunk_size = math.sqrt(area_per_chunk)

    # Ensure the chunk size is not too large or too small
    min_size = 0.05  # 0.05 degrees = 3 arcmin
    max_size = search_radius_deg

    chunk_size = max(min_size, min(max_size, chunk_size))

    logger.info(f"Estimated {total_sources:.0f} total sources, {num_chunks:.1f} chunks")
    logger.info(f"Optimal chunk size: {chunk_size:.4f}°")

    return chunk_size


def _generate_healpix_chunks(
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    pixel_size_deg: float,
    max_chunks: int = 64,
) -> List[Dict[str, Any]]:
    """Generate HEALPix chunks covering the search area.

    This function divides a circular search area into HEALPix cells for
    efficient spatial chunking of large queries.

    Args:
        center_ra: RA of the search area center (degrees)
        center_dec: Dec of the search area center (degrees)
        radius_deg: Radius of the search area (degrees)
        pixel_size_deg: Approximate size of each pixel (degrees)
        max_chunks: Maximum number of chunks to create

    Returns:
        List of dictionaries with HEALPix chunk definitions
    """
    try:
        import healpy as hp
        from astropy import units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        logger.error("healpy and astropy are required for HEALPix chunking")
        return []

    # Calculate appropriate HEALPix resolution (nside)
    # The pixel size corresponds approximately to the square root of the pixel area
    # For HEALPix, pixel area ≈ 4π/(12*nside²) steradians
    # Convert desired pixel size from degrees to radians
    pixel_size_rad = np.radians(pixel_size_deg)

    # Calculate nside that gives pixels of approximately the desired size
    # nside must be a power of 2
    pixel_area_rad = pixel_size_rad**2
    nside_estimate = np.sqrt((4 * np.pi) / (12 * pixel_area_rad))

    # Find nearest power of 2
    nside = 2 ** int(np.round(np.log2(nside_estimate)))

    # Limit nside to a reasonable range: 2^0 (1) to 2^10 (1024)
    nside = max(1, min(1024, nside))

    # Create a HEALPix instance
    logger.info(f"Using HEALPix with nside={nside}")

    # Convert center and radius to a cone
    SkyCoord(ra=center_ra * u.degree, dec=center_dec * u.degree, frame="icrs")

    # Find HEALPix pixels within the search radius
    ipix_disc = hp.query_disc(
        nside=nside,
        vec=hp.ang2vec(
            np.radians(90 - center_dec),  # theta: 0=north pole, π/2=equator
            np.radians(center_ra),  # phi: azimuthal angle, 0-2π
        ),
        radius=np.radians(radius_deg),
        inclusive=True,
        nest=True,  # Use nested pixel ordering
    )

    # Check if we have too many pixels
    if len(ipix_disc) > max_chunks:
        logger.warning(
            f"HEALPix chunking produced {len(ipix_disc)} pixels, "
            f"which exceeds max_chunks={max_chunks}. "
            f"Reducing to a coarser resolution."
        )

        # Reduce nside until we have fewer pixels than max_chunks
        while len(ipix_disc) > max_chunks and nside > 1:
            nside = nside // 2
            ipix_disc = hp.query_disc(
                nside=nside,
                vec=hp.ang2vec(np.radians(90 - center_dec), np.radians(center_ra)),
                radius=np.radians(radius_deg),
                inclusive=True,
                nest=True,
            )

        logger.info(f"Adjusted to nside={nside}, resulting in {len(ipix_disc)} pixels")

    # Create the chunk definitions
    chunks = []
    for i, ipix in enumerate(ipix_disc):
        # Get the boundaries of this HEALPix pixel
        vertices = hp.boundaries(nside, ipix, step=1, nest=True)

        # Convert from HEALPix convention (theta, phi) to celestial (ra, dec)
        # hp.vec2ang returns (theta, phi)
        thetas, phis = hp.vec2ang(vertices)

        # Convert to RA/Dec (degrees)
        ra_vertices = np.degrees(phis)
        dec_vertices = 90 - np.degrees(thetas)  # Convert colatitude to latitude

        # Calculate approximate center of the pixel
        pixel_center = hp.pix2ang(nside, ipix, nest=True)
        pixel_theta, pixel_phi = pixel_center
        pixel_ra = np.degrees(pixel_phi)
        pixel_dec = 90 - np.degrees(pixel_theta)  # Convert colatitude to latitude

        # Find the min/max RA/Dec for the pixel
        ra_min, ra_max = np.min(ra_vertices), np.max(ra_vertices)
        dec_min, dec_max = np.min(dec_vertices), np.max(dec_vertices)

        # Handle RA wrap-around
        if (ra_max - ra_min) > 180:
            # Likely crossing the 0/360 boundary
            ra_wrapped = ra_vertices % 360
            ra_min, ra_max = np.min(ra_wrapped), np.max(ra_wrapped)

        # Create the chunk definition
        chunk = {
            "id": f"healpix_{nside}_{ipix}",
            "type": "healpix",
            "ipix": int(ipix),
            "nside": int(nside),
            "center_ra": float(pixel_ra),
            "center_dec": float(pixel_dec),
            "ra_min": float(ra_min),
            "ra_max": float(ra_max),
            "dec_min": float(dec_min),
            "dec_max": float(dec_max),
        }

        chunks.append(chunk)

    logger.info(f"Generated {len(chunks)} HEALPix chunks with nside={nside}")
    return chunks
