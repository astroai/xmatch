import math
import time
import numpy as np

def _angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)
    dlon = ra2_rad - ra1_rad
    dlat = dec2_rad - dec1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return math.degrees(c)

def _angular_distance_1d(angle_diff: float, half_width: float) -> float:
    return max(0, abs(angle_diff) - half_width)

def _is_box_within_circle_orig(
    box_ra: float,
    box_dec: float,
    box_ra_half_width: float,
    box_dec_half_height: float,
    circle_ra: float,
    circle_dec: float,
    circle_radius: float,
) -> bool:
    corners = [
        (box_ra - box_ra_half_width, box_dec - box_dec_half_height),
        (box_ra - box_ra_half_width, box_dec + box_dec_half_height),
        (box_ra + box_ra_half_width, box_dec - box_dec_half_height),
        (box_ra + box_ra_half_width, box_dec + box_dec_half_height),
    ]

    for corner_ra, corner_dec in corners:
        dist = _angular_distance(corner_ra, corner_dec, circle_ra, circle_dec)
        if dist <= circle_radius:
            return True

    if (
        box_ra - box_ra_half_width <= circle_ra <= box_ra + box_ra_half_width
        and box_dec - box_dec_half_height <= circle_dec <= box_dec + box_dec_half_height
    ):
        return True

    min_dist_to_edge = min(
        abs(circle_dec - (box_dec - box_dec_half_height)),
        abs(circle_dec - (box_dec + box_dec_half_height)),
        abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width)),
    )

    return min_dist_to_edge <= circle_radius

def _is_box_within_circle_fast(
    box_ra: float,
    box_dec: float,
    box_ra_half_width: float,
    box_dec_half_height: float,
    circle_ra: float,
    circle_dec: float,
    circle_radius: float,
) -> bool:
    # Quick declination bounds check (latitude)
    # The distance in declination is directly comparable to degrees
    min_dec = box_dec - box_dec_half_height - circle_radius
    max_dec = box_dec + box_dec_half_height + circle_radius

    if circle_dec < min_dec or circle_dec > max_dec:
        return False

    # Original logic
    corners = [
        (box_ra - box_ra_half_width, box_dec - box_dec_half_height),
        (box_ra - box_ra_half_width, box_dec + box_dec_half_height),
        (box_ra + box_ra_half_width, box_dec - box_dec_half_height),
        (box_ra + box_ra_half_width, box_dec + box_dec_half_height),
    ]

    for corner_ra, corner_dec in corners:
        dist = _angular_distance(corner_ra, corner_dec, circle_ra, circle_dec)
        if dist <= circle_radius:
            return True

    if (
        box_ra - box_ra_half_width <= circle_ra <= box_ra + box_ra_half_width
        and box_dec - box_dec_half_height <= circle_dec <= box_dec + box_dec_half_height
    ):
        return True

    min_dist_to_edge = min(
        abs(circle_dec - (box_dec - box_dec_half_height)),
        abs(circle_dec - (box_dec + box_dec_half_height)),
        abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width)),
    )

    return min_dist_to_edge <= circle_radius

boxes = [(np.random.rand()*360, np.random.rand()*180 - 90, 1.0, 1.0) for _ in range(1000)]
circle_ra = 180.0
circle_dec = 0.0
circle_radius = 5.0

# Print the boxes where they differ
for b in boxes:
    orig = _is_box_within_circle_orig(*b, circle_ra, circle_dec, circle_radius)
    fast = _is_box_within_circle_fast(*b, circle_ra, circle_dec, circle_radius)
    if orig != fast:
        print(f"Box: {b}, orig: {orig}, fast: {fast}")
        print(f"Dec range: {b[1]-b[3]} to {b[1]+b[3]}")
        print(f"Circle dec: {circle_dec}, radius: {circle_radius}")
        print(f"Min dist to edge orig logic:")

        box_ra, box_dec, box_ra_half_width, box_dec_half_height = b
        print(f"  edge1: {abs(circle_dec - (box_dec - box_dec_half_height))}")
        print(f"  edge2: {abs(circle_dec - (box_dec + box_dec_half_height))}")
        print(f"  edge3: {abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width))}")

        min_dist_to_edge = min(
            abs(circle_dec - (box_dec - box_dec_half_height)),
            abs(circle_dec - (box_dec + box_dec_half_height)),
            abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width)),
        )
        print(f"  min_dist_to_edge: {min_dist_to_edge} <= {circle_radius} : {min_dist_to_edge <= circle_radius}")
