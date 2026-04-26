import time
import math
import numpy as np

def _angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
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

def _angular_distance_np(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)
    dlon = ra2_rad - ra1_rad
    dlat = dec2_rad - dec1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c)

def _is_box_within_circle(
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
    return False

def _is_box_within_circle_optimized(
    box_ra: float,
    box_dec: float,
    box_ra_half_width: float,
    box_dec_half_height: float,
    circle_ra: float,
    circle_dec: float,
    circle_radius: float,
) -> bool:
    # Try simple bounding box first before calculating haversine
    # The maximum distance in RA can be larger due to cosine of dec, but we can do a rough check

    # Check if the circle is completely outside the box's approximate bounding region
    min_dec = box_dec - box_dec_half_height - circle_radius
    max_dec = box_dec + box_dec_half_height + circle_radius

    if circle_dec < min_dec or circle_dec > max_dec:
        return False

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
    return False

boxes = [(np.random.rand()*360, np.random.rand()*180 - 90, 1.0, 1.0) for _ in range(100000)]
circle_ra = 180.0
circle_dec = 0.0
circle_radius = 5.0

t0 = time.time()
count1 = 0
for box in boxes:
    if _is_box_within_circle(*box, circle_ra, circle_dec, circle_radius):
        count1 += 1
print("Original:", time.time() - t0, "count:", count1)

t0 = time.time()
count2 = 0
for box in boxes:
    if _is_box_within_circle_optimized(*box, circle_ra, circle_dec, circle_radius):
        count2 += 1
print("Optimized bounding box pre-check:", time.time() - t0, "count:", count2)
