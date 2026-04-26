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

def _is_box_within_circle(
    box_ra: float,
    box_dec: float,
    box_ra_half_width: float,
    box_dec_half_height: float,
    circle_ra: float,
    circle_dec: float,
    circle_radius: float,
) -> bool:
    # Check bounding box in declination first (fast, simple)
    min_dec = box_dec - box_dec_half_height - circle_radius
    max_dec = box_dec + box_dec_half_height + circle_radius

    if circle_dec < min_dec or circle_dec > max_dec:
        return False

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
    def _angular_distance_1d(angle_diff: float, half_width: float) -> float:
        return max(0, abs(angle_diff) - half_width)

    min_dist_to_edge = min(
        abs(circle_dec - (box_dec - box_dec_half_height)),
        abs(circle_dec - (box_dec + box_dec_half_height)),
        abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width)),
    )

    return min_dist_to_edge <= circle_radius

boxes = [(np.random.rand()*360, np.random.rand()*180 - 90, 1.0, 1.0) for _ in range(1000000)]
circle_ra = 180.0
circle_dec = 0.0
circle_radius = 5.0

t0 = time.time()
count = 0
for box in boxes:
    if _is_box_within_circle(*box, circle_ra, circle_dec, circle_radius):
        count += 1
print("With declination bounds check:", time.time() - t0, "count:", count)

def _is_box_within_circle_orig(
    box_ra: float,
    box_dec: float,
    box_ra_half_width: float,
    box_dec_half_height: float,
    circle_ra: float,
    circle_dec: float,
    circle_radius: float,
) -> bool:
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
    def _angular_distance_1d(angle_diff: float, half_width: float) -> float:
        return max(0, abs(angle_diff) - half_width)

    min_dist_to_edge = min(
        abs(circle_dec - (box_dec - box_dec_half_height)),
        abs(circle_dec - (box_dec + box_dec_half_height)),
        abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width)),
    )

    return min_dist_to_edge <= circle_radius

t0 = time.time()
count2 = 0
for box in boxes:
    if _is_box_within_circle_orig(*box, circle_ra, circle_dec, circle_radius):
        count2 += 1
print("Original:", time.time() - t0, "count:", count2)
