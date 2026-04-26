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

# The problem with the previous optimization test is that _is_box_within_circle checks if the circle INTERSECTS the box at all.
# The original code logic is buggy!
# Look at original:
# min_dist_to_edge = min(
#     abs(circle_dec - (box_dec - box_dec_half_height)),
#     abs(circle_dec - (box_dec + box_dec_half_height)),
#     abs(_angular_distance_1d((circle_ra - box_ra + 180) % 360 - 180, box_ra_half_width)),
# )
# It returns True if ANY of these 3 distances is <= radius.
# This means if circle_dec is close to the box's BOTTOM edge, it returns True NO MATTER WHAT RA it has!
# Wait! This means the original code has a BUG where it includes boxes that are far away in RA just because they are close in Dec to the circle!
# Let's verify this bug.

print(f"Original bug check: box at RA=0, Dec=0. Circle at RA=180, Dec=0. Circle radius=5")
print(f"Result: {_is_box_within_circle_orig(0, 0, 1, 1, 180, 0, 5)}")
