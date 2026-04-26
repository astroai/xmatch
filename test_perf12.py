import time
import math

def _is_box_within_circle_orig(
    box_ra: float,
    box_dec: float,
    box_ra_half_width: float,
    box_dec_half_height: float,
    circle_ra: float,
    circle_dec: float,
    circle_radius: float,
) -> bool:
    # calculate something
    pass

def _angular_distance_1d(angle_diff: float, half_width: float) -> float:
    return max(0, abs(angle_diff) - half_width)

print(_angular_distance_1d((180 - 0 + 180) % 360 - 180, 1.0))
