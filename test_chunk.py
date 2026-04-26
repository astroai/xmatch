import time
import math
from xmatch.chunking import _is_box_within_circle

print("Original logic check. Box RA=0, Dec=0. Circle RA=180, Dec=0, radius=5.")
print("Result:", _is_box_within_circle(0, 0, 1, 1, 180, 0, 5))
