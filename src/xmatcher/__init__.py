"""xmatcher: cross-match astronomical catalogues (local files, remote archives, HATS).

Key high-level operations:

* :meth:`CrossMatch.crossmatch` — classic two-catalogue match.
* :meth:`CrossMatch.crossmatch_multi` — N-catalogue pairwise intersection.
* :meth:`CrossMatch.union_match` — build a master union catalogue (full outer joins).
* :meth:`CrossMatch.nway_match` — Bayesian N-way simultaneous crossmatching.
* :meth:`CrossMatch.fof_match` — Friends-of-Friends transitive closure merging
  multi-survey detections into object bundles.
* :meth:`CrossMatch.crossmatch_request` — typed entry point for new code.
"""

from .candidates import (
    normalize_candidate_hypotheses,
    verify_candidate_release,
    write_candidate_release,
)
from .crossmatch import CrossMatch
from .exceptions import ConfigError, CrossMatchError, InputError, StiltsError, TapError
from .io_utils import (
    FrameLike,
    astropy_table_to_polars,
    is_hats_dir,
    polars_to_astropy,
    scan_frame,
    write_frame,
)
from .matchers import MatchSpec
from .observations import (
    namespaced_source_id,
    normalize_photometry,
    normalize_property_evidence,
    required_observation_columns,
    source_inventory,
    validate_source_inventory,
    verify_observation_release,
    write_observation_release,
)
from .request import MatchRequest, SideOverrides  # noqa: F401
from .sources import CatalogueSource

__version__ = "0.5.1"

__all__ = [
    # Main entry points
    "CrossMatch",
    "CatalogueSource",
    "MatchSpec",
    "MatchRequest",
    "SideOverrides",
    "source_inventory",
    "namespaced_source_id",
    "normalize_photometry",
    "normalize_property_evidence",
    "required_observation_columns",
    "validate_source_inventory",
    "write_observation_release",
    "verify_observation_release",
    "write_candidate_release",
    "verify_candidate_release",
    "normalize_candidate_hypotheses",
    # Exceptions
    "CrossMatchError",
    "ConfigError",
    "InputError",
    "TapError",
    "StiltsError",
    # I/O helpers (Arrow interop)
    "FrameLike",
    "is_hats_dir",
    "scan_frame",
    "write_frame",
    "astropy_table_to_polars",
    "polars_to_astropy",
    # Meta
    "__version__",
]
