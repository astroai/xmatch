"""Catalogue crossmatch orchestrator.

Resolves inputs into :class:`~xmatcher.sources.CatalogueSource` objects, selects a
strategy from the input types, executes it on the appropriate backend, and
returns the result as a polars frame (eager by default, lazy on request).

Top-level operations:

* :meth:`CrossMatch.crossmatch` — classic two-catalogue match.
* :meth:`CrossMatch.crossmatch_multi` — N-catalogue pairwise intersection.
* :meth:`CrossMatch.union_match` — build a master union catalogue via sequential
  full outer joins across all catalogues.
* :meth:`CrossMatch.nway_match` — Bayesian N-way simultaneous crossmatching.
* :meth:`CrossMatch.fof_match` — Friends-of-Friends transitive closure across
  all catalogues, merging multi-survey detections into object bundles.
* :meth:`CrossMatch.crossmatch_request` — typed entry point for new code.
"""

import difflib
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from itertools import product as cartesian_product
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import auth, io_utils
from .astro_utils import (
    coord_arrays,
    find_coord_columns,
    require_finite_coordinates,
    sky_extent,
    sky_extent_from_frame,
)
from .bayes import compute_nway_p_match, fit_empirical_kde
from .discovery import (
    catalogue_entry_from_schema,
    endpoint_archive,
    get_table_schema,
    guess_endpoint,
    looks_like_table_id,
)
from .exceptions import ConfigError, CrossMatchError, InputError
from .matchers import (
    _RIGHT_SUFFIX,
    PMATCH_COLUMN,
    _arcsec_to_chord,
    _pos_sigma_arcsec,
    _radec_to_xyz,
    _validate_coordinate_frames,
    id_join,
    sky_match,
)
from .request import MatchRequest, MatchSpec
from .sources import ASTROMETRIC_COVARIANCE_KEYS, CatalogueSource
from .user_config import bundled_config_path, load_merged_config

logger = logging.getLogger(__name__)

FrameInput = str | Path | pl.DataFrame | pl.LazyFrame


def _find_default_config_path() -> Path | None:
    """Prefer bundled package yaml (user overlay is merged separately)."""
    bundled = bundled_config_path()
    if bundled.is_file():
        return bundled
    cwd = Path.cwd() / "xmatcher.yaml"
    return cwd if cwd.is_file() else None


DEFAULT_CONFIG_PATH = _find_default_config_path()

# Defaults shared with the CLI for the engine='ray-union' route (mirror + plan).
DEFAULT_SYNC_RATE_LIMIT = 1.0
DEFAULT_MIN_FREE_GB = 10.0
DEFAULT_SYNC_WORKERS = 8
DEFAULT_HATS_THRESHOLD = 100_000
DEFAULT_RADIUS_ARCSEC = 1.0

# Geometry/identity overrides the ray-union route honours per catalogue
# (mirrors the MatchRequest.from_params side mapping; --columns-* filtering
# is sequential-path output concern the union output does not implement).
_SIDE_OVERRIDE_KEYS = (
    "ra_column",
    "dec_column",
    "id_column",
    "frame",
    "ra_err_column",
    "dec_err_column",
    "corr_column",
    "astrometric_covariance_columns",
    "pos_err_units",
    "default_pos_error_arcsec",
    "epoch",
    "epoch_column",
    "pm_ra_column",
    "pm_dec_column",
    "parallax_column",
    "radial_velocity_column",
    "endpoint",
)


def _side_overrides(params: dict[str, Any], side: int) -> dict[str, Any]:
    """Extract ``--<key>-<side>`` params as ``{key: value}`` for one catalogue."""
    suffix = f"_{side}"
    out: dict[str, Any] = {"endpoint": params["endpoint"]} if params.get("endpoint") else {}
    for key, value in params.items():
        if (
            key.endswith(suffix)
            and key[: -len(suffix)] in _SIDE_OVERRIDE_KEYS
            and value is not None
        ):
            out[key[: -len(suffix)]] = value
    return out


def _joined_column_name(
    schema_names: set[str],
    original: str,
    suffix: str,
    added: set[str] | None = None,
) -> str | None:
    """Resolve a right-hand catalogue's column name inside a chained result.

    ``_rename_right`` suffixes *only* the columns that collide with the
    accumulator's, so a survey whose coordinate columns are named differently
    from the first catalogue's keeps its own name (``RAJ2000``, ``ra_icrs``,
    …).  Chained matching has to look up the name the result actually carries
    instead of assuming ``f"{first_catalogue_name}{suffix}"``.

    ``added`` (the names the last match step contributed) is the unambiguous
    source when the pre-step schema is known; otherwise the suffixed name is
    preferred and the plain name is the fallback.
    """
    suffixed = f"{original}{suffix}"
    if added:
        if suffixed in added:
            return suffixed
        if original in added:
            return original
    if suffixed in schema_names:
        return suffixed
    if original in schema_names:
        return original
    return None


def _angsep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in degrees."""
    d1, d2 = np.radians((dec1, dec2))
    h = (
        np.sin(np.radians(dec1 - dec2) / 2.0) ** 2
        + np.cos(d1) * np.cos(d2) * np.sin(np.radians(ra1 - ra2) / 2.0) ** 2
    )
    return float(np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))))


def _cone_filter_frame(
    frame: pl.DataFrame,
    ra_col: str,
    dec_col: str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
) -> pl.DataFrame:
    """Keep rows within ``radius_deg`` of the cone centre (vectorised polars).

    The pixel-level cone test in :func:`_try_mirrored_cone` is deliberately
    over-inclusive; this exact radial filter runs over the concatenated
    partitions (with a tiny float epsilon).
    """
    dec_r = np.radians(dec_deg)
    dec_expr = pl.col(dec_col).cast(pl.Float64)
    ra_expr = pl.col(ra_col).cast(pl.Float64)
    haversine = ((dec_expr - dec_deg).radians() / 2.0).sin().pow(
        2
    ) + dec_expr.radians().cos() * np.cos(dec_r) * ((ra_expr - ra_deg).radians() / 2.0).sin().pow(2)
    dist = 2.0 * haversine.clip(0.0, 1.0).sqrt().arcsin()
    return frame.filter(dist <= np.nextafter(np.radians(radius_deg), np.inf))


def _try_mirrored_cone(
    src: CatalogueSource,
    columns: list[str] | None,
    ra: float,
    dec: float,
    radius_deg: float,
) -> pl.DataFrame | None:
    """Serve a cone download from the mirrored HATS cache copy, when present.

    Returns a DataFrame when a mirror exists, is on local storage, and holds
    every requested column; ``None`` falls back to the live remote (no
    mirror, missing columns — e.g. target-epoch motion columns — or an
    unreadable local copy).  When ``columns`` is ``None`` (plain sky match,
    TAP ``SELECT *``) the mirror's own column set is served.
    """
    if src.access_method not in ("tap", "hats"):
        return None
    try:
        from .mirror import locate_mirrored
        from .storage import default_cache_root

        root, rel = locate_mirrored(src, cache_root=default_cache_root())
    except CrossMatchError:
        return None  # not mirrored yet — plain remote download
    if str(root).startswith("vos:") or not Path(root).is_dir():
        return None  # vos: roots need staging; the ray-union engine owns those
    try:
        from .hats_native import _ra_dec_columns, _read_properties, list_hats_pixels
        from .ray_union import _pixel_center_deg, _pixel_diagonal_deg

        hats_dir = Path(root) / rel
        partitions = list_hats_pixels(hats_dir)
        if not partitions:
            return None
        schema: set | None = None
        first_files: list[Path] | None = None
        for _order, _pix, part_path in partitions:
            files = [part_path] if part_path.is_file() else sorted(part_path.glob("*.parquet"))
            if files:
                schema = set(pl.scan_parquet(files[0]).collect_schema().names())
                first_files = files
                break
        if schema is None or first_files is None:
            return None
        want = set(columns) if columns else schema
        if not want.issubset(schema):
            return None
        ra_col, dec_col = _ra_dec_columns(src, hats_dir)
        if ra_col not in schema or dec_col not in schema:
            return None
        ring_ordering = _read_properties(hats_dir).get("hats_ordering", "NESTED").upper() == "RING"
        lazy_parts: list[pl.LazyFrame] = []
        for order, pix, part_path in partitions:
            if ring_ordering:
                from .ray_union import _cdshealpix

                pix = int(_cdshealpix().from_ring(np.asarray([pix], dtype=np.uint64), order)[0])
            cen_ra, cen_dec = _pixel_center_deg(order, pix)
            if _angsep_deg(cen_ra, cen_dec, ra, dec) > radius_deg + _pixel_diagonal_deg(order, pix):
                continue
            if part_path.is_file():
                lazy_parts.append(pl.scan_parquet(part_path))
            else:
                files = sorted(part_path.glob("*.parquet"))
                if files:
                    lazy_parts.append(pl.scan_parquet(files))
        if not lazy_parts:
            return pl.scan_parquet(first_files).head(0).collect()
        frame = pl.concat(lazy_parts, how="vertical").collect()
        out = _cone_filter_frame(frame, ra_col, dec_col, ra, dec, radius_deg)
        logger.info(
            "served cone for '%s' from the mirrored HATS cache (%d rows)", src.name, out.height
        )
        return out
    except Exception as exc:  # corrupted/incomplete cache -> live remote
        logger.warning(
            "mirrored HATS copy of '%s' unreadable (%s); falling back to the remote",
            src.name,
            exc,
        )
        return None


class CrossMatch:
    """Configuration holder and crossmatch entry point."""

    def __init__(self, config_file: str | Path | None = None, **kwargs):
        self._explicit_config = config_file is not None
        if config_file is None:
            if DEFAULT_CONFIG_PATH is None:
                raise ConfigError("No xmatcher.yaml found; pass config_file explicitly.")
            self.config_file = DEFAULT_CONFIG_PATH
        else:
            self.config_file = Path(config_file)
            if not self.config_file.is_file():
                raise ConfigError(f"Config file not found: {self.config_file}")

        self.config = self._load_config()
        self.archives_config = self.config.get("archives", {})
        self.catalogues_config = self.config.get("catalogues", {})
        self.aliases_config = self.config.get("catalogue_aliases", {})
        self.stilts_config = self.config.get("stilts_config", {})

        self.stilts_cmd_base = kwargs.get(
            "stilts_cmd_base", self.stilts_config.get("stilts_cmd_base")
        )
        self.stilts_java_opts = kwargs.get("java_opts", self.stilts_config.get("java_opts"))
        self.stilts_tmpdir = kwargs.get("tmpdir", self.stilts_config.get("tmpdir"))
        self.n_workers = kwargs.get("n_workers", os.cpu_count() or 1)
        # Optional default TAP endpoint for ad-hoc table-id resolution.
        self.default_endpoint: str | None = kwargs.get("endpoint")

        self.auth_config = auth.load_auth_config()
        self.last_spill_stats: dict[str, int] | None = None
        self._validate_config()
        logger.info("CrossMatch initialised from %s", self.config_file)

    # ------------------------------------------------------------------ config
    def _load_config(self) -> dict[str, Any]:
        if self._explicit_config:
            config, path = load_merged_config(
                config_file=self.config_file, include_user_overlay=False
            )
            self.config_file = path
            return config
        config, path = load_merged_config(include_user_overlay=True)
        self.config_file = path
        return config

    def _validate_config(self) -> None:
        if not isinstance(self.config, dict):
            raise ConfigError("Configuration must be a mapping.")
        for key in ("archives", "catalogues"):
            if key not in self.config:
                raise ConfigError(f"Missing required top-level key: '{key}'")
            if not isinstance(self.config[key], dict):
                raise ConfigError(f"Top-level key '{key}' must be a mapping.")

        canonical_names: dict[str, str] = {}
        for name in self.catalogues_config:
            if not isinstance(name, str):
                raise ConfigError(f"Catalogue names must be strings, got {name!r}.")
            folded = name.casefold()
            previous = canonical_names.get(folded)
            if previous is not None and previous != name:
                raise ConfigError(f"Catalogue names '{previous}' and '{name}' differ only by case.")
            canonical_names[folded] = name

        for name, cat in self.catalogues_config.items():
            if not isinstance(cat, dict):
                raise ConfigError(f"Catalogue '{name}' must be a mapping.")
            for required in (
                "archive",
                "service_id",
                "access_identifier",
                "ra_column",
                "dec_column",
            ):
                if required not in cat:
                    raise ConfigError(f"Catalogue '{name}' is missing '{required}'.")
            if cat["archive"] not in self.archives_config:
                raise ConfigError(
                    f"Catalogue '{name}' references unknown archive '{cat['archive']}'."
                )
            if cat["service_id"] not in self.archives_config[cat["archive"]]:
                raise ConfigError(
                    f"Catalogue '{name}' references unknown service '{cat['service_id']}'."
                )
            for field_name in ("parallax_column", "radial_velocity_column"):
                value = cat.get(field_name)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ConfigError(
                        f"Catalogue '{name}' has invalid '{field_name}'; expected a column name."
                    )
            covariance_columns = cat.get("astrometric_covariance_columns")
            if covariance_columns is not None:
                if not isinstance(covariance_columns, dict) or set(covariance_columns) != set(
                    ASTROMETRIC_COVARIANCE_KEYS
                ):
                    raise ConfigError(
                        f"Catalogue '{name}' has invalid astrometric_covariance_columns; "
                        f"expected exactly {list(ASTROMETRIC_COVARIANCE_KEYS)}."
                    )
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in covariance_columns.values()
                ):
                    raise ConfigError(
                        f"Catalogue '{name}' has empty astrometric covariance column names."
                    )

        alias_names: dict[str, str] = {}
        for alias, target in self.aliases_config.items():
            if not isinstance(alias, str):
                raise ConfigError(f"Alias names must be strings, got {alias!r}.")
            folded = alias.casefold()
            previous = alias_names.get(folded)
            if previous is not None and previous != alias:
                raise ConfigError(f"Alias names '{previous}' and '{alias}' differ only by case.")
            alias_names[folded] = alias
            if target not in self.catalogues_config:
                raise ConfigError(f"Alias '{alias}' points to unknown catalogue '{target}'.")

    def get_catalogue_config(self, name: str) -> dict[str, Any]:
        name = self.resolve_name(name)
        if name not in self.catalogues_config:
            exc = CrossMatchError(f"Catalogue '{name}' not found in configuration.")
            exc.source = name
            raise exc
        cat = self.catalogues_config[name]
        archive = cat.get("archive")
        service_id = cat.get("service_id")
        if not archive or not service_id:
            raise CrossMatchError(f"Catalogue '{name}' is missing 'archive' or 'service_id'.")
        if archive not in self.archives_config:
            exc = CrossMatchError(f"Archive '{archive}' (for '{name}') not found.")
            exc.source = archive
            raise exc
        service = self.archives_config[archive].get(service_id)
        if not isinstance(service, dict):
            exc = CrossMatchError(f"Service '{service_id}' (for '{name}') not found.")
            exc.source = service_id
            raise exc
        resolved = dict(service)
        resolved.update(cat)
        resolved["_catalogue_name"] = name
        resolved["_archive_name"] = archive
        return resolved

    def resolve_name(self, name: str) -> str:
        folded = name.casefold()
        alias = next(
            (target for key, target in self.aliases_config.items() if key.casefold() == folded),
            None,
        )
        if alias is not None:
            return alias
        return next(
            (canonical for canonical in self.catalogues_config if canonical.casefold() == folded),
            name.lower(),
        )

    def find_catalogue_by_access_id(self, table_id: str) -> str | None:
        """Return catalogue key whose ``access_identifier``/``table_name`` matches *table_id*.

        Lets users paste the ACCESS column from ``xmatcher list`` and still get the
        bundled entry (columns, errors, epoch) instead of a bare TAP_SCHEMA probe.
        """
        needle = table_id.strip().strip('"').lower()
        if not needle:
            return None
        for name, cat in self.catalogues_config.items():
            for key in ("access_identifier", "table_name"):
                val = cat.get(key)
                if val is None:
                    continue
                if str(val).strip().strip('"').lower() == needle:
                    return name
        return None

    def suggest(self, name: str, *, n: int = 3, cutoff: float = 0.4) -> list[str]:
        """Return catalogue or alias names similar to ``name`` for hinting.

        Compares ``name`` against the union of ``catalogues_config`` and
        ``aliases_config`` using :mod:`difflib`, **case-insensitively**,
        while preserving the original (display) casing of matched
        candidates in the returned list.

        Implementation: build a ``lowercase -> original`` dict from the
        pool (``catalogues_config`` + ``aliases_config``), match the
        caller's ``name.lower()`` against the lowercase keys, and map the
        matched lowercase keys back to their original casing.  This makes
        the helper robust to YAML configs that use mixed-case catalogue
        names (e.g. ``Gaia_DR3``) rather than the project's default
        lowercase convention.

        Note: when the pool contains two entries differing only in case
        (e.g. ``Gaia_DR3`` and ``gaia_dr3``), the dict collapses them
        and the first-listed entry wins — this is intentional; configs
        shouldn't ship case-only duplicates.

        The default ``cutoff=0.4`` is intentionally permissive; lower it
        if you see noise, raise it if you'd rather the helper stay silent
        on sloppy typos.  Returns an empty list when nothing is close
        enough — callers can simply elide the "did you mean?" suffix in
        that case.
        """
        pool = list(self.catalogues_config) + list(self.aliases_config)
        if not pool:
            return []
        lower_to_orig = {name.lower(): name for name in pool}
        matches = difflib.get_close_matches(
            name.lower(),
            list(lower_to_orig),
            n=n,
            cutoff=cutoff,
        )
        return [lower_to_orig[m] for m in matches]

    # ----------------------------------------------------------------- sources
    def resolve_source(self, value: FrameInput, overrides: dict[str, Any]) -> CatalogueSource:
        """Build a CatalogueSource from a path/name/frame plus per-side overrides.

        Unknown strings that look like TAP/VizieR table ids (e.g. ``II/349/ps1``
        or ``ls_dr10.tractor``) are resolved on the fly via TAP_SCHEMA when an
        endpoint can be guessed or is passed as ``overrides['endpoint']`` /
        ``CrossMatch(endpoint=…)``.
        """
        if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            lf = value.lazy() if isinstance(value, pl.DataFrame) else value
            return self._local_source("frame", lf=lf, overrides=overrides)

        text = str(value)
        resolved = self.resolve_name(text)
        if resolved in self.catalogues_config:
            return self._remote_source(self.get_catalogue_config(resolved), overrides)

        if text.startswith(("http://", "https://", "vos:")):
            # Remote HATS catalogue over HTTP / vos: — mirrored on demand by the
            # ray-union pipeline and `xmatcher sync`.
            name = text.rstrip("/").rsplit("/", 1)[-1] or "remote-hats"
            return CatalogueSource(
                name=name,
                is_local=False,
                access_method="hats",
                access_identifier=text,
                ra_column=overrides.get("ra_column"),
                dec_column=overrides.get("dec_column"),
                id_column=overrides.get("id_column"),
                release_namespace=overrides.get("release_namespace"),
                release_metadata=overrides.get("release_metadata", {}),
                photometry=overrides.get("photometry", []),
                property_evidence=overrides.get("property_evidence", []),
                frame=overrides.get("frame", "icrs"),
            )

        path = Path(text)
        if io_utils.is_hats_dir(path):
            return self._hats_source(path, overrides)
        if path.is_file():
            return self._local_source(path.stem, path=path, overrides=overrides)

        # Prefer a configured catalogue when the user pastes an ACCESS id from
        # `xmatcher list` (e.g. II/349/ps1 → ps1), before falling back to ad-hoc TAP.
        by_access = self.find_catalogue_by_access_id(text)
        if by_access is not None:
            return self._remote_source(self.get_catalogue_config(by_access), overrides)

        if looks_like_table_id(text):
            # Prefer the table-id heuristic (VizieR slash vs schema.table) over a
            # global --endpoint meant for the other match operand.
            endpoint = guess_endpoint(text) or overrides.get("endpoint") or self.default_endpoint
            if endpoint:
                return self.source_from_table_id(text, endpoint=endpoint, overrides=overrides)

        # Defer the "did you mean?" rendering to the CLI: the exception
        # carries `.source` so callers can decide how loudly to hint.
        raise InputError(
            f"'{text}' is not a file, HATS dir, or known catalogue.",
            source=text,
        )

    def source_from_table_id(
        self,
        table_id: str,
        *,
        endpoint: str,
        overrides: dict[str, Any] | None = None,
    ) -> CatalogueSource:
        """Build a remote :class:`CatalogueSource` from a TAP table id (no YAML write)."""
        overrides = overrides or {}
        archive_name, service_id, tap_url = endpoint_archive(endpoint)
        if archive_name is None:
            # Endpoint has no bundled archive (e.g. cadc) — still allow TAP download.
            archive_name = endpoint.lower()
        auth_session = self.auth_config.get_auth_session(archive_name)
        schema = get_table_schema(tap_url, table_id, auth_session=auth_session)
        if schema["columns_count"] == 0:
            raise InputError(
                f"No columns found for table '{table_id}' on endpoint '{endpoint}'.",
                source=table_id,
            )
        # Prefer explicit RA/Dec overrides over TAP_SCHEMA heuristics.
        if overrides.get("ra_column"):
            schema = dict(schema)
            schema["ra_column"] = overrides["ra_column"]
        if overrides.get("dec_column"):
            schema = dict(schema)
            schema["dec_column"] = overrides["dec_column"]
        if archive_name not in self.archives_config:
            # Synthesize a minimal archive so _remote_source can pick up tap URL.
            name, entry = catalogue_entry_from_schema(
                table_id,
                schema,
                archive=archive_name,
                service_id=service_id,
            )
            cfg = {
                **entry,
                "access_method": "tap",
                "access_url": tap_url,
                "_catalogue_name": name,
                "_archive_name": archive_name,
            }
            return self._remote_source(cfg, overrides)

        name, entry = catalogue_entry_from_schema(
            table_id,
            schema,
            archive=archive_name,
            service_id=service_id,
        )
        # Merge archive service fields the same way as configured catalogues.
        service = self.archives_config[archive_name].get(service_id, {})
        cfg = dict(service) if isinstance(service, dict) else {}
        cfg.update(entry)
        cfg["_catalogue_name"] = name
        cfg["_archive_name"] = archive_name
        return self._remote_source(cfg, overrides)

    def _resolve_coords(self, columns, overrides):
        ra = overrides.get("ra_column") or find_coord_columns(columns)[0]
        dec = overrides.get("dec_column") or find_coord_columns(columns)[1]
        return ra, dec

    def _local_source(self, name, *, path=None, lf=None, overrides) -> CatalogueSource:
        src = CatalogueSource(name=name, is_local=True, path=path)
        if lf is not None:
            src = src.with_frame(lf)
            src.name = name
        src.id_column = overrides.get("id_column")
        src.release_namespace = overrides.get("release_namespace")
        src.release_metadata = overrides.get("release_metadata", {})
        src.photometry = overrides.get("photometry", [])
        src.property_evidence = overrides.get("property_evidence", [])
        src.ra_err_column = overrides.get("ra_err_column")
        src.dec_err_column = overrides.get("dec_err_column")
        src.corr_column = overrides.get("corr_column")
        src.astrometric_covariance_columns = overrides.get("astrometric_covariance_columns")
        src.pos_err_units = overrides.get("pos_err_units") or "arcsec"
        src.default_pos_error_arcsec = overrides.get("default_pos_error_arcsec")
        src.epoch = overrides.get("epoch")
        src.epoch_column = overrides.get("epoch_column")
        src.pm_ra_column = overrides.get("pm_ra_column")
        src.pm_dec_column = overrides.get("pm_dec_column")
        src.parallax_column = overrides.get("parallax_column")
        src.radial_velocity_column = overrides.get("radial_velocity_column")
        src.frame = overrides.get("frame", "icrs")
        # Apply explicit overrides first; fall back to auto-detection. Note that
        # we no longer raise on missing RA/Dec here so ``id_join`` flows can
        # operate on tables without spatial columns; the hard error is now
        # emitted from :func:`xmatcher.matchers.sky_match`.
        src.ra_column = overrides.get("ra_column")
        src.dec_column = overrides.get("dec_column")
        if src.ra_column is None or src.dec_column is None:
            try:
                detected_ra, detected_dec = self._resolve_coords(src.columns(), overrides)
            except Exception:
                detected_ra, detected_dec = None, None
            if src.ra_column is None:
                src.ra_column = detected_ra
            if src.dec_column is None:
                src.dec_column = detected_dec
        return src

    def _resolve_fallback(
        self, value: Any, *, primary_name: str, primary_access: str | None
    ) -> str | CatalogueSource:
        """Resolve one ``fallback:`` entry of a catalogue config.

        Known catalogue/alias names resolve to their full
        :class:`CatalogueSource`; raw ``http(s)://`` / ``vos:`` URLs are kept
        for remote-HATS sources as direct mirror endpoints.  Resolution is
        config-only (no TAP_SCHEMA probes), so ``describe`` / ``list`` stay
        side-effect-free; the schema gate runs at sync time, not here.
        """
        text = str(value).strip()
        if text.startswith(("http://", "https://", "vos:")):
            if primary_access != "hats":
                raise CrossMatchError(
                    f"fallback '{text}' for '{primary_name}': raw URLs are only valid "
                    "for HATS catalogues"
                )
            return text
        resolved = self.resolve_name(text)
        if resolved not in self.catalogues_config:
            raise CrossMatchError(
                f"fallback '{text}' for '{primary_name}' is not a known catalogue"
            )
        return self._remote_source(self.get_catalogue_config(resolved), {})

    def _remote_source(self, cfg: dict[str, Any], overrides) -> CatalogueSource:
        src = CatalogueSource(
            name=cfg["_catalogue_name"],
            is_local=False,
            ra_column=overrides.get("ra_column") or cfg.get("ra_column"),
            dec_column=overrides.get("dec_column") or cfg.get("dec_column"),
            id_column=overrides.get("id_column") or cfg.get("id_column"),
            release_namespace=overrides.get("release_namespace", cfg.get("release_namespace")),
            release_metadata=overrides.get("release_metadata", cfg.get("release_metadata", {})),
            photometry=overrides.get("photometry", cfg.get("photometry", [])),
            property_evidence=overrides.get("property_evidence", cfg.get("property_evidence", [])),
            frame=overrides.get("frame", cfg.get("frame", "icrs")),
            ra_err_column=overrides.get("ra_err_column") or cfg.get("ra_err_column"),
            dec_err_column=overrides.get("dec_err_column") or cfg.get("dec_err_column"),
            corr_column=overrides.get("corr_column") or cfg.get("corr_column"),
            astrometric_covariance_columns=overrides.get("astrometric_covariance_columns")
            or cfg.get("astrometric_covariance_columns"),
            pos_err_units=cfg.get("pos_err_units", "arcsec"),
            default_pos_error_arcsec=cfg.get("default_pos_error_arcsec"),
            epoch=overrides.get("epoch") if "epoch" in overrides else cfg.get("epoch"),
            epoch_column=overrides.get("epoch_column") or cfg.get("epoch_column"),
            pm_ra_column=overrides.get("pm_ra_column") or cfg.get("pm_ra_column"),
            pm_dec_column=overrides.get("pm_dec_column") or cfg.get("pm_dec_column"),
            parallax_column=overrides.get("parallax_column") or cfg.get("parallax_column"),
            radial_velocity_column=overrides.get("radial_velocity_column")
            or cfg.get("radial_velocity_column"),
            access_method=cfg.get("access_method"),
            archive=cfg.get("_archive_name"),
            access_identifier=cfg.get("access_identifier") or cfg.get("table_name"),
            tap_url=cfg.get("access_url") or cfg.get("tap_url"),
            default_columns=cfg.get("default_columns"),
        )
        raw_fallbacks = cfg.get("fallback")
        if isinstance(raw_fallbacks, str):
            raw_fallbacks = [raw_fallbacks]
        src.fallbacks = [
            self._resolve_fallback(fb, primary_name=src.name, primary_access=src.access_method)
            for fb in (raw_fallbacks or [])
        ]
        return src

    def _hats_source(self, path: Path, overrides) -> CatalogueSource:
        return CatalogueSource(
            name=path.name,
            is_local=False,
            access_method="hats",
            access_identifier=str(path),
            path=path,
            ra_column=overrides.get("ra_column"),
            dec_column=overrides.get("dec_column"),
            id_column=overrides.get("id_column"),
            release_namespace=overrides.get("release_namespace"),
            release_metadata=overrides.get("release_metadata", {}),
            photometry=overrides.get("photometry", []),
            property_evidence=overrides.get("property_evidence", []),
            frame=overrides.get("frame", "icrs"),
            epoch=overrides.get("epoch"),
            epoch_column=overrides.get("epoch_column"),
            pm_ra_column=overrides.get("pm_ra_column"),
            pm_dec_column=overrides.get("pm_dec_column"),
            parallax_column=overrides.get("parallax_column"),
            radial_velocity_column=overrides.get("radial_velocity_column"),
            astrometric_covariance_columns=overrides.get("astrometric_covariance_columns"),
        )

    # --------------------------------------------------------------- execution
    def crossmatch(
        self,
        catalogue_1_input: FrameInput,
        catalogue_2_input: FrameInput,
        output_file: str | Path | None = None,
        *,
        lazy: bool = False,
        progress_cb: Callable[[str], None] | None = None,
        **params,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Run a two-catalogue crossmatch.

        For typed request objects see :meth:`crossmatch_request` with
        :class:`~xmatcher.request.MatchRequest`.

        ``progress_cb``, when supplied, is forwarded to :meth:`_download_remote`
        so the CLI can render a spinner during TAP/CDS downloads.
        """
        engine_choice = (params.get("engine") or "auto").lower()
        if engine_choice == "ray-union":
            # the distributed N-way join *is* the union engine: two inputs go
            # through the same full-outer-join pipeline as any N
            return self.union_match(
                [catalogue_1_input, catalogue_2_input],
                output_file=output_file,
                lazy=lazy,
                progress_cb=progress_cb,
                **params,
            )
        req = MatchRequest.from_params(
            catalogue_1_input,
            catalogue_2_input,
            output_file=output_file,
            lazy=lazy,
            **params,
        )
        hats_threshold = params.pop("hats_threshold", 100_000)
        return self.crossmatch_request(req, hats_threshold=hats_threshold, progress_cb=progress_cb)

    def crossmatch_multi(
        self,
        catalogues: list[FrameInput],
        output_file: str | Path | None = None,
        *,
        lazy: bool = False,
        progress_cb: Callable[[str], None] | None = None,
        **params: Any,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Run an N-catalogue crossmatch using sequential pairwise matching.

        Catalogue 1 and 2 are matched first; the result is then matched against
        catalogue 3, then catalogue 4, and so on.  All pairwise steps share the
        same ``MatchSpec`` derived from ``**params``.

        The first catalogue's RA/Dec columns are used as the spatial reference
        throughout the chain.  Column-name collisions on each successive right
        side are disambiguated with ``_2``, ``_3``, … suffixes.

        Only local files/frames are fully supported for catalogues beyond the
        first two; remote catalogues in positions 3+ require ``ra``, ``dec``,
        and ``radius_deg`` in ``params`` so they can be downloaded.
        """
        return self._multi_match_impl(
            catalogues, output_file, lazy, progress_cb=progress_cb, **params
        )

    def union_match(
        self,
        catalogues: list[FrameInput],
        output_file: str | Path | None = None,
        *,
        lazy: bool = False,
        progress_cb: Callable[[str], None] | None = None,
        **params: Any,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Build a master union catalogue via sequential full outer joins.

        Every catalogue's measurements are preserved — unmatched sources from
        any catalogue appear in the output with nulls for the other catalogues'
        columns. This is the core primitive for building a "gigantic union
        catalogue of all measurements of all sources on the sky."

        Unlike :meth:`crossmatch_multi` (which defaults to inner joins and
        progressively filters), this method uses ``join_type="1or2"`` (full
        outer) at every step, so rows from *every* catalogue survive even if
        they have no positional counterpart in any other catalogue.

        A ``_src_cats`` column is added identifying which catalogue(s)
        contributed to each row (a bitmask string like ``"1+2"``, ``"1+2+3"``,
        ``"3"`` for unmatched).  A ``sep_arcsec`` column holds the latest
        pairwise separation (or null for unmatched rows).

        Parameters
        ----------
        catalogues : list of FrameInput
            2+ catalogues (paths, names, or in-memory frames).
        output_file : str or Path, optional
            Stream final result to file.
        lazy : bool
            Return a ``LazyFrame`` instead of collecting.
        **params
            Same as :meth:`crossmatch` (``radius_arcsec``, ``matcher``,
            ``engine``, ``ra``/``dec``/``radius_deg``, …).

        Returns
        -------
        pl.DataFrame or pl.LazyFrame or None (when output_file is set)
            Master union catalogue with columns from all inputs plus
            ``_src_cats`` and ``sep_arcsec``.

        Examples
        --------
        >>> cm = CrossMatch()
        >>> master = cm.union_match(
        ...     ["gaia", "allwise_dl", "twomass.csv"],
        ...     ra=180, dec=-30, radius_deg=0.01, radius_arcsec=1.5,
        ... )
        >>> # Every row has: cat-1 cols (Gaia), cat-2 cols (_2 suffix),
        >>> # cat-3 cols (_3 suffix), _src_cats, sep_arcsec
        >>> print(master["_src_cats"].value_counts())

        Notes
        -----
        When every catalogue is a remote survey and no ``ra``/``dec``/
        ``radius_deg`` is given, :meth:`union_match` auto-routes to
        ``engine='ray-union'``: the full tables are mirrored and joined
        sky-wide (an ``output_file`` is required). A region or an explicit
        ``engine`` keeps the sequential in-process path.
        """  # Force full outer join at every pairwise step.
        union_params = dict(params)
        union_params.setdefault("join_type", "1or2")

        # Full-sky unions of N remote surveys: with no region the sequential
        # path has no local footprint to download around, so route all-remote
        # union requests to the distributed engine — it mirrors full tables
        # and needs no position. An explicit engine or any region keeps the
        # sequential in-process path.
        engine_choice = str(params.get("engine") or "auto").lower()
        region_given = (
            params.get("ra") is not None
            and params.get("dec") is not None
            and params.get("radius_deg") is not None
        )
        if engine_choice == "auto" and not region_given:
            try:
                all_remote = all(not self.resolve_source(cat, {}).is_local for cat in catalogues)
            except CrossMatchError:
                all_remote = False
            if all_remote:
                if output_file is None:
                    raise CrossMatchError(
                        "Full-sky unions of remote surveys write the joined HATS catalogue "
                        "to disk; pass -o/--output <out>.hats and xmatcher will auto-route "
                        "to engine='ray-union'."
                    )
                logger.info(
                    "auto-routed %d remote surveys to engine='ray-union' "
                    "(no region given; full-sky union)",
                    len(catalogues),
                )
                union_params["engine"] = "ray-union"
        return self._multi_match_impl(
            catalogues,
            output_file,
            lazy,
            progress_cb=progress_cb,
            union_match=True,
            **union_params,
        )

    def fof_match(
        self,
        catalogues: list[FrameInput],
        output_file: str | Path | None = None,
        *,
        radius_arcsec: float = 1.0,
        hats_threshold: int = 100_000,
        progress_cb: Callable[[str], None] | None = None,
        **params: Any,
    ) -> pl.DataFrame | None:
        """Friends-of-Friends transitive closure across all catalogues.

        Matches every pair of catalogues,
        then applies transitive closure: if source A matches B and B matches
        C (in different catalogues), {A, B, C} becomes a single **bundle** —
        one row in the output representing one physical object.

        Unlike :meth:`crossmatch_multi` (which chains pairwise inner joins
        and progressively filters) and :meth:`union_match` (which builds a
        giant table via outer joins), FoF builds a graph from all pairwise
        matches and collapses each connected component into one merged row.

        Parameters
        ----------
        catalogues : list of FrameInput
            2+ catalogues (paths, names, or in-memory frames).  Catalogue 1
            selects the components retained in the output; each retained
            component contains at least one catalogue-1 source.
        output_file : str or Path, optional
            Write result to file.
        radius_arcsec : float
            Search radius for pairwise spatial matching.
        hats_threshold : int
            Max rows per HEALPix pixel for .hats output.
        **params
            Passed through to :meth:`resolve_source` and per-pair matching
            (``engine``, ``ra``, ``dec``, ``radius_deg``, …).

        Returns
        -------
        pl.DataFrame or None
            One row per connected component (bundle).  Columns:

            * ``bundle_id`` — unique integer per bundle.
            * ``n_cats`` — how many catalogues contributed to this bundle.
            * ``_src_cats`` — e.g. ``"1+2+3"``.
            * Data columns from the primary catalogue are preserved where
              possible; columns from other catalogues are averaged (numerics)
              or first-value (strings).
        """
        if len(catalogues) < 2:
            raise CrossMatchError(
                "fof_match requires at least 2 catalogues, got %d" % len(catalogues),
            )

        # Resolve all sources (reuse shared multi-match plumbing).
        req = MatchRequest.from_params(
            catalogues[0],
            catalogues[1],
            output_file=None,
            lazy=True,
            radius_arcsec=radius_arcsec,
            find="all",  # FoF needs ALL candidates
            **{k: v for k, v in params.items() if k != "find"},
        )

        sources: list[CatalogueSource] = []
        sources.append(self.resolve_source(req.cat1, req.side1.as_dict()))
        sources.append(self.resolve_source(req.cat2, req.side2.as_dict()))
        for i, cat_input in enumerate(catalogues[2:], start=3):
            sources.append(self.resolve_source(cat_input, _side_overrides(params, i)))

        _validate_coordinate_frames(
            sources, target_epoch=req.spec.target_epoch, id_join=req.id_join
        )

        first_src = sources[0]
        n_total = len(sources)

        # Ensure first source is local.
        if not first_src.is_local and first_src.access_method in ("tap", "cds_xmatch"):
            downloaded = self._download_remote(first_src, req, prefix="1", progress_cb=progress_cb)
            first_src = first_src.with_frame(downloaded.lazy())
            sources[0] = first_src
        elif first_src.access_method == "hats" and not first_src.is_local:
            from . import hats_native

            first_src = first_src.with_frame(hats_native.load_hats_all(first_src).lazy())
            sources[0] = first_src

        # Download or load remaining sources.
        for i, src in enumerate(sources):
            if i == 0:
                continue
            if src.access_method == "hats" and not src.is_local:
                from . import hats_native

                sources[i] = src.with_frame(hats_native.load_hats_all(src).lazy())
                continue
            if src.is_local:
                continue
            downloaded = self._download_remote(
                src,
                req,
                prefix=str(i + 1),
                region_from=first_src.lazy(),
                local=first_src,
                progress_cb=progress_cb,
            )
            sources[i] = src.with_frame(downloaded.lazy())

        frames = [s.lazy().collect() for s in sources]
        orig_cols = [list(f.columns) for f in frames]
        if req.spec.target_epoch is not None:
            from . import matchers as _matchers

            target_ep = float(req.spec.target_epoch)
            empty_src = CatalogueSource(
                name="_empty", is_local=True, ra_column="_ra", dec_column="_dec"
            )
            empty_df = pl.DataFrame(
                {"_ra": pl.Series([], dtype=pl.Float64), "_dec": pl.Series([], dtype=pl.Float64)}
            )
            for i in range(n_total):
                if req.spec.matcher == "skyerr":
                    from .out_of_core import _align_epoch, _validate_sigma

                    frames[i] = _align_epoch(
                        frames[i].lazy(),
                        sources[i],
                        target_ep,
                        propagate_covariance=True,
                        pm_prior=req.spec.pm_prior,
                        magnitude_column=req.spec.pm_prior_magnitude_column,
                    ).collect()
                    _validate_sigma(frames[i].lazy(), _matchers._EPOCH_SIGMA, sources[i])
                else:
                    frames[i], _ = _matchers._apply_proper_motion(
                        frames[i],
                        empty_df,
                        sources[i],
                        empty_src,
                        target_ep,
                        propagate_covariance=req.spec.matcher == "skyellipse",
                        fallback_policy=req.spec.fallback_policy,
                    )
                    if req.spec.pm_prior:
                        frames[i], _, sources[i], _ = _matchers._apply_pm_drift_prior(
                            sources[i],
                            empty_src,
                            frames[i],
                            empty_df,
                            target_ep,
                            magnitude_column=req.spec.pm_prior_magnitude_column,
                        )

        for frame, source in zip(frames, sources, strict=True):
            require_finite_coordinates(
                frame[source.ra_column or "ra"].to_numpy(),
                frame[source.dec_column or "dec"].to_numpy(),
                label=f"catalogue '{source.name}'",
            )
        # --- All cross-catalogue edges -----------------------------------
        # Build a union-find structure across all rows.
        # Node IDs: [0..N0) for cat0, [N0..N0+N1) for cat1, etc.
        from scipy.spatial import cKDTree

        offsets = [0]
        for f in frames:
            offsets.append(offsets[-1] + f.height)
        total_nodes = offsets[-1]

        # Union-find parent array.
        parent = np.arange(total_nodes, dtype=np.int64)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        # All cross-catalogue edges are needed for genuine transitive closure.
        p_ra_col = sources[0].ra_column or "ra"
        p_dec_col = sources[0].dec_column or "dec"
        chord_max = _arcsec_to_chord(radius_arcsec)

        engine_choice = (req.engine or "auto").lower()
        n_edges = 0
        for i in range(n_total - 1):
            for j in range(i + 1, n_total):
                offset_j = offsets[j]
                if engine_choice == "ray":
                    from .ray_engine import ray_zone_match

                    l_idx, r_idx, _ = ray_zone_match(
                        frames[i],
                        frames[j],
                        sources[i],
                        sources[j],
                        replace(req.spec, find="all"),
                    )
                    for pi, nb in zip(l_idx.tolist(), r_idx.tolist(), strict=True):
                        union(offsets[i] + int(pi), offset_j + int(nb))
                        n_edges += 1
                elif req.spec.matcher != "sky":
                    from . import matchers as _matchers

                    l_idx, r_idx, _ = _matchers._scipy_match(
                        frames[i],
                        frames[j],
                        sources[i],
                        sources[j],
                        replace(req.spec, find="all"),
                    )
                    for pi, nb in zip(l_idx.tolist(), r_idx.tolist(), strict=True):
                        union(offsets[i] + int(pi), offset_j + int(nb))
                        n_edges += 1
                else:
                    p_ra = frames[i][sources[i].ra_column or "ra"].to_numpy().astype(float)
                    p_dec = frames[i][sources[i].dec_column or "dec"].to_numpy().astype(float)
                    p_xyz = _radec_to_xyz(p_ra, p_dec)
                    s_ra = frames[j][sources[j].ra_column or "ra"].to_numpy().astype(float)
                    s_dec = frames[j][sources[j].dec_column or "dec"].to_numpy().astype(float)
                    if len(s_ra) == 0 or len(p_ra) == 0:
                        continue
                    s_xyz = _radec_to_xyz(s_ra, s_dec)
                    tree = cKDTree(s_xyz)
                    idx_lists = tree.query_ball_point(p_xyz, r=chord_max, workers=-1)
                    for pi, neighbors in enumerate(idx_lists):
                        if not neighbors:
                            continue
                        node_p = offsets[i] + pi
                        for nb in neighbors:
                            union(node_p, offset_j + int(nb))
                            n_edges += 1

        frames = [f.select(cols) for f, cols in zip(frames, orig_cols, strict=True)]

        logger.info(
            "FoF: %d edges across %d catalogues (%d total nodes).",
            n_edges,
            n_total,
            total_nodes,
        )

        if n_edges == 0:
            # No cross-catalogue matches: every primary source is its own bundle.
            primary_cols0 = list(frames[0].columns)
            rows0: list = []
            for pi in range(frames[0].height):
                row0: dict = {"bundle_id": pi, "n_cats": 1, "_src_cats": "1"}
                for c in primary_cols0:
                    row0[c] = frames[0][c][pi]
                rows0.append(row0)
            result0 = pl.DataFrame(rows0)
            if output_file:
                io_utils.write_frame(
                    result0,
                    output_file,
                    ra_column=p_ra_col,
                    dec_column=p_dec_col,
                    hats_threshold=hats_threshold,
                )
                return None
            return result0

        # --- Find connected components -------------------------------------
        component = np.full(total_nodes, -1, dtype=np.int64)
        comp_id = 0
        for node in range(total_nodes):
            root = find(node)
            if component[root] < 0:
                component[root] = comp_id
                comp_id += 1
        # Propagate component IDs to all nodes.
        for node in range(total_nodes):
            component[node] = component[find(node)]

        n_bundles = comp_id
        logger.info("FoF: %d connected components (bundles).", n_bundles)

        # --- Build output: one row per bundle ------------------------------
        # For each bundle, collect contributing rows from each catalogue.
        # Output: bundle_id, n_cats, _src_cats, and merged data columns.

        # Gather primary catalogue columns as the output schema backbone.
        primary_cols = list(frames[0].columns)
        # Pre-build per-catalogue column lookup: cat_idx -> {orig_name: output_name}
        all_cols_seen = set(primary_cols)
        per_cat_cols: dict[int, dict[str, str]] = {}
        for j in range(1, n_total):
            cat_map: dict[str, str] = {}
            for c in frames[j].columns:
                if c in all_cols_seen:
                    suffixed = f"{c}_{j + 1}"
                else:
                    suffixed = c
                cat_map[c] = suffixed
                all_cols_seen.add(suffixed)
            per_cat_cols[j] = cat_map

        rows: list = []
        for bid in range(n_bundles):
            mask = component == bid
            cat_members: list[str] = []
            total_contributing = 0

            # Build merged row values.
            row: dict = {"bundle_id": bid, "n_cats": 0, "_src_cats": ""}

            # --- Primary catalogue: skip components with no primary member ---
            primary_mask = mask[offsets[0] : offsets[1]]
            primary_indices = np.where(primary_mask)[0]
            if len(primary_indices) == 0:
                # Component has no primary-catalogue node — skip it.
                # (Can happen when a secondary source is isolated — no edges
                # to any primary source.)
                continue
            cat_members.append("1")
            total_contributing += 1
            # Take the first matched primary row.
            pi = int(primary_indices[0])
            for c in primary_cols:
                row[c] = frames[0][c][pi]

            # --- Secondary catalogues: aggregate contributions ---------------
            for j in range(1, n_total):
                j_mask = mask[offsets[j] : offsets[j + 1]]
                j_indices = np.where(j_mask)[0]
                if len(j_indices) == 0:
                    continue
                cat_members.append(str(j + 1))
                total_contributing += 1
                cat_map = per_cat_cols[j]
                for orig_c, suf_name in cat_map.items():
                    vals = frames[j][orig_c].to_numpy()[j_indices]
                    if vals.dtype.kind in ("f", "i", "u"):
                        row[suf_name] = float(np.nanmean(vals.astype(float)))
                    else:
                        row[suf_name] = vals[0]

            row["n_cats"] = total_contributing
            row["_src_cats"] = "+".join(cat_members)
            rows.append(row)

        result = pl.DataFrame(rows)

        if output_file:
            io_utils.write_frame(
                result,
                output_file,
                ra_column=p_ra_col,
                dec_column=p_dec_col,
                hats_threshold=hats_threshold,
            )
            return None
        return result

    # ------------------------------------------------------------------ #
    # Shared multi-catalogue implementation (crossmatch_multi + union_match).
    # ------------------------------------------------------------------ #
    def _ray_union_multi(
        self,
        catalogues: list[FrameInput],
        output_file: str | Path | None,
        progress_cb: Callable[[str], None] | None = None,
        *,
        union_requested: bool = False,
        **params: Any,
    ) -> None:
        """Distributed N-survey full-outer join via :mod:`xmatcher.ray_union`.

        Every input is mirrored into the cache first (:func:`.mirror.ensure_mirrored`),
        then the HATS-sharded Ray pipeline runs.  Writes the joined HATS catalogue
        at ``output_file`` and returns ``None`` (like every other output-file path
        in this module).  A ``vos:`` output is staged in a local directory (an
        existing remote tree is pulled back first, so resume/reruns continue) and
        uploaded on success.
        """
        if output_file is None:
            raise CrossMatchError("engine='ray-union' requires an output file (-o/--output .hats).")
        unsupported = [
            key
            for key in (
                "id_join",
                "extra_distance_cols",
                "prior_columns",
                "probabilistic",
                "filter_expr",
                "columns_1",
                "columns_2",
                "memory_budget_bytes",
                "ra",
                "dec",
                "radius_deg",
                "region",
            )
            if params.get(key) is not None
            and params.get(key) is not False
            and params.get(key) != []
            and params.get(key) != {}
        ]
        if unsupported:
            raise CrossMatchError(
                "engine='ray-union' does not support "
                + ", ".join(unsupported)
                + "; use a pairwise or sequential engine for these options."
            )
        engine_choice = (params.get("engine") or "auto").lower()
        matcher = params.get("matcher") or "sky"
        if matcher not in ("sky", "skyerr", "skyellipse"):
            raise CrossMatchError(
                f"engine='ray-union' supports matcher in ('sky', 'skyerr', 'skyellipse'), got matcher='{matcher}'."
            )
        if params.get("pm_prior") and params.get("target_epoch") is None:
            raise CrossMatchError(
                "engine='ray-union' does not support missing-motion priors without target_epoch; "
                "set target_epoch or use the local candidate-release pipeline."
            )
        if params.get("target_epoch") is not None:
            pre_sources = [
                self.resolve_source(cat, _side_overrides(params, i))
                for i, cat in enumerate(catalogues, start=1)
            ]
            has_motion = any(
                (s.epoch is not None or s.epoch_column is not None)
                and ((s.pm_ra_column and s.pm_dec_column) or bool(params.get("pm_prior")))
                for s in pre_sources
            )
            if not has_motion:
                raise CrossMatchError(
                    "engine='ray-union' motion alignment requires source epoch and "
                    "proper-motion metadata (or pm_prior with epoch)."
                )
        # engine='ray-union' *is* the union engine, so a default join type is
        # treated as the full outer join (only an explicit non-outer choice is
        # an error); engine='ray' needs --union/--join 1or2 to mean "union".
        join_type = params.get("join_type") or "1and2"
        if (union_requested or engine_choice == "ray-union") and join_type == "1and2":
            join_type = "1or2"
        if join_type not in ("1or2", "all"):
            raise CrossMatchError(
                "engine='ray-union' supports full-outer join types '1or2'/'all' only; "
                f"got join_type='{join_type}'."
            )

        from . import ray_union
        from .mirror import ensure_mirrored, locate_mirrored
        from .storage import all_cache_roots, assert_headroom

        cache_cfg = self.config.get("cache", {}) if isinstance(self.config, dict) else {}
        roots = all_cache_roots(cache_cfg)
        if params.get("cache_root"):
            # an explicit --cache-root wins the primary slot; configured
            # replicas still ride along as fallback copies.
            roots = [params["cache_root"], *(r for r in roots if r != params["cache_root"])]
        cache_root = roots[0]
        synclimit = params.get("synclimit")
        rate_limit = float(
            synclimit
            if synclimit is not None
            else (cache_cfg.get("rate_limit_rps") or DEFAULT_SYNC_RATE_LIMIT)
        )
        threads = params.get("threads")
        workers = int(threads if threads is not None else DEFAULT_SYNC_WORKERS)
        min_free = float(
            params.get("min_free_gb")
            or os.environ.get("XMATCHER_MIN_FREE_GB")
            or cache_cfg.get("min_free_gb")
            or DEFAULT_MIN_FREE_GB
        )
        # Fail-fast headroom: a multi-day run must not die at 95% into a full disk.
        assert_headroom(cache_root, min_free, f"cache root '{cache_root}'")
        # A vos: output has no POSIX path: the engine writes a local staging
        # dir (any existing remote tree is pulled back first, so resume and
        # reruns continue where they left off) and the tree is uploaded on
        # success.
        out_text = str(output_file)
        vos_out = out_text.startswith("vos:")
        if vos_out:
            from .mirror import _copy_tree_files
            from .storage import LocalStorage, open_storage

            vos_root, _, _name = out_text.rpartition("/")
            if not vos_root or not _name:
                raise CrossMatchError(
                    f"Invalid vos: output '{output_file}' (need a container and a name)."
                )
            vos_storage = open_storage(vos_root)
            staging = Path(tempfile.mkdtemp(prefix="xmatcher-union-vos-"))
            out_dir = staging / _name
            if vos_storage.list(""):
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    _copy_tree_files(vos_storage, LocalStorage(str(staging)), "")
                    logger.info("staged existing vos: output from %s to %s", vos_root, out_dir)
                except Exception as exc:  # noqa: BLE001 - best-effort pre-stage
                    logger.warning("could not pre-stage existing vos: output: %s", exc)
            engine_output = str(out_dir)
        else:
            out_dir = Path(out_text)
            out_dir.mkdir(parents=True, exist_ok=True)
            engine_output = out_text
        assert_headroom(str(out_dir.parent), min_free, f"output directory '{out_dir.parent}'")

        # Append-only audit trail for day-scale runs: <out>/run.jsonl gains one
        # JSON object per event (attempt, mirror progress, chunk progress, done).
        run_log = out_dir / "run.jsonl"
        import json as _json

        def _append_run(event: str, **fields: Any) -> None:
            try:
                with open(run_log, "a") as fh:
                    fh.write(
                        _json.dumps(
                            {
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "event": event,
                                **fields,
                            },
                            default=str,
                        )
                        + "\n"
                    )
            except OSError:  # never let logging kill the run
                pass

        def progress(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)
            _append_run("progress", msg=msg)

        fresh_after = params.get("fresh_after")
        if fresh_after is not None:
            fresh_after = float(fresh_after)
        retries = int(params.get("retries") or 0)

        # Driver-level retry: mirror gaps refill incrementally and finished
        # chunks are skipped on re-entry, so a failed attempt (driver OOM,
        # SSH drop, a day-long WAN outage) resumes from where it died.
        attempt = 0
        while True:
            attempt += 1
            try:
                _append_run("attempt_start", attempt=attempt)
                # Per-catalogue column overrides like MatchRequest.from_params
                # (--ra1/--dec1/--id1 apply to the first catalogue, --ra2/--dec2/--id2
                # to the second); extras beyond catalogue 2 resolve like the sequential
                # path — with no overrides.
                sources: list[CatalogueSource] = []
                for i, cat in enumerate(catalogues, start=1):
                    overrides = _side_overrides(params, i)
                    sources.append(self.resolve_source(cat, overrides))
                _validate_coordinate_frames(sources, target_epoch=params.get("target_epoch"))
                if params.get("target_epoch") is not None:
                    has_motion = any(
                        (s.epoch is not None or s.epoch_column is not None)
                        and ((s.pm_ra_column and s.pm_dec_column) or bool(params.get("pm_prior")))
                        for s in sources
                    )
                    if not has_motion:
                        raise CrossMatchError(
                            "engine='ray-union' motion alignment requires source epoch and "
                            "proper-motion metadata (or pm_prior with epoch)."
                        )
                if not params.get("no_sync"):
                    sources = [
                        ensure_mirrored(
                            src,
                            cache_root=cache_root,
                            replica_roots=roots[1:],
                            rate_limit_rps=rate_limit,
                            workers=workers,
                            force=bool(params.get("force_sync")),
                            progress_cb=progress,
                            hats_threshold=int(
                                params.get("hats_threshold", DEFAULT_HATS_THRESHOLD)
                            ),
                            fresh_after=fresh_after,
                            min_free_gb=min_free,
                        )
                        for src in sources
                    ]
                else:
                    # --no-sync: require cached HATS copies up front, with the actionable
                    # "run xmatcher sync" error instead of a path-shaped crash downstream,
                    # and repoint each remote source at the first surviving copy
                    # (primary root, then replicas) so the union reads a replica when
                    # the primary is gone instead of failing on an empty directory.
                    from .mirror import _mirrored_source
                    from .storage import open_storage

                    for i, src in enumerate(sources):
                        if src.access_method == "tap" or (
                            src.access_method == "hats"
                            and (src.access_identifier or "").startswith(
                                ("http://", "https://", "vos:")
                            )
                        ):
                            root, rel = locate_mirrored(src, cache_roots=roots)
                            sources[i] = _mirrored_source(src, open_storage(root), rel, root)
                _append_run("mirror_done", attempt=attempt, sources=[s.name for s in sources])
                ray_union.ray_union_match(
                    sources,
                    sep_arcsec=float(params.get("radius_arcsec", DEFAULT_RADIUS_ARCSEC)),
                    output_file=engine_output,
                    hats_threshold=int(params.get("hats_threshold", DEFAULT_HATS_THRESHOLD)),
                    task_rows=params.get("task_rows"),
                    chunk_memory_gb=params.get("chunk_memory_gb"),
                    max_tuples=params.get("max_tuples"),
                    cache_root=cache_root,
                    progress_cb=progress,
                    matcher=matcher,
                    max_error=float(params.get("max_error", 1.0)),
                    target_epoch=params.get("target_epoch"),
                    pm_prior=bool(params.get("pm_prior", False)),
                    pm_prior_magnitude_column=params.get("pm_prior_magnitude_column"),
                    fallback_policy=str(params.get("fallback_policy", "warn")),
                )
                self.last_ray_union_plan = ray_union.last_plan()  # for tests / doctor
                if vos_out:
                    _copy_tree_files(LocalStorage(str(staging)), vos_storage, "")
                    logger.info("uploaded union output to %s", out_text)
                    _append_run("vos_uploaded", output=out_text)
                _append_run("done", attempt=attempt)
                return None
            except Exception as exc:  # noqa: BLE001
                if attempt > retries:
                    _append_run("failed", attempt=attempt, error=str(exc))
                    raise
                delay = min(10.0 * 2 ** (attempt - 1), 300.0)
                logger.warning(
                    "ray-union attempt %d/%d failed (%s: %s); backing off %.0fs and resuming "
                    "(mirror gaps refill incrementally, finished chunks are skipped)",
                    attempt,
                    retries + 1,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                _append_run("attempt_failed", attempt=attempt, error=str(exc), backoff_s=delay)
                time.sleep(delay)

    def _multi_match_impl(
        self,
        catalogues: list[FrameInput],
        output_file: str | Path | None,
        lazy: bool,
        *,
        union_match: bool = False,
        progress_cb: Callable[[str], None] | None = None,
        **params: Any,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        if len(catalogues) < 2:
            raise CrossMatchError("At least two catalogues are required for crossmatching.")

        engine_choice = (params.get("engine") or "auto").lower()
        if engine_choice == "ray-union":
            # the distributed N-way join *is* the union engine: route
            # unconditionally (README's `--engine ray-union` examples never
            # pass --union), regardless of the default join_type.
            self._ray_union_multi(
                catalogues,
                output_file,
                progress_cb=progress_cb,
                union_requested=union_match,
                **params,
            )
            return None
        if (
            engine_choice == "ray"
            and (union_match or (params.get("join_type") or "1and2") in ("1or2", "all"))
            and output_file is not None
            and (str(output_file).endswith(".hats") or str(output_file).startswith("vos:"))
            and (params.get("matcher") or "sky") in ("sky", "skyerr", "skyellipse")
            and not params.get("extra_distance_cols")
            and not params.get("prior_columns")
            and not params.get("probabilistic")
            and not any(isinstance(c, (pl.DataFrame, pl.LazyFrame)) for c in catalogues)
        ):
            self._ray_union_multi(
                catalogues,
                output_file,
                progress_cb=progress_cb,
                union_requested=union_match,
                **params,
            )
            return None

        req = MatchRequest.from_params(
            catalogues[0],
            catalogues[1],
            output_file=None,
            lazy=True,
            **params,
        )
        if req.memory_budget_bytes is not None:
            if union_match and len(catalogues) == 2:
                from .out_of_core import match_to_output, preflight_request, spill_required

                self.last_spill_stats = None
                preflight_request(req)
                left = self.resolve_source(req.cat1, req.side1.as_dict())
                right = self.resolve_source(req.cat2, req.side2.as_dict())
                _validate_coordinate_frames(
                    [left, right], target_epoch=req.spec.target_epoch, id_join=req.id_join
                )
                if left.is_local and right.is_local and spill_required(req, left, right):
                    spill_req = replace(req, output_file=output_file, lazy=False)
                    self.last_spill_stats = match_to_output(
                        spill_req, left, right, source_tags=True
                    )
                    return None
            elif len(catalogues) > 2:
                raise CrossMatchError(
                    "Bounded-memory union matching currently supports exactly two local "
                    "CSV/Parquet catalogues; N-way spill matching is not yet supported. "
                    "Use --engine ray-union for distributed N-way unions."
                )
            else:
                raise CrossMatchError(
                    "Bounded-memory matching currently supports pairwise crossmatch or "
                    "two-catalogue union output only."
                )

        # Resolve ALL sources up front.
        sources: list[CatalogueSource] = []
        sources.append(self.resolve_source(req.cat1, req.side1.as_dict()))
        sources.append(self.resolve_source(req.cat2, req.side2.as_dict()))
        for i, cat_input in enumerate(catalogues[2:], start=3):
            sources.append(self.resolve_source(cat_input, _side_overrides(params, i)))

        _validate_coordinate_frames(
            sources, target_epoch=req.spec.target_epoch, id_join=req.id_join
        )

        first_src = sources[0]
        n_total = len(sources)

        # Ensure first source is local.
        if not first_src.is_local and first_src.access_method in ("tap", "cds_xmatch"):
            downloaded = self._download_remote(first_src, req, prefix="1", progress_cb=progress_cb)
            first_src = first_src.with_frame(downloaded.lazy())
            sources[0] = first_src

        hats_threshold = params.pop("hats_threshold", 100_000)

        # Download remote sources beyond the first two.
        for i, src in enumerate(sources):
            if src.is_local or src.access_method == "hats":
                continue
            if i <= 1:
                continue
            downloaded = self._download_remote(
                src,
                req,
                prefix=str(i + 1),
                region_from=first_src.lazy(),
                local=first_src,
                progress_cb=progress_cb,
            )
            sources[i] = src.with_frame(downloaded.lazy())

        first_ra = first_src.ra_column or "ra"
        first_dec = first_src.dec_column or "dec"

        # First pair.
        accum_lf = self._dispatch(sources[0], sources[1], req)
        pair_cols = set(accum_lf.collect_schema().names())
        # The result-side names of catalogue 2's coordinate columns: only
        # suffixes when they collide with catalogue 1's names.
        kept_cols = set(sources[0].columns()) if sources[0].is_local else set()
        pair_added = (pair_cols - kept_cols) if kept_cols else None
        prev_ra = _joined_column_name(pair_cols, sources[1].ra_column or "ra", "_2", pair_added)
        prev_dec = _joined_column_name(pair_cols, sources[1].dec_column or "dec", "_2", pair_added)
        if prev_ra is None or prev_dec is None:
            if union_match:
                raise CrossMatchError(
                    f"Could not locate catalogue '{sources[1].name}' coordinate columns "
                    "in the match result; pass --ra2/--dec2 to name them explicitly."
                )
            logger.debug(
                "catalogue '%s' coordinate columns not found in the pair result; "
                "later steps will not re-position its rows",
                sources[1].name,
            )
        if union_match:
            # Per-row detection: which catalogue(s) contributed?  Requires BOTH
            # RA and Dec to be non-null on a side.
            accum_lf = accum_lf.with_columns(
                pl.when(
                    pl.col(prev_ra).is_not_null()
                    & pl.col(prev_dec).is_not_null()
                    & pl.col(first_ra).is_not_null()
                )
                .then(pl.lit("1+2"))
                .when(pl.col(prev_ra).is_not_null())
                .then(pl.lit("2"))
                .otherwise(pl.lit("1"))
                .alias("_src_cats")
            )

        # Remaining catalogues.
        for i in range(2, n_total):
            right_src = sources[i]
            suffix = f"_{i + 1}"

            # Coalesce spatial columns for unmatched rows.  ``prev_ra``/``prev_dec``
            # are the result-side names of the catalogue added by the previous
            # step, whatever they happen to be called.
            accum_cols = set(accum_lf.collect_schema().names())
            if i == 2:
                ra_candidates = [first_ra, prev_ra]
                dec_candidates = [first_dec, prev_dec]
            else:
                ra_candidates = ["_accum_ra", prev_ra]
                dec_candidates = ["_accum_dec", prev_dec]

            ra_present = [c for c in ra_candidates if c and c in accum_cols]
            dec_present = [c for c in dec_candidates if c and c in accum_cols]
            if ra_present:
                accum_lf = accum_lf.with_columns(
                    pl.coalesce([pl.col(c) for c in ra_present]).alias("_accum_ra")
                )
            if dec_present:
                accum_lf = accum_lf.with_columns(
                    pl.coalesce([pl.col(c) for c in dec_present]).alias("_accum_dec")
                )

            accum_src = replace(first_src, ra_column="_accum_ra", dec_column="_accum_dec")

            if right_src.access_method == "hats":
                from . import hats_native

                accum_lf = hats_native.hats_native_crossmatch(
                    accum_src,
                    right_src,
                    req.spec,
                    engine=req.engine,
                    local_lf1=accum_lf,
                    right_suffix=suffix,
                ).lazy()
            else:
                right_lf = right_src.lazy()
                accum_lf = (
                    self._local_match(
                        accum_src,
                        right_src,
                        accum_lf,
                        right_lf,
                        req,
                        right_suffix=suffix,
                    )
                    .collect()
                    .lazy()
                )

            step_cols = set(accum_lf.collect_schema().names())
            step_added = step_cols - accum_cols
            rN_ra_in_result = _joined_column_name(
                step_cols, right_src.ra_column or "ra", suffix, step_added
            )
            rN_dec_in_result = _joined_column_name(
                step_cols, right_src.dec_column or "dec", suffix, step_added
            )
            if rN_ra_in_result is None or rN_dec_in_result is None:
                if union_match:
                    raise CrossMatchError(
                        f"Could not locate catalogue '{right_src.name}' coordinate columns "
                        "in the match result; add them with --ra/--dec for that catalogue."
                    )
                logger.debug(
                    "catalogue '%s' coordinate columns not found in the result; "
                    "later steps will not re-position its rows",
                    right_src.name,
                )

            if union_match:
                # Per-row detection: did this accumulator row match the new
                # catalogue?  Check the right side's coordinate columns only
                # (they are null for an unmatched right row).
                cat_tag = f"+{i + 1}"
                cat_only = str(i + 1)
                accum_lf = accum_lf.with_columns(
                    pl.when(
                        pl.col(rN_ra_in_result).is_not_null() & pl.col("_src_cats").is_not_null()
                    )
                    .then(pl.col("_src_cats") + pl.lit(cat_tag))
                    .when(pl.col(rN_ra_in_result).is_not_null())
                    .then(pl.lit(cat_only))
                    .otherwise(pl.col("_src_cats"))
                    .alias("_src_cats")
                )

            prev_ra, prev_dec = rN_ra_in_result, rN_dec_in_result

        if output_file:
            io_utils.write_frame(
                accum_lf,
                output_file,
                ra_column=first_ra,
                dec_column=first_dec,
                hats_threshold=hats_threshold,
            )
            return None
        return accum_lf if lazy else accum_lf.collect()

    def nway_match(
        self,
        catalogues: list[FrameInput],
        output_file: str | Path | None = None,
        *,
        radius_arcsec: float = 1.0,
        prior_columns: list[str] | None = None,
        max_tuples_per_source: int = 10_000,
        chunk_size: int = 50_000,
        hats_threshold: int = 100_000,
        progress_cb: Callable[[str], None] | None = None,
        **params: Any,
    ) -> pl.DataFrame | None:
        """Bayesian N-way multi-catalogue crossmatching (Budavári & Szalay 2008).

        Matches *catalogues* simultaneously, computing an N-way Bayes factor
        for each tuple of sources and returning a ``p_match`` column in [0, 1].

        Unlike :meth:`crossmatch_multi` (which chains pairwise inner joins),
        this method builds cross-product tuples from catalogue 1 × catalogue 2
        × … × catalogue N and scores them with the full N-way spatial +
        photometric posterior.

        Parameters
        ----------
        catalogues : list of FrameInput
            2+ catalogues (paths, names, or in-memory frames).
        output_file : str or Path, optional
            Write result to file (.parquet/.csv/.fits/.hats).  When ``None``
            (default), returns the DataFrame.
        radius_arcsec : float
            Search radius (arcsec) for pairwise spatial pre-selection.
        prior_columns : list of str, optional
            Photometric columns for the photometric prior component.
        max_tuples_per_source : int
            Maximum Cartesian-product tuples per primary source before
            truncating with a warning (default 10 000).
        chunk_size : int
            Score tuples in batches of this size. Sources, candidate tuples
            and result batches are materialized in memory.
        hats_threshold : int
            Max rows per HEALPix pixel for .hats output (default 100 000).
        **params
            Passed through to :meth:`resolve_source` (``ra``, ``dec``,
            ``radius_deg`` for remote downloads).

        Returns
        -------
        pl.DataFrame or None
            One row per matched N-tuple, with columns from all catalogues
            (collisions get ``_2``, ``_3``, … suffixes) plus ``p_match``.
            Returns ``None`` when ``output_file`` is set.
        """
        if len(catalogues) < 2:
            raise CrossMatchError(
                "nway_match requires at least 2 catalogues, got %d" % len(catalogues)
            )

        if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
            raise CrossMatchError("nway_match radius_arcsec must be positive and finite")
        for name, value, minimum in (
            ("chunk_size", chunk_size, 1),
            ("max_tuples_per_source", max_tuples_per_source, 0),
        ):
            if (
                isinstance(value, bool)
                or not np.isfinite(value)
                or int(value) != value
                or value < minimum
            ):
                raise CrossMatchError(f"{name} must be an integer >= {minimum}")

        # Resolve all sources and ensure they are local (download if remote).
        sources = [
            self.resolve_source(cat_input, _side_overrides(params, i))
            for i, cat_input in enumerate(catalogues, start=1)
        ]
        target_epoch = params.get("target_epoch")
        pm_prior = bool(params.get("pm_prior", False))
        pm_prior_magnitude_column = params.get("pm_prior_magnitude_column")
        fallback_policy = str(params.get("fallback_policy", "warn"))
        _validate_coordinate_frames(sources, target_epoch=target_epoch)
        for i, src in enumerate(sources):
            if not src.is_local:
                if src.access_method in ("tap", "cds_xmatch"):
                    req = MatchRequest.from_params(
                        catalogues[0],
                        catalogues[0],  # dummy cat1/cat2
                        ra=params.get("ra"),
                        dec=params.get("dec"),
                        radius_deg=params.get("radius_deg"),
                        radius_arcsec=radius_arcsec,
                        target_epoch=target_epoch,
                        probabilistic=True,
                        prior_columns=prior_columns,
                    )
                    downloaded = self._download_remote(
                        src, req, prefix=str(i + 1), progress_cb=progress_cb
                    )
                    src = src.with_frame(downloaded.lazy())
                elif src.access_method == "hats":
                    from . import hats_native

                    src = src.with_frame(hats_native.load_hats_all(src).lazy())
                else:
                    raise CrossMatchError(
                        f"Catalogue {i + 1} ('{src.name}') is not local; "
                        f"nway_match requires local or downloadable catalogues."
                    )
            sources[i] = src

        frames = [s.lazy().collect() for s in sources]
        orig_cols = [list(f.columns) for f in frames]
        n_cats = len(sources)

        if target_epoch is not None:
            from . import matchers as _matchers

            target_ep = float(target_epoch)
            empty_src = CatalogueSource(
                name="_empty", is_local=True, ra_column="_ra", dec_column="_dec"
            )
            empty_df = pl.DataFrame(
                {"_ra": pl.Series([], dtype=pl.Float64), "_dec": pl.Series([], dtype=pl.Float64)}
            )
            for i in range(n_cats):
                frames[i], _ = _matchers._apply_proper_motion(
                    frames[i],
                    empty_df,
                    sources[i],
                    empty_src,
                    target_ep,
                    propagate_covariance=True,
                    fallback_policy=fallback_policy,
                )
                if pm_prior:
                    frames[i], _, sources[i], _ = _matchers._apply_pm_drift_prior(
                        sources[i],
                        empty_src,
                        frames[i],
                        empty_df,
                        target_ep,
                        magnitude_column=pm_prior_magnitude_column,
                    )

        for frame, source in zip(frames, sources, strict=True):
            require_finite_coordinates(
                frame[source.ra_column or "ra"].to_numpy(),
                frame[source.dec_column or "dec"].to_numpy(),
                label=f"catalogue '{source.name}'",
            )
        prior_kdes = []
        for column in prior_columns or []:
            if any(column not in frame.columns for frame in frames):
                raise CrossMatchError(
                    f"N-way prior column '{column}' must exist in every catalogue"
                )
            values = np.concatenate([frame[column].cast(pl.Float64).to_numpy() for frame in frames])
            prior_kdes.append(fit_empirical_kde(values, np.empty(0)))

        # Collect pairwise spatial match indices (primary × each other cat).
        p_ra = sources[0].ra_column or "ra"
        p_dec = sources[0].dec_column or "dec"
        engine_choice = (params.get("engine") or "auto").lower()
        all_pairs: list[tuple[np.ndarray, np.ndarray]] = []

        for j in range(1, n_cats):
            if engine_choice == "ray":
                from .ray_engine import ray_zone_match

                p_idx_j, o_idx_j, _ = ray_zone_match(
                    frames[0],
                    frames[j],
                    sources[0],
                    sources[j],
                    MatchSpec(radius_arcsec=radius_arcsec, find="all"),
                )
                all_pairs.append((p_idx_j.astype(int), o_idx_j.astype(int)))
            else:
                from scipy.spatial import cKDTree

                l_ra_np = frames[0][p_ra].to_numpy()
                l_dec_np = frames[0][p_dec].to_numpy()
                r_ra_np = frames[j][sources[j].ra_column or "ra"].to_numpy()
                r_dec_np = frames[j][sources[j].dec_column or "dec"].to_numpy()

                if len(l_ra_np) == 0 or len(r_ra_np) == 0:
                    all_pairs.append((np.array([], int), np.array([], int)))
                    continue

                l_xyz = _radec_to_xyz(l_ra_np, l_dec_np)
                r_xyz = _radec_to_xyz(r_ra_np, r_dec_np)
                chord_max = _arcsec_to_chord(radius_arcsec)
                tree = cKDTree(r_xyz)
                idx_lists = tree.query_ball_point(l_xyz, r=chord_max, workers=-1)

                p_idx_parts, o_idx_parts = [], []
                for pi, neighbors in enumerate(idx_lists):
                    if neighbors:
                        p_idx_parts.append(np.full(len(neighbors), pi, dtype=int))
                        o_idx_parts.append(np.asarray(neighbors, dtype=int))
                if p_idx_parts:
                    all_pairs.append((np.concatenate(p_idx_parts), np.concatenate(o_idx_parts)))
                else:
                    all_pairs.append((np.array([], int), np.array([], int)))

        # If any non-primary catalogue has no matches, return empty.
        if any(len(p[0]) == 0 for p in all_pairs):
            empty = _build_empty_nway_result(
                [f.select(cols) for f, cols in zip(frames, orig_cols, strict=True)], n_cats
            )
            if output_file:
                io_utils.write_frame(
                    empty,
                    output_file,
                    ra_column=p_ra,
                    dec_column=p_dec,
                    hats_threshold=hats_threshold,
                )
                return None
            return empty

        # Pre-compute sigma arrays for all catalogues before stripping temporary columns.
        cat_ra_names = [sources[i].ra_column or "ra" for i in range(n_cats)]
        cat_dec_names = [sources[i].dec_column or "dec" for i in range(n_cats)]
        sigmas_all = []
        for i, src in enumerate(sources):
            from . import matchers as _matchers

            if _matchers._PROPAGATED_COV_EE in frames[i].columns:
                covariance = _matchers._pos_covariance(frames[i], src)
                sigma = np.sqrt(covariance[0] + covariance[1])
            else:
                sigma = _pos_sigma_arcsec(frames[i], src)
            if sigma is None:
                raise CrossMatchError(
                    f"N-way catalogue '{src.name}' needs declared positional errors or an explicit default_pos_error_arcsec"
                )
            else:
                # Gaussian weights need per-axis sigma, not radial RMS.
                sigma = np.maximum(sigma / np.sqrt(2.0), 1e-6)
            sigmas_all.append(sigma)
        frames = [f.select(cols) for f, cols in zip(frames, orig_cols, strict=True)]

        # Pre-group candidate indices by primary index in O(N_pairs) time.
        pairs_by_pi: list[dict[int, np.ndarray]] = []
        for p_idx, o_idx in all_pairs:
            order = np.argsort(p_idx, kind="stable")
            p_sorted = p_idx[order]
            o_sorted = o_idx[order]
            uniq, first_pos = np.unique(p_sorted, return_index=True)
            splits = np.split(o_sorted, first_pos[1:])
            pairs_by_pi.append({int(k): v for k, v in zip(uniq, splits, strict=True)})

        # Chunked cartesian-product tuple iteration.
        n_primary = frames[0].height
        chunk_batches: list[list] = []
        chunk_tuples: list = []
        total_tuples = 0
        truncated_sources = 0

        for pi in range(n_primary):
            matches_per_cat = [by_pi.get(pi) for by_pi in pairs_by_pi]
            if any(m is None or len(m) == 0 for m in matches_per_cat):
                continue

            prod_size = 1
            for m in matches_per_cat:
                assert m is not None
                prod_size *= len(m)
            if prod_size == 0:
                continue

            if max_tuples_per_source and prod_size > max_tuples_per_source:
                truncated_sources += 1
                if truncated_sources == 1:
                    logger.warning(
                        "Per-source tuple count %d exceeds max_tuples_per_source=%d; "
                        "truncating (further warnings suppressed).",
                        prod_size,
                        max_tuples_per_source,
                    )
                cap = max_tuples_per_source
                for count, combo in enumerate(cartesian_product(*matches_per_cat)):  # type: ignore[arg-type]
                    if count >= cap:
                        break
                    indices = [pi] + list(combo)
                    chunk_tuples.append(indices)
                    total_tuples += 1
                    if len(chunk_tuples) >= chunk_size:
                        chunk_batches.append(chunk_tuples)
                        chunk_tuples = []
            else:
                for combo in cartesian_product(*matches_per_cat):  # type: ignore[arg-type]
                    indices = [pi] + list(combo)
                    chunk_tuples.append(indices)
                    total_tuples += 1
                    if len(chunk_tuples) >= chunk_size:
                        chunk_batches.append(chunk_tuples)
                        chunk_tuples = []

        if chunk_tuples:
            chunk_batches.append(chunk_tuples)

        if not chunk_batches:
            empty = _build_empty_nway_result(frames, n_cats)
            if output_file:
                io_utils.write_frame(
                    empty,
                    output_file,
                    ra_column=p_ra,
                    dec_column=p_dec,
                    hats_threshold=hats_threshold,
                )
                return None
            return empty

        if engine_choice == "ray":
            try:
                import ray  # noqa: PLC0415

                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True, log_to_driver=False)

                @ray.remote
                def _ray_nway_chunk(
                    batch: list,
                    n_c: int,
                    frames_val: list,
                    ra_names: list,
                    dec_names: list,
                    sigmas_val: list,
                    rad_arcsec: float,
                    priors: list | None,
                    population_kdes: list,
                ) -> pl.DataFrame:
                    return _process_nway_chunk(
                        batch,
                        n_c,
                        frames_val,
                        ra_names,
                        dec_names,
                        sigmas_val,
                        rad_arcsec,
                        priors,
                        population_kdes,
                    )

                frames_ref = ray.put(frames)
                sigmas_ref = ray.put(sigmas_all)
                futures = [
                    _ray_nway_chunk.remote(
                        batch,
                        n_cats,
                        frames_ref,
                        cat_ra_names,
                        cat_dec_names,
                        sigmas_ref,
                        radius_arcsec,
                        prior_columns,
                        prior_kdes,
                    )
                    for batch in chunk_batches
                ]
                result_chunks = list(ray.get(futures))
            except ImportError:
                result_chunks = [
                    _process_nway_chunk(
                        batch,
                        n_cats,
                        frames,
                        cat_ra_names,
                        cat_dec_names,
                        sigmas_all,
                        radius_arcsec,
                        prior_columns,
                        prior_kdes,
                    )
                    for batch in chunk_batches
                ]
        else:
            result_chunks = [
                _process_nway_chunk(
                    batch,
                    n_cats,
                    frames,
                    cat_ra_names,
                    cat_dec_names,
                    sigmas_all,
                    radius_arcsec,
                    prior_columns,
                    prior_kdes,
                )
                for batch in chunk_batches
            ]

        if truncated_sources:
            logger.info(
                "%d/%d primary sources had tuples truncated at %d per source.",
                truncated_sources,
                n_primary,
                max_tuples_per_source,
            )

        logger.info(
            "nway_match: %d tuples from %d catalogues across %d primary sources.",
            total_tuples,
            n_cats,
            n_primary,
        )
        result = pl.concat(result_chunks, how="vertical")

        if output_file:
            io_utils.write_frame(
                result,
                output_file,
                ra_column=p_ra,
                dec_column=p_dec,
                hats_threshold=hats_threshold,
            )
            return None
        return result

    def crossmatch_request(
        self,
        req: MatchRequest,
        *,
        hats_threshold: int = 100_000,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Run a crossmatch from a typed :class:`~xmatcher.request.MatchRequest`.

        This is the preferred entry point for new code. It gives mypy and IDEs
        full visibility into every parameter.

        ``progress_cb``, when supplied, is forwarded to
        :meth:`_dispatch` so the CLI can render a spinner during TAP/CDS
        downloads.
        """
        if req.memory_budget_bytes is not None:
            from .out_of_core import preflight_request

            preflight_request(req)
        self.last_spill_stats = None
        src1 = self.resolve_source(req.cat1, req.side1.as_dict())
        src2 = self.resolve_source(req.cat2, req.side2.as_dict())
        _validate_coordinate_frames(
            [src1, src2], target_epoch=req.spec.target_epoch, id_join=req.id_join
        )

        if req.memory_budget_bytes is not None and src1.is_local and src2.is_local:
            from .out_of_core import match_to_output, spill_required

            if spill_required(req, src1, src2):
                self.last_spill_stats = match_to_output(req, src1, src2)
                return None

        result_lf = self._dispatch(src1, src2, req, progress_cb=progress_cb)

        if req.output_file:
            io_utils.write_frame(
                result_lf,
                req.output_file,
                ra_column=src1.ra_column or "ra",
                dec_column=src1.dec_column or "dec",
                hats_threshold=hats_threshold,
            )
            return None
        return result_lf if req.lazy else result_lf.collect()

    # ----------------------------------------------------------------- dispatch
    def _dispatch(
        self,
        src1: CatalogueSource,
        src2: CatalogueSource,
        req: MatchRequest,
        *,
        right_suffix: str = _RIGHT_SUFFIX,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pl.LazyFrame:
        _validate_coordinate_frames(
            [src1, src2], target_epoch=req.spec.target_epoch, id_join=req.id_join
        )
        if src1.access_method == "hats" or src2.access_method == "hats":
            from . import hats_native

            lf1 = src1.lazy() if src1.is_local else None
            lf2 = src2.lazy() if src2.is_local else None
            return hats_native.hats_native_crossmatch(
                src1,
                src2,
                req.spec,
                engine=req.engine,
                local_lf1=lf1,
                local_lf2=lf2,
                right_suffix=right_suffix,
            ).lazy()

        if src1.is_local and src2.is_local:
            return self._local_match(src1, src2, src1.lazy(), src2.lazy(), req)

        if src1.is_local != src2.is_local:
            return self._local_vs_remote(src1, src2, req, progress_cb=progress_cb)

        return self._remote_vs_remote(src1, src2, req, progress_cb=progress_cb)

    def _id_columns(self, src1: CatalogueSource, src2: CatalogueSource, req: MatchRequest):
        if not req.id_join:
            return None
        id1 = req.id_column_1 or src1.id_column
        id2 = req.id_column_2 or src2.id_column
        if not id1 or not id2:
            raise CrossMatchError(
                "ID join requested but id columns are unknown. Provide --id1 and --id2."
            )
        return id1, id2

    def _local_match(
        self,
        src1: CatalogueSource,
        src2: CatalogueSource,
        lf1: pl.LazyFrame,
        lf2: pl.LazyFrame,
        req: MatchRequest,
        *,
        right_suffix: str = _RIGHT_SUFFIX,
    ) -> pl.LazyFrame:
        _validate_coordinate_frames(
            [src1, src2], target_epoch=req.spec.target_epoch, id_join=req.id_join
        )
        ids = self._id_columns(src1, src2, req)
        if ids:
            return id_join(lf1, lf2, ids[0], ids[1], req.spec.join_type, suffix=right_suffix)
        return sky_match(
            src1,
            src2,
            lf1,
            lf2,
            req.spec,
            engine=req.engine,
            stilts_cmd_base=self.stilts_cmd_base,
            java_opts=self.stilts_java_opts,
            tmpdir=self.stilts_tmpdir,
            right_suffix=right_suffix,
        )

    def _local_vs_remote(
        self,
        src1: CatalogueSource,
        src2: CatalogueSource,
        req: MatchRequest,
        *,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pl.LazyFrame:
        local, remote = (src1, src2) if src1.is_local else (src2, src1)
        local_lf = local.lazy()
        remote_prefix = "2" if src1.is_local else "1"

        if (
            remote.access_method == "cds_xmatch"
            and not req.id_join
            and req.spec.target_epoch is None
            and not req.probabilistic
            and not req.spec.prior_columns
            and req.spec.matcher == "sky"
            and req.spec.find == "all"
            and req.spec.join_type == "1and2"
            and not req.spec.extra_distance_cols
            and not req.spec.filter_expr
            and not req.side1.columns
            and not req.side2.columns
            and req.engine in (None, "auto")
            and req.ra is None
            and req.dec is None
            and req.radius_deg is None
        ):
            from .remote_cds import cds_xmatch_local_remote

            result = cds_xmatch_local_remote(local, remote, local_lf, req.spec)
            return result.lazy()

        remote_lf = self._download_remote(
            remote,
            req,
            prefix=remote_prefix,
            region_from=local_lf,
            local=local,
            progress_cb=progress_cb,
        ).lazy()
        downloaded = remote.with_frame(remote_lf)
        new1 = src1 if src1.is_local else downloaded
        new2 = downloaded if src1.is_local else src2
        return self._local_match(new1, new2, new1.lazy(), new2.lazy(), req)

    def _remote_vs_remote(
        self,
        src1: CatalogueSource,
        src2: CatalogueSource,
        req: MatchRequest,
        *,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pl.LazyFrame:
        if (
            src1.access_method == "tap"
            and src2.access_method == "tap"
            and src1.tap_url == src2.tap_url
            and not req.id_join
            and req.spec.target_epoch is None
            and not req.probabilistic
            and not req.spec.prior_columns
            and req.spec.matcher == "sky"
            and req.spec.find == "all"
            and req.spec.join_type == "1and2"
            and not req.spec.extra_distance_cols
            and not req.spec.filter_expr
            and not req.side1.columns
            and not req.side2.columns
            and req.engine in (None, "auto")
            and req.ra is None
            and req.dec is None
            and req.radius_deg is None
        ):
            from .remote_tap import tap_self_join

            auth_session = self.auth_config.get_auth_session(src1.archive)
            return tap_self_join(src1, src2, req.spec, auth_session=auth_session).lazy()

        lf1 = self._download_remote(src1, req, prefix="1", progress_cb=progress_cb).lazy()
        lf2 = self._download_remote(src2, req, prefix="2", progress_cb=progress_cb).lazy()
        new1 = src1.with_frame(lf1)
        new2 = src2.with_frame(lf2)
        return self._local_match(new1, new2, lf1, lf2, req)

    def _download_remote(
        self,
        src: CatalogueSource,
        req: MatchRequest,
        *,
        prefix: str,
        region_from=None,
        local=None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pl.DataFrame:
        ra = req.ra
        dec = req.dec
        radius_deg = req.radius_deg
        if (
            req.spec.matcher in {"skyerr", "skyellipse"} or req.spec.target_epoch is not None
        ) and any(value is None for value in (ra, dec, radius_deg)):
            raise CrossMatchError(
                "Remote uncertainty/epoch matching needs an explicit region (ra, dec, radius_deg); unknown remote errors or motion cannot define a safe automatic halo"
            )
        if region_from is not None and (ra is None or dec is None):
            # Use the polars-native aggregate version when given a LazyFrame so
            # the local table's RA/Dec columns never need to materialise.
            import polars as pl

            if isinstance(region_from, pl.LazyFrame) and local.ra_column and local.dec_column:
                extent = sky_extent_from_frame(region_from, local.ra_column, local.dec_column)
            elif local.ra_column and local.dec_column:
                ra_arr, dec_arr = coord_arrays(region_from, local.ra_column, local.dec_column)
                extent = sky_extent(ra_arr, dec_arr)
            else:
                extent = None
            if extent:
                ra, dec = extent["ra_center_deg"], extent["dec_center_deg"]
                radius_deg = min(180.0, extent["radius_deg"] + req.spec.radius_arcsec / 3600.0)
        if ra is None or dec is None or radius_deg is None:
            raise CrossMatchError(
                f"Downloading remote catalogue '{src.name}' needs a region "
                f"(provide ra/dec and radius_deg)."
            )

        require_finite_coordinates([ra], [dec], label="remote region")
        if not np.isfinite(radius_deg) or not 0.0 < radius_deg <= 180.0:
            raise CrossMatchError("remote region radius_deg must be finite and in (0, 180]")

        columns = (
            req.side1.columns if prefix == "1" else req.side2.columns if prefix == "2" else None
        )
        # A projection must retain geometry and all inputs needed by the scorer.
        # With no requested/default projection, SELECT * already includes them.
        projection = columns if columns is not None else src.default_columns
        required: list[str | None] = [src.ra_column, src.dec_column]
        if req.id_join:
            required.append(src.id_column)
        if (
            req.spec.matcher in {"skyerr", "skyellipse"}
            or req.probabilistic
            or req.spec.prior_columns
        ):
            required.extend((src.ra_err_column, src.dec_err_column, src.corr_column))
            if src.astrometric_covariance_columns:
                required.extend(src.astrometric_covariance_columns.values())
        required.extend(req.spec.prior_columns)
        required.extend(req.spec.extra_distance_cols)
        if req.spec.target_epoch is not None:
            required.extend(
                (
                    src.pm_ra_column,
                    src.pm_dec_column,
                    src.epoch_column,
                    src.parallax_column,
                    src.radial_velocity_column,
                )
            )
        if projection is not None:
            columns = list(dict.fromkeys([*projection, *filter(None, required)]))
        mirrored = _try_mirrored_cone(src, columns, ra, dec, radius_deg)
        if mirrored is not None:
            return mirrored
        auth_session = self.auth_config.get_auth_session(src.archive)
        if src.access_method == "tap":
            from .remote_tap import download_from_tap

            return download_from_tap(
                src,
                ra=ra,
                dec=dec,
                radius_deg=radius_deg,
                columns=columns,
                auth_session=auth_session,
                progress_cb=progress_cb,
            )
        if src.access_method == "cds_xmatch":
            from .remote_cds import download_from_cds

            return download_from_cds(
                src,
                ra=ra,
                dec=dec,
                radius_arcsec=radius_deg * 3600.0,
                columns=columns,
                progress_cb=progress_cb,
            )
        raise CrossMatchError(f"Cannot download from access method '{src.access_method}'.")


# --------------------------------------------------------------------------- #
# nway_match helpers (module-level for pickling safety)
# --------------------------------------------------------------------------- #
def _build_empty_nway_result(frames: list, n_cats: int) -> pl.DataFrame:
    """Return an empty N-way frame while retaining each measurement's dtype."""
    schema = dict(frames[0].schema)
    for j in range(1, n_cats):
        for column, dtype in frames[j].schema.items():
            name = column
            while name in schema:
                name += f"_{j + 1}"
            schema[name] = dtype
    schema[PMATCH_COLUMN] = pl.Float64
    return pl.DataFrame(schema=schema)


def _process_nway_chunk(
    chunk_tuples: list,
    n_cats: int,
    frames: list,
    cat_ra_names: list,
    cat_dec_names: list,
    sigmas_all: list,
    radius_arcsec: float,
    prior_columns: list | None,
    prior_kdes: list | None = None,
) -> pl.DataFrame:
    """Process one chunk of N-way tuples: compute p_match and build frame."""
    # Extract per-catalogue index arrays from the chunk.
    indices_per_cat = [
        np.array([t[i] for t in chunk_tuples], dtype=np.int64) for i in range(n_cats)
    ]

    ras = [
        frames[i][cat_ra_names[i]].to_numpy()[indices_per_cat[i]].astype(float)
        for i in range(n_cats)
    ]
    decs = [
        frames[i][cat_dec_names[i]].to_numpy()[indices_per_cat[i]].astype(float)
        for i in range(n_cats)
    ]
    sigmas = [sigmas_all[i][indices_per_cat[i]] for i in range(n_cats)]

    if prior_columns:
        prior_data = []
        for col in prior_columns:
            col_vals = []
            for i, f in enumerate(frames):
                if col in f.columns:
                    col_vals.append(f[col].to_numpy()[indices_per_cat[i]].astype(float))
                else:
                    col_vals.append(np.full(len(indices_per_cat[i]), np.nan))
            prior_data.append(col_vals)
        p_match = compute_nway_p_match(
            ras,
            decs,
            sigmas,
            radius_arcsec,
            prior_columns=prior_data,
            prior_kdes=prior_kdes,
        )
    else:
        p_match = compute_nway_p_match(ras, decs, sigmas, radius_arcsec)

    # Build chunk result frame.
    # Track ALL seen column names so collisions between catalogues 2+3
    # (e.g. RAJ2000 in both AllWISE and USNO) are suffixed correctly.
    result_parts = []
    seen_columns: set = set()
    for i, f in enumerate(frames):
        suffix = f"_{i + 1}" if i > 0 else ""
        part = f.gather(indices_per_cat[i])
        if suffix:
            names = set(seen_columns)
            renames = {}
            for column in part.columns:
                name = column
                while name in names:
                    name += suffix
                names.add(name)
                renames[column] = name
            part = part.rename(renames)
        seen_columns.update(part.columns)
        result_parts.append(part)

    result = result_parts[0].hstack(
        [column for frame in result_parts[1:] for column in frame.get_columns()]
    )
    result = result.with_columns(pl.Series(PMATCH_COLUMN, p_match))
    return result
