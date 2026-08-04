"""Catalogue crossmatch orchestrator.

Resolves inputs into :class:`~xmatch.sources.CatalogueSource` objects, selects a
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
import multiprocessing
from dataclasses import replace
from itertools import product as cartesian_product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import polars as pl

from . import auth, io_utils
from .astro_utils import coord_arrays, find_coord_columns, sky_extent, sky_extent_from_frame
from .bayes import compute_nway_p_match
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
    id_join,
    sky_match,
)
from .request import MatchRequest
from .sources import ASTROMETRIC_COVARIANCE_KEYS, CatalogueSource
from .user_config import bundled_config_path, load_merged_config

logger = logging.getLogger(__name__)

FrameInput = Union[str, Path, pl.DataFrame, pl.LazyFrame]


def _find_default_config_path() -> Optional[Path]:
    """Prefer bundled package yaml (user overlay is merged separately)."""
    bundled = bundled_config_path()
    if bundled.is_file():
        return bundled
    cwd = Path.cwd() / "xmatch.yaml"
    return cwd if cwd.is_file() else None


DEFAULT_CONFIG_PATH = _find_default_config_path()


class CrossMatch:
    """Configuration holder and crossmatch entry point."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None, **kwargs):
        self._explicit_config = config_file is not None
        if config_file is None:
            if DEFAULT_CONFIG_PATH is None:
                raise ConfigError("No xmatch.yaml found; pass config_file explicitly.")
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
        self.n_workers = kwargs.get("n_workers", multiprocessing.cpu_count())
        # Optional default TAP endpoint for ad-hoc table-id resolution.
        self.default_endpoint: Optional[str] = kwargs.get("endpoint")

        self.auth_config = auth.load_auth_config()
        self.last_spill_stats: Optional[Dict[str, int]] = None
        self._validate_config()
        logger.info("CrossMatch initialised from %s", self.config_file)

    # ------------------------------------------------------------------ config
    def _load_config(self) -> Dict[str, Any]:
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

        for alias, target in self.aliases_config.items():
            if target not in self.catalogues_config:
                raise ConfigError(f"Alias '{alias}' points to unknown catalogue '{target}'.")

    def get_catalogue_config(self, name: str) -> Dict[str, Any]:
        name = self.resolve_name(name)
        if name not in self.catalogues_config:
            raise CrossMatchError(f"Catalogue '{name}' not found in configuration.")
        cat = self.catalogues_config[name]
        archive = cat.get("archive")
        service_id = cat.get("service_id")
        if not archive or not service_id:
            raise CrossMatchError(f"Catalogue '{name}' is missing 'archive' or 'service_id'.")
        if archive not in self.archives_config:
            raise CrossMatchError(f"Archive '{archive}' (for '{name}') not found.")
        service = self.archives_config[archive].get(service_id)
        if not isinstance(service, dict):
            raise CrossMatchError(f"Service '{service_id}' (for '{name}') not found.")
        resolved = dict(service)
        resolved.update(cat)
        resolved["_catalogue_name"] = name
        resolved["_archive_name"] = archive
        return resolved

    def resolve_name(self, name: str) -> str:
        return self.aliases_config.get(name.lower(), name.lower())

    def find_catalogue_by_access_id(self, table_id: str) -> Optional[str]:
        """Return catalogue key whose ``access_identifier``/``table_name`` matches *table_id*.

        Lets users paste the ACCESS column from ``xmatch list`` and still get the
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

    def suggest(self, name: str, *, n: int = 3, cutoff: float = 0.4) -> List[str]:
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
    def resolve_source(self, value: FrameInput, overrides: Dict[str, Any]) -> CatalogueSource:
        """Build a CatalogueSource from a path/name/frame plus per-side overrides.

        Unknown strings that look like TAP/VizieR table ids (e.g. ``II/349/ps1``
        or ``ls_dr10.tractor``) are resolved on the fly via TAP_SCHEMA when an
        endpoint can be guessed or is passed as ``overrides['endpoint']`` /
        ``CrossMatch(endpoint=…)``.
        """
        if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            lf = value.lazy() if isinstance(value, pl.DataFrame) else value
            return self._local_source("frame", lf=lf, overrides=overrides)

        # pandas frame support without importing pandas eagerly.
        if value.__class__.__module__.startswith("pandas"):
            return self._local_source("frame", lf=pl.from_pandas(value).lazy(), overrides=overrides)

        text = str(value)
        resolved = self.resolve_name(text)
        if resolved in self.catalogues_config:
            return self._remote_source(self.get_catalogue_config(resolved), overrides)

        path = Path(text)
        if io_utils.is_hats_dir(path):
            return self._hats_source(path, overrides)
        if path.is_file():
            return self._local_source(path.stem, path=path, overrides=overrides)

        # Prefer a configured catalogue when the user pastes an ACCESS id from
        # `xmatch list` (e.g. II/349/ps1 → ps1), before falling back to ad-hoc TAP.
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
        overrides: Optional[Dict[str, Any]] = None,
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
        # Apply explicit overrides first; fall back to auto-detection. Note that
        # we no longer raise on missing RA/Dec here so ``id_join`` flows can
        # operate on tables without spatial columns; the hard error is now
        # emitted from :func:`xmatch.matchers.sky_match`.
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

    def _remote_source(self, cfg: Dict[str, Any], overrides) -> CatalogueSource:
        return CatalogueSource(
            name=cfg["_catalogue_name"],
            is_local=False,
            ra_column=overrides.get("ra_column") or cfg.get("ra_column"),
            dec_column=overrides.get("dec_column") or cfg.get("dec_column"),
            id_column=overrides.get("id_column") or cfg.get("id_column"),
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
        output_file: Optional[Union[str, Path]] = None,
        *,
        lazy: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
        **params,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        """Run a crossmatch (backward-compatible spread-args entry point).

        For new code prefer :meth:`crossmatch_request` with a typed
        :class:`~xmatch.request.MatchRequest`.

        ``progress_cb``, when supplied, is forwarded to :meth:`_download_remote`
        so the CLI can render a spinner during TAP/CDS downloads.
        """
        req = MatchRequest.from_legacy(
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
        catalogues: List[FrameInput],
        output_file: Optional[Union[str, Path]] = None,
        *,
        lazy: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
        **params: Any,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
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
        catalogues: List[FrameInput],
        output_file: Optional[Union[str, Path]] = None,
        *,
        lazy: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
        **params: Any,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
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
        """  # Force full outer join at every pairwise step.
        union_params = dict(params)
        union_params.setdefault("join_type", "1or2")
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
        catalogues: List[FrameInput],
        output_file: Optional[Union[str, Path]] = None,
        *,
        radius_arcsec: float = 1.0,
        hats_threshold: int = 100_000,
        progress_cb: Optional[Callable[[str], None]] = None,
        **params: Any,
    ) -> Optional[pl.DataFrame]:
        """Friends-of-Friends transitive closure across all catalogues.

        Matches the first catalogue against every other catalogue pairwise,
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
            acts as the spatial hub — all pairwise matches go through it.
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
        req = MatchRequest.from_legacy(
            catalogues[0],
            catalogues[1],
            output_file=None,
            lazy=True,
            radius_arcsec=radius_arcsec,
            find="all",  # FoF needs ALL candidates
            **{k: v for k, v in params.items() if k != "find"},
        )

        sources: List[CatalogueSource] = []
        sources.append(self.resolve_source(req.cat1, req.side1.as_dict()))
        sources.append(self.resolve_source(req.cat2, req.side2.as_dict()))
        for cat_input in catalogues[2:]:
            sources.append(self.resolve_source(cat_input, {}))

        first_src = sources[0]
        n_total = len(sources)

        # Ensure first source is local.
        if not first_src.is_local and first_src.access_method in ("tap", "cds_xmatch"):
            downloaded = self._download_remote(first_src, req, prefix="1", progress_cb=progress_cb)
            first_src = first_src.with_frame(downloaded.lazy())
            sources[0] = first_src

        # Download remaining remote sources.
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

        frames = [s.lazy().collect() for s in sources]

        # --- Pairwise matches: cat1 × each other catalogue -----------------
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

        # Match cat1 (index 0) against each other catalogue.
        p_ra_col = sources[0].ra_column or "ra"
        p_dec_col = sources[0].dec_column or "dec"
        p_ra = frames[0][p_ra_col].to_numpy().astype(float)
        p_dec = frames[0][p_dec_col].to_numpy().astype(float)
        p_xyz = _radec_to_xyz(p_ra, p_dec)
        chord_max = _arcsec_to_chord(radius_arcsec)

        n_edges = 0
        for j in range(1, n_total):
            s_ra = frames[j][sources[j].ra_column or "ra"].to_numpy().astype(float)
            s_dec = frames[j][sources[j].dec_column or "dec"].to_numpy().astype(float)
            s_xyz = _radec_to_xyz(s_ra, s_dec)
            tree = cKDTree(s_xyz)
            idx_lists = tree.query_ball_point(p_xyz, r=chord_max, workers=-1)
            offset_j = offsets[j]
            for pi, neighbors in enumerate(idx_lists):
                if not neighbors:
                    continue
                node_p = offsets[0] + pi
                for nb in neighbors:
                    union(node_p, offset_j + int(nb))
                    n_edges += 1

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
        per_cat_cols: Dict[int, Dict[str, str]] = {}
        for j in range(1, n_total):
            cat_map: Dict[str, str] = {}
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
            cat_members: List[str] = []
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
    def _multi_match_impl(
        self,
        catalogues: List[FrameInput],
        output_file: Optional[Union[str, Path]],
        lazy: bool,
        *,
        union_match: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None,
        **params: Any,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        if len(catalogues) < 2:
            raise CrossMatchError("At least two catalogues are required for crossmatching.")

        req = MatchRequest.from_legacy(
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
                if left.is_local and right.is_local and spill_required(req, left, right):
                    spill_req = replace(req, output_file=output_file, lazy=False)
                    self.last_spill_stats = match_to_output(
                        spill_req, left, right, source_tags=True
                    )
                    return None
            elif len(catalogues) > 2:
                raise CrossMatchError(
                    "Bounded-memory union matching currently supports exactly two local "
                    "CSV/Parquet catalogues; N-way spill matching is not yet supported."
                )
            else:
                raise CrossMatchError(
                    "Bounded-memory matching currently supports pairwise crossmatch or "
                    "two-catalogue union output only."
                )

        # Resolve ALL sources up front.
        sources: List[CatalogueSource] = []
        sources.append(self.resolve_source(req.cat1, req.side1.as_dict()))
        sources.append(self.resolve_source(req.cat2, req.side2.as_dict()))
        for cat_input in catalogues[2:]:
            sources.append(self.resolve_source(cat_input, {}))

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
        if union_match:
            # Per-row detection: which catalogue(s) contributed?
            # Check if the right side's spatial columns are non-null.
            r2_ra = sources[1].ra_column or "ra"
            r2_dec = sources[1].dec_column or "dec"
            # Determine the renamed right-side RA/Dec columns in the result.
            # If they collide with left columns, _build_result renames them
            # with a ``_2`` suffix; mirror that suffixing here so the
            # ``_src_cats`` tag accurately reflects which side(s) contributed
            # (i.e. requires *both* RA and Dec non-null on every side).
            left_cols = set(sources[0].columns()) if sources[0].is_local else set()
            r2_ra_in_result = f"{r2_ra}_2" if r2_ra in left_cols else r2_ra
            r2_dec_in_result = f"{r2_dec}_2" if r2_dec in left_cols else r2_dec
            accum_lf = accum_lf.with_columns(
                pl.when(
                    pl.col(r2_ra_in_result).is_not_null()
                    & pl.col(r2_dec_in_result).is_not_null()
                    & pl.col(first_ra).is_not_null()
                )
                .then(pl.lit("1+2"))
                .when(pl.col(r2_ra_in_result).is_not_null())
                .then(pl.lit("2"))
                .otherwise(pl.lit("1"))
                .alias("_src_cats")
            )

        # Remaining catalogues.
        for i in range(2, n_total):
            right_src = sources[i]
            suffix = f"_{i + 1}"

            # Coalesce spatial columns for unmatched rows.
            accum_cols = set(accum_lf.collect_schema().names())
            if i == 2:
                ra_candidates = [first_ra, f"{first_ra}_2"]
                dec_candidates = [first_dec, f"{first_dec}_2"]
            else:
                ra_candidates = ["_accum_ra", f"{first_ra}_{i}"]
                dec_candidates = ["_accum_dec", f"{first_dec}_{i}"]

            ra_present = [c for c in ra_candidates if c in accum_cols]
            dec_present = [c for c in dec_candidates if c in accum_cols]
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
                from . import hats_source

                accum_lf = hats_source.hats_crossmatch(
                    accum_src,
                    right_src,
                    req.spec,
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

            if union_match:
                # Per-row detection: did accumulator row match the new catalogue?
                # Use the suffixed column name since right-side columns are renamed.
                rN_ra = right_src.ra_column or "ra"
                rN_ra_in_result = f"{rN_ra}{suffix}"
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
        catalogues: List[FrameInput],
        output_file: Optional[Union[str, Path]] = None,
        *,
        radius_arcsec: float = 1.0,
        prior_columns: Optional[List[str]] = None,
        max_tuples_per_source: int = 10_000,
        chunk_size: int = 50_000,
        hats_threshold: int = 100_000,
        progress_cb: Optional[Callable[[str], None]] = None,
        **params: Any,
    ) -> Optional[pl.DataFrame]:
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
            Process tuples in batches of this size to bound memory.
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

        # Resolve all sources and ensure they are local (download if remote).
        sources: List[CatalogueSource] = []
        for i, cat_input in enumerate(catalogues):
            src = self.resolve_source(cat_input, {})
            if not src.is_local:
                if src.access_method in ("tap", "cds_xmatch"):
                    req = MatchRequest.from_legacy(
                        catalogues[0],
                        catalogues[0],  # dummy cat1/cat2
                        ra=params.get("ra"),
                        dec=params.get("dec"),
                        radius_deg=params.get("radius_deg"),
                        radius_arcsec=radius_arcsec,
                    )
                    downloaded = self._download_remote(
                        src, req, prefix=str(i + 1), progress_cb=progress_cb
                    )
                    src = src.with_frame(downloaded.lazy())
                else:
                    raise CrossMatchError(
                        f"Catalogue {i + 1} ('{src.name}') is not local; "
                        f"nway_match requires local or downloadable catalogues."
                    )
            sources.append(src)

        frames = [s.lazy().collect() for s in sources]
        n_cats = len(sources)

        # Collect pairwise spatial match indices (primary × each other cat).
        p_ra = sources[0].ra_column or "ra"
        p_dec = sources[0].dec_column or "dec"
        all_pairs: list = []

        for j in range(1, n_cats):
            from scipy.spatial import cKDTree

            l_ra_np = frames[0][p_ra].to_numpy()
            l_dec_np = frames[0][p_dec].to_numpy()
            r_ra_np = frames[j][sources[j].ra_column or "ra"].to_numpy()
            r_dec_np = frames[j][sources[j].dec_column or "dec"].to_numpy()

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

        # Pre-compute sigma arrays for all catalogues.
        cat_ra_names = [sources[i].ra_column or "ra" for i in range(n_cats)]
        cat_dec_names = [sources[i].dec_column or "dec" for i in range(n_cats)]
        sigmas_all = []
        for i, src in enumerate(sources):
            sigma = _pos_sigma_arcsec(frames[i], src)
            if sigma is None:
                sigma = np.full(frames[i].height, 0.5, dtype=float)
            sigmas_all.append(sigma)

        # Chunked cartesian-product tuple iteration.
        n_primary = frames[0].height
        result_chunks: list = []
        chunk_tuples: list = []
        total_tuples = 0
        truncated_sources = 0

        for pi in range(n_primary):
            matches_per_cat = []
            for cat_j in range(len(all_pairs)):
                p_idx, o_idx = all_pairs[cat_j]
                mask = p_idx == pi
                matches_per_cat.append(o_idx[mask])
            if any(len(m) == 0 for m in matches_per_cat):
                continue

            prod_size = 1
            for m in matches_per_cat:
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
                for count, combo in enumerate(cartesian_product(*matches_per_cat)):
                    if count >= cap:
                        break
                    indices = [pi] + list(combo)
                    chunk_tuples.append(indices)
                    total_tuples += 1
                    if len(chunk_tuples) >= chunk_size:
                        result_chunks.append(
                            _process_nway_chunk(
                                chunk_tuples,
                                n_cats,
                                frames,
                                cat_ra_names,
                                cat_dec_names,
                                sigmas_all,
                                radius_arcsec,
                                prior_columns,
                            )
                        )
                        chunk_tuples = []
            else:
                for combo in cartesian_product(*matches_per_cat):
                    indices = [pi] + list(combo)
                    chunk_tuples.append(indices)
                    total_tuples += 1
                    if len(chunk_tuples) >= chunk_size:
                        result_chunks.append(
                            _process_nway_chunk(
                                chunk_tuples,
                                n_cats,
                                frames,
                                cat_ra_names,
                                cat_dec_names,
                                sigmas_all,
                                radius_arcsec,
                                prior_columns,
                            )
                        )
                        chunk_tuples = []

        if chunk_tuples:
            result_chunks.append(
                _process_nway_chunk(
                    chunk_tuples,
                    n_cats,
                    frames,
                    cat_ra_names,
                    cat_dec_names,
                    sigmas_all,
                    radius_arcsec,
                    prior_columns,
                )
            )

        if not result_chunks:
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
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        """Run a crossmatch from a typed :class:`~xmatch.request.MatchRequest`.

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
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> pl.LazyFrame:
        if src1.access_method == "hats" or src2.access_method == "hats":
            if req.spec.target_epoch is not None:
                raise CrossMatchError(
                    "target-epoch propagation is not yet supported by the HATS/LSDB engine"
                )
            from . import hats_source

            lf1 = src1.lazy() if src1.is_local else None
            lf2 = src2.lazy() if src2.is_local else None
            return hats_source.hats_crossmatch(
                src1,
                src2,
                req.spec,
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
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> pl.LazyFrame:
        local, remote = (src1, src2) if src1.is_local else (src2, src1)
        local_lf = local.lazy()
        remote_prefix = "2" if src1.is_local else "1"

        if (
            remote.access_method == "cds_xmatch"
            and not req.id_join
            and req.spec.target_epoch is None
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
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> pl.LazyFrame:
        if (
            src1.access_method == "tap"
            and src2.access_method == "tap"
            and src1.tap_url == src2.tap_url
            and not req.id_join
            and req.spec.target_epoch is None
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
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> pl.DataFrame:
        ra = req.ra
        dec = req.dec
        radius_deg = req.radius_deg
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
                radius_deg = extent["radius_deg"] + req.spec.radius_arcsec / 3600.0
        if ra is None or dec is None or radius_deg is None:
            raise CrossMatchError(
                f"Downloading remote catalogue '{src.name}' needs a region "
                f"(provide ra/dec and radius_deg)."
            )

        columns = (
            req.side1.columns if prefix == "1" else req.side2.columns if prefix == "2" else None
        )
        required: List[Optional[str]] = []
        if req.spec.matcher in {"skyerr", "skyellipse"}:
            required.extend((src.ra_err_column, src.dec_err_column, src.corr_column))
        if req.spec.target_epoch is not None:
            required.extend(
                (
                    src.ra_column,
                    src.dec_column,
                    src.pm_ra_column,
                    src.pm_dec_column,
                    src.epoch_column,
                    src.parallax_column,
                    src.radial_velocity_column,
                )
            )
            if req.spec.matcher == "skyellipse" and src.astrometric_covariance_columns:
                required.extend(src.astrometric_covariance_columns.values())
        if required:
            columns = list(
                dict.fromkeys([*(columns or src.default_columns or []), *filter(None, required)])
            )
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
def _build_empty_nway_result(frames: list, n_cats: int) -> "pl.DataFrame":
    """Return an empty result frame with the correct schema for nway_match."""
    cols = list(frames[0].columns)
    for j in range(1, n_cats):
        suffix = f"_{j + 1}"
        for c in frames[j].columns:
            cols.append(c + suffix if c in cols else c)
    cols.append("p_match")
    return pl.DataFrame(schema={c: pl.Float64 for c in cols}).clear()


def _process_nway_chunk(
    chunk_tuples: list,
    n_cats: int,
    frames: list,
    cat_ra_names: list,
    cat_dec_names: list,
    sigmas_all: list,
    radius_arcsec: float,
    prior_columns: Optional[list],
) -> "pl.DataFrame":
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
            overlap = seen_columns & set(part.columns)
            part = part.rename({c: f"{c}{suffix}" for c in overlap})
        seen_columns.update(part.columns)
        result_parts.append(part)

    result = pl.concat(result_parts, how="horizontal")
    result = result.with_columns(pl.Series(PMATCH_COLUMN, p_match))
    return result
