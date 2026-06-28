"""Catalogue crossmatch orchestrator.

Resolves inputs into :class:`~xmatch.sources.CatalogueSource` objects, selects a
strategy from the input types, executes it on the appropriate backend, and
returns the result as a polars frame (eager by default, lazy on request).
"""

import difflib
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import yaml

from . import auth, io_utils
from .astro_utils import coord_arrays, find_coord_columns, sky_extent, sky_extent_from_frame
from .exceptions import ConfigError, CrossMatchError, InputError
from .matchers import _RIGHT_SUFFIX, id_join, sky_match
from .request import MatchRequest
from .sources import CatalogueSource

logger = logging.getLogger(__name__)

FrameInput = Union[str, Path, pl.DataFrame, pl.LazyFrame]


def _find_default_config_path() -> Optional[Path]:
    candidates = [Path(__file__).parent / "xmatch.yaml", Path.cwd() / "xmatch.yaml"]
    try:
        from importlib.resources import files

        candidates.insert(0, Path(str(files("xmatch") / "xmatch.yaml")))
    except Exception:
        pass
    candidates += [
        Path.home() / ".config" / "xmatch" / "xmatch.yaml",
        Path.home() / ".xmatch" / "xmatch.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


DEFAULT_CONFIG_PATH = _find_default_config_path()


class CrossMatch:
    """Configuration holder and crossmatch entry point."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None, **kwargs):
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

        self.auth_config = auth.load_auth_config()
        self._validate_config()
        logger.info("CrossMatch initialised from %s", self.config_file)

    # ------------------------------------------------------------------ config
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_file, "r") as fh:
                config = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Error parsing {self.config_file}: {exc}") from exc
        if not isinstance(config, dict):
            raise ConfigError("Configuration file is not a YAML mapping.")
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

        for alias, target in self.aliases_config.items():
            if target not in self.catalogues_config:
                raise ConfigError(f"Alias '{alias}' points to unknown catalogue '{target}'.")

    def get_catalogue_config(self, name: str) -> Dict[str, Any]:
        name = name.lower()
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

    def suggest(self, name: str, *, n: int = 3, cutoff: float = 0.4) -> List[str]:
        """Return catalogue or alias names similar to ``name`` for hinting.

        Compares ``name`` (case-insensitively) against the union of
        ``catalogues_config`` and ``aliases_config`` using :mod:`difflib`. The
        default ``cutoff=0.4`` is intentionally permissive; lower it if you
        see noise, raise it if you'd rather the helper stay silent on sloppy
        typos.

        Returns an empty list when nothing is close enough — callers can simply
        elide the "did you mean?" suffix in that case.
        """
        pool = list(self.catalogues_config) + list(self.aliases_config)
        return difflib.get_close_matches(name.lower(), pool, n=n, cutoff=cutoff)

    # ----------------------------------------------------------------- sources
    def resolve_source(self, value: FrameInput, overrides: Dict[str, Any]) -> CatalogueSource:
        """Build a CatalogueSource from a path/name/frame plus per-side overrides."""
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

        # Defer the "did you mean?" rendering to the CLI: the exception
        # carries `.source` so callers can decide how loudly to hint.
        raise InputError(
            f"'{text}' is not a file, HATS dir, or known catalogue.",
            source=text,
        )

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
            ra_err_column=cfg.get("ra_err_column"),
            dec_err_column=cfg.get("dec_err_column"),
            corr_column=cfg.get("corr_column"),
            pos_err_units=cfg.get("pos_err_units", "arcsec"),
            default_pos_error_arcsec=cfg.get("default_pos_error_arcsec"),
            epoch=cfg.get("epoch"),
            epoch_column=cfg.get("epoch_column"),
            pm_ra_column=cfg.get("pm_ra_column"),
            pm_dec_column=cfg.get("pm_dec_column"),
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
        )

    # --------------------------------------------------------------- execution
    def crossmatch(
        self,
        catalogue_1_input: FrameInput,
        catalogue_2_input: FrameInput,
        output_file: Optional[Union[str, Path]] = None,
        *,
        lazy: bool = False,
        **params,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        """Run a crossmatch (backward-compatible spread-args entry point).

        For new code prefer :meth:`crossmatch_request` with a typed
        :class:`~xmatch.request.MatchRequest`.
        """
        req = MatchRequest.from_legacy(
            catalogue_1_input,
            catalogue_2_input,
            output_file=output_file,
            lazy=lazy,
            **params,
        )
        return self.crossmatch_request(req)

    def crossmatch_multi(
        self,
        catalogues: List[FrameInput],
        output_file: Optional[Union[str, Path]] = None,
        *,
        lazy: bool = False,
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
        if len(catalogues) < 2:
            raise CrossMatchError("At least two catalogues are required for crossmatching.")

        # Build a shared request for every pairwise step.
        req = MatchRequest.from_legacy(
            catalogues[0],
            catalogues[1],
            output_file=None,
            lazy=True,
            **params,
        )

        # ------------------------------------------------------------------ #
        # Resolve ALL sources up front.
        # ------------------------------------------------------------------ #
        sources: List[CatalogueSource] = []
        sources.append(self.resolve_source(req.cat1, req.side1.as_dict()))
        sources.append(self.resolve_source(req.cat2, req.side2.as_dict()))
        for cat_input in catalogues[2:]:
            sources.append(self.resolve_source(cat_input, {}))

        first_src = sources[0]

        # Ensure first source is local so we can use its frame for region
        # inference and as the spatial reference throughout the chain.
        if not first_src.is_local and first_src.access_method in ("tap", "cds_xmatch"):
            downloaded = self._download_remote(first_src, req, prefix="1")
            first_src = first_src.with_frame(downloaded.lazy())
            sources[0] = first_src

        # Download remote (tap/cds_xmatch) sources beyond the first two when
        # possible.  HATS sources at any position are handled by _dispatch /
        # hats_crossmatch below — skip them here.
        #
        # NOTE: when *first_src* is a HATS catalogue, region inference via
        # ``first_src.lazy()`` below is impossible for TAP/CDS downloads at
        # positions 3+.  Work around this by passing explicit ``ra``, ``dec``,
        # and ``radius_deg`` in the params.  The clearer error surfaces from
        # ``_download_remote`` rather than a bare ``ValueError``.
        for i, src in enumerate(sources):
            if src.is_local or src.access_method == "hats":
                continue
            if i <= 1:
                continue  # handled by _dispatch below
            # For catalogues 3+, download the region inferred from the first
            # source (or explicit ra/dec/radius_deg from params).
            downloaded = self._download_remote(
                src,
                req,
                prefix=str(i + 1),
                region_from=first_src.lazy(),
                local=first_src,
            )
            sources[i] = src.with_frame(downloaded.lazy())

        # ------------------------------------------------------------------ #
        # First pair: use _dispatch so local/remote/hats all work.
        # ------------------------------------------------------------------ #
        accum_lf = self._dispatch(sources[0], sources[1], req)

        # ------------------------------------------------------------------ #
        # Remaining catalogues: local match against the accumulator.
        # Each successive right side gets its own suffix: _2, _3, _4, …
        # An eager checkpoint prevents O(N²) query-graph re-evaluation.
        # ------------------------------------------------------------------ #
        for i in range(2, len(sources)):
            right_src = sources[i]
            suffix = f"_{i + 1}"  # catalogue 3 → _3, catalogue 4 → _4, …

            if right_src.access_method == "hats":
                # HATS at position 3+: route via hats_crossmatch with the
                # accumulated frame as the local left side.
                from . import hats_source

                accum_lf = hats_source.hats_crossmatch(
                    first_src, right_src, req.spec,
                    local_lf1=accum_lf,
                    right_suffix=suffix,
                ).lazy()
            else:
                right_lf = right_src.lazy()
                accum_lf = (
                    self._local_match(
                        first_src,  # spatial reference: always cat-1 RA/Dec
                        right_src,
                        accum_lf,
                        right_lf,
                        req,
                        right_suffix=suffix,
                    )
                    .collect()
                    .lazy()  # checkpoint: materialise before next iteration
                )

        if output_file:
            io_utils.write_frame(accum_lf, output_file)
            return None
        return accum_lf if lazy else accum_lf.collect()

    def crossmatch_request(
        self,
        req: MatchRequest,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        """Run a crossmatch from a typed :class:`~xmatch.request.MatchRequest`.

        This is the preferred entry point for new code. It gives mypy and IDEs
        full visibility into every parameter.
        """
        src1 = self.resolve_source(req.cat1, req.side1.as_dict())
        src2 = self.resolve_source(req.cat2, req.side2.as_dict())

        result_lf = self._dispatch(src1, src2, req)

        if req.output_file:
            io_utils.write_frame(result_lf, req.output_file)
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
    ) -> pl.LazyFrame:
        if src1.access_method == "hats" or src2.access_method == "hats":
            from . import hats_source

            lf1 = src1.lazy() if src1.is_local else None
            lf2 = src2.lazy() if src2.is_local else None
            return hats_source.hats_crossmatch(
                src1, src2, req.spec,
                local_lf1=lf1, local_lf2=lf2,
                right_suffix=right_suffix,
            ).lazy()

        if src1.is_local and src2.is_local:
            return self._local_match(src1, src2, src1.lazy(), src2.lazy(), req)

        if src1.is_local != src2.is_local:
            return self._local_vs_remote(src1, src2, req)

        return self._remote_vs_remote(src1, src2, req)

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

    def _local_vs_remote(self, src1: CatalogueSource, src2: CatalogueSource, req: MatchRequest) -> pl.LazyFrame:
        local, remote = (src1, src2) if src1.is_local else (src2, src1)
        local_lf = local.lazy()
        remote_prefix = "2" if src1.is_local else "1"

        if remote.access_method == "cds_xmatch" and not req.id_join:
            from .remote_cds import cds_xmatch_local_remote

            result = cds_xmatch_local_remote(local, remote, local_lf, req.spec)
            return result.lazy()

        remote_lf = self._download_remote(
            remote, req, prefix=remote_prefix, region_from=local_lf, local=local
        ).lazy()
        downloaded = remote.with_frame(remote_lf)
        new1 = src1 if src1.is_local else downloaded
        new2 = downloaded if src1.is_local else src2
        return self._local_match(new1, new2, new1.lazy(), new2.lazy(), req)

    def _remote_vs_remote(self, src1: CatalogueSource, src2: CatalogueSource, req: MatchRequest) -> pl.LazyFrame:
        if (
            src1.access_method == "tap"
            and src2.access_method == "tap"
            and src1.tap_url == src2.tap_url
            and not req.id_join
        ):
            from .remote_tap import tap_self_join

            auth_session = self.auth_config.get_auth_session(src1.archive)
            return tap_self_join(src1, src2, req.spec, auth_session=auth_session).lazy()

        lf1 = self._download_remote(src1, req, prefix="1").lazy()
        lf2 = self._download_remote(src2, req, prefix="2").lazy()
        new1 = src1.with_frame(lf1)
        new2 = src2.with_frame(lf2)
        return self._local_match(new1, new2, lf1, lf2, req)

    def _download_remote(
        self, src: CatalogueSource, req: MatchRequest, *, prefix: str, region_from=None, local=None
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
            req.side1.columns
            if prefix == "1"
            else req.side2.columns
            if prefix == "2"
            else None
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
            )
        if src.access_method == "cds_xmatch":
            from .remote_cds import download_from_cds

            return download_from_cds(
                src,
                ra=ra,
                dec=dec,
                radius_arcsec=radius_deg * 3600.0,
                columns=columns,
            )
        raise CrossMatchError(f"Cannot download from access method '{src.access_method}'.")
