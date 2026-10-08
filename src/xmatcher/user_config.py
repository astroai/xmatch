"""User config path, deep-merge over bundled YAML, and ``adopt`` helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError


def bundled_config_path() -> Path:
    """Return the package-shipped ``xmatcher.yaml`` path."""
    try:
        from importlib.resources import files

        return Path(str(files("xmatcher") / "xmatcher.yaml"))
    except Exception:
        return Path(__file__).parent / "xmatcher.yaml"


def user_config_path() -> Path:
    """Canonical user overlay path (``~/.config/xmatcher/xmatcher.yaml``)."""
    return Path.home() / ".config" / "xmatcher" / "xmatcher.yaml"


def find_user_config_path() -> Path | None:
    """Return the user overlay path if it exists, else ``None``."""
    path = user_config_path()
    return path if path.is_file() else None


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* onto a shallow copy of *base*.

    Mapping values are merged; other values replace. Lists replace entirely.
    """
    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Error parsing {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file is not a YAML mapping: {path}")
    return data


def load_merged_config(
    *,
    config_file: Path | None = None,
    include_user_overlay: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Load config: explicit file alone, else bundled ∪ user ∪ cwd overlays.

    Returns ``(config_dict, primary_path)`` where *primary_path* is the
    explicit file, the last overlay applied, or the bundled path.
    """
    if config_file is not None:
        path = Path(config_file)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return load_yaml_mapping(path), path

    bundled = bundled_config_path()
    if not bundled.is_file():
        raise ConfigError("No bundled xmatcher.yaml found; pass config_file explicitly.")
    config = load_yaml_mapping(bundled)
    primary = bundled
    if include_user_overlay:
        user = find_user_config_path()
        if user is not None:
            config = deep_merge(config, load_yaml_mapping(user))
            primary = user
        cwd_cfg = Path.cwd() / "xmatcher.yaml"
        # Project-local overlay (skip if it is the bundled file or already merged).
        if cwd_cfg.is_file() and cwd_cfg.resolve() not in {
            bundled.resolve(),
            *([user.resolve()] if user is not None else []),
        }:
            config = deep_merge(config, load_yaml_mapping(cwd_cfg))
            primary = cwd_cfg
    return config, primary


def format_catalogue_yaml(name: str, entry: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize one catalogue entry as an indented YAML block under ``catalogues:``."""
    pad = " " * indent
    # Dump just the entry mapping, then prefix each line.
    body = yaml.safe_dump(
        {name: entry},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip()
    lines = body.splitlines()
    # safe_dump already emits `name:` at column 0; re-indent under catalogues.
    return "\n".join(pad + line if line else line for line in lines) + "\n"


def append_catalogue_to_user_config(
    name: str,
    entry: dict[str, Any],
    *,
    path: Path | None = None,
    alias: str | None = None,
    archives: dict[str, Any] | None = None,
) -> Path:
    """Create or update the user YAML with one catalogue (and optional alias).

    Ensures ``archives`` exist (copied from *archives* or the bundled file)
    so validation succeeds when the user file is loaded alone or as overlay.
    """
    dest = path or user_config_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        data = load_yaml_mapping(dest)
    else:
        data = {}

    # Always merge needed archives so a second adopt from another endpoint
    # (e.g. cds then noao_datalab) does not leave a dangling archive ref.
    existing_archives = data.setdefault("archives", {})
    if not isinstance(existing_archives, dict):
        raise ConfigError(f"'archives' in {dest} must be a mapping.")
    if archives:
        for archive_name, archive_body in archives.items():
            if archive_name not in existing_archives:
                existing_archives[archive_name] = archive_body
    elif not existing_archives:
        bundled = load_yaml_mapping(bundled_config_path())
        data["archives"] = bundled.get("archives", {})

    cats = data.setdefault("catalogues", {})
    if not isinstance(cats, dict):
        raise ConfigError(f"'catalogues' in {dest} must be a mapping.")
    if name in cats:
        raise ConfigError(f"Catalogue '{name}' already exists in {dest}.")
    cats[name] = entry

    needed_archive = entry.get("archive")
    if needed_archive and needed_archive not in data.get("archives", {}):
        if archives and needed_archive in archives:
            data.setdefault("archives", {})[needed_archive] = archives[needed_archive]
        else:
            bundled = load_yaml_mapping(bundled_config_path())
            bundled_archives = bundled.get("archives", {})
            if needed_archive in bundled_archives:
                data.setdefault("archives", {})[needed_archive] = bundled_archives[needed_archive]
            else:
                raise ConfigError(f"Cannot adopt '{name}': archive '{needed_archive}' is unknown.")

    if alias:
        aliases = data.setdefault("catalogue_aliases", {})
        if not isinstance(aliases, dict):
            raise ConfigError(f"'catalogue_aliases' in {dest} must be a mapping.")
        aliases[alias.lower()] = name

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temporary = Path(fh.name)
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, dest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dest
