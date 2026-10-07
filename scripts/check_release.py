#!/usr/bin/env python
"""Check built release contents against the current source tree (stdlib only)."""

import argparse
import re
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check_tag(tag: str, version: str) -> None:
    expected = f"v{version}"
    _require(tag == expected, f"release tag {tag!r} does not match project version {expected!r}")


def package_files(root: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("*.py", "*.yaml", "fixtures/*.json")
        for path in (root / "src/xmatch").glob(pattern)
    )


def required_sdist_files(
    root: Path, package: list[Path], project: dict | None = None
) -> list[Path]:
    required = [
        root / filename
        for filename in (
            "pyproject.toml",
            "MANIFEST.in",
            "pixi.toml",
            "pixi.lock",
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "LICENSE",
            ".gitignore",
            ".pre-commit-config.yaml",
        )
    ]
    required += package
    for directory, pattern in (
        ("tests", "*.py"),
        ("tests/inputs", "*.csv"),
        ("docs", "*.md"),
        ("docs", "*.csv"),
        ("docs", "*.png"),
        ("scripts", "*.py"),
        ("scripts", "*.sh"),
    ):
        required.extend(sorted((root / directory).glob(pattern)))
    if project is not None:
        for pattern in project.get("license-files", []):
            required.extend(sorted(root.glob(pattern)))
    return sorted(set(required))


def check_wheel(wheel_path: Path, root: Path, project: dict, package: list[Path]) -> None:
    expected_package = {path.relative_to(root / "src").as_posix() for path in package}
    expected_license_files = {
        path.relative_to(root).as_posix()
        for pattern in project.get("license-files", [])
        for path in root.glob(pattern)
    }

    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        _require(len(names) == len(set(names)), "wheel contains duplicate member names")
        actual_package = {
            name for name in names if name.startswith("xmatch/") and not name.endswith("/")
        }
        _require(
            actual_package == expected_package,
            f"wheel package contents differ: missing={sorted(expected_package - actual_package)}, "
            f"extra={sorted(actual_package - expected_package)}",
        )
        for path in package:
            member = path.relative_to(root / "src").as_posix()
            _require(wheel.read(member) == path.read_bytes(), f"wheel package file differs: {path}")
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        _require(
            len(metadata_paths) == 1,
            f"expected one wheel METADATA, found {metadata_paths}",
        )
        dist_info = metadata_paths[0].rsplit("/", 1)[0]
        allowed_names = (
            expected_package
            | {
                f"{dist_info}/{filename}"
                for filename in ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt")
            }
            | {f"{dist_info}/licenses/{relative}" for relative in expected_license_files}
        )
        unexpected_names = {
            name for name in names if not name.endswith("/") and name not in allowed_names
        }
        _require(
            not unexpected_names,
            f"wheel contents differ: extra={sorted(unexpected_names)}",
        )
        metadata = Parser().parsestr(wheel.read(metadata_paths[0]).decode())
        wheel_name = re.sub(r"[-_.]+", "-", metadata.get("Name", "")).lower()
        project_name = re.sub(r"[-_.]+", "-", project["name"]).lower()
        _require(wheel_name == project_name, f"wheel Name differs: {metadata.get('Name')!r}")
        _require(metadata.get("Version") == project["version"], "wheel Version differs")
        _require(
            metadata.get("License-Expression") == project["license"],
            "wheel License-Expression differs",
        )
        _require(
            metadata.get("Requires-Python") == project["requires-python"],
            "wheel Requires-Python differs",
        )
        extras = project.get("optional-dependencies", {})
        _require(
            set(metadata.get_all("Provides-Extra", [])) == set(extras),
            "wheel extras differ from pyproject.toml",
        )
        expected_dependencies = project["dependencies"] + [
            f'{requirement}; extra == "{extra}"'
            for extra, requirements in extras.items()
            for requirement in requirements
        ]

        def normalize(requirement: str) -> str:
            return re.sub(r"\s+", "", requirement).replace("'", '"')

        _require(
            {normalize(requirement) for requirement in metadata.get_all("Requires-Dist", [])}
            == {normalize(requirement) for requirement in expected_dependencies},
            "wheel dependencies differ from pyproject.toml",
        )

        declared_license_files = set(metadata.get_all("License-File", []))
        _require(
            declared_license_files == expected_license_files,
            f"wheel License-File metadata differs: expected={sorted(expected_license_files)}, "
            f"found={sorted(declared_license_files)}",
        )
        license_prefix = f"{metadata_paths[0].rsplit('/', 1)[0]}/licenses/"
        actual_license_members = {
            name.removeprefix(license_prefix)
            for name in names
            if name.startswith(license_prefix) and not name.endswith("/")
        }
        _require(
            actual_license_members == expected_license_files,
            f"wheel license files differ: missing={sorted(expected_license_files - actual_license_members)}, "
            f"extra={sorted(actual_license_members - expected_license_files)}",
        )
        for relative in expected_license_files:
            member = f"{license_prefix}{relative}"
            _require(
                wheel.read(member) == (root / relative).read_bytes(),
                f"wheel license file differs from source: {relative}",
            )


def _is_generated_sdist_metadata(name: str, archive_root: str, project_name: str) -> bool:
    relative = PurePosixPath(name).relative_to(archive_root)
    parts = relative.parts
    egg_info = re.sub(r"[-.]+", "_", project_name).lower() + ".egg-info"
    return relative.as_posix() in {"PKG-INFO", "setup.cfg"} or (
        len(parts) == 3
        and parts[0] == "src"
        and parts[1] == egg_info
        and parts[2]
        in {
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        }
    )


def check_sdist(sdist_path: Path, root: Path, required: list[Path], project_name: str) -> None:
    with tarfile.open(sdist_path) as archive:
        members = archive.getmembers()
        roots = {member.name.split("/", 1)[0] for member in members}
        _require(len(roots) == 1, f"expected one source archive root, found {roots}")
        archive_root = roots.pop()
        file_members = [member for member in members if not member.isdir()]
        member_names = [member.name for member in file_members]
        _require(
            len(member_names) == len(set(member_names)),
            "source archive contains duplicate member names",
        )
        _require(
            all(member.isfile() for member in file_members),
            "source archive contains a non-regular file member",
        )
        expected_names = {
            f"{archive_root}/{path.relative_to(root).as_posix()}" for path in required
        }
        unexpected_names = {
            name
            for name in member_names
            if name not in expected_names
            and not _is_generated_sdist_metadata(name, archive_root, project_name)
        }
        _require(
            not unexpected_names,
            f"source archive contents differ: extra={sorted(unexpected_names)}",
        )
        for path in required:
            name = f"{archive_root}/{path.relative_to(root).as_posix()}"
            member_stream = archive.extractfile(name)
            _require(member_stream is not None, f"source archive is missing {path}")
            with member_stream:
                _require(member_stream.read() == path.read_bytes(), f"source file differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--tag", help="require this release tag to match the project version")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    if args.tag is not None:
        check_tag(args.tag, project["version"])
    artifacts = list(args.dist.glob("*.whl"))
    _require(len(artifacts) == 1, f"expected one wheel in {args.dist}, found {artifacts}")
    wheel_path = artifacts[0]
    artifacts = list(args.dist.glob("*.tar.gz"))
    _require(
        len(artifacts) == 1,
        f"expected one source archive in {args.dist}, found {artifacts}",
    )
    sdist_path = artifacts[0]
    package = package_files(root)
    required = required_sdist_files(root, package, project)
    check_wheel(wheel_path, root, project, package)
    check_sdist(sdist_path, root, required, project["name"])
    print(f"Release contents verified: {len(package)} wheel files, {len(required)} source files")


if __name__ == "__main__":
    main()
