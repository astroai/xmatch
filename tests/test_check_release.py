"""Offline tests for release archive integrity checks."""

import io
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.check_release import (
    check_sdist,
    check_tag,
    check_wheel,
    package_files,
    required_sdist_files,
)

ROOT = Path(__file__).resolve().parents[1]


def _project():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _wheel_path(tmp_path):
    project = _project()
    return tmp_path / f"{project['name']}-{project['version']}-py3-none-any.whl"


def _sdist_path(tmp_path):
    project = _project()
    return tmp_path / f"{project['name']}-{project['version']}.tar.gz"


def _egg_info_dir(project):
    return f"{re.sub(r'[-.]+', '_', project['name']).lower()}.egg-info"


def test_release_tag_matches_project_version():
    version = _project()["version"]
    check_tag(f"v{version}", version)


@pytest.mark.parametrize("tag", ["0.5.0", "v0.5.0", "v0.5.1-rc1"])
def test_release_tag_rejects_other_versions(tag):
    version = _project()["version"]
    with pytest.raises(SystemExit, match="does not match project version"):
        check_tag(tag, version)


def test_release_checker_tracks_import_package_name():
    package = package_files(ROOT)
    package_root = ROOT / "src" / "xmatcher"

    assert package_root / "__init__.py" in package
    assert all(path.is_relative_to(package_root) for path in package)


def _wheel(path, *, extra=(), license_bytes=None, include_license=True, corrupt=None):
    project = _project()
    package = package_files(ROOT)
    dist_info = f"{project['name']}-{project['version']}.dist-info"
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"License-Expression: {project['license']}",
        f"Requires-Python: {project['requires-python']}",
    ]
    license_paths = [
        candidate for pattern in project["license-files"] for candidate in ROOT.glob(pattern)
    ]
    lines.extend(
        f"License-File: {candidate.relative_to(ROOT).as_posix()}" for candidate in license_paths
    )
    for dependency in project["dependencies"]:
        lines.append(f"Requires-Dist: {dependency}")
    for extra_name, dependencies in project.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra_name}")
        lines.extend(
            f'Requires-Dist: {dependency}; extra == "{extra_name}"' for dependency in dependencies
        )

    with zipfile.ZipFile(path, "w") as wheel:
        for source in package:
            name = source.relative_to(ROOT / "src").as_posix()
            if name == corrupt:
                wheel.writestr(name, "corrupt\n")
            else:
                wheel.write(source, name)
        for name in extra:
            wheel.writestr(name, "unexpected = True\n")
        if include_license:
            for source in license_paths:
                data = source.read_bytes() if license_bytes is None else license_bytes
                wheel.writestr(f"{dist_info}/licenses/{source.relative_to(ROOT).as_posix()}", data)
        wheel.writestr(f"{dist_info}/METADATA", "\n".join(lines) + "\n")


def _sdist(path, *, extra=()):
    project = _project()
    package = package_files(ROOT)
    required = required_sdist_files(ROOT, package, project)
    archive_root = f"{project['name']}-{project['version']}"
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo(archive_root)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for source in required:
            archive.add(source, f"{archive_root}/{source.relative_to(ROOT).as_posix()}")
        for relative in extra:
            payload = b"unexpected\n"
            info = tarfile.TarInfo(f"{archive_root}/{relative}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        generated = tarfile.TarInfo(f"{archive_root}/src/{_egg_info_dir(project)}/SOURCES.txt")
        generated_data = b"generated metadata\n"
        generated.size = len(generated_data)
        archive.addfile(generated, io.BytesIO(generated_data))
    return required


def test_wheel_accepts_expected_package_and_license(tmp_path):
    path = _wheel_path(tmp_path)
    _wheel(path)
    check_wheel(path, ROOT, _project(), package_files(ROOT))


def test_wheel_rejects_unlisted_package_files(tmp_path):
    path = _wheel_path(tmp_path)
    _wheel(path, extra=(f"{_project()['name']}/obsolete.py",))
    with pytest.raises(SystemExit, match="wheel package contents differ"):
        check_wheel(path, ROOT, _project(), package_files(ROOT))


@pytest.mark.parametrize(
    ("include_license", "license_bytes"),
    [(False, None), (True, b"wrong license\n")],
)
def test_wheel_rejects_omitted_or_changed_license(tmp_path, include_license, license_bytes):
    path = _wheel_path(tmp_path)
    _wheel(path, license_bytes=license_bytes, include_license=include_license)
    with pytest.raises(SystemExit, match="license"):
        check_wheel(path, ROOT, _project(), package_files(ROOT))


@pytest.mark.parametrize(
    "extra",
    [
        "unexpected.txt",
        f"src/{_egg_info_dir(_project())}/unexpected.txt",
        f"src/{_egg_info_dir(_project())}/nested/SOURCES.txt",
    ],
)
def test_sdist_rejects_unlisted_source_payload(tmp_path, extra):
    path = _sdist_path(tmp_path)
    required = _sdist(path, extra=(extra,))
    with pytest.raises(SystemExit, match="source archive contents differ"):
        check_sdist(path, ROOT, required, _project()["name"])


@pytest.mark.parametrize(
    "extra",
    [
        "unexpected.py",
        "startup.pth",
        f"{_project()['name']}-{_project()['version']}.data/purelib/other.py",
    ],
)
def test_wheel_rejects_payload_outside_package(tmp_path, extra):
    path = _wheel_path(tmp_path)
    _wheel(path, extra=(extra,))
    with pytest.raises(SystemExit, match="wheel contents differ"):
        check_wheel(path, ROOT, _project(), package_files(ROOT))


def test_sdist_accepts_declared_sources_and_generated_metadata(tmp_path):
    path = _sdist_path(tmp_path)
    required = _sdist(path)
    check_sdist(path, ROOT, required, _project()["name"])


def test_wheel_integrity_checks_still_run_under_python_optimized_mode(tmp_path):
    path = _wheel_path(tmp_path)
    _wheel(path, corrupt=f"{_project()['name']}/association.py")

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from pathlib import Path; import tomllib; "
            "from scripts.check_release import check_wheel, package_files; "
            "root = Path.cwd(); project = tomllib.loads((root / 'pyproject.toml').read_text())['project']; "
            f"check_wheel(Path({str(path)!r}), root, project, package_files(root))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "wheel package file differs" in result.stderr
