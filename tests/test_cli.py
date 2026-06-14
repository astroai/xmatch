import polars as pl
import pytest

from xmatch.cli import main


@pytest.fixture
def local_files(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.parquet"
    pl.DataFrame({"id": [1, 2], "ra": [10.0, 20.0], "dec": [5.0, 6.0]}).write_csv(a)
    pl.DataFrame({"id": [1, 2], "ra": [10.00005, 20.00005], "dec": [5.0, 6.0]}).write_parquet(b)
    return a, b


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "xmatch" in capsys.readouterr().out


def test_cli_requires_two_catalogues(capsys):
    assert main([]) == 2
    assert "two catalogues" in capsys.readouterr().err


def test_cli_list(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Available catalogues:" in out
    assert "gaia_esa" in out


def test_cli_describe(capsys):
    assert main(["--describe", "gaia_esa"]) == 0
    out = capsys.readouterr().out
    assert "Catalogue: gaia_esa" in out


def test_cli_describe_unknown(capsys):
    assert main(["--describe", "unknown_catalogue"]) == 1
    err = capsys.readouterr().err
    assert "Catalogue 'unknown_catalogue' not found." in err


def test_cli_match_to_stdout(local_files, capsys):
    a, b = local_files
    assert main([str(a), str(b), "-r", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "sep_arcsec" in out
    assert out.count("\n") >= 3  # header + 2 rows


def test_cli_match_to_file(local_files, tmp_path):
    a, b = local_files
    out = tmp_path / "out.parquet"
    assert main([str(a), str(b), "-r", "1.0", "-o", str(out)]) == 0
    assert pl.read_parquet(out).height == 2


def test_cli_unknown_catalogue_returns_error(capsys):
    assert main(["nope_not_real", "also_not_real"]) == 1
    assert "Error" in capsys.readouterr().err
