import pytest
from xmatch.cli import main

def test_no_dangling_whitespace(capsys):
    assert main(["--describe", "bogus_catalogue_with_no_close_matches"]) == 1
    err = capsys.readouterr().err
    assert err == "  Catalogue 'bogus_catalogue_with_no_close_matches' not found.\n"

def test_no_trailing_whitespace(capsys):
    assert main(["bogus1", "bogus2"]) == 1
    err = capsys.readouterr().err
    # The output should not contain "  \n" or " \n"
    assert " \n" not in err
    assert "  \n" not in err
