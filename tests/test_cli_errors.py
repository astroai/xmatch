from io import StringIO
from unittest.mock import patch

from xmatch.cli import main


def test_missing_catalogues_prints_to_stderr():
    stderr = StringIO()
    stdout = StringIO()

    with patch("sys.stderr", stderr), patch("sys.stdout", stdout):
        # Pass no arguments, which will trigger the missing catalogues error
        ret = main(["xmatch"])

        err_out = stderr.getvalue()
        std_out = stdout.getvalue()

        assert ret == 1
        assert "Error: Two catalogues are required for matching." in err_out
        assert "Use 'xmatch --help'" in err_out
        assert "Error:" not in std_out

def test_catalogue_not_found_prints_to_stderr():
    stderr = StringIO()
    stdout = StringIO()

    with patch("sys.stderr", stderr), patch("sys.stdout", stdout):
        ret = main(["xmatch", "--describe", "nonexistent_catalogue"])

        err_out = stderr.getvalue()
        std_out = stdout.getvalue()

        assert ret == 0 # Current behavior for list/describe commands
        assert "Catalogue 'nonexistent_catalogue' not found." in err_out
        assert "not found" not in std_out
