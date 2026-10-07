"""TAP ownership checks use the real PyVO client over the local UWS protocol."""

from unittest.mock import patch

import pytest
import pyvo

from xmatch import tap
from xmatch.exceptions import TapError

from . import tap_fake
from .tap_fake import FakeTAPServer, make_rows


def test_tap_success_deletes_only_its_own_job():
    server = FakeTAPServer(make_rows(3))
    try:
        service = pyvo.dal.TAPService(server.url)
        unrelated = service.submit_job("SELECT id FROM t ORDER BY id LIMIT 1")
        try:
            result = tap.execute_tap_query(service, "SELECT id FROM t ORDER BY id LIMIT 3")
            assert list(result["id"]) == [0, 1, 2]
            assert set(server.jobs) == {unrelated.job_id}
        finally:
            unrelated.delete()
    finally:
        server.shutdown()


def test_tap_server_error_deletes_its_job():
    server = FakeTAPServer(make_rows(3))
    try:
        service = pyvo.dal.TAPService(server.url)
        with (
            patch.object(tap_fake, "_JOB_XML", tap_fake._JOB_XML.replace("COMPLETED", "ERROR")),
            pytest.raises(TapError, match="did not complete: ERROR"),
        ):
            tap.execute_tap_query(service, "SELECT id FROM t")
        assert server.jobs == {}
    finally:
        server.shutdown()


@pytest.mark.parametrize("error", [RuntimeError("progress failed"), KeyboardInterrupt()])
def test_tap_callback_failure_or_cancel_deletes_its_job(error):
    server = FakeTAPServer(make_rows(3))
    try:
        service = pyvo.dal.TAPService(server.url)

        def interrupt(_status):
            raise error

        with pytest.raises(TapError if isinstance(error, Exception) else KeyboardInterrupt):
            tap.execute_tap_query(service, "SELECT id FROM t", progress_cb=interrupt)
        assert server.jobs == {}
    finally:
        server.shutdown()
