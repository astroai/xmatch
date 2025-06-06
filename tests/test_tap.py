import pytest
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from unittest.mock import patch, MagicMock, call
import pyvo

# Adjust import based on project structure
from xmatch.tap import (
    TapError,
    get_tap_service,
    execute_tap_query
)

# --- Fixtures for TAP tests ---

@pytest.fixture
def sample_query_df():
    """Sample DataFrame for building spatial queries."""
    return pd.DataFrame({
        'ra': [10.0, 20.0],
        'dec': [5.0, -5.0]
    })

@pytest.fixture
def mock_tap_service():
    """Fixture for a mocked pyvo.dal.TAPService."""
    service = MagicMock(spec=pyvo.dal.TAPService)
    service.baseurl = "mock://tap"
    # Mock the search method for synchronous queries
    mock_result = MagicMock(spec=pyvo.dal.DALResults)
    mock_table = Table({'a': [1, 2], 'b': [3, 4]})
    mock_result.to_table.return_value = mock_table
    service.search.return_value = mock_result

    # Mock async job methods
    mock_job = MagicMock(spec=pyvo.dal.AsyncTAPJob)
    mock_job.phase = 'COMPLETED'
    mock_job.run.return_value = None
    mock_job.wait.return_value = None
    mock_job.fetch_result.return_value = mock_result # Reuse the sync result mock
    service.run_async.return_value = mock_job

    return service

# --- Tests for get_tap_service ---

@patch('pyvo.dal.TAPService')
def test_get_tap_service_new_connection(MockTAPService, mock_tap_service):
    """Test creating a new TAP service connection."""
    MockTAPService.return_value = mock_tap_service
    url = "http://new.tap/service"

    # Clear cache before test
    from xmatch import tap
    tap._tap_service_cache = {}

    service = get_tap_service(url)

    MockTAPService.assert_called_once_with(url)
    assert service == mock_tap_service
    assert url in tap._tap_service_cache

@patch('pyvo.dal.TAPService')
def test_get_tap_service_cached_connection(MockTAPService, mock_tap_service):
    """Test retrieving a cached TAP service connection."""
    MockTAPService.return_value = mock_tap_service
    url = "http://cached.tap/service"

    # Clear cache and add the service
    from xmatch import tap
    tap._tap_service_cache = {url: mock_tap_service}

    service = get_tap_service(url)

    # TAPService should NOT be called again
    MockTAPService.assert_not_called()
    assert service == mock_tap_service

@patch('pyvo.dal.TAPService')
def test_get_tap_service_connection_error(MockTAPService):
    """Test handling connection error."""
    MockTAPService.side_effect = pyvo.dal.DALServiceError("Connection failed")
    url = "http://fail.tap/service"

    # Clear cache
    from xmatch import tap
    tap._tap_service_cache = {}

    with pytest.raises(TapError, match="Failed to connect"):        get_tap_service(url)

@patch('pyvo.dal.TAPService')
def test_get_tap_service_cache_key_auth(MockTAPService, mock_tap_service):
    """Test that auth parameters change the cache key."""
    MockTAPService.return_value = mock_tap_service
    url = "http://auth.tap/service"

    # Clear cache
    from xmatch import tap
    tap._tap_service_cache = {}

    service1 = get_tap_service(url, user="user1", password="pass1")
    service2 = get_tap_service(url, user="user2", password="pass2")
    service3 = get_tap_service(url, user="user1", password="pass1") # Same as first

    # Should be two distinct calls to TAPService
    assert MockTAPService.call_count == 2
    calls = MockTAPService.call_args_list
    assert calls[0] == call(url, user="user1", password="pass1")
    assert calls[1] == call(url, user="user2", password="pass2")

    # Check cache keys (internal detail, but useful for verification)
    assert len(tap._tap_service_cache) == 2
    key1 = f"{url}_password_pass1_user_user1"
    key2 = f"{url}_password_pass2_user_user2"
    assert key1 in tap._tap_service_cache
    assert key2 in tap._tap_service_cache

    # Check returned services
    assert service1 == mock_tap_service
    assert service2 == mock_tap_service
    assert service3 == service1 # Should retrieve from cache

# --- Tests for execute_tap_query ---

def test_execute_tap_query_sync_success(mock_tap_service):
    """Test successful synchronous TAP query execution."""
    mock_tap_service.search.return_value = MagicMock(to_table=MagicMock(return_value=Table([{'a': 1}]))) # Mock successful result
    query = "SELECT * FROM table"
    result = execute_tap_query(mock_tap_service, query)
    mock_tap_service.search.assert_called_once_with(query=query)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result['a'].iloc[0] == 1

def test_execute_tap_query_sync_error(mock_tap_service):
    """Test error handling during synchronous TAP query execution."""
    mock_tap_service.search.side_effect = pyvo.dal.DALQueryError("Sync Query Error")
    query = "SELECT * FROM table"
    with pytest.raises(TapError, match="Sync Query Error"):
        execute_tap_query(mock_tap_service, query, max_retries=1)
    assert mock_tap_service.search.call_count == 1 # Should try once

# --- Tests for execute_tap_query (formerly async, now unified) ---

def test_execute_tap_query_async_success(mock_tap_service):
    """Test successful TAP query (previously async mode)."""
    # Mock the async job pattern
    mock_job = MagicMock()
    mock_job.phase = 'COMPLETED'
    mock_job.fetch_result.return_value = MagicMock(to_table=MagicMock(return_value=Table([{'b': 2}]))) 
    mock_tap_service.submit_job.return_value = mock_job

    query = "SELECT * FROM table"
    # The function now handles sync/async internally based on service capabilities/behavior
    # We might need to adjust mocks if internal logic changed significantly
    result = execute_tap_query(mock_tap_service, query)

    # Assertion might need adjustment depending on how sync/async is now handled.
    # Assuming it might still use submit_job/fetch_result for potentially long queries:
    mock_tap_service.submit_job.assert_called_once_with(query=query)
    mock_job.run.assert_called_once()
    mock_job.wait.assert_called_once()
    mock_job.fetch_result.assert_called_once()
    assert isinstance(result, pd.DataFrame)
    assert result['b'].iloc[0] == 2

def test_execute_tap_query_async_retry_and_success(mock_tap_service):
    """Test retry mechanism for TAP query (previously async mode)."""
    mock_job_error = MagicMock()
    mock_job_error.phase = 'ERROR'
    mock_job_success = MagicMock()
    mock_job_success.phase = 'COMPLETED'
    mock_job_success.fetch_result.return_value = MagicMock(to_table=MagicMock(return_value=Table([{'c': 3}]))) 

    # Fail first time, succeed second time
    mock_tap_service.submit_job.side_effect = [mock_job_error, mock_job_success]

    query = "SELECT * FROM table"
    result = execute_tap_query(mock_tap_service, query, max_retries=2)

    assert mock_tap_service.submit_job.call_count == 2
    # Check interactions with the successful job
    mock_job_success.run.assert_called_once()
    mock_job_success.wait.assert_called_once()
    mock_job_success.fetch_result.assert_called_once()
    assert isinstance(result, pd.DataFrame)
    assert result['c'].iloc[0] == 3

def test_execute_tap_query_async_timeout(mock_tap_service):
    """Test timeout during TAP query (previously async mode)."""
    mock_job = MagicMock()
    mock_job.wait.side_effect = TimeoutError("Job timed out")
    mock_tap_service.submit_job.return_value = mock_job

    query = "SELECT * FROM table"
    with pytest.raises(TapError, match="timed out"):
        execute_tap_query(mock_tap_service, query, max_retries=1)

    mock_tap_service.submit_job.assert_called_once()
    mock_job.run.assert_called_once()
    mock_job.wait.assert_called_once()
    mock_job.delete.assert_called_once() # Check if job is deleted on timeout
    mock_job.fetch_result.assert_not_called()

def test_execute_tap_query_async_error_phase(mock_tap_service):
    """Test handling of ERROR phase during TAP query (previously async mode)."""
    mock_job = MagicMock()
    mock_job.phase = 'ERROR'
    mock_tap_service.submit_job.return_value = mock_job

    query = "SELECT * FROM table"
    with pytest.raises(TapError, match="TAP job failed with phase ERROR"):
        execute_tap_query(mock_tap_service, query, max_retries=1)

    mock_tap_service.submit_job.assert_called_once()
    mock_job.run.assert_called_once()
    mock_job.wait.assert_called_once()
    mock_job.delete.assert_called_once() # Check if job is deleted on error
    mock_job.fetch_result.assert_not_called()

# TODO: Add tests for higher-level functions like perform_tap_id_join, perform_tap_spatial_join
# These will involve mocking get_tap_service and execute_tap_query.