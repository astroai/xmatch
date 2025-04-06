import pytest
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from unittest.mock import patch, MagicMock, call
import pyvo

# Adjust import based on project structure
from src.xmatch.tap import (
    build_spatial_query,
    perform_local_id_join_df,
    perform_local_spatial_join_df,
    ensure_j2000,
    get_tap_service,
    execute_tap_query,
    TapError
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

def test_build_spatial_query_basic(sample_query_df):
    """Test basic spatial query construction."""
    query = build_spatial_query(
        df=sample_query_df,
        tap_table="gaia.dr3",
        ra="ra",
        dec="dec",
        radius=1.5, # arcsec
        columns=["source_id", "g_mag"]
    )
    assert "SELECT source_id, g_mag FROM gaia.dr3" in query
    assert "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 10.0, 5.0, 0.00041666))" in query
    assert "OR 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 20.0, -5.0, 0.00041666))" in query
    # Check radius conversion (1.5 arcsec = 1.5/3600 deg)
    assert np.isclose(1.5 / 3600, 0.00041666, atol=1e-8)

def test_build_spatial_query_all_columns(sample_query_df):
    """Test query when all columns are requested."""
    query = build_spatial_query(
        df=sample_query_df,
        tap_table="vizier.cat",
        ra="ra",
        dec="dec",
        radius=2.0,
        columns=None # Request all columns
    )
    assert "SELECT * FROM vizier.cat" in query
    assert "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 10.0, 5.0, 0.00055555))" in query
    assert np.isclose(2.0 / 3600, 0.00055555, atol=1e-8)

def test_build_spatial_query_with_schema(sample_query_df):
    """Test query construction with an explicit schema."""
    query = build_spatial_query(
        df=sample_query_df,
        tap_table="gaia_source",
        ra="ra",
        dec="dec",
        radius=1.0,
        columns=["ra", "dec"],
        tap_schema="gaiadr3"
    )
    assert "SELECT ra, dec FROM gaiadr3.gaia_source" in query
    assert "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 10.0, 5.0, 0.00027777))" in query
    assert np.isclose(1.0 / 3600, 0.00027777, atol=1e-8)

def test_build_spatial_query_empty_df():
    """Test query construction with an empty input DataFrame."""
    empty_df = pd.DataFrame({'ra': [], 'dec': []})
    query = build_spatial_query(
        df=empty_df,
        tap_table="any.table",
        ra="ra",
        dec="dec",
        radius=1.0,
        columns=["col1"]
    )
    # Should produce a query that returns no results but is syntactically valid
    assert "SELECT col1 FROM any.table WHERE 1=0" in query 

# --- Fixtures for local join tests ---

@pytest.fixture
def df1_local():
    """First DataFrame for local join tests."""
    return pd.DataFrame({
        'id1': [1, 2, 3, 4],
        'ra1': [10.0, 20.0, 30.0, 40.0],
        'dec1': [5.0, -5.0, 50.0, -50.0],
        'val1': ['a', 'b', 'c', 'd']
    })

@pytest.fixture
def df2_local_id():
    """Second DataFrame for local ID join tests."""
    return pd.DataFrame({
        'id2': [2, 4, 5, 6],
        'name': ['apple', 'banana', 'orange', 'grape'],
        'val2': [100, 200, 300, 400]
    })

@pytest.fixture
def df2_local_spatial():
    """Second DataFrame for local spatial join tests."""
    return pd.DataFrame({
        'id2': [10, 11, 12, 13],
        # RA/Dec slightly offset from df1_local for matching
        'ra2': [10.0001, 20.0002, 35.0, 40.0001],
        'dec2': [5.0001, -5.0002, 55.0, -50.0001],
        'val2': [1000, 2000, 3000, 4000]
    })

# --- Tests for perform_local_id_join_df ---

def test_perform_local_id_join_df_basic(df1_local, df2_local_id):
    """Test basic ID join between two DataFrames."""
    result_df = perform_local_id_join_df(
        input_df=df1_local,
        catalogue_2_df=df2_local_id,
        id_column_1='id1',
        id_column_2='id2',
        columns=None # Keep all columns from df2
    )
    assert len(result_df) == 2 # Should match on id=2 and id=4
    assert list(result_df.columns) == ['id1', 'ra1', 'dec1', 'val1', 'id2', 'name', 'val2']
    assert sorted(result_df['id1'].tolist()) == [2, 4]
    assert result_df[result_df['id1'] == 2]['name'].iloc[0] == 'apple'
    assert result_df[result_df['id1'] == 4]['val2'].iloc[0] == 200

def test_perform_local_id_join_df_select_cols(df1_local, df2_local_id):
    """Test ID join selecting specific columns from df2."""
    result_df = perform_local_id_join_df(
        input_df=df1_local,
        catalogue_2_df=df2_local_id,
        id_column_1='id1',
        id_column_2='id2',
        columns=['name'] # Only keep 'name' from df2
    )
    assert len(result_df) == 2
    assert list(result_df.columns) == ['id1', 'ra1', 'dec1', 'val1', 'id2', 'name']
    assert sorted(result_df['id1'].tolist()) == [2, 4]

def test_perform_local_id_join_df_no_match(df1_local):
    """Test ID join when there are no matching IDs."""
    df2_no_match = pd.DataFrame({'id2': [5, 6, 7], 'val2': [50, 60, 70]})
    result_df = perform_local_id_join_df(
        input_df=df1_local,
        catalogue_2_df=df2_no_match,
        id_column_1='id1',
        id_column_2='id2'
    )
    assert len(result_df) == 0

# --- Tests for perform_local_spatial_join_df ---

def test_perform_local_spatial_join_df_basic(df1_local, df2_local_spatial):
    """Test basic spatial join between two DataFrames."""
    radius_arcsec = 1.0
    result_df = perform_local_spatial_join_df(
        input_df=df1_local,
        catalogue_2_df=df2_local_spatial,
        ra='ra1',
        dec='dec1',
        radius=radius_arcsec,
        columns=None # Keep all columns from df2
    )
    # Should match id1=1 -> id2=10, id1=2 -> id2=11, id1=4 -> id2=13
    assert len(result_df) == 3
    expected_cols = ['id1', 'ra1', 'dec1', 'val1', 'idx', 'sep_arcsec', 'id2', 'ra2', 'dec2', 'val2']
    assert sorted(list(result_df.columns)) == sorted(expected_cols)
    assert sorted(result_df['id1'].tolist()) == [1, 2, 4]
    assert result_df[result_df['id1'] == 1]['id2'].iloc[0] == 10
    assert result_df[result_df['id1'] == 2]['id2'].iloc[0] == 11
    assert result_df[result_df['id1'] == 4]['id2'].iloc[0] == 13
    assert np.all(result_df['sep_arcsec'] <= radius_arcsec)

def test_perform_local_spatial_join_df_select_cols(df1_local, df2_local_spatial):
    """Test spatial join selecting specific columns from df2."""
    radius_arcsec = 1.0
    result_df = perform_local_spatial_join_df(
        input_df=df1_local,
        catalogue_2_df=df2_local_spatial,
        ra='ra1',
        dec='dec1',
        radius=radius_arcsec,
        columns=['id2', 'val2'] # Select specific columns
    )
    assert len(result_df) == 3
    expected_cols = ['id1', 'ra1', 'dec1', 'val1', 'idx', 'sep_arcsec', 'id2', 'val2']
    assert sorted(list(result_df.columns)) == sorted(expected_cols)
    assert sorted(result_df['id1'].tolist()) == [1, 2, 4]

def test_perform_local_spatial_join_df_no_match(df1_local):
    """Test spatial join when no coordinates are within radius."""
    df2_far = pd.DataFrame({
        'id2': [20, 21],
        'ra2': [180.0, 190.0],
        'dec2': [0.0, 10.0],
        'val2': [5000, 6000]
    })
    result_df = perform_local_spatial_join_df(
        input_df=df1_local,
        catalogue_2_df=df2_far,
        ra='ra1',
        dec='dec1',
        radius=1.0,
    )
    assert len(result_df) == 0 

# --- Tests for get_tap_service ---

@patch('pyvo.dal.TAPService')
def test_get_tap_service_new_connection(MockTAPService, mock_tap_service):
    """Test creating a new TAP service connection."""
    MockTAPService.return_value = mock_tap_service
    url = "http://new.tap/service"

    # Clear cache before test
    from src.xmatch import tap
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
    from src.xmatch import tap
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
    from src.xmatch import tap
    tap._tap_service_cache = {}

    with pytest.raises(TapError, match="Failed to connect"):        get_tap_service(url)

@patch('pyvo.dal.TAPService')
def test_get_tap_service_cache_key_auth(MockTAPService, mock_tap_service):
    """Test that auth parameters change the cache key."""
    MockTAPService.return_value = mock_tap_service
    url = "http://auth.tap/service"

    # Clear cache
    from src.xmatch import tap
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
    """Test successful synchronous TAP query."""
    query = "SELECT a, b FROM table"
    result = execute_tap_query(mock_tap_service, query, is_async=False)

    mock_tap_service.search.assert_called_once_with(query=query, language='ADQL')
    mock_tap_service.run_async.assert_not_called()
    assert isinstance(result, Table)
    assert list(result.colnames) == ['a', 'b']
    assert len(result) == 2

def test_execute_tap_query_sync_error(mock_tap_service):
    """Test handling error during synchronous TAP query."""
    query = "SELECT * FROM bad_table"
    mock_tap_service.search.side_effect = pyvo.dal.DALQueryError("Table not found")

    with pytest.raises(TapError, match="TAP query failed"):
        execute_tap_query(mock_tap_service, query, is_async=False)

    mock_tap_service.search.assert_called_once_with(query=query, language='ADQL')

@patch('time.sleep', return_value=None) # Mock time.sleep to speed up test
def test_execute_tap_query_async_success(mock_sleep, mock_tap_service):
    """Test successful asynchronous TAP query."""
    query = "SELECT a, b FROM large_table"
    result = execute_tap_query(mock_tap_service, query, is_async=True, retry_delay=1, timeout=10)

    mock_tap_service.run_async.assert_called_once_with(query=query, language='ADQL')
    # Check wait and fetch_result were called on the job object
    job = mock_tap_service.run_async.return_value
    job.wait.assert_called_once_with(timeout=10)
    job.fetch_result.assert_called_once()
    mock_tap_service.search.assert_not_called()

    assert isinstance(result, Table)
    assert list(result.colnames) == ['a', 'b']

@patch('time.sleep', return_value=None) # Mock time.sleep
def test_execute_tap_query_async_retry_and_success(mock_sleep, mock_tap_service):
    """Test async query that requires retrying due to PENDING/EXECUTING phases."""
    query = "SELECT * FROM complex_table"
    job = mock_tap_service.run_async.return_value

    # Simulate job phases: PENDING -> EXECUTING -> COMPLETED
    job.phase_sequence = ['PENDING', 'EXECUTING', 'COMPLETED']
    def phase_side_effect(*args, **kwargs):
        # Pop from the start of the sequence
        current_phase = job.phase_sequence.pop(0)
        job.phase = current_phase
        if current_phase != 'COMPLETED':
            # wait should raise TimeoutError if not completed within polling interval
            # but execute_tap_query catches it and retries
            raise TimeoutError("Still running")
        # Only succeed on the last call

    job.wait.side_effect = phase_side_effect

    result = execute_tap_query(mock_tap_service, query, is_async=True, retry_delay=1, timeout=10)

    mock_tap_service.run_async.assert_called_once_with(query=query, language='ADQL')
    assert job.wait.call_count == 3 # Called for PENDING, EXECUTING, COMPLETED
    job.fetch_result.assert_called_once()
    assert result is not None

@patch('time.sleep', return_value=None)
def test_execute_tap_query_async_timeout(mock_sleep, mock_tap_service):
    """Test async query that times out."""
    query = "SELECT * FROM very_large_table"
    job = mock_tap_service.run_async.return_value
    job.phase = 'EXECUTING' # Stays in executing phase
    # wait keeps raising TimeoutError
    job.wait.side_effect = TimeoutError("Job taking too long")

    with pytest.raises(TapError, match="timed out"):
        execute_tap_query(mock_tap_service, query, is_async=True, retry_delay=1, timeout=5)

    mock_tap_service.run_async.assert_called_once_with(query=query, language='ADQL')
    # Check wait was called multiple times based on timeout/delay
    assert job.wait.call_count > 1
    job.fetch_result.assert_not_called()

@patch('time.sleep', return_value=None)
def test_execute_tap_query_async_error_phase(mock_sleep, mock_tap_service):
    """Test async query that fails with an ERROR phase."""
    query = "SELECT * FROM error_table"
    job = mock_tap_service.run_async.return_value
    job.phase = 'ERROR' # Job fails immediately or during execution
    # Mock wait to reflect the error state immediately
    job.wait.side_effect = lambda *args, **kwargs: None # No timeout if error

    with pytest.raises(TapError, match="TAP job failed with phase ERROR"):
        execute_tap_query(mock_tap_service, query, is_async=True, retry_delay=1, timeout=10)

    mock_tap_service.run_async.assert_called_once_with(query=query, language='ADQL')
    # Wait might be called once before checking phase
    assert job.wait.call_count <= 1
    job.fetch_result.assert_not_called()

# TODO: Add tests for higher-level functions like perform_tap_id_join, perform_tap_spatial_join
# These will involve mocking get_tap_service and execute_tap_query. 