# xmatch

A flexible tool for cross-matching astronomical catalogues using various methods (STILTS, TAP, CDS XMatch).

## Features

- Cross-match local files (Parquet, FITS, CSV) or remote catalogues defined in configuration.
- Supports multiple matching backends:
  - Local matching via STILTS `tmatch2` (sky, skyerr, skyellipse).
  - Remote TAP service joins (fixed radius).
  - Remote CDS XMatch service (requires `astroquery`).
- Strategy selection based on input types (local/remote) and service capabilities.
- Handles epoch propagation between catalogues with proper motion (e.g., Gaia J2016.0 vs J2000.0) for local/downloaded matches.
- Spatial chunking for large remote catalogues using HEALPix (requires `astropy-healpix`).
- Configuration via YAML files (`catalogues.yaml`, `auth.yaml`).
- Command-line interface and Python API.

## Installation

Requires Java (for STILTS) and Python >= 3.10.

```bash
# Install from PyPI (if published)
# pip install xmatch

# Install from source
git clone https://github.com/sfabbro/xmatch.git
cd xmatch
pip install .

# For development
pip install -e ".[dev]"
```

**STILTS Setup:** Ensure the `stilts` command is available in your system PATH, or set the `STILTS_JAR` environment variable, or provide the path via the `stilts_cmd_base` configuration setting in `catalogues.yaml` or Python API.

## Usage

### Command Line Interface (`xmatch`)

```bash
# Basic: Match two configured catalogues (e.g., gaia_esa vs delve_noao)
xmatch gaia_esa delve_noao --output gaia_delve_match.parquet --radius 1.5

# Match a local file against a configured remote catalogue
xmatch /path/to/my_sources.csv gaia_esa --output my_gaia_match.parquet --radius 2.0

# Use CDS XMatch service (if remote catalogue configured for it)
xmatch /path/to/my_sources.csv galex_cds --output my_galex_match.parquet --radius 3.0

# Specify matcher explicitly (for local/download strategies)
xmatch gaia_esa ukidsslas_noao --output gaia_ukidss_skyerr.parquet --matcher skyerr --max-error 5.0

# Run remote spatial chunking (requires ra, dec, radius)
xmatch gaia_esa vhs_cds --output gaia_vhs_chunked.parquet --radius 1.0 --ra 150.1 --dec 2.5 --strategy remote_spatial_chunked_match --nside 64

# List available catalogues
xmatch --list-catalogues

# Describe a catalogue
xmatch --describe gaia_esa

# See all options
xmatch --help
```

### Python API

```python
from xmatch import CrossMatch

# Initialize with default config
cm = CrossMatch()

# Perform cross-match (local file vs configured catalogue)
result_df = cm.crossmatch(
    catalogue_1_input='my_local_sources.parquet',
    catalogue_2_input='gaia_esa',
    output_file='local_vs_gaia.parquet', # Optional: save directly
    radius_arcsec=1.5,
    matcher='skyerr', # Optional: suggest matcher for local/download
    max_error=5.0    # Optional: separation for skyerr/skyellipse
)

# Perform spatial chunked remote match
result_df_chunked = cm.crossmatch(
    catalogue_1_input='gaia_esa',
    catalogue_2_input='vhs_cds',
    strategy='remote_spatial_chunked_match', # Force strategy
    ra=150.1, # Required for spatial chunking
    dec=2.5,  # Required for spatial chunking
    radius_arcsec=1.0, # Defines area for chunking
    nside=64 # Optional: HEALPix nside for chunking
)

print(result_df_chunked.head())
```

## Advanced Error Ellipse Matching

The xmatch library provides comprehensive support for error ellipse matching, including correlation between position errors. This is especially important for high-precision astrometric catalogs like Gaia.

### Error Ellipse Matching Types

1. **Simple position match** (`sky`): Fixed radius search, ignoring error information.
2. **Error ellipse without correlation** (`skyerr`): Uses RA/Dec errors but assumes no correlation.
3. **Full error ellipse with correlation** (`skyellipse`): Uses the complete error ellipse including correlation.

### Command Line Examples

```bash
# Match with explicit skyerr matcher (using error ellipses without correlation)
xmatch gaia_dr3 legacy_dr10 --output gaia_legacy_skyerr.parquet --matcher skyerr

# Match with skyellipse matcher (full error ellipse including correlation)
xmatch gaia_dr3 ps1 --output gaia_ps1_skyellipse.parquet --matcher skyellipse

# Match with explicit max-error scaling factor (N-sigma criterion)
xmatch gaia_dr3 des_dr2 --output gaia_des_skyerr.parquet --matcher skyerr --max-error 5.0
```

### Python API Examples

```python
from xmatch import CrossMatch

cm = CrossMatch()

# Example 1: Match with skyerr when both catalogs have position errors but no correlation
result_df = cm.crossmatch(
    catalogue_1_input='gaia_dr3',
    catalogue_2_input='legacy_dr10',
    matcher='skyerr',  # Use error ellipses without correlation
    max_error=3.0,     # 3-sigma matching criterion
    output_file='gaia_legacy_skyerr.parquet'
)

# Example 2: Match with skyellipse when both catalogs have position errors AND correlation
result_df = cm.crossmatch(
    catalogue_1_input='gaia_dr3',
    catalogue_2_input='catalog_with_correlations',
    matcher='skyellipse',  # Use full error ellipses with correlation
    max_error=5.0,         # 5-sigma matching criterion
    output_file='gaia_correlated_skyellipse.parquet'
)

# Example 3: Let xmatch automatically select the best matcher based on available error information
result_df = cm.crossmatch(
    catalogue_1_input='gaia_dr3',
    catalogue_2_input='wise_allwise',
    # No matcher specified - will auto-select based on available error columns in configs
    output_file='gaia_wise_auto.parquet'
)
```

### Catalog Configuration for Error Ellipse Matching

To use error ellipse matching, your catalog configurations should include:

```yaml
catalogues:
  gaia_dr3:
    description: "Gaia Data Release 3"
    # Required base columns
    ra_column: "ra"
    dec_column: "dec"
    # Error columns for skyerr/skyellipse matchers
    ra_err_column: "ra_error"
    dec_err_column: "dec_error"
    # Correlation column required for skyellipse matcher
    corr_column: "ra_dec_corr"
    # Specify error units if not in arcseconds
    pos_err_units: "mas"  # Options: "mas" (milliarcsec), "arcsec" (default), "deg" (degrees)
    # Optional: Default position error to use if column values are missing
    default_pos_error_arcsec: 0.1
    # Other catalog configuration...
```

### Error Handling Behavior

- If both catalogs have error columns, the `skyerr` matcher is used automatically.
- If both catalogs have error columns AND correlation columns, the `skyellipse` matcher is used automatically.
- If error information is incomplete or missing, the basic `sky` matcher is used as fallback.
- Different error units are automatically converted (e.g., milliarcseconds to arcseconds).
- When using a matcher that requires information not available in the catalogs, the system will validate and report errors.

## Configuration

- `catalogues.yaml`: Defines data archives (TAP/CDS services) and specific catalogues (access details, columns, errors, epoch). See the file for structure and examples.
- `auth.yaml`: Stores credentials for authenticated services (e.g., NOIRLab Data Lab TAP). Uses the `keyring` library for secure storage. Create this file manually if needed (see `auth.py` for expected format).

## License

This project is licensed under the MIT License.
