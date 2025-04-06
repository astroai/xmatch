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

## Configuration

- `catalogues.yaml`: Defines data archives (TAP/CDS services) and specific catalogues (access details, columns, errors, epoch). See the file for structure and examples.
- `auth.yaml`: Stores credentials for authenticated services (e.g., NOIRLab Data Lab TAP). Uses the `keyring` library for secure storage. Create this file manually if needed (see `auth.py` for expected format).

## License

This project is licensed under the MIT License. 