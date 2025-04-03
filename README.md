# xmatch

A flexible tool for cross-matching astronomical catalogues, supporting multiple methods and formats.

## Features

- Support for multiple cross-matching methods:
  - STILTS (recommended for large catalogues)
  - TAP (Table Access Protocol)
  - Astropy (for smaller catalogues)
  - Astroquery (CDS XMatch service)
- Automatic method selection based on catalogue size and availability
- Configuration via YAML file
- Support for various input/output formats (FITS, Parquet, etc.)
- Chunked processing for large catalogues
- Progress tracking and logging

## Installation

```bash
pip install xmatch
```

## Usage

### Command Line Interface

Basic usage:
```bash
xmatch master_cat target_cat
```

Examples:
```bash
# Cross-match Gaia DR3 with GALEX
xmatch gaiadr3 galex

# Cross-match with custom radius and output file
xmatch gaiadr3 galex -r 3.0 -o gaia_galex.parquet

# Cross-match with specific columns
xmatch gaiadr3 galex -c "objid,FUVmag,NUVmag"

# Use specific method
xmatch gaiadr3 galex -m stilts

# Verbose output
xmatch gaiadr3 galex -v
```

### Python API

```python
from xmatch import CrossMatch

# Initialize cross-matcher
crossmatcher = CrossMatch()

# Perform cross-match
result = crossmatcher.crossmatch(
    master_cat="gaiadr3",
    target_cat="galex",
    radius=3.0,
    columns=["objid", "FUVmag", "NUVmag"]
)
```

## Configuration

The package uses a YAML configuration file to store catalogue information and cross-matching settings. By default, it looks for `catalogues.yaml` in the package's config directory.

Example configuration:
```yaml
catalogues:
  gaiadr3:
    name: "Gaia DR3"
    tap_url: "https://gea.esac.esa.int/tap-server/tap"
    tap_table: "gaia.dr3_source"
    vizier_id: "I/355/gaiadr3"
    ra_column: "ra"
    dec_column: "dec"
    id_column: "source_id"
    chunk_size: 100000
    max_upload_rows: 50000
    default_radius: 2.0  # arcseconds
```

## Supported Catalogues

- Gaia DR3
- GALEX AIS
- DESI Legacy Survey DR10
- VHS (VISTA Hemisphere Survey)
- UKIDSS DR11plus

More catalogues can be added to the configuration file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 