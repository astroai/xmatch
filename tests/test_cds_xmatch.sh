#!/bin/bash

# Test matching a local file against a remote catalogue using the CDS XMatch service via astroquery.

set -e

OUTPUT_FILE="tests/output_cds_xmatch.parquet"
CONFIG_FILE="src/xmatch/xmatch.yaml"
LOCAL_FILE="tests/test_cds_input.csv"

# Define the remote catalogue configured to use CDS XMatch
# This entry MUST exist in xmatch.yaml and point to the 'cds' archive's 'xmatch_service'.
# Example YAML snippet needed:
# catalogues:
#   twomass_cds_xmatch:
#     description: "2MASS PSC via CDS XMatch Service"
#     archive: cds
#     service_id: xmatch_service # IMPORTANT: Use the xmatch service
#     access_identifier: "II/246/out" # VizieR table ID for 2MASS PSC
#     ra_column: "RAJ2000" # Column name in the VizieR table
#     dec_column: "DEJ2000"
#     estimated_size: 'huge'
REMOTE_CATALOGUE="twomass_cds_xmatch"

# Check if the assumed catalogue name exists in the config file
if ! grep -q "$REMOTE_CATALOGUE:" "$CONFIG_FILE"; then
    echo "Warning: Catalogue entry '$REMOTE_CATALOGUE' might be missing or named differently in $CONFIG_FILE." >&2
    echo "         Please ensure it is defined correctly (using archive: cds, service_id: xmatch_service) for this test." >&2
    # Decide whether to exit or proceed cautiously
    # exit 1 
fi

RADIUS_ARCSEC=5.0

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command
# Expecting the 'cds_xmatch_local_remote' strategy.
python -m src.xmatch.cli $LOCAL_FILE $REMOTE_CATALOGUE \
    --config $CONFIG_FILE \
    --radius $RADIUS_ARCSEC \
    --ra1-col ra_in \
    --dec1-col dec_in \
    -o $OUTPUT_FILE 
    # Columns are less configurable via CDS XMatch, often returns fixed set + distance
    # --columns-1 name 
    # --columns-2 "_2MASS,Jmag" # Column selection might not be reliable here

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

# TODO: Check for expected number of matches (should be >= 2 based on input coords)

echo "test_cds_xmatch.sh PASSED (Basic check - file created)"

# rm -f $OUTPUT_FILE 