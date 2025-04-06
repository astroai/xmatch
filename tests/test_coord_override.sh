#!/bin/bash

# Test overriding default RA/Dec column names for local files.

set -e

OUTPUT_FILE="tests/output_coord_override.parquet"
FILE1="tests/test_id_join_1.csv" # Has columns 'ra', 'dec'
FILE2="tests/test_coord_override_2.csv" # Has columns 'ALPHA_J2000', 'DELTA_J2000'

RADIUS_ARCSEC=72 # Large enough radius (0.01 deg ~ 36 arcsec) to ensure matches

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command, overriding column names
python -m src.xmatch.cli $FILE1 $FILE2 \
    --radius $RADIUS_ARCSEC \
    --ra1-col ra \
    --dec1-col dec \
    --ra2-col ALPHA_J2000 \
    --dec2-col DELTA_J2000 \
    -o $OUTPUT_FILE \
    --columns-1 source_id,ra,dec \
    --columns-2 obj_id,extra_val

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

# TODO: Check for expected number of matches (should be 2 based on coords)

echo "test_coord_override.sh PASSED (Basic check - file created)"

# rm -f $OUTPUT_FILE 