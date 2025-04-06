#!/bin/bash

# Test joining two remote catalogues on the same TAP service (VizieR) using ADQL JOIN for spatial match.
# Uses 2MASS PSC (II/246/out) and ASKAP-POSSUM Pilot (II/281/askp) joined spatially.

set -e

OUTPUT_FILE="tests/output_remote_tap_sky_join.parquet"
CONFIG_FILE="src/xmatch/xmatch.yaml"

# Define the catalogues in config (ensure they exist with correct RA/Dec cols)
CAT1="twomass_psc_cds" # Assuming a name defined in xmatch.yaml for II/246/out
CAT2="askap_possum_cds" # Assuming a name defined in xmatch.yaml for II/281/askp

# Check if the assumed catalogue names exist in the config file
if ! grep -q "$CAT1:" "$CONFIG_FILE" || ! grep -q "$CAT2:" "$CONFIG_FILE"; then
    echo "Warning: Catalogue entries '$CAT1' or '$CAT2' might be missing or named differently in $CONFIG_FILE." >&2
    echo "         Please ensure they are defined correctly (including ra_column, dec_column) for this test." >&2
    # Decide whether to exit or proceed cautiously
    # exit 1 
fi

# Small radius for the test
RADIUS_ARCSEC=1.5

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command
# Expecting the 'remote_join' strategy to be selected.
python -m src.xmatch.cli $CAT1 $CAT2 \
    --config $CONFIG_FILE \
    --radius $RADIUS_ARCSEC \
    -o $OUTPUT_FILE \
    --columns-1 "_2MASS,RAJ2000,DEJ2000,Jmag" \
    --columns-2 "_2MASS,RAdeg,DEdeg,S1400" # Select columns including coords and IDs

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

# TODO: Add check for non-zero row count.

echo "test_remote_tap_sky_join.sh PASSED (Basic check - file created)"

# rm -f $OUTPUT_FILE 