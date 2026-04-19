#!/bin/bash
set -e

TEST_NAME="remote_tap_sky_join"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_remote_tap_sky_join.parquet"
CAT1="gaia_cds"
CAT2="twomass_psc_cds"
RADIUS=1.0

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command
# Should use remote TAP spatial join
xmatch $CAT1 $CAT2 \
    --radius $RADIUS \
    -o $OUTPUT_FILE \
    --columns-1 "Source,RA_ICRS,DE_ICRS,Gmag" \
    --columns-2 "_2MASS,Jmag,Hmag,Kmag"

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

echo "test_remote_tap_sky_join.sh PASSED (Basic check - file created)"
