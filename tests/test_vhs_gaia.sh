#!/bin/bash
set -e

TEST_NAME="vhs_gaia"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_${TEST_NAME}.parquet"
CAT1="vhs_cds"
CAT2="gaia_cds"
RADIUS=1.0

rm -f "$OUTPUT_FILE"

echo "Running test: $TEST_NAME"
xmatch "$CAT1" "$CAT2" \
    --radius "$RADIUS" \
    -o "$OUTPUT_FILE"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi
echo "Test $TEST_NAME PASSED (Basic check - file created)"
