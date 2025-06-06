#!/bin/bash
set -e

TEST_NAME="desils_gaia"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_${TEST_NAME}.parquet"
CAT1="desils_noao"
CAT2="gaia_noao"
RADIUS=1.0 # Example radius

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
# rm -f "$OUTPUT_FILE"