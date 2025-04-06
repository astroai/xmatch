#!/bin/bash
# Test: xmatch vhs_cds gaia_cds -o vhs_gaia.parquet

set -e

TEST_NAME="vhs_cds_vs_gaia_cds"
OUTPUT_FILE="tests/output_${TEST_NAME}.parquet"
CONFIG_FILE="src/xmatch/xmatch.yaml"
CAT1="vhs_cds"
CAT2="gaia_cds"
RADIUS=1.0 # Example radius

rm -f "$OUTPUT_FILE"

echo "Running test: $TEST_NAME"
python -m src.xmatch.cli "$CAT1" "$CAT2" \
    --config "$CONFIG_FILE" \
    --radius "$RADIUS" \
    -o "$OUTPUT_FILE"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi
echo "Test $TEST_NAME PASSED (Basic check - file created)"
# rm -f "$OUTPUT_FILE" 