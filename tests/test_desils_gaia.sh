#!/bin/bash
# Test: xmatch desils_noao gaia_noao -o desils_gaia.parquet

set -e

TEST_NAME="desils_noao_vs_gaia_noao"
OUTPUT_FILE="tests/output_${TEST_NAME}.parquet"
CONFIG_FILE="src/xmatch/xmatch.yaml"
CAT1="desils_noao"
CAT2="gaia_noao"
RADIUS=1.0 # Example radius

# Check if the assumed catalogue names exist in the config file
if ! grep -q "$CAT1:" "$CONFIG_FILE" || ! grep -q "$CAT2:" "$CONFIG_FILE"; then
    echo "Warning: Catalogue entries '$CAT1' or '$CAT2' might be missing or named differently in $CONFIG_FILE." >&2
    echo "         Please ensure they are defined correctly (including ra_column, dec_column) for this test." >&2
    # Decide whether to exit or proceed cautiously
    # exit 1 
fi

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