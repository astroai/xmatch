#!/bin/bash

# Test joining two local CSV files based on ID columns
# Expected: rows with source_id 102, 103, 105 should be joined.

# Ensure script fails on error
set -e

# Define expected output file
OUTPUT_FILE="tests/output_local_id_join.parquet"

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command
python -m src.xmatch.cli tests/test_id_join_1.csv tests/test_id_join_2.csv \
    --join-on-ids source_id:catalog_id \
    -o $OUTPUT_FILE \
    --columns-1 source_id,ra,mag_g \
    --columns-2 extra_data,comment

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

# TODO: Add more specific checks (e.g., using pandas in a Python script to check row count/content)

echo "test_local_id_join.sh PASSED (Basic check - file created)"

# Optional: Clean up output file
# rm -f $OUTPUT_FILE 