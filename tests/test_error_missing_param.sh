#!/bin/bash

# Test that the CLI exits with an error if neither --radius nor --join-on-ids is provided for matching.

set +e # Disable exit on error, as we expect an error

FILE1="tests/test_id_join_1.csv"
FILE2="tests/test_coord_override_2.csv"
OUTPUT_FILE="tests/output_error_missing_param.parquet"

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the command - expect failure
python -m src.xmatch.cli $FILE1 $FILE2 -o $OUTPUT_FILE > /dev/null 2>&1
EXIT_CODE=$?

set -e # Re-enable exit on error

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo "Error: Command succeeded unexpectedly. Expected failure due to missing --radius/--join-on-ids." >&2
    # Clean up the erroneously created file if it exists
    rm -f $OUTPUT_FILE
    exit 1
else
    echo "test_error_missing_param.sh PASSED (Command failed as expected)"
fi

# Ensure output file was NOT created
if [ -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was created unexpectedly on failure." >&2
    rm -f $OUTPUT_FILE
    exit 1
fi 