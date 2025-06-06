#!/bin/bash
set -e

TEST_NAME="parallel_local_remote"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_parallel_local_remote.parquet"
LOCAL_FILE="$testdir/inputs/my_sources.csv"
REMOTE_CAT="gaia_cds" 
RADIUS=1.0

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command with parallel processing
xmatch $LOCAL_FILE $REMOTE_CAT \
    --radius $RADIUS \
    -o $OUTPUT_FILE \
    -vv # Added verbose to see parallel processing messages

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

echo "test_parallel_local_remote.sh PASSED (Basic check - file created)"