#!/bin/bash
set -e

TEST_NAME="local_vs_remote_id_join"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_local_vs_remote_id_join.parquet"
LOCAL_FILE="$testdir/inputs/test_local_vs_remote_id_1.csv" # Has 'gaia_id' column matching source_id
REMOTE_CAT="gaia_esa" # Has 'source_id' column in the actual remote database

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command with ID-based join
xmatch $LOCAL_FILE $REMOTE_CAT \
    --join-on-ids "{'cat1':'gaia_id', 'cat2':'source_id'}" \
    -o $OUTPUT_FILE

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

echo "test_local_vs_remote_id_join.sh PASSED (Basic check - file created)"
