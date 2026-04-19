#!/bin/bash
set -e

TEST_NAME="local_id_join"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_local_id_join.parquet"
FILE1="$testdir/inputs/test_id_join_1.csv" # Example file with shared ID column
FILE2="$testdir/inputs/test_id_join_2.csv" # Example file with shared ID column

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command with ID-based join
xmatch $FILE1 $FILE2 \
    --join-on-ids "{'cat1':'source_id', 'cat2':'match_id'}" \
    -o $OUTPUT_FILE

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

echo "test_local_id_join.sh PASSED (Basic check - file created)"
