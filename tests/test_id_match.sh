#!/bin/bash
set -e

TEST_NAME="id_match"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_id_match.parquet"
FILE1="test_id_join_1.csv"
FILE2="test_id_join_2.csv"

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the command with ID match on 'source_id' and 'ref_id' columns
xmatch $FILE1 $FILE2 \
    --join-on-ids "{'cat1':'source_id', 'cat2':'catalog_id'}" \
    -o $OUTPUT_FILE

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

echo "test_id_match.sh PASSED (Basic check - file created)"