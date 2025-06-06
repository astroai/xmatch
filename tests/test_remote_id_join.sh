#!/bin/bash
set -e

TEST_NAME="remote_id_join"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_remote_id_join.parquet"

# Test joining two catalogues by ID column.

CAT1="gaia_noao"
CAT2="ukidsslas_noao"
JOIN_KEYS="source_id:sourceid"

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command with ID-based join
xmatch $CAT1 $CAT2 \
    --join-on-ids "{'cat1':'$CAT1', 'cat2':'$CAT2'}" \
    -o $OUTPUT_FILE

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

echo "test_remote_id_join.sh PASSED (Basic check - file created)"