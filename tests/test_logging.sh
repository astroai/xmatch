#!/bin/bash
set -e

TEST_NAME="logging_test"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_logging_test.parquet"
LOG_FILE="$testdir/outputs/xmatch_test.log"
CAT1="gaia_cds"
CAT2="twomass_psc_cds"
RADIUS=1.0

# Clean up previous runs
rm -f "$OUTPUT_FILE" "$LOG_FILE"

# Run the xmatch command with increased verbosity and log file
xmatch "$CAT1" "$CAT2" \
    --radius "$RADIUS" \
    -o "$OUTPUT_FILE" \
    -vvv \
    --log-file="$LOG_FILE"

# Check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

# Check: Ensure log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file $LOG_FILE was not created." >&2
    exit 1
fi

# Check: Ensure log file has debug content
if ! grep -q "DEBUG" "$LOG_FILE"; then
    echo "Warning: Log file $LOG_FILE doesn't contain expected DEBUG output." >&2
    # Non-fatal warning
fi

echo "test_logging.sh PASSED (Basic check - files created)"

# Clean up
# rm -f "$OUTPUT_FILE" "$LOG_FILE"