#!/bin/bash

# Test verbose logging and logging to a file.

set -e

FILE1="tests/test_id_join_1.csv"
FILE2="tests/test_coord_override_2.csv"
OUTPUT_FILE="tests/output_logging_test.parquet"
LOG_FILE="tests/output_logging_test.log"
RADIUS_ARCSEC=72

# Clean up previous run
rm -f $OUTPUT_FILE
rm -f $LOG_FILE

# Run the command with verbose flags and log file output
python -m src.xmatch.cli $FILE1 $FILE2 \
    --radius $RADIUS_ARCSEC \
    --ra1-col ra \
    --dec1-col dec \
    --ra2-col ALPHA_J2000 \
    --dec2-col DELTA_J2000 \
    -o $OUTPUT_FILE \
    --log-file $LOG_FILE \
    -vv # Max verbosity (DEBUG)

# Basic checks
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file $LOG_FILE was not created." >&2
    exit 1
fi

# Check if log file contains DEBUG messages (simple check)
if ! grep -q "DEBUG" "$LOG_FILE"; then
    echo "Error: Log file $LOG_FILE does not appear to contain DEBUG level messages." >&2
    exit 1
fi

echo "test_logging.sh PASSED (Basic checks - files created, DEBUG log found)"

# Optional: Clean up output files
# rm -f $OUTPUT_FILE $LOG_FILE 