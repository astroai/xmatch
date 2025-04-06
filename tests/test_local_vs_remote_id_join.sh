#!/bin/bash

# Test joining a local CSV file against a configured remote table based on ID columns.
# Uses ukidss_gaia_xmatch_noao from config, expecting specific ID column names.
# NOTE: The local IDs are fake, so we likely expect 0 matches, but the test checks if the command runs and produces an output file.

set -e

OUTPUT_FILE="tests/output_local_vs_remote_id_join.parquet"
CONFIG_FILE="src/xmatch/xmatch.yaml"
LOCAL_FILE="tests/test_local_vs_remote_id_1.csv"
REMOTE_CATALOGUE="ukidss_gaia_xmatch_noao"

# ID columns: local file column : remote catalogue column from config
# The remote column name MUST match the one defined in the YAML for that catalogue.
JOIN_KEYS="local_id:UKIDSSDR11PLUS_LASSOURCE_SOURCEID" 

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command
# We expect the strategy to be download_and_match (downloads remote, joins locally)
# because ukidss_gaia_xmatch_noao is likely defined under noao_datalab TAP,
# which doesn't support remote ID joins against arbitrary uploaded lists in this implementation.
python -m src.xmatch.cli $LOCAL_FILE $REMOTE_CATALOGUE \
    --config $CONFIG_FILE \
    --join-on-ids "$JOIN_KEYS" \
    -o $OUTPUT_FILE \
    --columns-1 local_id,local_val \
    --columns-2 GAIADR3_GAIA_SOURCE_SOURCEID,separation # Select the Gaia ID and separation from the remote table

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

# TODO: Add check for expected number of rows (likely 0 or small number if any IDs accidentally match)

echo "test_local_vs_remote_id_join.sh PASSED (Basic check - file created)"

# rm -f $OUTPUT_FILE 