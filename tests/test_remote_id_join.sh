#!/bin/bash
# Test the remote ID-based JOIN functionality
# Joins UKIDSS LAS Source table with the pre-computed UKIDSS x GaiaDR3 crossmatch table
# on the NOAO Data Lab TAP service based on matching UKIDSS source IDs.

echo "Running Remote ID Join Test: UKIDSS LAS + UKIDSSxGaia XMatch (NOAO)"

# Define the catalogues involved (names from xmatch.yaml)
CAT1="ukidsslas_noao"
CAT2="ukidss_gaia_xmatch_noao"

# Define the join keys (cat1 maps to CAT1 column, cat2 maps to CAT2 column)
JOIN_KEY_CAT1="sourceid"
JOIN_KEY_CAT2="UKIDSSDR11PLUS_LASSOURCE_SOURCEID"

# Define output file
OUTPUT_FILE="tests/remote_id_join_output.parquet"

# Columns to select from UKIDSS (CAT1)
COLS1="yaperflux3,hapermag3"
# Columns to select from the XMatch table (CAT2) - primarily the Gaia ID
COLS2="GAIADR3_GAIA_SOURCE_SOURCEID"

# Perform the ID join
xmatch $CAT1 $CAT2 \
       --join-on-ids "$JOIN_KEY_CAT1:$JOIN_KEY_CAT2" \
       -o $OUTPUT_FILE \
       --columns1 "$COLS1" \
       --columns2 "$COLS2" \
       -v

# Basic check: See if the output file was created
if [ -f "$OUTPUT_FILE" ]; then
  echo "Remote ID Join test completed. Output file created: $OUTPUT_FILE"
  # Optional: Add command to inspect the first few rows of the parquet file
  # e.g., using pandas/pyarrow if available in the test environment
  # python -c "import pandas as pd; print(pd.read_parquet('$OUTPUT_FILE').head())"
else
  echo "Remote ID Join test FAILED. Output file was not created."
  exit 1
fi

exit 0 