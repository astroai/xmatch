#!/bin/bash
set -e

TEST_NAME="remote_tap_id_join"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_remote_tap_id_join.parquet"

# Test joining two remote catalogues on the same TAP service (VizieR) using ADQL JOIN for IDs.
# Uses 2MASS PSC (II/246/out) and ASKAP-POSSUM Pilot (II/281/askp) joined on _2MASS designation.

# Define the catalogues in config (or ensure they exist with correct IDs)
CAT1="twomass_psc_cds" # Assuming a name defined in xmatch.yaml for II/246/out
CAT2="askap_possum_cds" # Assuming a name defined in xmatch.yaml for II/281/askp

# ID columns from the *remote* tables, as defined in their YAML entries
# These MUST match the id_column keys in the YAML for CAT1 and CAT2 respectively.
JOIN_KEYS="_2MASS:_2MASS"

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command
# Expecting the 'remote_join' strategy to be selected.
xmatch $CAT1 $CAT2 \
    --join-on-ids "$JOIN_KEYS" \
    -o $OUTPUT_FILE \
    --columns-1 "_2MASS,Jmag,Hmag,Kmag" \
    --columns-2 "_2MASS,S1400,e_S1400,Type" # Select columns from both

# Basic check: Ensure output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file $OUTPUT_FILE was not created." >&2
    exit 1
fi

# TODO: Add check for non-zero row count, as we expect matches here.

echo "test_remote_tap_id_join.sh PASSED (Basic check - file created)"

# rm -f $OUTPUT_FILE
