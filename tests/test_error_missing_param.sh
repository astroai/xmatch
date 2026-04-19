#!/bin/bash
set -e

TEST_NAME="error_missing_param"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_error_missing_param.parquet"
CAT1="gaia_cds"
CAT2="twomass_psc_cds"
# Deliberately NOT providing radius

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command without required radius parameter
# Should fail with non-zero exit code
if xmatch $CAT1 $CAT2 -o $OUTPUT_FILE 2>/dev/null; then
    echo "Error: Command succeeded but should have failed due to missing radius parameter." >&2
    exit 1
else
    echo "Command correctly failed due to missing radius parameter"
fi

echo "test_error_missing_param.sh PASSED (correctly detected missing parameter)"
