#!/bin/bash
set -e

TEST_NAME="error_invalid_input"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

OUTPUT_FILE="$testdir/outputs/output_error_invalid_input.parquet"
CAT1="nonexistent_catalogue" # This doesn't exist in the config
CAT2="gaia_cds" # This one exists
RADIUS=1.0

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command with a nonexistent catalogue
# Should fail with non-zero exit code
if xmatch $CAT1 $CAT2 --radius $RADIUS -o $OUTPUT_FILE 2>/dev/null; then
    echo "Error: Command succeeded but should have failed due to invalid catalogue." >&2
    exit 1
else
    echo "Command correctly failed due to invalid catalogue"
fi

# Now try with a nonexistent file
if xmatch nonexistent_file.csv $CAT2 --radius $RADIUS -o $OUTPUT_FILE 2>/dev/null; then
    echo "Error: Command succeeded but should have failed due to nonexistent file." >&2
    exit 1
else
    echo "Command correctly failed due to nonexistent file"
fi

echo "test_error_invalid_input.sh PASSED (correctly detected invalid inputs)"