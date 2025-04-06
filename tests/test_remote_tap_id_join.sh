#!/bin/bash

# Test joining two remote catalogues on the same TAP service (VizieR) using ADQL JOIN for IDs.
# Uses 2MASS PSC (II/246/out) and ASKAP-POSSUM Pilot (II/281/askp) joined on _2MASS designation.

set -e

OUTPUT_FILE="tests/output_remote_tap_id_join.parquet"
CONFIG_FILE="src/xmatch/xmatch.yaml"

# Define the catalogues in config (or ensure they exist with correct IDs)
CAT1="twomass_psc_cds" # Assuming a name defined in xmatch.yaml for II/246/out
CAT2="askap_possum_cds" # Assuming a name defined in xmatch.yaml for II/281/askp

# We need to ensure xmatch.yaml has entries for these with correct config:
# catalogues:
#   twomass_psc_cds:
#     description: "2MASS Point Source Catalog via CDS VizieR TAP"
#     archive: cds
#     service_id: tap_service
#     access_identifier: "II/246/out"
#     # ... other necessary columns like ra, dec
#     id_column: "_2MASS" # The join key
#     ra_column: "RAJ2000"
#     dec_column: "DEJ2000"
#     estimated_size: 'huge'
#   askap_possum_cds:
#     description: "ASKAP POSSUM Pilot Survey Catalogue via CDS VizieR TAP"
#     archive: cds
#     service_id: tap_service
#     access_identifier: "II/281/askp"
#     id_column: "_2MASS" # The join key
#     ra_column: "RAdeg"
#     dec_column: "DEdeg"
#     estimated_size: 'medium'

# Check if the assumed catalogue names exist in the config file
if ! grep -q "$CAT1:" "$CONFIG_FILE" || ! grep -q "$CAT2:" "$CONFIG_FILE"; then
    echo "Warning: Catalogue entries '$CAT1' or '$CAT2' might be missing or named differently in $CONFIG_FILE." >&2
    echo "         Please ensure they are defined correctly for this test." >&2
    # Decide whether to exit or proceed cautiously
    # exit 1 
fi

# ID columns from the *remote* tables, as defined in their YAML entries
# These MUST match the id_column keys in the YAML for CAT1 and CAT2 respectively.
JOIN_KEYS="_2MASS:_2MASS" 

# Clean up previous run
rm -f $OUTPUT_FILE

# Run the xmatch command
# Expecting the 'remote_join' strategy to be selected.
python -m src.xmatch.cli $CAT1 $CAT2 \
    --config $CONFIG_FILE \
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