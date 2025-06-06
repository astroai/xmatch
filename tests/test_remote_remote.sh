#!/bin/bash
set -e

TEST_NAME="remote_remote"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

# Test cross-matching between two remote catalogues
# Note: This might take a while depending on the query complexity and server load.

# Crossmatch ESA Gaia with CDS GALEX AIS
xmatch gaia_esa galex_cds \
       -o "$testdir/outputs/remote_remote_output.parquet" \
       -r 1.0 \
       --columns-2 \"FUVmag,NUVmag\" \
       -v