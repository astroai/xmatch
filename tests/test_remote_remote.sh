#!/bin/bash
# Test cross-matching between two remote catalogues
# Note: This might take a while depending on the query complexity and server load.

# Crossmatch ESA Gaia with CDS GALEX AIS
xmatch gaia_esa galex_cds \
       -o tests/remote_remote_output.parquet \
       -r 1.0 \
       --columns-2 \"FUVmag,NUVmag\" \
       -v 