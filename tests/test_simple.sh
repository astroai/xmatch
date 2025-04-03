#!/bin/bash

xmatch my_sources.csv GAIA_CDS_XMATCH \
       -o quick_test_output.parquet \
       -r 2.0 \
       --config=$HOME/src/xmatch/src/xmatch/config/catalogues.yaml \
       --log-level DEBUG \
       --id-column-1 source_id \
       --columns-2 "Source,Gmag,BPmag,RPmag,parallax"
