#!/bin/bash

xmatch my_sources.csv gaia_cds \
       -o quick_test_output.parquet \
       -r 2.0 \
       --config=$HOME/src/xmatch/src/xmatch/xmatch.yaml \
       -vv \
       --columns-2 "Source,Gmag,BPmag,RPmag,parallax"
