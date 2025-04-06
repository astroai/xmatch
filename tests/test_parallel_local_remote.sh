#!/bin/bash
# Test cross-matching a local file against a remote catalogue using the --parallel flag
# This tests the process-based parallelism, not specific Healpix chunking.

xmatch tests/my_sources.csv gaia_esa \
       -o tests/parallel_local_remote_output.parquet \
       -r 2.0 \
       -v 