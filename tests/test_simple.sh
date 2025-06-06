#!/bin/bash
set -e

TEST_NAME="simple"

# Get the directory where the script is located
testdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$testdir/outputs"

# Simple test of basic functionality
# Run this test with the working directory set to the project root

xmatch "$testdir/inputs/simple_sources.csv" gaia_cds \
       -o "$testdir/outputs/quick_test_output.parquet" \
       -r 2.0 \
       -vv \
       --columns-2 "Source,Gmag,BPmag,RPmag,Plx"
