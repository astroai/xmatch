#!/bin/bash
# Test ID-based matching between two local CSV files

xmatch tests/id_cat1.csv tests/id_cat2.csv \
       --join-on-ids id1:id2 \
       -o tests/id_match_output.parquet \
       -vv 