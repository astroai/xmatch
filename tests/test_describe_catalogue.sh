#!/bin/bash

# Test the --describe-catalogue informational command.

set -e

# Run the command to describe a specific catalogue entry
xmatch --describe-catalogue gaia_cds

# Check the exit code (non-zero would trigger the set -e if it failed)
echo "test_describe_catalogue.sh PASSED (command returned successful exit code)"
