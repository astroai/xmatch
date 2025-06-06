#!/bin/bash

# Test the --list-catalogues informational command.

set -e

# Run the command to list all catalogues
xmatch --list-catalogues

# Check the exit code (non-zero would trigger the set -e if it failed)
echo "test_list_catalogues.sh PASSED (command returned successful exit code)"