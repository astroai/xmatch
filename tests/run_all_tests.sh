#!/bin/bash

# Master script to run all .sh tests in the tests/ directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assume the project root is one level above the tests directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Changing to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || exit 1

# Ensure the tests directory exists (relative to project root)
TEST_DIR="tests"
if [ ! -d "$TEST_DIR" ]; then
  echo "Error: Tests directory '$TEST_DIR' not found in project root." >&2
  exit 1
fi

echo "Running all shell tests in $TEST_DIR/ ..."
echo "---------------------"

FAILED_TESTS=()
# Find all .sh files in the tests directory, excluding this runner script
TEST_FILES=($(find "$TEST_DIR" -maxdepth 1 -name '*.sh' -not -name '$(basename "${BASH_SOURCE[0]}")'))
TOTAL_TESTS=${#TEST_FILES[@]}

if [ $TOTAL_TESTS -eq 0 ]; then
    echo "No test scripts (*.sh) found in $TEST_DIR/ (excluding $(basename "${BASH_SOURCE[0]}"))."
    exit 0
fi

# Loop through test scripts
for test_script in "${TEST_FILES[@]}"; do
  TEST_NAME=$(basename "$test_script")
  echo "[ RUNNING ] $TEST_NAME"

  # Make executable
  chmod +x "$test_script"
  
  # Run the test script.
  # Individual scripts should have 'set -e' to exit on error.
  if "$test_script"; then
    # If the script exits with 0, it's considered passed.
    echo "[ PASSED  ] $TEST_NAME"
  else
    # If the script exits with non-zero, it failed.
    EXIT_CODE=$?
    echo "[ FAILED  ] $TEST_NAME (Exit code: $EXIT_CODE)"
    FAILED_TESTS+=("$TEST_NAME")
  fi
  echo "---------------------"
done


# --- Report Summary ---
PASSED_COUNT=$((TOTAL_TESTS - ${#FAILED_TESTS[@]}))

echo "====================="
echo "Test Summary:"
echo "Total tests found: $TOTAL_TESTS"
echo "Passed: $PASSED_COUNT"
echo "Failed: ${#FAILED_TESTS[@]}"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
  echo "====================="
  echo "ALL TESTS PASSED" >> /dev/stderr # Print final status prominently
  echo "====================="
  exit 0
else
  echo "=====================" >> /dev/stderr
  echo "FAILED TESTS:" >> /dev/stderr
  for failed_test in "${FAILED_TESTS[@]}"; do
    echo "  - $failed_test" >> /dev/stderr
  done
  echo "=====================" >> /dev/stderr
  exit 1
fi 