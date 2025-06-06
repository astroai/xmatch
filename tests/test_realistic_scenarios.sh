#!/bin/bash
set -e

# Test realistic astronomical catalog cross-matching scenarios
echo "Running realistic astronomical catalog cross-matching tests..."
python -m pytest -xvs test_realistic_xmatch.py

# Success message if all tests pass
echo "✅ All realistic cross-matching tests passed successfully!"