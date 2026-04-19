#!/bin/bash
set -e

# Test error ellipse matching capabilities
echo "Running error ellipse matching tests..."
python -m pytest -xvs test_error_matching.py

# Success message if all tests pass
echo "✅ All error ellipse matching tests passed successfully!"
