"""Test script to verify CrossMatch class functionality."""

import sys

from xmatch.crossmatch import CrossMatch


def test_class():
    """Simple test to verify CrossMatch class can be instantiated and methods called."""
    try:
        cm = CrossMatch()
        # Check if the crossmatch method exists
        has_crossmatch = hasattr(cm, "crossmatch")
        print(f"CrossMatch instance has 'crossmatch' method: {has_crossmatch}")

        # List available methods
        methods = [m for m in dir(cm) if not m.startswith("_")]
        print(f"Available public methods: {methods}")

        # Check private methods that might be related
        private_methods = [m for m in dir(cm) if m.startswith("_") and "crossmatch" in m.lower()]
        print(f"Related private methods: {private_methods}")

        return has_crossmatch
    except Exception as e:
        print(f"Error testing CrossMatch: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    test_class()
