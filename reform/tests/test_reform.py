"""
Unit and regression test for the reform package.
"""

# Import package, test suite, and other packages as needed
import reform
import pytest
import sys

def test_reform_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "reform" in sys.modules
