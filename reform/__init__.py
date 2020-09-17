"""
Replica Exchange FOR openmM (REFORM)
A simple implementation of replica exchange MD simulations for OpenMM.
"""

# Add imports here
from .reform import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
