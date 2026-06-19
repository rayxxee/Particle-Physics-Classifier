"""
pytest configuration and shared fixtures for the Particle Physics Classifier test suite.
"""
import sys
from pathlib import Path

# Ensure the project root is on the Python path
# (needed when running pytest from the project root)
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
