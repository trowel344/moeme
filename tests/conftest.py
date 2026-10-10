"""Make `src/` importable so tests run from a clean checkout."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
