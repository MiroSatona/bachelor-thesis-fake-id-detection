"""
Atuhor: Martin Zůbek
Date: 2026-03-28
File: conftest.py
Description: This file is used for configuration of the path to the root of the project for pytest.
"""

import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))