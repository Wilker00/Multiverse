"""
tools/parallel_rollout.py

CLI wrapper for parallel rollout execution.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from core.parallel_rollout import _main_cli


if __name__ == "__main__":
    _main_cli()
