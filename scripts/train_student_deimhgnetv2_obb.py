#!/usr/bin/env python3
"""Compatibility wrapper for the root-level OBB student training entry."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_student_deimhgnetv2_obb import main


if __name__ == "__main__":
    main()
