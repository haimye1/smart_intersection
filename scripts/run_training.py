#!/usr/bin/env python3
"""
Run the full smart intersection anomaly pipeline.
"""

import os
import sys

# נתיב לתיקיית הסקריפט (D:\scripts)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# נתיב לשורש הפרויקט (D:\)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# להוסיף את שורש הפרויקט ל-PYTHONPATH
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# עכשיו אפשר לייבא ישירות את הפייפליין מה-root
from train_pipeline import run_full_pipeline


def main():
    run_full_pipeline()


if __name__ == "__main__":
    main()
