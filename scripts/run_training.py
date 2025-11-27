#!/usr/bin/env python3
import os, sys

# מקבל נתיב הפרויקט
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# מוסיף את src ל-PYTHONPATH
sys.path.insert(0, SRC_DIR)

# עכשיו ניתן לייבא את החבילה
from smart_intersection.train_pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
