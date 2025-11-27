
"""Run the full smart intersection anomaly pipeline.

Usage in Colab:

    !git clone https://github.com/haimye1/smart_intersection.git
    %cd smart_intersection
    !pip install -q -r requirements.txt
    !python scripts/run_training.py

Assumes the normal dataset is available at the path in config.DATA_PATH.
"""

from smart_intersection import run_full_pipeline


def main():
    _ = run_full_pipeline()


if __name__ == "__main__":
    main()
