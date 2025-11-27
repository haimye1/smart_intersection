
import os

DATA_PATH = os.environ.get(
    "SMART_INTERSECTION_DATA_PATH",
    "/content/smart_intersection_normal.csv",
)

MODEL_DIR = os.environ.get(
    "SMART_INTERSECTION_MODEL_DIR",
    "models_smart_intersection",
)

RANDOM_STATE  = int(os.environ.get("SMART_INTERSECTION_RANDOM_STATE", 42))
WINDOW_SIZE   = int(os.environ.get("SMART_INTERSECTION_WINDOW_SIZE", 60))
WINDOW_STEP   = int(os.environ.get("SMART_INTERSECTION_WINDOW_STEP", 10))
BATCH_SIZE    = int(os.environ.get("SMART_INTERSECTION_BATCH_SIZE", 128))
AE_LATENT_DIM = int(os.environ.get("SMART_INTERSECTION_AE_LATENT_DIM", 8))
AE_EPOCHS     = int(os.environ.get("SMART_INTERSECTION_AE_EPOCHS", 20))
AE_LR         = float(os.environ.get("SMART_INTERSECTION_AE_LR", 1e-3))

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
