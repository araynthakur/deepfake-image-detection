"""Central settings. Change values here if you want a bigger / smaller run."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # optional: manually unzipped datasets go here
MANIFEST = DATA_DIR / "manifest.csv"  # created by prepare_data.py
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "best_model.keras"
META_PATH = MODELS_DIR / "meta.json"

# The three Kaggle datasets from the project
DATASETS = {
    "realvsfake_81k": "wish096/realvsfake-81k-by-wish",
    "real_vs_fake_130k": "shreyanshpatel1/130k-real-vs-fake-face",
    "faces_140k": "xhlulu/140k-real-and-fake-faces",
}

CLASS_NAMES = ["REAL", "DEEPFAKE"]   # Real = 0, Fake = 1 (as in the slides)

# ---- data ----
SEED = 42
MAX_PER_CLASS = 10000   # images per class used from the combined pool (0 = use everything)
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

# ---- model / training ----
IMG_SIZE = 224          # EfficientNetB4's native size is 380 (better, but much slower)
BATCH_SIZE = 32
EPOCHS_HEAD = 3         # phase 1: train only the new classifier head
EPOCHS_FINE = 7         # phase 2: fine-tune the top of EfficientNetB4
LR_HEAD = 1e-3
LR_FINE = 1e-5
FINE_TUNE_LAYERS = 80   # how many of the last EfficientNetB4 layers to unfreeze

# Only for pipeline testing without internet (set DF_WEIGHTS=none)
WEIGHTS = None if os.getenv("DF_WEIGHTS", "").lower() == "none" else "imagenet"
