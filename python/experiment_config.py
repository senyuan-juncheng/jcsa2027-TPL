import os
from pathlib import Path

# Files
DATA_DIR = Path("../npy")
X_PATH = DATA_DIR / "X.npy"
META_PATH = DATA_DIR / "meta.csv"
VOCAB_PATH = DATA_DIR / "vocab.json"

# Experiment
MODEL = os.environ.get("MODEL", "baseline").lower()
SEED = int(os.environ.get("SEED", 123))
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Network
CNN_CHANNELS = (8, 16, 8)
GRU_HIDDEN = 256
GRU_LAYERS = 2
DROPOUT = 0.2
TIME_DOWNSAMPLE = 4
FREQ_DOWNSAMPLE = 16 if MODEL == "baseline" else 8

# Split
TRAIN_FRAC = 0.85
VAL_FRAC = 0.05
TEST_FRAC = 0.10
