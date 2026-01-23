from pathlib import Path

# project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw" / "NEU-DET"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "validation"

MODEL_PATH = BASE_DIR / "models" / "defect_detector_finetuned.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 10
LEARNING_RATE = 1e-3
