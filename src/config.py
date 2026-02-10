from src.utils import load_config

CONFIG_PATH = "config/local.yaml" 

cfg = load_config(CONFIG_PATH)

# project
SEED = cfg["project"]["seed"]

# data
TRAIN_DIR = cfg["data"]["train_dir"]
VAL_DIR = cfg["data"]["val_dir"]

# model
IMG_SIZE = tuple(cfg["model"]["img_size"])
NUM_CLASSES = cfg["model"]["num_classes"]
MODEL_PATH = cfg["model"]["model_path"]

# training
BATCH_SIZE = cfg["training"]["batch_size"]

# inference
CONFIDENCE_THRESHOLD = cfg["inference"]["confidence_threshold"]

