from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "StanfordDogs"
IMAGES_DIR = DATA_DIR / "Images"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"

NUM_CLASSES = 120
IMAGE_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 4

SEED = 42

HEAD_EPOCHS = 10
HEAD_LR = 1e-3

FINETUNE_EPOCHS = 30
FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4
PATIENCE = 7

MODELS = [
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet50",
]