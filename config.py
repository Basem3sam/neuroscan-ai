#config.py
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10  # Increased from 1 for better results
LR = 1e-4
KNN_NEIGHBORS = 5
RANDOM_SEED = 42
NUM_CLASSES = 2

MODEL_SAVE_PATH = "models/cnn_model.pth"
KNN_SAVE_PATH = "models/knn_model.joblib"
KMEANS_SAVE_PATH = "models/kmeans_model.joblib" # Add this for consistency

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")