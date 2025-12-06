# main.py
import torch
from cnn import CNN
import joblib
from dataset import get_dataloaders
from config import DEVICE, NUM_CLASSES as num_classes
# ----------------------------
# 1️⃣ Load pretrained models
# ----------------------------
cnn_model = CNN(num_classes)
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=DEVICE))
cnn_model.to(DEVICE)
cnn_model.eval()

knn_model = joblib.load("knn_model.joblib")
kmeans_model = joblib.load("kmeans_model.joblib")

# ----------------------------
# 2️⃣ Load data (optional demo)
# ----------------------------
train_loader, test_loader, classes = get_dataloaders()

# ----------------------------
# 3️⃣ Feature extraction helper
# ----------------------------
def extract_features(img_tensor):
    with torch.no_grad():
        _, feats = cnn_model(img_tensor.to(DEVICE))
    return feats.cpu().numpy()

# ----------------------------
# 4️⃣ Choose pipeline
# ----------------------------
pipeline = input("Choose pipeline (cnn / knn / kmeans / cnn_knn / kmeans_knn): ").strip().lower()

# For demo, take the first batch from test_loader
images, labels = next(iter(test_loader))

if pipeline == "cnn":
    with torch.no_grad():
        outputs, _ = cnn_model(images.to(DEVICE))
        preds = outputs.argmax(dim=1).cpu().numpy()
    print("CNN predictions:", preds)

elif pipeline == "knn":
    feats = extract_features(images)
    preds = knn_model.predict(feats)
    print("KNN predictions:", preds)

elif pipeline == "kmeans":
    feats = extract_features(images)
    clusters = kmeans_model.predict(feats)
    print("K-Means clusters:", clusters)

elif pipeline == "cnn_knn":
    feats = extract_features(images)
    preds = knn_model.predict(feats)
    print("CNN → KNN predictions:", preds)

elif pipeline == "kmeans_knn":
    feats = extract_features(images)
    clusters = kmeans_model.predict(feats).reshape(-1, 1)
    preds = knn_model.predict(clusters)
    print("K-Means → KNN predictions:", preds)

else:
    print("Invalid pipeline choice")
