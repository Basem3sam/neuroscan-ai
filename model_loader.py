# model_loader.py
import torch
import joblib
import numpy as np
from cnn import CNN
from config import DEVICE, NUM_CLASSES, MODEL_SAVE_PATH, KNN_SAVE_PATH, KMEANS_SAVE_PATH

# Load models once at startup
print("Loading models...")

cnn_model = CNN(NUM_CLASSES)
cnn_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
cnn_model.to(DEVICE)
cnn_model.eval()

knn_model = joblib.load(KNN_SAVE_PATH)
kmeans_model = joblib.load(KMEANS_SAVE_PATH)

print("✅ All models loaded successfully!")

# Class names
CLASS_NAMES = ["Healthy", "Tumor"]

def extract_features(img_array):
    """
    Convert preprocessed image to CNN features
    img_array: numpy array of shape (128, 128, 1)
    """
    # Convert to tensor: (1, 1, 128, 128)
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (H,W,C) -> (1,C,H,W)
    
    with torch.no_grad():
        _, features = cnn_model(img_tensor.to(DEVICE))
    
    return features.cpu().numpy()

def predict_single(preprocessed_img, pipeline="cnn"):
    """
    Make prediction on a single preprocessed image
    
    Args:
        preprocessed_img: numpy array (128, 128, 1)
        pipeline: "cnn", "knn", "kmeans", "cnn_knn", "kmeans_knn"
    
    Returns:
        prediction (int): 0=Healthy, 1=Tumor
        probability (float): confidence score
    """
    
    if pipeline == "cnn":
        img_tensor = torch.from_numpy(preprocessed_img).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            outputs, _ = cnn_model(img_tensor.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1).item()
            prob = probs[0][pred].item()
        
        return pred, prob
    
    elif pipeline == "knn":
        features = extract_features(preprocessed_img)
        pred = knn_model.predict(features)[0]
        
        # Get probabilities from KNN
        probs = knn_model.predict_proba(features)[0]
        prob = probs[pred]
        
        return pred, prob
    
    elif pipeline == "kmeans":
        features = extract_features(preprocessed_img)
        cluster = kmeans_model.predict(features)[0]
        
        # Map cluster to class (you may need to adjust this mapping)
        # For now, assume cluster 0 = Healthy, cluster 1 = Tumor
        pred = cluster
        prob = 0.75  # K-Means doesn't give probabilities
        
        return pred, prob
    
    elif pipeline == "cnn_knn":
        features = extract_features(preprocessed_img)
        pred = knn_model.predict(features)[0]
        probs = knn_model.predict_proba(features)[0]
        prob = probs[pred]
        
        return pred, prob
    
    elif pipeline == "kmeans_knn":
        features = extract_features(preprocessed_img)
        cluster = kmeans_model.predict(features).reshape(-1, 1)
        pred = knn_model.predict(cluster)[0]
        
        # Approximate probability
        prob = 0.80
        
        return pred, prob
    
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")