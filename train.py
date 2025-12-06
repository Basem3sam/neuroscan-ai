# train.py
import torch
import numpy as np
import joblib
from dataset import get_dataloaders
from cnn import train_cnn
from knn import train_knn
from kmeans_classifier import train_kmeans
from config import (
    MODEL_SAVE_PATH, KNN_SAVE_PATH, KMEANS_SAVE_PATH, 
    DEVICE, NUM_CLASSES, KNN_NEIGHBORS
)

def get_features_and_labels(model, loader):
    """
    Runs data through the CNN to get feature vectors (fc1 output)
    and corresponding labels for KNN/K-Means training.
    """
    model.eval()
    features_list = []
    labels_list = []

    print("Extracting features from CNN...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            
            # Get the second return value (feats) from your CNN class
            _, feats = model(images) 
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    # Concatenate all batches into one large array
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    
    return features, labels

def main():
    # 1. Prepare Data
    print("Loading Data...")
    train_loader, test_loader, classes = get_dataloaders()
    print(f"Classes detected: {classes}")
    
    # Update NUM_CLASSES dynamically based on dataset
    current_num_classes = len(classes)

    # 2. Train CNN
    print("Starting CNN Training...")
    cnn_model = train_cnn(train_loader, current_num_classes)
    
    # Save CNN
    torch.save(cnn_model.state_dict(), MODEL_SAVE_PATH)
    print(f"CNN Model saved to {MODEL_SAVE_PATH}")

    # 3. Extract Features for Hybrid Models
    # We use the trained CNN to turn images into vectors
    print("Extracting features for hybrid training...")
    train_feats, train_labels = get_features_and_labels(cnn_model, train_loader)
    test_feats, test_labels = get_features_and_labels(cnn_model, test_loader)

    # 4. Train KNN
    print(f"Training KNN (k={KNN_NEIGHBORS})...")
    knn_model, knn_acc = train_knn(train_feats, train_labels, test_feats, test_labels, k=KNN_NEIGHBORS)
    joblib.dump(knn_model, KNN_SAVE_PATH) # Save KNN

    # 5. Train K-Means
    print(f"Training K-Means (clusters={current_num_classes})...")
    kmeans_model, kmeans_acc = train_kmeans(train_feats, train_labels, test_feats, test_labels, n_clusters=current_num_classes)
    joblib.dump(kmeans_model, KMEANS_SAVE_PATH) # Save K-Means

    print("\n--- Final Results ---")
    print(f"KNN Accuracy: {knn_acc:.4f}")
    print(f"K-Means Accuracy: {kmeans_acc:.4f}")

if __name__ == "__main__":
    main()