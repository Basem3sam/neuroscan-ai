# knn.py
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(train_features, train_labels, test_features, test_labels, k):
    """
    Train KNN, return both model and accuracy
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)

    preds = knn.predict(test_features)
    acc = accuracy_score(test_labels, preds)

    print("KNN Accuracy:", acc)
    joblib.dump(knn, "knn_model.joblib")

    return knn, acc
