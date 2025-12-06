# kmeans_classifier.py
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def train_kmeans(train_features, train_labels, test_features, test_labels, n_clusters):
    """
    Train K-Means, return both model and accuracy
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_features)

    preds = kmeans.predict(test_features)

    # Map clusters to labels for accuracy
    mapping = {}
    for cluster in range(n_clusters):
        indices = np.where(preds == cluster)[0]
        if len(indices) == 0:
            continue
        most_common = np.bincount(test_labels[indices]).argmax()
        mapping[cluster] = most_common

    preds_mapped = np.array([mapping[c] for c in preds])

    acc = accuracy_score(test_labels, preds_mapped)
    print("K-Means Accuracy:", acc)

    joblib.dump(kmeans, "kmeans_model.joblib")

    return kmeans, acc
