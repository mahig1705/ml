
# =========================================================
# CLUSTERING TASK
# Algorithms:
# KMeans
# KMedoids
# Expectation Maximization (Gaussian Mixture)
# =========================================================

import numpy as np
import pandas as pd
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

try:
    from sklearn_extra.cluster import KMedoids
except:
    KMedoids = None

# ---------------------------------------------------------
# COMMON FUNCTION
# ---------------------------------------------------------
def clustering_metrics(name, X, labels):
    print(f"\n===== {name} =====")
    print("Silhouette Score:", silhouette_score(X, labels))

    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title(name)
    plt.show()

# =========================================================
# 1) MANUAL DATASET
# =========================================================
print("\n================ MANUAL DATASET =================")

X_manual = np.array([
    [randint(1, 30), randint(1, 30)]
    for _ in range(200)
])

scaler = StandardScaler()
X_manual = scaler.fit_transform(X_manual)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_manual)
clustering_metrics("KMeans", X_manual, labels)

# KMedoids
if KMedoids:
    kmedoids = KMedoids(n_clusters=3, random_state=42)
    labels = kmedoids.fit_predict(X_manual)
    clustering_metrics("KMedoids", X_manual, labels)

# EM
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X_manual)
clustering_metrics("Expectation Maximization", X_manual, labels)

# =========================================================
# 2) SKLEARN DATASET
# =========================================================
print("\n================ SKLEARN DATASET =================")

X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=42
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
clustering_metrics("KMeans", X, labels)

if KMedoids:
    kmedoids = KMedoids(n_clusters=3, random_state=42)
    labels = kmedoids.fit_predict(X)
    clustering_metrics("KMedoids", X, labels)

gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)
clustering_metrics("Expectation Maximization", X, labels)
