
# =========================================================
# DIMENSIONALITY REDUCTION
# Algorithms:
# PCA
# SVD
# LDA
# =========================================================

import numpy as np
import pandas as pd
from random import randint

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# COMMON FUNCTION
# ---------------------------------------------------------
def visualize(title, X_reduced, y=None):
    plt.figure(figsize=(6,5))

    if y is not None:
        plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y)
    else:
        plt.scatter(X_reduced[:,0], X_reduced[:,1])

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

# =========================================================
# 1) MANUAL DATASET
# =========================================================
print("\n================ MANUAL DATASET =================")

X_manual = np.array([
    [randint(1,100), randint(1,100),
     randint(1,100), randint(1,100)]
    for _ in range(200)
])

y_manual = np.array([randint(0,1) for _ in range(200)])

scaler = StandardScaler()
X_manual = scaler.fit_transform(X_manual)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_manual)

print("\nPCA Explained Variance:", pca.explained_variance_ratio_)
visualize("PCA", X_pca)

# SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_manual)

print("\nSVD Explained Variance:", svd.explained_variance_ratio_)
visualize("SVD", X_svd)

# LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_manual, y_manual)

print("\nLDA Shape:", X_lda.shape)

# =========================================================
# 2) SKLEARN DATASET
# =========================================================
print("\n================ SKLEARN DATASET =================")

iris = load_iris()

X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("\nPCA Explained Variance:", pca.explained_variance_ratio_)
visualize("PCA - Iris", X_pca, y)

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

print("\nSVD Explained Variance:", svd.explained_variance_ratio_)
visualize("SVD - Iris", X_svd, y)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

visualize("LDA - Iris", X_lda, y)
