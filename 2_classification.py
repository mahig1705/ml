
# =========================================================
# CLASSIFICATION TASK
# Algorithms:
# Decision Tree
# KNN
# Naive Bayes
# SVM Linear
# SVM Non Linear
# Bagging
# Random Forest
# Boosting (AdaBoost, GradientBoost, XGBoost,
# CatBoost, LightGBM)
# =========================================================

import numpy as np
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Optional Libraries
try:
    from xgboost import XGBClassifier
except:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except:
    LGBMClassifier = None

# ---------------------------------------------------------
# COMMON FUNCTION
# ---------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(f"\n================ {name} ================")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # Cross Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=kfold,
        scoring='accuracy'
    )

    print("Cross Validation Scores:", cv_scores)
    print("Average CV Score:", cv_scores.mean())

# =========================================================
# 1) MANUAL DATASET
# =========================================================
print("\n================ MANUAL DATASET =================")

data = {
    "Age": [randint(18, 60) for _ in range(200)],
    "Salary": [randint(20000, 100000) for _ in range(200)],
    "Experience": [randint(0, 20) for _ in range(200)],
}

df = pd.DataFrame(data)

df["Purchased"] = (
    (df["Salary"] > 50000) &
    (df["Experience"] > 5)
).astype(int)

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM Linear": SVC(kernel='linear'),
    "SVM RBF": SVC(kernel='rbf'),
    "Bagging": BaggingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boost": GradientBoostingClassifier(),
}

if XGBClassifier:
    models["XGBoost"] = XGBClassifier(eval_metric='logloss')

if CatBoostClassifier:
    models["CatBoost"] = CatBoostClassifier(verbose=0)

if LGBMClassifier:
    models["LightGBM"] = LGBMClassifier()

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

# =========================================================
# 2) SKLEARN DATASET
# =========================================================
print("\n================ SKLEARN DATASET =================")

X, y = make_classification(
    n_samples=500,
    n_features=6,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

# Optional Visualization
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Classification Dataset")
plt.show()
