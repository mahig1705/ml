
# =========================================================
# REGRESSION TASK
# Algorithms:
# 1. Linear Regression
# 2. Multiple Linear Regression
# =========================================================

import numpy as np
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# COMMON FUNCTION
# ---------------------------------------------------------
def evaluate_regression(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n===== REGRESSION METRICS =====")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2  :", r2_score(y_test, y_pred))

    # K-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=kfold, scoring='r2')

    print("\nCross Validation R2 Scores:", cv_scores)
    print("Average CV Score:", cv_scores.mean())

    return y_pred

# =========================================================
# 1) MANUAL DATASET USING randint
# =========================================================
print("\n================ MANUAL DATASET =================")

data = {
    "Hours_Studied": [randint(1, 10) for _ in range(100)],
    "Sleep_Hours": [randint(4, 10) for _ in range(100)],
    "Attendance": [randint(60, 100) for _ in range(100)],
}

df = pd.DataFrame(data)

# Target Variable
df["Marks"] = (
    5 * df["Hours_Studied"]
    + 2 * df["Sleep_Hours"]
    + 0.5 * df["Attendance"]
    + np.random.randint(-10, 10, 100)
)

X = df.drop("Marks", axis=1)
y = df["Marks"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()

y_pred = evaluate_regression(model, X_train, X_test, y_train, y_test)

# Optional Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Manual Dataset Regression")
plt.show()

# =========================================================
# 2) SKLEARN DATASET
# =========================================================
print("\n================ SKLEARN DATASET =================")

X, y = make_regression(
    n_samples=300,
    n_features=4,
    noise=15,
    random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()

y_pred = evaluate_regression(model, X_train, X_test, y_train, y_test)

# Optional Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Sklearn Dataset Regression")
plt.show()
