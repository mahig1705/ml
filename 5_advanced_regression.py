
# =========================================================
# ADVANCED REGRESSION TASK
# Algorithms:
# DecisionTreeRegressor
# RandomForestRegressor
# SVR
# AdaBoostRegressor
# GradientBoostingRegressor
# XGBoost Regressor
# CatBoost Regressor
# LightGBM Regressor
# =========================================================

import numpy as np
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
except:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except:
    LGBMRegressor = None

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
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2  :", r2_score(y_test, y_pred))

    # Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring='r2'
    )

    print("Cross Validation Scores:", scores)
    print("Average CV Score:", scores.mean())

# =========================================================
# 1) MANUAL DATASET
# =========================================================
print("\n================ MANUAL DATASET =================")

data = {
    "Area": [randint(500, 5000) for _ in range(200)],
    "Bedrooms": [randint(1, 6) for _ in range(200)],
    "Age": [randint(1, 20) for _ in range(200)],
}

df = pd.DataFrame(data)

df["Price"] = (
    50 * df["Area"]
    + 100000 * df["Bedrooms"]
    - 5000 * df["Age"]
    + np.random.randint(-50000, 50000, 200)
)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "SVR": SVR(),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
}

if XGBRegressor:
    models["XGBoost Regressor"] = XGBRegressor()

if CatBoostRegressor:
    models["CatBoost Regressor"] = CatBoostRegressor(verbose=0)

if LGBMRegressor:
    models["LightGBM Regressor"] = LGBMRegressor()

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

# =========================================================
# 2) SKLEARN DATASET
# =========================================================
print("\n================ SKLEARN DATASET =================")

X, y = make_regression(
    n_samples=500,
    n_features=5,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

# Optional Visualization
plt.scatter(y_test, models["Random Forest Regressor"].fit(X_train, y_train).predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest Regression")
plt.show()
