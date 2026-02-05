# train_lgbm_binary_maxpower.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------- CONFIG --------
INPUT_CSV = "C:\\Users\\PC\\Desktop\\test\\train_binary.csv"
TARGET_COL = "direction"
TRAIN_FRAC = 0.98   # Use 98% for training
OUTPUT_MODEL = "lgbm_model_binary_max.pkl"

# -------- LightGBM Highest Accuracy Settings --------
LGB_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,      # lower = better generalization
    "n_estimators": 5000,       # more trees = deeper learning
    "num_leaves": 512,          # more splits
    "max_depth": -1,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 0.05,
    "verbosity": -1,
    "num_threads": -1           # use all CPU cores
}

# -------- Load ----------
df = pd.read_csv(INPUT_CSV)
print("Loaded:", INPUT_CSV, "rows:", len(df))

# Ensure binary direction exists
if TARGET_COL not in df.columns:
    raise SystemExit("Target column 'direction' not found. Abort.")

# Features = everything except target
features = [c for c in df.columns if c != TARGET_COL]

# Numeric convert
for c in features + [TARGET_COL]:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Fill missing
df = df.ffill().bfill()

# Final X,y
X = df[features].copy()
y = df[TARGET_COL].astype(int)

# Chronological split
n = len(df)
train_n = int(n * TRAIN_FRAC)
X_train, y_train = X.iloc[:train_n], y.iloc[:train_n]
X_test,  y_test  = X.iloc[train_n:], y.iloc[train_n:]

print(f"Rows total: {n}, train: {len(X_train)}, test: {len(X_test)}")

# Train
model = lgb.LGBMClassifier(**LGB_PARAMS)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, OUTPUT_MODEL)
print("✅ Saved model ->", OUTPUT_MODEL)