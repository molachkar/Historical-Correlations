import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import time

# ==========================
# CONFIG
# ==========================

DATA_PATHS = [
    "xauusd_train_pruned.csv",
    "xauusd_val_pruned.csv",
    "xauusd_test_pruned.csv"
]

TARGET = "y_next_log_return"
DATE_COL = "Date"
N_FOLDS = 5
SEED = 42

# ==========================
# METRICS
# ==========================

def ic(y, p):
    if np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0,1])

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

# ==========================
# LOAD DATA
# ==========================

print("Loading data...")

dfs = []
for path in DATA_PATHS:
    dfs.append(pd.read_csv(path))

df = pd.concat(dfs, ignore_index=True)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

print("Total rows:", len(df))
print("Total features:", len(df.columns) - 2)

# ==========================
# PREP FEATURES
# ==========================

X = df.drop(columns=[TARGET, DATE_COL])
y = df[TARGET].astype("float64")

# Convert object columns to category
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category")

# ==========================
# MODEL BUILDER
# ==========================

def build_model():
    return lgb.LGBMRegressor(
        n_estimators=30000,
        learning_rate=0.03,
        num_leaves=64,
        min_data_in_leaf=80,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1
    )

# ==========================
# WALK-FORWARD CV
# ==========================

n = len(df)
fold_size = n // (N_FOLDS + 1)

results = []
importance_storage = []

print("\nStarting Walk-Forward CV\n")

for fold in range(N_FOLDS):

    train_end = fold_size * (fold + 1)
    test_end = fold_size * (fold + 2)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[train_end:test_end]
    y_test = y.iloc[train_end:test_end]

    print(f"Fold {fold+1}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    model = build_model()

    start = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l2",
        categorical_feature="auto",
        callbacks=[lgb.early_stopping(300)]
    )

    duration = round(time.time() - start, 2)

    best_iter = model.best_iteration_

    preds = model.predict(X_test, num_iteration=best_iter)

    fold_ic = ic(y_test, preds)
    fold_rmse = rmse(y_test, preds)

    print(f"IC: {round(fold_ic,4)} | RMSE: {round(fold_rmse,6)} | Iter: {best_iter} | Time: {duration}s")
    print("-"*50)

    results.append(fold_ic)

    imp = model.booster_.feature_importance(importance_type="gain")
    importance_storage.append(pd.Series(imp, index=X.columns))

# ==========================
# SUMMARY
# ==========================

print("\n========================")
print("CV SUMMARY")
print("========================")

print("IC per fold:", [round(r,4) for r in results])
print("Mean IC:", round(np.nanmean(results),4))
print("Std IC :", round(np.nanstd(results),4))

importance_df = pd.concat(importance_storage, axis=1)
importance_df.columns = [f"Fold_{i+1}" for i in range(N_FOLDS)]

mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)

print("\nTop Stable Features:")
print(mean_importance.head(10))

print("\nDone.")