import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# -------- CONFIG --------
INPUT_CSV = "C:\\Users\\PC\\Desktop\\MML\\clean.csv"
TARGET_COL = "direction"
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
OUTPUT_MODEL = "lightgbm-direction-model.pkl"

# -------- LOAD --------
df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {len(df)} rows")

# -------- DROP LEAKAGE COLUMNS --------
drop_cols = [
    'xauusd-close', 'xauusd-open', 'xauusd-high', 'xauusd-low',
    'gold_return', 'date', 'time', 'timestamp'
]
drop_cols = [col for col in drop_cols if col in df.columns]
print(f"Dropping: {drop_cols}")

X = df.drop(columns=drop_cols + [TARGET_COL], errors='ignore')
y = df[TARGET_COL].astype(int)

# -------- CLEAN DATA --------
X = X.replace([np.inf, -np.inf], np.nan)
X = X.ffill().bfill()

# Remove any remaining NaN rows
valid_idx = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_idx].reset_index(drop=True)
y = y[valid_idx].reset_index(drop=True)

print(f"Clean data: {len(X)} rows, {len(X.columns)} features")

# -------- CHRONOLOGICAL SPLIT --------
n = len(X)
train_n = int(n * TRAIN_FRAC)
val_n = int(n * (TRAIN_FRAC + VAL_FRAC))

X_train = X.iloc[:train_n]
y_train = y.iloc[:train_n]
X_val = X.iloc[train_n:val_n]
y_val = y.iloc[train_n:val_n]
X_test = X.iloc[val_n:]
y_test = y.iloc[val_n:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# -------- CLASS BALANCE --------
class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
print(f"Scale pos weight: {scale_pos_weight:.3f}")

# -------- OPTIMIZED LIGHTGBM PARAMETERS --------
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'num_leaves': 20,              # Increased from 15 (more features now)
    'max_depth': 5,                # Increased from 4
    'min_child_samples': 80,       # Reduced from 100 (more data variety)
    'min_child_weight': 0.01,
    'subsample': 0.75,             # Increased from 0.7
    'subsample_freq': 5,
    'colsample_bytree': 0.75,      # Increased from 0.7 (more features to sample)
    'reg_alpha': 1.5,              # Reduced from 2.0
    'reg_lambda': 1.5,             # Reduced from 2.0
    'scale_pos_weight': scale_pos_weight,
    'max_bin': 128,
    'min_split_gain': 0.05,        # Reduced from 0.1
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# -------- TRAIN WITH EARLY STOPPING --------
model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

# -------- EVALUATE --------
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_val = model.predict_proba(X_val)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print("\nTRAIN SET:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"AUC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")

print("\nVALIDATION SET:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"AUC: {roc_auc_score(y_val, y_pred_proba_val):.4f}")

print("\nTEST SET:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['Down', 'Up']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

# -------- THRESHOLD OPTIMIZATION --------
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

thresholds = np.arange(0.35, 0.70, 0.05)
best_threshold = 0.5
best_accuracy = 0

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba_test >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    cm = confusion_matrix(y_test, y_pred_thresh)
    
    # Calculate precision for class 1 (up)
    precision_up = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    recall_up = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    print(f"Threshold {thresh:.2f}: Acc={acc:.4f}, Precision(Up)={precision_up:.4f}, Recall(Up)={recall_up:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = thresh

print(f"\nBest Threshold: {best_threshold:.2f} with Accuracy: {best_accuracy:.4f}")

# -------- FEATURE IMPORTANCE --------
fi = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 25 FEATURES")
print("="*60)
print(fi.head(25).to_string(index=False))

# -------- SAVE --------
joblib.dump(model, OUTPUT_MODEL)
joblib.dump(list(X.columns), 'feature_names.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl')
fi.to_csv('feature_importance.csv', index=False)

print("\n" + "="*60)
print(f"Saved: {OUTPUT_MODEL}")
print(f"Saved: feature_names.pkl")
print(f"Saved: best_threshold.pkl")
print(f"Saved: feature_importance.csv")
print("="*60)