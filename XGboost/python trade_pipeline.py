import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

TRAIN_PATH = "xauusd_train_pruned.csv"
VAL_PATH   = "xauusd_val_pruned.csv"
TEST_PATH  = "xauusd_test_pruned.csv"

DATE_COL   = "Date"
TARGET_COL = "y_next_log_return"

Z_WINDOW      = 252
VOL_WINDOW    = 20
Z_THRESHOLD   = 0.8
SMOOTH_ALPHA  = 0.2
MAX_LEVERAGE  = 1.0
COST_PER_TURN = 0.0005
SEED = 42

def ic(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

def sharpe(daily_returns):
    r = np.asarray(daily_returns, float)
    r = r[~np.isnan(r)]
    if r.size < 2 or np.std(r) == 0:
        return np.nan
    return float((np.mean(r) / np.std(r)) * np.sqrt(252))

def max_drawdown(equity_curve):
    eq = np.asarray(equity_curve, float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(np.min(dd))

def make_xy(df):
    X = df.drop(columns=[DATE_COL, TARGET_COL], errors="ignore").copy()
    y = df[TARGET_COL].astype("float64").copy()

    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")
        else:
            if not np.issubdtype(X[c].dtype, np.number) and str(X[c].dtype) != "category":
                X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")

    # drop accidental leakage
    leak = [c for c in X.columns if c.startswith("y_")]
    X.drop(columns=leak, inplace=True, errors="ignore")
    return X, y

def align_categories(X_ref, X_other, cat_cols):
    # freeze category set from X_ref and apply to X_other
    for c in cat_cols:
        cats = X_ref[c].cat.categories
        X_other[c] = X_other[c].astype("category")
        X_other[c] = X_other[c].cat.set_categories(cats)
    return X_other

# -----------------------------
# LOAD
# -----------------------------
train = pd.read_csv(TRAIN_PATH)
val   = pd.read_csv(VAL_PATH)
test  = pd.read_csv(TEST_PATH)

for df in (train, val, test):
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
    df.sort_values(DATE_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

dev = pd.concat([train, val], ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)

X_dev, y_dev   = make_xy(dev)
X_test, y_test = make_xy(test)

# identify categorical columns from dev
cat_cols = [c for c in X_dev.columns if str(X_dev[c].dtype) == "category"]

# align test categories to dev categories
X_test = align_categories(X_dev, X_test, cat_cols)

# -----------------------------
# TRAIN FINAL MODEL
# -----------------------------
model = lgb.LGBMRegressor(
    n_estimators=50000,
    learning_rate=0.02,
    num_leaves=128,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=1.0,
    lambda_l2=1.0,
    max_bin=255,
    random_state=SEED,
    n_jobs=-1,
    objective="regression",
    verbosity=-1
)

print("Training final model on train+val...")
model.fit(X_dev, y_dev, categorical_feature=cat_cols if cat_cols else "auto")

# -----------------------------
# TEST METRICS
# -----------------------------
print("Predicting on test...")
pred_test = model.predict(X_test)

print("\n--- TEST PREDICTION METRICS ---")
print("IC   :", round(ic(y_test, pred_test), 4))
print("RMSE :", round(rmse(y_test, pred_test), 6))
print("MAE  :", round(float(mean_absolute_error(y_test, pred_test)), 6))

# -----------------------------
# BUILD SIGNAL USING DEV+TEST HISTORY
# -----------------------------
X_all = pd.concat([X_dev, X_test], axis=0, ignore_index=True)

# align combined categories too (safe)
X_all = align_categories(X_dev, X_all, cat_cols)

pred_all = model.predict(X_all)

sig_all = pd.DataFrame({
    DATE_COL: pd.concat([dev[DATE_COL], test[DATE_COL]], ignore_index=True),
    "pred": pred_all,
    TARGET_COL: pd.concat([y_dev, y_test], ignore_index=True),
})

mu = sig_all["pred"].shift(1).rolling(Z_WINDOW, min_periods=50).mean()
sd = sig_all["pred"].shift(1).rolling(Z_WINDOW, min_periods=50).std(ddof=0)
sig_all["pred_z"] = (sig_all["pred"] - mu) / sd.replace(0, np.nan)

sig_all["vol"] = sig_all[TARGET_COL].shift(1).rolling(VOL_WINDOW, min_periods=10).std(ddof=0)

sig_all["signal"] = 0
sig_all.loc[sig_all["pred_z"] >  Z_THRESHOLD, "signal"] = 1
sig_all.loc[sig_all["pred_z"] < -Z_THRESHOLD, "signal"] = -1

eps = 1e-12
sig_all["pos_raw"] = sig_all["signal"] / (sig_all["vol"] + eps)
sig_all["pos_raw"] = sig_all["pos_raw"].clip(-MAX_LEVERAGE, MAX_LEVERAGE).fillna(0.0)

pos = np.zeros(len(sig_all), dtype=float)
for i in range(1, len(sig_all)):
    pos[i] = (1.0 - SMOOTH_ALPHA) * pos[i-1] + SMOOTH_ALPHA * sig_all["pos_raw"].iloc[i]
sig_all["position"] = pos

# -----------------------------
# STRATEGY METRICS ON TEST ONLY
# -----------------------------
sig = sig_all.iloc[len(dev):].copy().reset_index(drop=True)

sig["position_lag"] = sig["position"].shift(1).fillna(0.0)
sig["turnover"] = (sig["position"] - sig["position"].shift(1)).abs().fillna(0.0)
sig["cost"] = sig["turnover"] * COST_PER_TURN
sig["strategy_ret"] = sig["position_lag"] * sig[TARGET_COL] - sig["cost"]
sig["equity"] = (1.0 + sig["strategy_ret"].fillna(0.0)).cumprod()

print("\n--- TEST STRATEGY METRICS (with costs) ---")
print("Sharpe       :", round(sharpe(sig['strategy_ret']), 3))
print("Max Drawdown :", round(max_drawdown(sig['equity']), 3))
print("Avg Turnover :", round(float(sig['turnover'].mean()), 4))
print("Final Equity :", round(float(sig['equity'].iloc[-1]), 3))
print("Trade days % :", round(float((sig['signal'] != 0).mean() * 100), 2))

out = sig[[DATE_COL, "signal", "position"]].copy()
out.to_csv("signals_for_metatrader.csv", index=False)
print("\nSaved: signals_for_metatrader.csv")