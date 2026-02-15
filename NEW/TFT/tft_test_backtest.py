import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

# ==============================
# CONFIG
# ==============================

CHECKPOINT_PATH = "tft-best.ckpt"
TEST_PATH = "test_prepared.csv"

ENCODER_LENGTH = 120
PREDICTION_LENGTH = 3
TARGET = "gold_return_10d"

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(TEST_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["series_id"] = 0
df["time_idx"] = np.arange(len(df))

categorical_cols = [
    "Vol_Regime",
    "Return_Regime",
    "Volume_Regime",
    "Trend_Regime",
    "Risk_State",
    "Market_State",
    "Bull_Bear_Label",
]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# ==============================
# BUILD DATASET
# ==============================

dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target=TARGET,
    group_ids=["series_id"],
    max_encoder_length=ENCODER_LENGTH,
    max_prediction_length=PREDICTION_LENGTH,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=[c for c in df.columns if c not in ["Date", TARGET, "series_id"] + categorical_cols],
    time_varying_known_categoricals=[],
    time_varying_unknown_categoricals=categorical_cols,
    target_normalizer=GroupNormalizer(groups=["series_id"]),
)

loader = dataset.to_dataloader(train=False, batch_size=64)

# ==============================
# LOAD MODEL
# ==============================

model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

# ==============================
# PREDICT
# ==============================

predictions = []
actuals = []

with torch.no_grad():
    for batch in loader:
        x, y = batch
        out = model(x)
        preds = out.prediction.cpu().numpy()
        predictions.append(preds.reshape(-1))
        actuals.append(y[0].cpu().numpy().reshape(-1))

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# ==============================
# METRICS
# ==============================

mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
corr = np.corrcoef(predictions, actuals)[0, 1]
directional_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))

print("\n===== TEST METRICS =====")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Correlation:", round(corr, 4))
print("Directional Accuracy:", round(directional_accuracy, 4))

# ==============================
# BACKTEST STRATEGY
# ==============================

signals = np.sign(predictions)
strategy_returns = signals * actuals

equity_curve = np.cumprod(1 + strategy_returns / 100)

sharpe = np.mean(strategy_returns) / np.std(strategy_returns)

print("Sharpe (raw):", round(sharpe, 4))

# ==============================
# PLOTS
# ==============================

plt.figure()
plt.plot(actuals, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Predicted vs Actual Returns")
plt.show()

plt.figure()
plt.plot(equity_curve)
plt.title("Strategy Equity Curve")
plt.show()
