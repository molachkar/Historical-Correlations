"""
TFT Training Script — Google Colab Ready
=========================================
Trains a Temporal Fusion Transformer on gold_return_10d.

Requirements:
    pip install pytorch-forecasting pytorch-lightning

Usage:
    Upload train_prepared.csv and val_prepared.csv to Colab, then run.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Installs (uncomment in Colab)
# ─────────────────────────────────────────────────────────────────────────────
# !pip install -q pytorch-forecasting pytorch-lightning

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED, workers=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET         = "gold_return_10d"
DATE_COL       = "Date"
TIME_COL       = "time_idx"
GROUP_COL      = "series_id"
ENCODER_LENGTH = 60
PRED_LENGTH    = 10
BATCH_SIZE     = 64
MAX_EPOCHS     = 100
OUTPUT_DIR     = "./tft_out"

KNOWN_CAT_COLS = [
    "Return_Regime",
    "Volume_Regime",
    "Vol_Regime",
    "Trend_Regime",
    "Risk_State",
    "Market_State",
    "Bull_Bear_Label",
]

EXCLUDED_COLS = {DATE_COL, TIME_COL, GROUP_COL, TARGET}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data ...")
train_raw = pd.read_csv("train_prepared.csv")
val_raw   = pd.read_csv("val_prepared.csv")
print(f"    train raw shape : {train_raw.shape}")
print(f"    val   raw shape : {val_raw.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge → assign time_idx → split back
#    time_idx must be continuous across both splits (no gap at the boundary)
# ─────────────────────────────────────────────────────────────────────────────
train_len = len(train_raw)

full_df = pd.concat([train_raw, val_raw], ignore_index=True)
full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL])
full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)
full_df[TIME_COL]  = np.arange(len(full_df))   # 0, 1, 2, ... no gaps
full_df[GROUP_COL] = "0"                        # single series, string first

train_df = full_df.iloc[:train_len].copy()
val_df   = full_df.iloc[train_len:].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Fix dtypes
#    - Known categoricals: str → category  (catches pandas 2.x str dtype too)
#    - GROUP_COL: str → category
# ─────────────────────────────────────────────────────────────────────────────
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str).astype("category")
    for col in KNOWN_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")
    # Safety: catch any remaining raw string columns
    for col in df.columns:
        if col in EXCLUDED_COLS or col in KNOWN_CAT_COLS or col == GROUP_COL:
            continue
        if pd.api.types.is_string_dtype(df[col]):
            print(f"    [warn] Unexpected string col '{col}' → category")
            df[col] = df[col].astype(str).astype("category")
    return df

train_df = fix_dtypes(train_df)
val_df   = fix_dtypes(val_df)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Detect real-valued feature columns
# ─────────────────────────────────────────────────────────────────────────────
cat_cols  = [c for c in KNOWN_CAT_COLS if c in train_df.columns]
real_cols = [
    c for c in train_df.columns
    if c not in EXCLUDED_COLS
    and c not in cat_cols
    and c != GROUP_COL
    and pd.api.types.is_numeric_dtype(train_df[c])
]

print(f"\n[2] Features detected")
print(f"    Real  ({len(real_cols)}) : {real_cols}")
print(f"    Cat   ({len(cat_cols)})  : {cat_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Build TimeSeriesDataSet
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Building TimeSeriesDataSet ...")

training_dataset = TimeSeriesDataSet(
    train_df,
    time_idx=TIME_COL,
    target=TARGET,
    group_ids=[GROUP_COL],
    min_encoder_length=ENCODER_LENGTH,
    max_encoder_length=ENCODER_LENGTH,
    min_prediction_length=PRED_LENGTH,
    max_prediction_length=PRED_LENGTH,
    time_varying_known_reals=[TIME_COL],
    time_varying_unknown_reals=real_cols + [TARGET],
    static_categoricals=cat_cols,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Validation inherits all scalers/encoders — never re-fitted
validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    val_df,
    predict=False,
    stop_randomization=True,
)

print(f"    Training samples   : {len(training_dataset):,}")
print(f"    Validation samples : {len(validation_dataset):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
train_loader = training_dataset.to_dataloader(
    train=True,
    batch_size=BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
)
val_loader = validation_dataset.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Building TFT model ...")

model = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.2,
    hidden_continuous_size=32,
    loss=QuantileLoss(),
    optimizer="adam",
    log_interval=10,
    reduce_on_plateau_patience=3,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"    Parameters : {n_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Callbacks
# ─────────────────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    verbose=True,
)

checkpoint = ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    filename="tft-best",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Trainer
#     precision=32  → AMP disabled (avoids overflow on financial targets)
#     gradient_clip → prevents exploding gradients common in TFT
# ─────────────────────────────────────────────────────────────────────────────
device    = "gpu" if torch.cuda.is_available() else "cpu"
n_devices = 1

print(f"\n[5] Training on {device.upper()} ...")

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator=device,
    devices=n_devices,
    precision=32,                   # AMP disabled
    gradient_clip_val=0.1,
    callbacks=[early_stop, checkpoint],
    enable_progress_bar=True,
    enable_model_summary=True,
    log_every_n_steps=10,
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

# ─────────────────────────────────────────────────────────────────────────────
# 12. Results
# ─────────────────────────────────────────────────────────────────────────────
best_ckpt = checkpoint.best_model_path
best_loss = checkpoint.best_model_score

print("\n" + "=" * 50)
print("  Training Complete")
print("=" * 50)
print(f"  Best val_loss  : {best_loss:.6f}")
print(f"  Checkpoint     : {best_ckpt}")
print("=" * 50 + "\n")