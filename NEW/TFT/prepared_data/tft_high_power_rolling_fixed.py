#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

TARGET = "gold_return_10d"
CAT_COLS = ["Vol_Regime", "Trend_Regime", "Risk_State",
            "Market_State", "Bull_Bear_Label"]


def load_data():
    train = pd.read_csv("train_prepared.csv")
    val   = pd.read_csv("val_prepared.csv")
    test  = pd.read_csv("test_prepared.csv")

    for df in [train, val, test]:
        df["Date"] = pd.to_datetime(df["Date"])

    return train, val, test


def prepare_full_dataframe(train, val, test):
    full = pd.concat([train, val, test]).sort_values("Date").reset_index(drop=True)
    full["series_id"] = "XAUUSD"
    full["time_idx"] = np.arange(len(full))
    full["dow"] = full["Date"].dt.dayofweek
    full["month"] = full["Date"].dt.month
    return full


def build_dataset(df):

    df = df.copy()

    for col in CAT_COLS:
        if col in df.columns:
            df.loc[:, col] = df[col].astype("category")

    df.loc[:, "series_id"] = df["series_id"].astype("category")

    known_reals = ["time_idx", "dow", "month"]

    exclude = {"Date", TARGET, *known_reals, "series_id", *CAT_COLS}
    unknown_reals = [c for c in df.columns
                     if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["series_id"],
        max_encoder_length=120,
        max_prediction_length=3,
        static_categoricals=["series_id"],
        time_varying_unknown_categoricals=[c for c in CAT_COLS if c in df.columns],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals + [TARGET],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return dataset


def train_fold(train_df, val_df, batch_size, max_epochs):

    training = build_dataset(train_df)
    validation = TimeSeriesDataSet.from_dataset(training, val_df,
                                                stop_randomization=True)

    train_loader = training.to_dataloader(train=True,
                                          batch_size=batch_size)
    val_loader = validation.to_dataloader(train=False,
                                          batch_size=batch_size)

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=256,
        attention_head_size=8,
        dropout=0.30,
        hidden_continuous_size=128,
        learning_rate=2e-4,
        optimizer="adamw",
        loss=QuantileLoss(),
        reduce_on_plateau_patience=3,
    )

    early_stop = EarlyStopping(monitor="val_loss",
                               patience=6,
                               mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=32,
        gradient_clip_val=0.3,
        callbacks=[early_stop],
        enable_model_summary=False,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_loss"].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=40)
    args = parser.parse_args()

    train, val, test = load_data()
    full = prepare_full_dataframe(train, val, test)

    fold_size = len(full) // 5
    val_losses = []

    for fold in range(3):
        split_point = fold_size * (fold + 2)

        train_df = full.iloc[:split_point].copy()
        val_df = full.iloc[split_point:split_point + fold_size].copy()

        print(f"\n===== Fold {fold+1} =====")

        loss = train_fold(
            train_df,
            val_df,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
        )

        print(f"Fold {fold+1} Val Loss: {loss}")
        val_losses.append(loss)

    print("\nAverage Validation Loss:", np.mean(val_losses))


if __name__ == "__main__":
    main()
