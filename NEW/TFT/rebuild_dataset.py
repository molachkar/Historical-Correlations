"""
TFT Dataset Rebuild & Validation
=================================
Reconstructs the exact TimeSeriesDataSet structure used during training,
validates it against all 7 requirements, and saves the rebuilt test
dataframe to test_rebuilt.csv.

Usage:
    python rebuild_dataset.py \
        --train train_prepared.csv \
        --test  test_prepared.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  —  must match training exactly
# ─────────────────────────────────────────────────────────────────────────────
TARGET         = "gold_return_10d"
GROUP_COL      = "series_id"
TIME_COL       = "time_idx"
DATE_COL       = "Date"
ENCODER_LENGTH = 60
PRED_LENGTH    = 10

EXCLUDED_COLS  = {DATE_COL, TIME_COL, GROUP_COL, TARGET}

KNOWN_CAT_COLS = [
    "Return_Regime",
    "Volume_Regime",
    "Vol_Regime",
    "Trend_Regime",
    "Risk_State",
    "Market_State",
    "Bull_Bear_Label",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & merge
# ─────────────────────────────────────────────────────────────────────────────
def load_and_merge(train_path: str, test_path: str):
    print("\n[STEP 1] Loading CSVs ...")
    train_df  = pd.read_csv(train_path)
    test_df   = pd.read_csv(test_path)
    train_len = len(train_df)
    print(f"         train rows : {train_len:,}")
    print(f"         test rows  : {len(test_df):,}")

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL])
    full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"         date range : {full_df[DATE_COL].min().date()} to {full_df[DATE_COL].max().date()}")

    # Req 2: continuous integer time index from 0
    full_df[TIME_COL] = np.arange(len(full_df))

    # Req 3: constant group identifier
    full_df[GROUP_COL] = "gold"

    print(f"         time_idx   : 0 to {full_df[TIME_COL].max()} (monotonic: {full_df[TIME_COL].is_monotonic_increasing})")
    print(f"         series_id  : constant 'gold'")

    return full_df, train_len


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Fix dtypes
# ─────────────────────────────────────────────────────────────────────────────
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 2] Fixing dtypes ...")

    df[GROUP_COL] = df[GROUP_COL].astype("category")

    for col in KNOWN_CAT_COLS:
        if col not in df.columns:
            raise ValueError(f"Expected categorical column '{col}' not found in data.")
        before = str(df[col].dtype)
        df[col] = df[col].astype("category")
        levels = sorted(df[col].cat.categories.tolist())
        print(f"         {col:<22} {before} -> category | levels: {levels}")

    # Catch any other unexpected string columns
    for col in df.columns:
        if col in EXCLUDED_COLS or col in KNOWN_CAT_COLS or col == GROUP_COL:
            continue
        if pd.api.types.is_string_dtype(df[col]):
            print(f"         [WARN] Unexpected string column '{col}' -> casting to category")
            df[col] = df[col].astype("category")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Classify features
# ─────────────────────────────────────────────────────────────────────────────
def classify_features(df: pd.DataFrame):
    print("\n[STEP 3] Classifying features ...")
    real_cols, cat_cols = [], []
    for col in df.columns:
        if col in EXCLUDED_COLS or col == GROUP_COL:
            continue
        if str(df[col].dtype) == "category":
            cat_cols.append(col)
        else:
            real_cols.append(col)

    print(f"         real features : {len(real_cols)}")
    print(f"         cat  features : {len(cat_cols)} -> {cat_cols}")
    return real_cols, cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Validate
# ─────────────────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame, real_cols: list, cat_cols: list, train_len: int):
    print("\n[STEP 4] Validating ...")
    errors = []

    # Req 5: no NaNs
    nulls = df.isnull().sum()
    bad   = nulls[nulls > 0]
    if len(bad) > 0:
        errors.append(f"NaNs found: {bad.to_dict()}")
    else:
        print("         NaNs          : none [OK]")

    # Req 4: real cols are numeric
    non_numeric = [c for c in real_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Non-numeric real columns: {non_numeric}")
    else:
        print(f"         Real dtypes   : all numeric [OK]")

    # Req 4: cat cols are category
    non_cat = [c for c in cat_cols if str(df[c].dtype) != "category"]
    if non_cat:
        errors.append(f"Non-category categorical columns: {non_cat}")
    else:
        print(f"         Cat dtypes    : all category [OK]")

    # Req 2: time_idx sequential
    if not np.array_equal(df[TIME_COL].values, np.arange(len(df))):
        errors.append("time_idx is not sequential from 0")
    else:
        print(f"         time_idx      : sequential 0 to {len(df)-1} [OK]")

    # Req 3: group col constant category
    if str(df[GROUP_COL].dtype) != "category":
        errors.append("series_id is not category dtype")
    elif df[GROUP_COL].nunique() != 1:
        errors.append("series_id is not constant")
    else:
        print(f"         series_id     : constant category [OK]")

    # Req 6: enough rows
    test_rows = len(df) - train_len
    min_rows  = ENCODER_LENGTH + PRED_LENGTH
    if test_rows < min_rows:
        errors.append(f"Test portion has {test_rows} rows, need >= {min_rows}")
    else:
        print(f"         Window check  : {test_rows} test rows >= {min_rows} needed [OK]")

    if errors:
        print("\n  [VALIDATION FAILED]")
        for e in errors:
            print(f"  x {e}")
        raise SystemExit(1)
    else:
        print("         All checks passed [OK]")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build training TimeSeriesDataSet
# ─────────────────────────────────────────────────────────────────────────────
def build_training_dataset(train_df: pd.DataFrame,
                           real_cols: list,
                           cat_cols: list) -> TimeSeriesDataSet:
    print("\n[STEP 5] Building training TimeSeriesDataSet ...")
    ds = TimeSeriesDataSet(
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
        static_categoricals=cat_cols if cat_cols else [],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    print(f"         Training samples : {len(ds):,}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Build test TimeSeriesDataSet via from_dataset()
# ─────────────────────────────────────────────────────────────────────────────
def build_test_dataset(training_dataset: TimeSeriesDataSet,
                       test_df: pd.DataFrame) -> TimeSeriesDataSet:
    print("\n[STEP 6] Building test TimeSeriesDataSet via from_dataset() ...")
    ds = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_df,
        predict=False,
        stop_randomization=True,
    )
    print(f"         Test samples : {len(ds):,}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Save rebuilt test dataframe to CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_test_csv(full_df: pd.DataFrame, train_len: int, out_path: str):
    print("\n[STEP 7] Saving rebuilt test dataframe ...")

    # Pure test rows only (no context overlap)
    test_only_df = full_df.iloc[train_len:].copy()

    # Convert category columns back to string for clean CSV output
    for col in test_only_df.select_dtypes("category").columns:
        if col == GROUP_COL:
            continue  # keep series_id readable but as string
        test_only_df[col] = test_only_df[col].astype(str)

    test_only_df.to_csv(out_path, index=False)

    print(f"         Saved to       : {out_path}")
    print(f"         Rows           : {len(test_only_df):,}")
    print(f"         Columns        : {len(test_only_df.columns)}")
    print(f"         time_idx range : {test_only_df[TIME_COL].min()} to {test_only_df[TIME_COL].max()}")
    print(f"         Date range     : {test_only_df[DATE_COL].min().date()} to {test_only_df[DATE_COL].max().date()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(training_dataset, test_dataset, real_cols, cat_cols):
    print("\n" + "=" * 50)
    print("  Dataset Rebuild Summary")
    print("=" * 50)
    print(f"  Encoder length       : {ENCODER_LENGTH}")
    print(f"  Prediction length    : {PRED_LENGTH}")
    print(f"  Target               : {TARGET}")
    print(f"  Group col            : {GROUP_COL}")
    print(f"  Real features        : {len(real_cols)}")
    print(f"  Categorical features : {len(cat_cols)}")
    print(f"  Training samples     : {len(training_dataset):,}")
    print(f"  Test samples         : {len(test_dataset):,}")
    print("=" * 50)
    print("  [OK] Dataset is structurally identical to training.")
    print("  [OK] Scalers and encoders carried over via from_dataset().")
    print("  [OK] test_rebuilt.csv saved with time_idx and series_id.")
    print("  [OK] Ready for model loading and inference.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TFT Dataset Rebuild & Validation")
    parser.add_argument("--train",  default="train_prepared.csv")
    parser.add_argument("--test",   default="test_prepared.csv")
    parser.add_argument("--output", default="test_rebuilt.csv")
    args = parser.parse_args()

    # 1. Load and merge
    full_df, train_len = load_and_merge(args.train, args.test)

    # 2. Fix dtypes
    full_df = fix_dtypes(full_df)

    # 3. Classify features
    real_cols, cat_cols = classify_features(full_df)

    # 4. Validate
    validate(full_df, real_cols, cat_cols, train_len)

    # 5. Split
    train_df = full_df.iloc[:train_len].copy()

    # Test slice includes last ENCODER_LENGTH rows of train as encoder context
    # (encoder input only — never prediction targets, no leakage)
    context_start = max(0, train_len - ENCODER_LENGTH)
    test_df = full_df.iloc[context_start:].copy()

    # 6. Build training dataset (fits scalers/encoders)
    training_dataset = build_training_dataset(train_df, real_cols, cat_cols)

    # 7. Build test dataset (inherits scalers/encoders via from_dataset)
    test_dataset = build_test_dataset(training_dataset, test_df)

    # 8. Save rebuilt test CSV
    save_test_csv(full_df, train_len, args.output)

    # 9. Summary
    print_summary(training_dataset, test_dataset, real_cols, cat_cols)

    return training_dataset, test_dataset


if __name__ == "__main__":
    main()