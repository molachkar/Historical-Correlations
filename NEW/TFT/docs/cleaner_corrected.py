"""\
LEAKAGE-SAFE DATA PREPARATION PIPELINE
- Causal indicators computed on full timeline (past-only: rolling/ewm)
- Target computed BEFORE global features
- Warmup automatically removed (includes lagged features)
- Global indicators derived from TRAIN only and applied to val/test
- Per-split row sanitization (drops any remaining NaNs; keeps schema identical)
Outputs: train/val/test CSVs + scaler.pkl + thresholds.json
"""

import os
import json
import pickle
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class Config:
    DATA_PATH: str = "master_dataset_clean.csv"
    OUTPUT_DIR: str = "prepared_data"

    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.20
    TEST_RATIO: float = 0.10

    PREDICTION_HORIZON: int = 10
    TARGET_COL: str = "gold_return_10d"
    PRICE_COL: str = "Close_XAUUSD"
    VOLUME_COL: str = "Volume_XAUUSD"
    VIX_COL: str = "Close_VIX"

    EMA_PERIODS = (9, 21, 50, 200)
    SMA_PERIODS = (9, 21, 50, 200)
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2


class TechnicalIndicators:
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # causal smoothing
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
        ma = series.rolling(window=period, min_periods=period).mean()
        sd = series.rolling(window=period, min_periods=period).std(ddof=0)
        upper = ma + num_std * sd
        lower = ma - num_std * sd
        width = upper - lower
        return upper, ma, lower, width


def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_data(cfg: Config) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(cfg.DATA_PATH)
    _require_cols(df, ["Date", cfg.PRICE_COL, cfg.VOLUME_COL, cfg.VIX_COL])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # numeric coercion for key columns
    for c in [cfg.PRICE_COL, cfg.VOLUME_COL, cfg.VIX_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  Loaded: {len(df)} rows x {df.shape[1]} columns")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def compute_technical_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("\nComputing technical indicators (causal, full timeline)...")
    price = df[cfg.PRICE_COL]

    for p in cfg.EMA_PERIODS:
        df[f"EMA_{p}"] = TechnicalIndicators.ema(price, p)
    for p in cfg.SMA_PERIODS:
        df[f"SMA_{p}"] = TechnicalIndicators.sma(price, p)

    df["RSI_14"] = TechnicalIndicators.rsi(price, cfg.RSI_PERIOD)

    macd, sig, hist = TechnicalIndicators.macd(price, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
    df["MACD"] = macd
    df["MACD_Signal"] = sig
    df["MACD_Hist"] = hist

    up, mid, low, width = TechnicalIndicators.bollinger(price, cfg.BB_PERIOD, cfg.BB_STD)
    df["BB_Upper"] = up
    df["BB_Middle"] = mid
    df["BB_Lower"] = low
    df["BB_Width"] = width

    df["Rolling_Std_20"] = price.rolling(window=20, min_periods=20).std(ddof=0)

    df["Close_Returns"] = price.pct_change()
    df["Log_Returns"] = np.log(price / price.shift(1))

    # lagged known-at-t feature must be computed BEFORE split
    df["VIX_Lagged"] = df[cfg.VIX_COL].shift(1)

    return df


def create_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("\nCreating target (future return)...")
    h = cfg.PREDICTION_HORIZON
    df[cfg.TARGET_COL] = (df[cfg.PRICE_COL].shift(-h) / df[cfg.PRICE_COL] - 1.0) * 100.0
    # drop tail rows where target is unknowable
    df = df[df[cfg.TARGET_COL].notna()].reset_index(drop=True)
    print(f"  Removed last {h} rows (no future label)")
    return df


def auto_remove_warmup(df: pd.DataFrame) -> pd.DataFrame:
    print("\nAuto-detecting warmup period...")
    # warmup must include lagged features + all rolling/ewm outputs
    indicator_prefixes = ("EMA_", "SMA_", "RSI", "MACD", "BB_", "Rolling_", "Close_Returns", "Log_Returns")
    indicator_cols = [c for c in df.columns if c.startswith(indicator_prefixes)]
    # also require VIX_Lagged
    if "VIX_Lagged" in df.columns:
        indicator_cols.append("VIX_Lagged")

    # find first index where all required cols are non-null
    mask = df[indicator_cols].notna().all(axis=1)
    if not mask.any():
        raise ValueError("Warmup removal failed: no row has all indicators available.")

    first_valid = int(np.argmax(mask.to_numpy()))
    if first_valid > 0:
        print(f"  Warmup rows removed: {first_valid}")
        df = df.iloc[first_valid:].reset_index(drop=True)
    else:
        print("  Warmup rows removed: 0")
    return df


def split_data(df: pd.DataFrame, cfg: Config):
    print("\nSplitting data (time-ordered 70/20/10)...")
    n = len(df)
    train_end = int(n * cfg.TRAIN_RATIO)
    val_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"  Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    print(f"  Val:   {len(val)} rows ({val['Date'].min().date()} to {val['Date'].max().date()})")
    print(f"  Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")
    return train, val, test


def _make_edges_from_train(train_series: pd.Series, probs, name: str):
    s = train_series.dropna().astype(float)
    if len(s) < 50:
        raise ValueError(f"Not enough non-NaN samples in train to compute edges for {name}.")

    qs = s.quantile(probs).to_numpy(dtype=float)
    edges = [-np.inf] + qs.tolist() + [np.inf]

    # ensure strictly increasing for finite edges
    out = [edges[0]]
    for i in range(1, len(edges) - 1):
        prev = out[-1]
        cur = float(edges[i])
        if not np.isfinite(cur):
            cur = prev
        if np.isfinite(prev):
            # bump if needed
            if cur <= prev:
                # epsilon relative to magnitude
                eps = max(1e-12, abs(prev) * 1e-12)
                cur = prev + eps
        out.append(cur)
    out.append(edges[-1])

    # final sanity: need at least 4 edges => 3 bins
    if len(out) < 4:
        raise ValueError(f"Edges collapsed for {name}.")
    return out


def _percentile_from_sorted(train_sorted: np.ndarray, x: float) -> float:
    if not np.isfinite(x) or train_sorted.size == 0:
        return np.nan
    pos = np.searchsorted(train_sorted, x, side="right")
    return float(np.clip(pos / train_sorted.size * 100.0, 0.0, 100.0))


def compute_global_indicators(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: Config):
    print("\nComputing global indicators (train-derived, frozen)...")
    thresholds = {}

    # TRAIN references
    price_tr = train[cfg.PRICE_COL].astype(float)
    vol_tr = train[cfg.VOLUME_COL].astype(float)
    ret_tr = train["Close_Returns"].astype(float)
    lret_tr = train["Log_Returns"].astype(float)
    sig_tr = train["Rolling_Std_20"].astype(float)

    # Stats for z-score/minmax
    thresholds["price_mean"] = float(price_tr.mean())
    thresholds["price_std"] = float(price_tr.std(ddof=0))
    thresholds["price_min"] = float(price_tr.min())
    thresholds["price_max"] = float(price_tr.max())
    thresholds["price_all_time_high"] = float(price_tr.max())
    thresholds["price_all_time_low"] = float(price_tr.min())

    thresholds["volume_mean"] = float(vol_tr.mean())
    thresholds["volume_std"] = float(vol_tr.std(ddof=0))
    thresholds["volume_min"] = float(vol_tr.min())
    thresholds["volume_max"] = float(vol_tr.max())

    thresholds["return_mean"] = float(ret_tr.mean())
    thresholds["return_std"] = float(ret_tr.std(ddof=0))
    thresholds["logreturn_mean"] = float(lret_tr.mean())
    thresholds["logreturn_std"] = float(lret_tr.std(ddof=0))

    # Train-derived edges (robust with -inf/+inf)
    thresholds["price_edges"] = _make_edges_from_train(price_tr, [0.25, 0.50, 0.75], "price")
    thresholds["volume_edges"] = _make_edges_from_train(vol_tr, [0.33, 0.67], "volume")
    thresholds["return_edges"] = _make_edges_from_train(ret_tr, [0.33, 0.67], "returns")
    thresholds["logreturn_edges"] = _make_edges_from_train(lret_tr, [0.33, 0.67], "log_returns")
    thresholds["vol_edges"] = _make_edges_from_train(sig_tr, [0.33, 0.67], "rolling_std")

    # Sorted arrays for percentile/rank lookup
    price_sorted = np.sort(price_tr.to_numpy())
    vol_sorted = np.sort(vol_tr.to_numpy())
    ret_sorted = np.sort(ret_tr.dropna().to_numpy())

    def apply_block(df: pd.DataFrame):
        # price percentiles/ranks
        df["Price_Percentile"] = df[cfg.PRICE_COL].astype(float).apply(lambda x: _percentile_from_sorted(price_sorted, x))
        df["Price_Rank_Pct"] = df["Price_Percentile"]
        df["Close_ZScore"] = (df[cfg.PRICE_COL].astype(float) - thresholds["price_mean"]) / (thresholds["price_std"] or np.nan)
        df["Normalized_Close"] = (df[cfg.PRICE_COL].astype(float) - thresholds["price_min"]) / ((thresholds["price_max"] - thresholds["price_min"]) or np.nan)

        df["Close_Quantile"] = pd.cut(df[cfg.PRICE_COL].astype(float), bins=thresholds["price_edges"], labels=False, include_lowest=True)
        df["Price_Bucket"] = pd.cut(df[cfg.PRICE_COL].astype(float), bins=thresholds["price_edges"], labels=False, include_lowest=True)

        df["Distance_From_AllTimeHigh"] = thresholds["price_all_time_high"] - df[cfg.PRICE_COL].astype(float)
        df["Pct_From_AllTimeHigh"] = (df[cfg.PRICE_COL].astype(float) / thresholds["price_all_time_high"] - 1.0) * 100.0
        df["Distance_From_AllTimeLow"] = df[cfg.PRICE_COL].astype(float) - thresholds["price_all_time_low"]
        df["Price_to_Peak_Ratio"] = df[cfg.PRICE_COL].astype(float) / thresholds["price_all_time_high"]

        # returns
        df["Return_Percentile"] = df["Close_Returns"].astype(float).apply(lambda x: _percentile_from_sorted(ret_sorted, x))
        df["Return_ZScore"] = (df["Close_Returns"].astype(float) - thresholds["return_mean"]) / (thresholds["return_std"] or np.nan)
        df["LogReturn_ZScore"] = (df["Log_Returns"].astype(float) - thresholds["logreturn_mean"]) / (thresholds["logreturn_std"] or np.nan)
        df["LogReturn_Quantile"] = pd.cut(df["Log_Returns"].astype(float), bins=thresholds["logreturn_edges"], labels=False, include_lowest=True)

        # regimes from train edges
        df["Return_Regime"] = "neutral"
        neg_edge = thresholds["return_edges"][1]
        pos_edge = thresholds["return_edges"][-2]
        df.loc[df["Close_Returns"] < neg_edge, "Return_Regime"] = "negative"
        df.loc[df["Close_Returns"] > pos_edge, "Return_Regime"] = "positive"

        df["Global_Momentum_Rank"] = df["Return_Percentile"]

        # volume
        df["Volume_Percentile"] = df[cfg.VOLUME_COL].astype(float).apply(lambda x: _percentile_from_sorted(vol_sorted, x))
        df["Volume_Rank_Pct"] = df["Volume_Percentile"]
        df["Volume_ZScore"] = (df[cfg.VOLUME_COL].astype(float) - thresholds["volume_mean"]) / (thresholds["volume_std"] or np.nan)
        df["Normalized_Volume"] = (df[cfg.VOLUME_COL].astype(float) - thresholds["volume_min"]) / ((thresholds["volume_max"] - thresholds["volume_min"]) or np.nan)
        df["Volume_Quantile"] = pd.cut(df[cfg.VOLUME_COL].astype(float), bins=thresholds["volume_edges"], labels=False, include_lowest=True)
        df["Volume_Bucket"] = pd.cut(df[cfg.VOLUME_COL].astype(float), bins=thresholds["volume_edges"], labels=False, include_lowest=True)

        df["Volume_Regime"] = "medium"
        low_edge = thresholds["volume_edges"][1]
        high_edge = thresholds["volume_edges"][-2]
        df.loc[df[cfg.VOLUME_COL] < low_edge, "Volume_Regime"] = "low"
        df.loc[df[cfg.VOLUME_COL] > high_edge, "Volume_Regime"] = "high"

        # volatility regime
        df["Vol_Regime"] = "medium"
        vlow = thresholds["vol_edges"][1]
        vhigh = thresholds["vol_edges"][-2]
        df.loc[df["Rolling_Std_20"] < vlow, "Vol_Regime"] = "low"
        df.loc[df["Rolling_Std_20"] > vhigh, "Vol_Regime"] = "high"

        # trend regime (causal)
        df["Trend_Regime"] = "neutral"
        df.loc[df["EMA_50"] > df["EMA_200"], "Trend_Regime"] = "uptrend"
        df.loc[df["EMA_50"] < df["EMA_200"], "Trend_Regime"] = "downtrend"

        # risk / market / bull-bear (uses lagged VIX already computed pre-split)
        df["Risk_State"] = "normal"
        df.loc[(df["Vol_Regime"] == "high") & (df["VIX_Lagged"] > 20), "Risk_State"] = "high_risk"
        df.loc[(df["Vol_Regime"] == "low") & (df["VIX_Lagged"] < 15), "Risk_State"] = "low_risk"

        df["Market_State"] = "neutral"
        df.loc[(df["Trend_Regime"] == "uptrend") & (df["Vol_Regime"] == "low"), "Market_State"] = "bull_quiet"
        df.loc[(df["Trend_Regime"] == "uptrend") & (df["Vol_Regime"] == "high"), "Market_State"] = "bull_volatile"
        df.loc[(df["Trend_Regime"] == "downtrend") & (df["Vol_Regime"] == "low"), "Market_State"] = "bear_quiet"
        df.loc[(df["Trend_Regime"] == "downtrend") & (df["Vol_Regime"] == "high"), "Market_State"] = "bear_volatile"

        df["Bull_Bear_Label"] = "neutral"
        df.loc[df["Trend_Regime"] == "uptrend", "Bull_Bear_Label"] = "bull"
        df.loc[df["Trend_Regime"] == "downtrend", "Bull_Bear_Label"] = "bear"

        return df

    train = apply_block(train)
    val = apply_block(val)
    test = apply_block(test)

    return train, val, test, thresholds


def _drop_any_nan(df: pd.DataFrame, name: str) -> pd.DataFrame:
    n0 = len(df)
    nans = df.isna().sum().sum()
    if nans == 0:
        print(f"  {name}: 0 NaNs")
        return df

    # drop rows with any NaN (safe; does not create leakage)
    df2 = df.dropna().reset_index(drop=True)
    dropped = n0 - len(df2)
    print(f"  {name}: dropped {dropped} rows containing NaNs")
    return df2


def fit_and_apply_scaler(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: Config):
    print("\nFitting scaler on train, applying to all...")

    # exclude non-numeric and target
    exclude = {
        "Date",
        cfg.TARGET_COL,
        "Vol_Regime",
        "Return_Regime",
        "Volume_Regime",
        "Trend_Regime",
        "Risk_State",
        "Market_State",
        "Bull_Bear_Label",
    }

    # numeric cols only
    numeric_cols = [c for c in train.columns if c not in exclude and pd.api.types.is_numeric_dtype(train[c])]

    scaler = StandardScaler()
    scaler.fit(train[numeric_cols])

    train_s = train.copy()
    val_s = val.copy()
    test_s = test.copy()

    train_s[numeric_cols] = scaler.transform(train[numeric_cols])
    val_s[numeric_cols] = scaler.transform(val[numeric_cols])
    test_s[numeric_cols] = scaler.transform(test[numeric_cols])

    print(f"  Scaled {len(numeric_cols)} numeric columns")
    return train_s, val_s, test_s, scaler, numeric_cols


def save_outputs(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, scaler, numeric_cols, thresholds, cfg: Config):
    print("\nSaving outputs...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(cfg.OUTPUT_DIR, "train_prepared.csv")
    val_path = os.path.join(cfg.OUTPUT_DIR, "val_prepared.csv")
    test_path = os.path.join(cfg.OUTPUT_DIR, "test_prepared.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"  {train_path} ({len(train)} rows x {train.shape[1]} cols)")
    print(f"  {val_path} ({len(val)} rows x {val.shape[1]} cols)")
    print(f"  {test_path} ({len(test)} rows x {test.shape[1]} cols)")

    scaler_path = os.path.join(cfg.OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "numeric_cols": numeric_cols}, f)
    print(f"  {scaler_path}")

    thresholds_path = os.path.join(cfg.OUTPUT_DIR, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  {thresholds_path}")


def main():
    cfg = Config()

    print("=" * 60)
    print("LEAKAGE-SAFE PIPELINE")
    print("=" * 60)

    df = load_data(cfg)
    df = compute_technical_indicators(df, cfg)
    df = create_target(df, cfg)
    df = auto_remove_warmup(df)

    train, val, test = split_data(df, cfg)
    train, val, test, thresholds = compute_global_indicators(train, val, test, cfg)

    print("\nFinal NaN cleanup (rows only)...")
    train = _drop_any_nan(train, "Train")
    val = _drop_any_nan(val, "Val")
    test = _drop_any_nan(test, "Test")

    # enforce identical schema
    cols = list(train.columns)
    if set(cols) != set(val.columns) or set(cols) != set(test.columns):
        raise RuntimeError("Column mismatch between train/val/test")
    val = val[cols]
    test = test[cols]

    train, val, test, scaler, numeric_cols = fit_and_apply_scaler(train, val, test, cfg)

    # hard fail if any NaNs remain
    if train.isna().any().any() or val.isna().any().any() or test.isna().any().any():
        raise RuntimeError("NaNs remain after preprocessing")

    save_outputs(train, val, test, scaler, numeric_cols, thresholds, cfg)

    print("\nDONE")


if __name__ == "__main__":
    main()
