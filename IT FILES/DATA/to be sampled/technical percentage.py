import pandas as pd
import numpy as np

# -------- CONFIG --------
INPUT_CSV = "C:\\Users\\PC\\Desktop\\MML\\live.csv"
OUTPUT_CSV = "C:\\Users\\PC\\Desktop\\MML\\clean_with_all_indicators.csv"

# -------- LOAD --------
df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {len(df)} rows")

# -------- TECHNICAL INDICATORS TO FIX --------
technical_indicators = [
    'rsi_7', 'rsi_14', 'rsi_21', 'rsi_28',
    'atr_14', 'ema_20', 'ema_50',
    'macd_line', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_bandwidth', 'bb_percent_b',
    'stoch_k', 'stoch_d', 'williams_r',
    'roc_10', 'roc_20', 'cci',
    'sma_20', 'sma_50', 'sma_200',
    'price_sma20_pct', 'price_sma50_pct', 'price_ema20_pct',
    'sma_20_50_cross', 'ema_20_50_cross',
    'psar', 'psar_trend',
    'std_10', 'std_20', 'momentum_10', 'momentum_20', 'tema'
]

# -------- FORWARD FILL ZEROS --------
print("\n" + "="*60)
print("FIXING ZERO VALUES IN TECHNICAL INDICATORS")
print("="*60)

for col in technical_indicators:
    if col in df.columns:
        # Count zeros before fixing
        zero_count_before = (df[col] == 0).sum()
        
        if zero_count_before > 0:
            # Replace zeros with NaN, then forward fill
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].ffill()
            
            # If still NaN at the start, backward fill
            df[col] = df[col].bfill()
            
            # Count zeros after fixing
            zero_count_after = (df[col] == 0).sum()
            nan_count_after = df[col].isna().sum()
            
            print(f"{col}: {zero_count_before} zeros → {zero_count_after} zeros, {nan_count_after} NaN")
        else:
            print(f"{col}: No zeros found")
    else:
        print(f"{col}: Column not found in dataset")

# -------- VERIFY --------
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
total_zeros = sum((df[col] == 0).sum() for col in technical_indicators if col in df.columns)
total_nans = sum(df[col].isna().sum() for col in technical_indicators if col in df.columns)
print(f"Total zeros remaining in technical indicators: {total_zeros}")
print(f"Total NaNs remaining in technical indicators: {total_nans}")

# -------- SAVE --------
df.to_csv(OUTPUT_CSV, index=False)
print("\n" + "="*60)
print(f"Saved: {OUTPUT_CSV}")
print(f"Total rows: {len(df)}")
print("="*60)