import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
sample_dir = os.path.join(script_dir, "pr sample")
results_dir = os.path.join(script_dir, "RESULTS")
os.makedirs(results_dir, exist_ok=True)

market_files = ['XAUUSD.csv', 'NASDAQ.csv', 'SP500.csv', 'RUSSELL2000.csv', 'VIX.csv', 'DXY.csv', 'EURUSD.csv', 'USDJPY.csv']
economic_files = ['CPI.csv', 'NFP.csv', 'PCE.csv','GEO.csv', 'UNEMPLOYMENT RATE.csv', 'FED RATE.csv', 'TIPS2y.csv', 'TIPS5y.csv', 'TIPS10y.csv']
indicator_files = ['INDICATORS.csv']

report = []

def log(msg):
    print(msg)
    report.append(msg)

def calculate_max_change(df, columns):
    max_changes = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            pct_change = df[col].pct_change().abs()
            max_changes[col] = pct_change.max()
    return max_changes

def clean_market_data(filepath, filename):
    log(f"Processing market data: {filename}")
    df = pd.read_csv(filepath)
    
    if 'Volume' in df.columns:
        first_50_volume = df['Volume'].head(50)
        if (first_50_volume == 0).all():
            df = df.drop(columns=['Volume'])
            log(f"  Dropped Volume column (all zeros)")
    
    numeric_cols = [col for col in df.columns if col != 'Date' and pd.api.types.is_numeric_dtype(df[col])]
    max_changes = calculate_max_change(df, numeric_cols)
    
    outliers_fixed = 0
    for col in numeric_cols:
        if col in max_changes and max_changes[col] > 0:
            tolerance = max_changes[col] * 1.5
            pct_change = df[col].pct_change().abs()
            outlier_mask = pct_change > tolerance
            if outlier_mask.any():
                df.loc[outlier_mask, col] = np.nan
                outliers_fixed += outlier_mask.sum()
    
    if outliers_fixed > 0:
        log(f"  Fixed {outliers_fixed} outliers")
    
    for col in numeric_cols:
        df[col] = df[col].ffill().bfill()
    
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    nan_count = df[numeric_cols].isna().sum().sum()
    if nan_count > 0:
        log(f"  Warning: {nan_count} NaN values remain")
    
    output_path = os.path.join(results_dir, filename)
    df.to_csv(output_path, index=False)
    log(f"  Saved to {filename}")
    return df

def clean_economic_data(filepath, filename):
    log(f"Processing economic data: {filename}")
    df = pd.read_csv(filepath)
    
    numeric_cols = [col for col in df.columns if col != 'Date' and pd.api.types.is_numeric_dtype(df[col])]
    max_changes = calculate_max_change(df, numeric_cols)
    
    outliers_fixed = 0
    for col in numeric_cols:
        if col in max_changes and max_changes[col] > 0:
            tolerance = max_changes[col] * 1.5
            pct_change = df[col].pct_change().abs()
            outlier_mask = pct_change > tolerance
            if outlier_mask.any():
                df.loc[outlier_mask, col] = np.nan
                outliers_fixed += outlier_mask.sum()
    
    if outliers_fixed > 0:
        log(f"  Fixed {outliers_fixed} outliers")
    
    for col in numeric_cols:
        df[col] = df[col].ffill().bfill()
    
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    nan_count = df[numeric_cols].isna().sum().sum()
    if nan_count > 0:
        log(f"  Warning: {nan_count} NaN values remain")
    
    output_path = os.path.join(results_dir, filename)
    df.to_csv(output_path, index=False)
    log(f"  Saved to {filename}")
    return df

def clean_indicators(filepath, filename):
    log(f"Processing indicators: {filename}")
    df = pd.read_csv(filepath)
    
    numeric_cols = [col for col in df.columns if col != 'Date' and pd.api.types.is_numeric_dtype(df[col])]
    
    for col in numeric_cols:
        df[col] = df[col].ffill().bfill()
    
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    nan_count = df[numeric_cols].isna().sum().sum()
    if nan_count > 0:
        log(f"  Warning: {nan_count} NaN values remain")
    
    output_path = os.path.join(results_dir, filename)
    df.to_csv(output_path, index=False)
    log(f"  Saved to {filename}")
    return df

log("Starting data cleaning process")
log(f"Input: {sample_dir}")
log(f"Output: {results_dir}")
log("")

all_files = os.listdir(sample_dir)
all_files = [f for f in all_files if f.endswith('.csv') and f != 'master_time_span.csv']

for filename in all_files:
    filepath = os.path.join(sample_dir, filename)
    
    try:
        if filename.lower() in [f.lower() for f in market_files]:
            clean_market_data(filepath, filename)
        elif filename.lower() in [f.lower() for f in economic_files]:
            clean_economic_data(filepath, filename)
        elif filename.lower() in [f.lower() for f in indicator_files]:
            clean_indicators(filepath, filename)
        else:
            log(f"Skipping unknown file: {filename}")
    except Exception as e:
        log(f"ERROR processing {filename}: {str(e)}")

log("")
log("Combining all cleaned files into master dataset")

cleaned_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
master_df = None

for filename in sorted(cleaned_files):
    filepath = os.path.join(results_dir, filename)
    df = pd.read_csv(filepath)
    
    if 'Date' not in df.columns:
        log(f"  Skipping {filename} (no Date column)")
        continue
    
    if master_df is None:
        master_df = df
        log(f"  Base: {filename} ({len(df.columns)} columns)")
    else:
        master_df = master_df.merge(df, on='Date', how='outer', suffixes=('', f'_{filename[:-4]}'))
        log(f"  Merged: {filename} (total: {len(master_df.columns)} columns)")

master_df = master_df.sort_values('Date').reset_index(drop=True)

master_path = os.path.join(results_dir, "master_dataset.csv")
master_df.to_csv(master_path, index=False)

log("")
log(f"Master dataset created: {len(master_df)} rows, {len(master_df.columns)} columns")
log(f"Saved to: master_dataset.csv")

report_path = os.path.join(results_dir, "cleaning_report.txt")
with open(report_path, 'w') as f:
    f.write("\n".join(report))

log(f"Report saved to: cleaning_report.txt")
log("Process complete")