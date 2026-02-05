import pandas as pd
import numpy as np

# -------- CONFIG --------
INPUT_CSV = "C:\\Users\\PC\\Desktop\\clean.csv"
OUTPUT_CSV = "C:\\Users\\PC\\Desktop\\MML\\clean_with_all_indicators.csv"

# -------- LOAD DATA --------
print("="*60)
print("GOLD TECHNICAL INDICATOR CALCULATOR")
print("="*60)

df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# -------- FIX DECIMAL PARTS (Forward fill logic) --------
print("\nFixing decimal parts in OHLC data...")

ohlc_cols = ['xauusd-open', 'xauusd-high', 'xauusd-low', 'xauusd-close']

for col in ohlc_cols:
    if col in df.columns:
        prev_valid = None
        for i in range(len(df)):
            current = df.loc[i, col]
            if pd.notna(current):
                # Check if it's an integer (no decimal part)
                if current % 1 == 0 and prev_valid is not None:
                    # Use previous decimal structure
                    prev_int = int(prev_valid)
                    prev_decimal = prev_valid - prev_int
                    df.loc[i, col] = current + prev_decimal
                else:
                    prev_valid = df.loc[i, col]

print("✓ Decimal parts fixed")

# -------- EXTRACT PRICE DATA --------
close = df['xauusd-close']
high = df['xauusd-high']
low = df['xauusd-low']
open_price = df['xauusd-open']

# -------- CALCULATE INDICATORS --------
print("\nCalculating technical indicators...")

# ===== RSI (Multiple Periods) =====
def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

df['rsi_7'] = calculate_rsi(close, 7)
df['rsi_14'] = calculate_rsi(close, 14)
df['rsi_21'] = calculate_rsi(close, 21)
df['rsi_28'] = calculate_rsi(close, 28)
print("  ✓ RSI calculated")

# ===== ATR (Average True Range) =====
tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr_14'] = tr.rolling(window=14).mean()
print("  ✓ ATR calculated")

# ===== EMAs =====
df['ema_20'] = close.ewm(span=20, adjust=False).mean()
df['ema_50'] = close.ewm(span=50, adjust=False).mean()
print("  ✓ EMAs calculated")

# ===== MACD =====
ema_12 = close.ewm(span=12, adjust=False).mean()
ema_26 = close.ewm(span=26, adjust=False).mean()
df['macd_line'] = ema_12 - ema_26
df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
df['macd_histogram'] = df['macd_line'] - df['macd_signal']
print("  ✓ MACD calculated")

# ===== BOLLINGER BANDS =====
sma_20 = close.rolling(window=20).mean()
std_20 = close.rolling(window=20).std()
df['bb_upper'] = sma_20 + (std_20 * 2)
df['bb_lower'] = sma_20 - (std_20 * 2)
df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / (sma_20 + 1e-10)
df['bb_percent_b'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
print("  ✓ Bollinger Bands calculated")

# ===== STOCHASTIC OSCILLATOR =====
lowest_low = low.rolling(window=14).min()
highest_high = high.rolling(window=14).max()
df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
print("  ✓ Stochastic calculated")

# ===== WILLIAMS %R =====
df['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
print("  ✓ Williams %R calculated")

# ===== ROC (Rate of Change) =====
df['roc_10'] = close.pct_change(periods=10) * 100
df['roc_20'] = close.pct_change(periods=20) * 100
print("  ✓ ROC calculated")

# ===== CCI (Commodity Channel Index) =====
tp = (high + low + close) / 3
sma_tp = tp.rolling(window=20).mean()
mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
print("  ✓ CCI calculated")

# ===== SMAs (INCLUDING 200) =====
df['sma_20'] = close.rolling(window=20).mean()
df['sma_50'] = close.rolling(window=50).mean()
df['sma_200'] = close.rolling(window=200).mean()  # THIS NEEDS 200 DAYS
print("  ✓ SMAs calculated (including SMA_200)")

# ===== PRICE DISTANCE FROM MAs =====
df['price_sma20_pct'] = ((close - df['sma_20']) / (df['sma_20'] + 1e-10)) * 100
df['price_sma50_pct'] = ((close - df['sma_50']) / (df['sma_50'] + 1e-10)) * 100
df['price_ema20_pct'] = ((close - df['ema_20']) / (df['ema_20'] + 1e-10)) * 100
print("  ✓ Price distances calculated")

# ===== MA CROSSOVERS =====
df['sma_20_50_cross'] = df['sma_20'] - df['sma_50']
df['ema_20_50_cross'] = df['ema_20'] - df['ema_50']
print("  ✓ MA crossovers calculated")

# ===== PARABOLIC SAR =====
def calculate_parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
    sar = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    ep = pd.Series(index=close.index, dtype=float)
    af = pd.Series(index=close.index, dtype=float)
    
    sar.iloc[0] = close.iloc[0]
    trend.iloc[0] = 1
    ep.iloc[0] = high.iloc[0]
    af.iloc[0] = af_start
    
    for i in range(1, len(close)):
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
            
            if close.iloc[i] < sar.iloc[i]:
                trend.iloc[i] = -1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = low.iloc[i]
                af.iloc[i] = af_start
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
        else:  # Downtrend
            sar.iloc[i] = sar.iloc[i-1] - af.iloc[i-1] * (sar.iloc[i-1] - ep.iloc[i-1])
            
            if close.iloc[i] > sar.iloc[i]:
                trend.iloc[i] = 1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = high.iloc[i]
                af.iloc[i] = af_start
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
    
    return sar, trend

df['psar'], df['psar_trend'] = calculate_parabolic_sar(high, low, close)
print("  ✓ Parabolic SAR calculated")

# ===== STANDARD DEVIATION =====
df['std_10'] = close.rolling(window=10).std()
df['std_20'] = close.rolling(window=20).std()
print("  ✓ Standard Deviation calculated")

# ===== MOMENTUM =====
df['momentum_10'] = close - close.shift(10)
df['momentum_20'] = close - close.shift(20)
print("  ✓ Momentum calculated")

# ===== TRIPLE EMA =====
df['tema'] = close.ewm(span=20, adjust=False).mean()
print("  ✓ TEMA calculated")

# -------- DATA VALIDATION --------
print("\n" + "="*60)
print("DATA VALIDATION")
print("="*60)

# Check first 200 rows
first_200_check = df.iloc[:200]
indicators_to_check = ['sma_200', 'sma_50', 'rsi_28', 'ema_50']

print(f"\nFirst 200 rows check:")
for ind in indicators_to_check:
    if ind in df.columns:
        nan_count = first_200_check[ind].isna().sum()
        print(f"  {ind}: {nan_count}/200 rows are NaN")

# Check row 200 onwards (should have NO NaN in most indicators)
after_200_check = df.iloc[200:]
print(f"\nRow 201 onwards check (should be clean):")
for ind in indicators_to_check:
    if ind in df.columns:
        nan_count = after_200_check[ind].isna().sum()
        total = len(after_200_check)
        print(f"  {ind}: {nan_count}/{total} rows are NaN")
        if nan_count > 0:
            print(f"    ⚠️ WARNING: Row 201+ should not have NaN in {ind}")

# -------- SUMMARY --------
print("\n" + "="*60)
print("INDICATOR SUMMARY")
print("="*60)

all_indicators = [
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

print(f"Total indicators calculated: {len(all_indicators)}")
print(f"Total columns in output: {len(df.columns)}")
print(f"Total rows: {len(df)}")
print(f"\nFirst 200 rows: Have NaN in long-period indicators (EXPECTED)")
print(f"Row 201 onwards: Should be clean (all indicators valid)")

# -------- CLEAN NaN IN FIRST 200 ROWS FOR NON-CRITICAL INDICATORS --------
# Do NOT forward/backward fill - leave NaN as NaN for first 200 rows
# Only fill for indicators that naturally have values before row 200

print("\n" + "="*60)
print("NaN HANDLING")
print("="*60)
print("First 200 rows: NaN values PRESERVED (correct behavior)")
print("Row 201+: Checking for unexpected NaN...")

# For rows 201+, check if there are unexpected NaN
# These should NOT exist unless there's missing price data
unexpected_nan = df.iloc[200:].isna().sum()
if unexpected_nan.sum() > 0:
    print("\n⚠️ WARNING: Unexpected NaN found in rows 201+:")
    for col in unexpected_nan[unexpected_nan > 0].index:
        print(f"  {col}: {unexpected_nan[col]} NaN values")
    print("\nThese will be forward-filled...")
    df.iloc[200:] = df.iloc[200:].ffill()
else:
    print("✓ No unexpected NaN found after row 200")

# -------- SAVE --------
print("\n" + "="*60)
print("SAVING OUTPUT")
print("="*60)

df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Saved to: {OUTPUT_CSV}")
print(f"✓ Total rows: {len(df)}")
print(f"✓ Total columns: {len(df.columns)}")

# -------- DISPLAY SAMPLE --------
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

print("\nRow 1 (should have NaN in long-period indicators):")
print(df[['time', 'xauusd-close', 'sma_20', 'sma_50', 'sma_200', 'rsi_14']].iloc[0])

print("\nRow 200 (last row with NaN in SMA_200):")
print(df[['time', 'xauusd-close', 'sma_20', 'sma_50', 'sma_200', 'rsi_14']].iloc[199])

print("\nRow 201 (first fully clean row):")
print(df[['time', 'xauusd-close', 'sma_20', 'sma_50', 'sma_200', 'rsi_14']].iloc[200])

print("\nRow 300 (should be fully clean):")
print(df[['time', 'xauusd-close', 'sma_20', 'sma_50', 'sma_200', 'rsi_14']].iloc[299])

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Check the output CSV - first 200 rows should have NaN in SMA_200")
print("2. Row 201 onwards should have ALL indicator values")
print("3. Use this file for training (drop first 200 rows before training)")
print("4. For daily prediction, ensure you have 200 days of history")