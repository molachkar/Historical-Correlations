import pandas as pd

# -------- FEATURE NAMES --------
# These are the 48 features your model needs
features = [
    # Price data (t-1 = previous day)
    'xauusd-close',  # Friday's close for Monday signal
    
    # Market indicators (t-1)
    'vix-close',
    'dxy-close',
    'eurusd-close',
    'futures-close',
    'usd-jpy close',
    'etfs',
    
    # Macro indicators (latest available)
    'DFII10',
    'DFF',
    'cpi',
    'GPR',
    'nfp',
    'pce',
    'un-rate',
    
    # Technical indicators (calculated from t-1 data)
    'rsi_7',
    'rsi_14',
    'rsi_21',
    'rsi_28',
    'atr_14',
    'ema_20',
    'ema_50',
    'macd_line',
    'macd_signal',
    'macd_histogram',
    'bb_upper',
    'bb_lower',
    'bb_bandwidth',
    'bb_percent_b',
    'stoch_k',
    'stoch_d',
    'williams_r',
    'roc_10',
    'roc_20',
    'cci',
    'sma_20',
    'sma_50',
    'sma_200',
    'price_sma20_pct',
    'price_sma50_pct',
    'price_ema20_pct',
    'sma_20_50_cross',
    'ema_20_50_cross',
    'psar',
    'psar_trend',
    'std_10',
    'std_20',
    'momentum_10',
    'momentum_20',
    'tema'
]

# -------- CREATE TEMPLATE --------
template_df = pd.DataFrame(columns=features)

# Add instruction rows as examples
instructions = {
    'xauusd-close': ['FRIDAY CLOSE', 'Use PREVIOUS trading day close'],
    'vix-close': ['FRIDAY CLOSE', 'Use PREVIOUS trading day close'],
    'dxy-close': ['FRIDAY CLOSE', 'Use PREVIOUS trading day close'],
    'eurusd-close': ['FRIDAY CLOSE', 'Use PREVIOUS trading day close'],
    'futures-close': ['FRIDAY CLOSE', 'Use PREVIOUS trading day close'],
    'usd-jpy close': ['FRIDAY CLOSE', 'Use PREVIOUS trading day close'],
    'etfs': ['FRIDAY CLOSE', 'Use PREVIOUS trading day close'],
    'DFII10': ['LATEST', 'Use most recent value'],
    'DFF': ['LATEST', 'Use most recent value'],
    'cpi': ['LATEST', 'Monthly data - use last published'],
    'GPR': ['LATEST', 'Use most recent value'],
    'nfp': ['LATEST', 'Monthly data - use last published'],
    'pce': ['LATEST', 'Monthly data - use last published'],
    'un-rate': ['LATEST', 'Monthly data - use last published'],
    'rsi_7': ['FROM TRADINGVIEW', 'Copy from TradingView using FRIDAY close'],
    'rsi_14': ['FROM TRADINGVIEW', 'Copy from TradingView using FRIDAY close'],
    'rsi_21': ['FROM TRADINGVIEW', 'Copy from TradingView using FRIDAY close'],
    'rsi_28': ['FROM TRADINGVIEW', 'Copy from TradingView using FRIDAY close'],
    'atr_14': ['FROM TRADINGVIEW', 'Copy from TradingView using FRIDAY close'],
    'ema_20': ['FROM TRADINGVIEW', 'Copy from TradingView using FRIDAY close'],
    'ema_50': ['FROM TRADINGVIEW', 'Copy from TradingView using FRIDAY close'],
    'macd_line': ['FROM TRADINGVIEW', 'MACD line value at FRIDAY close'],
    'macd_signal': ['FROM TRADINGVIEW', 'MACD signal value at FRIDAY close'],
    'macd_histogram': ['FROM TRADINGVIEW', 'MACD histogram value at FRIDAY close'],
    'bb_upper': ['FROM TRADINGVIEW', 'Upper Bollinger Band at FRIDAY close'],
    'bb_lower': ['FROM TRADINGVIEW', 'Lower Bollinger Band at FRIDAY close'],
    'bb_bandwidth': ['CALCULATE', '(bb_upper - bb_lower) / sma_20'],
    'bb_percent_b': ['CALCULATE', '(xauusd-close - bb_lower) / (bb_upper - bb_lower)'],
    'stoch_k': ['FROM TRADINGVIEW', 'Stochastic %K at FRIDAY close'],
    'stoch_d': ['FROM TRADINGVIEW', 'Stochastic %D at FRIDAY close'],
    'williams_r': ['FROM TRADINGVIEW', 'Williams %R at FRIDAY close'],
    'roc_10': ['FROM TRADINGVIEW', 'Rate of Change 10 at FRIDAY close'],
    'roc_20': ['FROM TRADINGVIEW', 'Rate of Change 20 at FRIDAY close'],
    'cci': ['FROM TRADINGVIEW', 'CCI (20) at FRIDAY close'],
    'sma_20': ['FROM TRADINGVIEW', 'SMA 20 at FRIDAY close'],
    'sma_50': ['FROM TRADINGVIEW', 'SMA 50 at FRIDAY close'],
    'sma_200': ['FROM TRADINGVIEW', 'SMA 200 at FRIDAY close'],
    'price_sma20_pct': ['CALCULATE', '((xauusd-close - sma_20) / sma_20) * 100'],
    'price_sma50_pct': ['CALCULATE', '((xauusd-close - sma_50) / sma_50) * 100'],
    'price_ema20_pct': ['CALCULATE', '((xauusd-close - ema_20) / ema_20) * 100'],
    'sma_20_50_cross': ['CALCULATE', 'sma_20 - sma_50'],
    'ema_20_50_cross': ['CALCULATE', 'ema_20 - ema_50'],
    'psar': ['FROM TRADINGVIEW', 'Parabolic SAR value at FRIDAY close'],
    'psar_trend': ['FROM TRADINGVIEW', '1 if uptrend, -1 if downtrend'],
    'std_10': ['FROM TRADINGVIEW', 'Standard Deviation 10 at FRIDAY close'],
    'std_20': ['FROM TRADINGVIEW', 'Standard Deviation 20 at FRIDAY close'],
    'momentum_10': ['CALCULATE', 'xauusd-close - close_10_days_ago'],
    'momentum_20': ['CALCULATE', 'xauusd-close - close_20_days_ago'],
    'tema': ['FROM TRADINGVIEW', 'Triple EMA 20 at FRIDAY close']
}

# Create instruction rows
instruction_row1 = {col: instructions.get(col, ['', ''])[0] for col in features}
instruction_row2 = {col: instructions.get(col, ['', ''])[1] for col in features}

template_df = pd.concat([
    pd.DataFrame([instruction_row1]),
    pd.DataFrame([instruction_row2]),
    template_df
], ignore_index=True)

# Add 5 empty data rows
for i in range(5):
    template_df = pd.concat([template_df, pd.DataFrame([[None]*len(features)], columns=features)], ignore_index=True)

# -------- SAVE --------
output_path = "C:\\Users\\PC\\Desktop\\MML\\today_data_template.csv"
template_df.to_csv(output_path, index=False)

print("="*60)
print("TEMPLATE CREATED")
print("="*60)
print(f"Saved to: {output_path}")
print(f"Total features: {len(features)}")
print("\nINSTRUCTIONS:")
print("1. For MONDAY signal → Use FRIDAY's data")
print("2. For TUESDAY signal → Use MONDAY's data")
print("3. Delete instruction rows (first 2 rows) before feeding to model")
print("4. Fill one row per prediction date")
print("5. Calculate derived features (marked CALCULATE) after getting base indicators")
print("="*60)

# -------- CREATE SEPARATE CALCULATION HELPER --------
calc_instructions = """
CALCULATION FORMULAS:
====================

After collecting data from TradingView, calculate these:

1. bb_bandwidth = (bb_upper - bb_lower) / sma_20

2. bb_percent_b = (xauusd-close - bb_lower) / (bb_upper - bb_lower)

3. price_sma20_pct = ((xauusd-close - sma_20) / sma_20) * 100

4. price_sma50_pct = ((xauusd-close - sma_50) / sma_50) * 100

5. price_ema20_pct = ((xauusd-close - ema_20) / ema_20) * 100

6. sma_20_50_cross = sma_20 - sma_50

7. ema_20_50_cross = ema_20 - ema_50

8. momentum_10 = xauusd-close - (close price from 10 days ago)

9. momentum_20 = xauusd-close - (close price from 20 days ago)

WEEKEND LOGIC:
==============
- For MONDAY signal: Use FRIDAY's closing data
- Gold doesn't trade on weekends, so Friday is the "previous day"
- All indicators should be calculated as of Friday's close

DATA SOURCE PRIORITY:
====================
1. Price data (xauusd, vix, dxy, etc.) → Use PREVIOUS trading day close
2. Technical indicators → Get from TradingView at PREVIOUS trading day close
3. Macro data (CPI, NFP, etc.) → Use latest published value (monthly)
"""

calc_file = "C:\\Users\\PC\\Desktop\\MML\\calculation_instructions.txt"
with open(calc_file, 'w') as f:
    f.write(calc_instructions)

print(f"\nCalculation instructions saved to: {calc_file}")