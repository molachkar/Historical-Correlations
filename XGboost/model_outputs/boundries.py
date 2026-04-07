import pandas as pd
import numpy as np

# Load your Out-Of-Fold predictions
df = pd.read_csv('cv_predictions_oof.csv')
df = df[df['has_prediction'] == True].copy()

# 1. Rolling Baseline (60 days is more reactive for Gold cycles)
WINDOW = 60
df['pred_mean'] = df['oof_prediction'].rolling(WINDOW).mean()
df['pred_std'] = df['oof_prediction'].rolling(WINDOW).std()
df['pred_z'] = (df['oof_prediction'] - df['pred_mean']) / df['pred_std']

# 2. Realistic XM Spread Cost (Approx 3.5 pips for XAUUSD)
# This is the "hurdle" the model must jump to be profitable
COST = 0.000175 

df_clean = df.dropna(subset=['pred_z']).copy()

# 3. Find the Best Threshold
results = []
for z in np.linspace(0.5, 3.0, 26):
    # Only trade if model is high-confidence (Z-score)
    trades = df_clean[df_clean['pred_z'].abs() > z].copy()
    if len(trades) < 20: continue # Need enough trades to be sure
    
    # Calculate profit: (Direction * Return) - Cost
    trades['profit'] = (np.sign(trades['pred_z']) * trades['y_next_log_return']) - COST
    win_rate = (trades['profit'] > 0).mean()
    
    results.append({'Z_Thresh': z, 'Win_Rate': win_rate, 'Total_Trades': len(trades), 'Avg_Ret': trades['profit'].mean()})

best = pd.DataFrame(results).sort_values('Avg_Ret', ascending=False).iloc[0]
print(f"Optimal Z-Threshold: {best['Z_Thresh']:.2f}")
print(f"Realistic Win Rate: {best['Win_Rate']*100:.1f}%")
print(f"Expected Trades: ~{best['Total_Trades'] / (len(df)/252):.1f} per year")