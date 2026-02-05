import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------- CONFIG --------
MODEL_PATH = "C:\\Users\\PC\\Desktop\\MML\\lightgbm-direction-model.pkl"
FEATURES_PATH = "C:\\Users\\PC\\Desktop\\MML\\feature_names.pkl"
THRESHOLD_PATH = "C:\\Users\\PC\\Desktop\\MML\\best_threshold.pkl"

# -------- LOAD MODEL --------
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
threshold = joblib.load(THRESHOLD_PATH)

print("="*60)
print("GOLD TRADING SIGNAL GENERATOR")
print("="*60)
print(f"Model loaded: {MODEL_PATH}")
print(f"Threshold: {threshold:.2f}")
print(f"Required features: {len(feature_names)}")
print("="*60)

# -------- PREPARE TODAY'S DATA --------
# REPLACE THIS WITH YOUR LIVE DATA SOURCE
# Example: Read from CSV with today's data
today_data_path = "C:\\Users\\PC\\Desktop\\MML\\today_data.csv"
today_df = pd.read_csv(today_data_path)

# Ensure all required features are present
missing_features = [f for f in feature_names if f not in today_df.columns]
if missing_features:
    print(f"ERROR: Missing features: {missing_features}")
    exit()

# Select only required features in correct order
X_today = today_df[feature_names].iloc[-1:].copy()  # Get last row (today)

# Handle any missing values
X_today = X_today.ffill().bfill()
X_today = X_today.replace([np.inf, -np.inf], np.nan).ffill().bfill()

print("\nToday's data loaded successfully")
print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

# -------- GENERATE PREDICTION --------
prediction_proba = model.predict_proba(X_today)[0]
prob_down = prediction_proba[0]
prob_up = prediction_proba[1]

print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)
print(f"Probability DOWN: {prob_down:.4f} ({prob_down*100:.2f}%)")
print(f"Probability UP:   {prob_up:.4f} ({prob_up*100:.2f}%)")

# -------- GENERATE SIGNAL --------
print("\n" + "="*60)
print("TRADING SIGNAL")
print("="*60)

if prob_up > threshold:
    signal = "BUY"
    confidence = prob_up
    print(f"📈 SIGNAL: {signal}")
    print(f"💪 CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"✅ TRADE: YES - Enter LONG position")
    print(f"🎯 Risk-Reward: 1:2")
    print(f"📊 Expected Win Rate: 89%")
else:
    signal = "HOLD"
    confidence = max(prob_down, prob_up)
    print(f"⏸️  SIGNAL: {signal}")
    print(f"⚠️  CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"❌ TRADE: NO - Stay out of market")
    print(f"📝 Reason: Confidence below threshold ({threshold:.2f})")

print("="*60)

# -------- SAVE SIGNAL LOG --------
log_entry = {
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'signal': signal,
    'prob_up': prob_up,
    'prob_down': prob_down,
    'confidence': confidence,
    'threshold': threshold
}

log_df = pd.DataFrame([log_entry])
log_file = "C:\\Users\\PC\\Desktop\\MML\\signal_log.csv"

try:
    existing_log = pd.read_csv(log_file)
    updated_log = pd.concat([existing_log, log_df], ignore_index=True)
    updated_log.to_csv(log_file, index=False)
except FileNotFoundError:
    log_df.to_csv(log_file, index=False)

print(f"\nSignal logged to: {log_file}")