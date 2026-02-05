import pandas as pd
import numpy as np

# === Load your dataset ===
df = pd.read_csv("C:\\Users\\PC\\Desktop\\MML\\test.csv")  # change filename

# === Ensure price column exists ===
price_col = "xauusd-close"  # adjust if needed

if price_col not in df.columns:
    raise ValueError(f"Column '{price_col}' not found in dataset!")

# === Create binary direction (1 = Up, 0 = Down) ===
df["direction"] = (df[price_col].shift(-1) > df[price_col]).astype(int)

# === OPTIONAL: Add RSI (14) ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["xauusd-RSI"] = compute_rsi(df[price_col])

# === OPTIONAL: Add ATR (14) — requires high/low/close ===
# If you DON’T have high/low, skip this part
if all(col in df.columns for col in ["xauusd-high", "xauusd-low", "xauusd-close"]):
    high = df["xauusd-high"]
    low = df["xauusd-low"]
    close = df["xauusd-close"]
    tr = pd.DataFrame({
        "H-L": high - low,
        "H-C": (high - close.shift()).abs(),
        "L-C": (low - close.shift()).abs()
    }).max(axis=1)
    df["xauusd-ATR"] = tr.rolling(14).mean()
else:
    print("⚠️ ATR skipped — no high/low columns found.")

# === Drop rows with NaN introduced by indicators ===
df = df.dropna().reset_index(drop=True)

# === Save updated file ===
df.to_csv("test.csv", index=False)

print("✅ Done! Saved as train_binary.csv")
