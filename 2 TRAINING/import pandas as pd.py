import pandas as pd
import numpy as np

# === CONFIG ===
INPUT_FILE = "C:\\Users\\PC\\Desktop\\train.csv"              # Your input file
OUTPUT_FILE = "train_with_direction.csv"  # Output file
THRESHOLD_METHOD = "auto"  # Options: "auto" for % of ATR, float for fixed move like 0.5

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE)

# Assume first column is XAUUSD-close
price = df.iloc[:, 0]

# === CALCULATE RETURNS ===
df['return'] = price.diff()

# === DETERMINE DIRECTION WITH THRESHOLD ===
if THRESHOLD_METHOD == "auto":
    # Auto threshold = 10% of median absolute return (adapts to volatility)
    threshold = df['return'].abs().median() * 0.1
else:
    threshold = float(THRESHOLD_METHOD)

df['direction'] = df['return'].apply(lambda x: 
                        1 if x > threshold else
                        -1 if x < -threshold else 
                        0)

# Remove first row (no previous price)
df = df.iloc[1:].reset_index(drop=True)

# === SAVE ===
df.to_csv(OUTPUT_FILE, index=False)

print(f"Done! Saved as {OUTPUT_FILE} with threshold = {threshold}")
