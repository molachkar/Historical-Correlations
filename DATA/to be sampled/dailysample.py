import pandas as pd

# === LOAD YOUR MONTHLY DATA ===
# Replace with your file path. Must contain a 'date' column.
df = pd.read_csv("C:\\Users\\PC\\Desktop\\MML\\to be sampled\\unemployment rate monthly.csv")

# === Ensure datetime format ===
df['time'] = pd.to_datetime(df['time'])

# === Set index ===
df = df.set_index('time')

# === Resample to Daily and Forward Fill ===
daily_df = df.resample('D').ffill()

# === Save Result ===
daily_df.reset_index().to_csv("monthly_to_daily_ffill.csv", index=False)

print("Done. Original rows:", len(df), "| Daily rows:", len(daily_df))