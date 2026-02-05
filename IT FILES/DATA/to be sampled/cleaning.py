import pandas as pd

# === LOAD YOUR BIG DATASET ===
df = pd.read_csv("C:\\Users\\PC\\Desktop\\MML\\combined.csv")

# === Replace 0 and NaN (empty) values with NaN for uniform handling ===
df = df.replace(0, pd.NA)

# === Forward fill each column independently ===
df_ffill = df.ffill()

# === Save cleaned dataset ===
df_ffill.to_csv("cleaned_forward_filled.csv", index=False)

print("Done. Rows:", len(df))
