import pandas as pd

# === LOAD YOUR FILES ===
# Replace these with your actual paths
skeleton = pd.read_csv("C:\\Users\\PC\\Desktop\\MML\\skeleton.csv")   # Must have a 'date' column
other_df = pd.read_csv("C:\\Users\\PC\\Desktop\\MML\\unemployment rate.csv")      # Any dataset to align

# === Ensure proper datetime formatting ===
skeleton['time'] = pd.to_datetime(skeleton['time'])
other_df['time'] = pd.to_datetime(other_df['time'])

# === Set index for alignment ===
skeleton = skeleton.set_index('time')
other_df = other_df.set_index('time')

# === Align using reindex ===
aligned_df = other_df.reindex(skeleton.index)

# === (Optional) Forward Fill or Leave NaNs ===
aligned_df_ffill = aligned_df.ffill()  # or .bfill()

# === Reset index & save ===
aligned_df_ffill.reset_index().to_csv("aligned_ffill.csv", index=False)

print("Alignment complete.")
print(f"Original rows: {len(other_df)} | After alignment: {len(aligned_df)}")