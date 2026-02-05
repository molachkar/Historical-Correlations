import pandas as pd

# === SETTINGS ===
file_path = "C:\\Users\\PC\\Desktop\\MML\\data.csv"     # <-- update
price_column = "xauusd-close"   # <-- update if needed
output_file = "test.csv"
threshold = 0.05  # 0.3%

# === LOAD WITHOUT CHANGING ANYTHING ===
df = pd.read_csv(file_path, dtype=str)  # Read everything as string to avoid auto conversion

# Convert price column to float manually
df[price_column] = df[price_column].astype(float)

# === COMPUTE DIRECTION ON RAW ORDER (NO SORTING / NO DATE CHANGE) ===
df['return'] = df[price_column].pct_change()

# Apply threshold-based classification
df['direction'] = df['return'].apply(lambda x: 1 if x > threshold else (0 if x < -threshold else None))

# Drop neutral rows
df = df.dropna(subset=['direction'])
df['direction'] = df['direction'].astype(int)

# Drop return column
df = df.drop(columns=['return'])

# === SAVE WITHOUT TOUCHING TIME FORMAT ===
df.to_csv(output_file, index=False)

print("✅ Labeled file saved as:", output_file)
print(df.head())
print(df['direction'].value_counts())
