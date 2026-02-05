# extract_feature_importance.py

import joblib
import pandas as pd

MODEL_PATH = "C:\\Users\\PC\\Desktop\\test\\lgbm_model_binary_max.pkl"  # change if different

print("Loading model...")
model = joblib.load(MODEL_PATH)

# Extract importance
fi = pd.DataFrame({
    "feature": model.feature_name_,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

fi.to_csv("feature_importance.csv", index=False)
print("✅ Saved feature_importance.csv")
print(fi.head(15))
