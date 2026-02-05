# predict_test_accuracy.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "C:\\Users\\PC\\Desktop\\test\\lgbm_model_binary_max.pkl"
TEST_CSV = "C:\\Users\\PC\\Desktop\\test\\train_binary.csv"  # your real test file

# Load
model = joblib.load(MODEL_PATH)
df = pd.read_csv(TEST_CSV)

# If your test set already contains true direction:
if "direction" in df.columns:
    X_test = df.drop(columns=["direction"])
    y_true = df["direction"]
else:
    X_test = df
    y_true = None

# Predict
y_pred = model.predict(X_test)

# Output
df["prediction"] = y_pred
df.to_csv("test_predictions.csv", index=False)
print("✅ Saved predictions as test_predictions.csv")

if y_true is not None:
    print("\nTest Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
