# src/train.py
# Full training pipeline for FactoryGuard (NO leakage, API-compatible)

import pandas as pd
from pathlib import Path
import joblib

from src.feature_engineering import create_features

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "factoryguard_production_data_20min.parquet"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading raw data...")
df = pd.read_parquet(RAW_PATH)

print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating features...")
df_fe = create_features(df)

# Drop rows with NaNs from rolling windows
df_fe = df_fe.dropna().reset_index(drop=True)

print(f"Feature shape after FE: {df_fe.shape}")

# ============================================================
# SORT FOR TIME-BASED SPLIT (CRITICAL)
# ============================================================

df_fe = df_fe.sort_values("timestamp").reset_index(drop=True)

# ============================================================
# TRAIN / TEST SPLIT (NO LEAKAGE)
# ============================================================

split_idx = int(len(df_fe) * 0.8)

train_df = df_fe.iloc[:split_idx]
test_df = df_fe.iloc[split_idx:]

y_train = train_df["failure_24h"]
y_test = test_df["failure_24h"]

X_train = train_df.drop(columns=["failure_24h", "machine_id", "timestamp"])
X_test = test_df.drop(columns=["failure_24h", "machine_id", "timestamp"])

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test : X={X_test.shape}, y={y_test.shape}")
print("\nTarget distribution:")
print(y_train.value_counts())

# ============================================================
# SCALING (FIT ON TRAIN ONLY)
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled (train-fit only)")

# ============================================================
# MODEL TRAINING (Logistic Regression Baseline)
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model training complete")

# ============================================================
# PREDICTIONS
# ============================================================

y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print(f"Train predictions (sample): {y_train_pred_proba[:5]}")
print(f"Test predictions  (sample): {y_test_pred_proba[:5]}")

# ============================================================
# SAVE MODEL + PREPROCESSOR
# ============================================================

model_path = MODELS_DIR / "model.joblib"
scaler_path = MODELS_DIR / "preprocessor.joblib"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModel saved to: {model_path}")
print(f"Preprocessor saved to: {scaler_path}")

# ============================================================
# SAVE TEST PREDICTIONS FOR EVALUATION
# ============================================================

test_out = X_test.copy()
test_out.insert(0, "y_test_true", y_test.values)
test_out.insert(1, "y_test_pred_proba", y_test_pred_proba)

predictions_path = PROCESSED_DIR / "test_predictions.parquet"
test_out.to_parquet(predictions_path, index=False)

print(f"Test predictions saved to: {predictions_path}")

print("\n✅ Training pipeline complete!")