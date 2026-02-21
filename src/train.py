# src/train.py
# Train XGBoost model for predictive maintenance (FactoryGuard AI)

import sys
from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# Path setup (IMPORTANT for imports)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from feature_engineering import create_features

RAW_PATH = BASE_DIR / "data" / "raw" / "factoryguard_production_data_20min.parquet"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Step 1: Load raw data
# ------------------------------------------------------------------
print("Loading raw data...")
df = pd.read_parquet(RAW_PATH)
print(f"Raw data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ------------------------------------------------------------------
# Step 2: Feature engineering
# ------------------------------------------------------------------
print("\nEngineering features...")
df_features = create_features(df)

# Drop NaNs from rolling features
df_features = df_features.dropna().reset_index(drop=True)
print(f"Features shape after dropping NaNs: {df_features.shape}")

# ------------------------------------------------------------------
# Step 3: Time-based train/test split (NO LEAKAGE)
# ------------------------------------------------------------------
print("\nPerforming time-based train/test split (no leakage)...")

df_features = df_features.sort_values("timestamp")

split_idx = int(len(df_features) * 0.8)

train_df = df_features.iloc[:split_idx]
test_df = df_features.iloc[split_idx:]

y_train = train_df["failure_24h"]
y_test = test_df["failure_24h"]

X_train = train_df.drop(columns=["failure_24h", "machine_id", "timestamp"])
X_test = test_df.drop(columns=["failure_24h", "machine_id", "timestamp"])

print(f"Train set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")
print("Target distribution (train):")
print(y_train.value_counts())

# ------------------------------------------------------------------
# Step 4: Scaling (fit on train only)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler (train-fit only)")

# ------------------------------------------------------------------
# Step 5: Handle extreme class imbalance
# ------------------------------------------------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ------------------------------------------------------------------
# Step 6: Train XGBoost model
# ------------------------------------------------------------------
print("\nTraining XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)

print("XGBoost training complete")

# ------------------------------------------------------------------
# Step 7: Generate predictions
# ------------------------------------------------------------------
y_train_pred_proba = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

print(f"Train predictions (sample): {y_train_pred_proba[:5]}")
print(f"Test predictions (sample): {y_test_pred_proba[:5]}")

# ------------------------------------------------------------------
# Step 8: Save model + scaler
# ------------------------------------------------------------------
model_path = MODELS_DIR / "xgboost_model.joblib"
scaler_path = MODELS_DIR / "preprocessor.joblib"

joblib.dump(xgb_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"XGBoost model saved to: {model_path}")
print(f"Preprocessor saved to: {scaler_path}")

# ------------------------------------------------------------------
# Step 9: Save test predictions for evaluation
# ------------------------------------------------------------------
test_out = X_test.copy()
test_out.insert(0, "y_test_true", y_test.values)
test_out.insert(1, "y_test_pred_proba", y_test_pred_proba)

predictions_path = PROCESSED_DIR / "test_predictions.parquet"
test_out.to_parquet(predictions_path, index=False)

print(f"Test predictions saved to: {predictions_path}")

print("\n✅ XGBoost training pipeline complete!")

