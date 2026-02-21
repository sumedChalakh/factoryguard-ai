# src/train.py
# Step 2a: Import libraries + Load raw data

import pandas as pd
from pathlib import Path
from src.feature_engineering import create_features

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "factoryguard_production_data_20min.parquet"
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading raw data...")
df = pd.read_parquet(RAW_PATH)
print(f"Raw data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Step 2b: Call create_features() + separate X and y
print("\nEngineering features...")
df_features = create_features(df)

# Drop rows with NaNs caused by rolling windows
df_features = df_features.dropna().reset_index(drop=True)
print(f"Features shape after dropping NaNs: {df_features.shape}")

# Separate features (X) and target (y)
# Target column is 'failure_24h' (failure in next 24 hours)
y = df_features['failure_24h'].copy()
X = df_features.drop(columns=['failure_24h', 'machine_id', 'timestamp']).copy()

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Step 2c: Train/test split + scaling
from sklearn.preprocessing import StandardScaler
print("\nPerforming time-based train/test split (no leakage)...")
# Sort by timestamp (critical)
df_features = df_features.sort_values("timestamp")

# 80% past → train, 20% future → test
split_idx = int(len(df_features) * 0.8)

train_df = df_features.iloc[:split_idx]
test_df = df_features.iloc[split_idx:]

y_train = train_df["failure_24h"]
y_test = test_df["failure_24h"]

X_train = train_df.drop(columns=["failure_24h", "machine_id", "timestamp"])
X_test = test_df.drop(columns=["failure_24h", "machine_id", "timestamp"])

print(f"Train set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")

# Scale features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler (train-fit only)")

# Step 2d: Train Logistic Regression baseline
from sklearn.linear_model import LogisticRegression

print("\nTraining Logistic Regression baseline...")
# Handle class imbalance with class_weight='balanced'
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Addresses class imbalance
)

lr_model.fit(X_train_scaled, y_train)
print("Logistic Regression training complete")

# Predictions for evaluation
y_train_pred_proba = lr_model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"Train predictions (sample): {y_train_pred_proba[:5]}")
print(f"Test predictions (sample): {y_test_pred_proba[:5]}")

# Step 2e: Save model + preprocessor to joblib
import joblib

print("\nSaving model and preprocessor...")
model_path = MODELS_DIR / "model.joblib"
scaler_path = MODELS_DIR / "preprocessor.joblib"

joblib.dump(lr_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Model saved to: {model_path}")
print(f"Preprocessor saved to: {scaler_path}")

# Save predictions for evaluation step
X_test.insert(0, 'y_test_true', y_test.values)
X_test.insert(1, 'y_test_pred_proba', y_test_pred_proba)
predictions_path = BASE_DIR / "data" / "processed" / "test_predictions.parquet"
predictions_path.parent.mkdir(parents=True, exist_ok=True)
X_test.to_parquet(predictions_path, index=False)

print(f"Test predictions saved to: {predictions_path}")
print("\n✅ Training pipeline complete!")

