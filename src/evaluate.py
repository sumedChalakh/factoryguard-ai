# src/evaluate.py
# Evaluate Logistic Regression model using PR-AUC

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = BASE_DIR / "data" / "processed" / "test_predictions.parquet"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading test predictions...")
df_preds = pd.read_parquet(PREDICTIONS_PATH)

# Extract true labels and predicted probabilities
y_test_true = df_preds['y_test_true'].values
y_test_pred_proba = df_preds['y_test_pred_proba'].values

print(f"Test set size: {len(y_test_true)}")
print(f"Target distribution: {pd.Series(y_test_true).value_counts().to_dict()}")

# Calculate PR-AUC (PRIMARY METRIC)
precision, recall, _ = precision_recall_curve(y_test_true, y_test_pred_proba)
pr_auc = auc(recall, precision)

print(f"\n{'='*50}")
print(f"PRIMARY METRIC - PR-AUC: {pr_auc:.4f}")
print(f"{'='*50}")

# Convert probabilities to binary predictions (threshold 0.5)
y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

# Calculate additional metrics for context
accuracy = accuracy_score(y_test_true, y_test_pred)
f1 = f1_score(y_test_true, y_test_pred)
roc_auc = roc_auc_score(y_test_true, y_test_pred_proba)

print(f"\nOther Metrics:")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - F1 Score: {f1:.4f}")
print(f"  - ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix & Classification Report
cm = confusion_matrix(y_test_true, y_test_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

print(f"\nClassification Report:")
print(classification_report(y_test_true, y_test_pred, target_names=['No Failure', 'Failure']))

# Save evaluation results to file
results = {
    'model': 'LogisticRegression',
    'metric': 'PR-AUC',
    'pr_auc': pr_auc,
    'accuracy': accuracy,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'tn': int(cm[0,0]),
    'fp': int(cm[0,1]),
    'fn': int(cm[1,0]),
    'tp': int(cm[1,1])
}

results_df = pd.DataFrame([results])
results_path = REPORTS_DIR / "evaluation_results.csv"
results_df.to_csv(results_path, index=False)

print(f"\n✅ Evaluation complete!")
print(f"Results saved to: {results_path}")

