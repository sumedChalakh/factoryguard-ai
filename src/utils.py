# src/utils.py

import shap
import numpy as np

def explain_with_shap(model, X_row):
    """
    Generate SHAP explanation for a single sample.

    Args:
        model: trained XGBoost model
        X_row: pandas DataFrame with ONE ROW

    Returns:
        dict with base_value and shap values
    """
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_row)

    return {
        "base_value": float(explainer.expected_value),
        "shap_values": dict(
            zip(X_row.columns, shap_values[0].tolist())
        )
    }
