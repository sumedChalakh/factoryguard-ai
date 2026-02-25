# src/utils.py
import shap
import numpy as np


def explain_with_shap(model, X_row):
    """
    Generate SHAP explanation for a SINGLE row.

    Args:
        model : trained sklearn model (LogisticRegression)
        X_row : pandas DataFrame with exactly ONE ROW
                (already scaled, same columns as training)

    Returns:
        dict with:
            - base_value
            - shap_values (feature -> contribution)
    """

    # Use a compatible masker/background for sklearn models.
    # For some SHAP versions, passing the model object directly without a
    # masker raises: "model is not callable".
    background = X_row.copy()

    try:
        explainer = shap.Explainer(model, background)
        shap_result = explainer(X_row)
    except TypeError:
        explainer = shap.Explainer(model.predict_proba, background)
        shap_result = explainer(X_row)

    values = np.asarray(shap_result.values)
    base_values = np.asarray(shap_result.base_values)

    if values.ndim == 3:
        shap_values = values[0, :, 1]
    else:
        shap_values = values[0]

    if base_values.ndim == 2:
        base_value = base_values[0, 1]
    elif base_values.ndim == 1 and base_values.size > 1:
        base_value = base_values[1]
    elif base_values.ndim == 1:
        base_value = base_values[0]
    else:
        base_value = base_values

    # Convert to JSON-serializable format
    return {
        "base_value": float(base_value),
        "shap_values": {
            feature: float(value)
            for feature, value in zip(X_row.columns, shap_values)
        }
    }