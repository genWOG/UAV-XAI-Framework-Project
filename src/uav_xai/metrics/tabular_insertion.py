import numpy as np
from dataclasses import dataclass

@dataclass
class TabularInsertionResult:
    fractions: np.ndarray
    confidences: np.ndarray
    auc: float

def compute_tabular_insertion(model, input_data, attribution, target, baseline_value=None, steps=20):
    """
    Insertion AUC: добавляем самые важные признаки в baseline и смотрим, как растет confidence.
    """
    if baseline_value is None:
        baseline_value = 0.0

    input_data = np.asarray(input_data).ravel()
    attribution = np.asarray(attribution).ravel()
    n_features = len(input_data)

    order = np.argsort(np.abs(attribution))[::-1]
    fractions = np.linspace(0, 1, steps + 1)
    confidences = []

    for frac in fractions:
        num_to_insert = int(round(frac * n_features))
        
        # Инициализируем baseline значениями
        if isinstance(baseline_value, np.ndarray) and baseline_value.ndim > 0:
            inserted_input = baseline_value.copy()
        else:
            inserted_input = np.full_like(input_data, baseline_value)
            
        if num_to_insert > 0:
            idx = order[:num_to_insert]
            inserted_input[idx] = input_data[idx]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([inserted_input])[0]
            conf = float(proba[target])
        else:
            conf = float(model.predict([inserted_input])[0] == target)
        confidences.append(conf)

    confidences = np.array(confidences)
    try:
        auc = float(np.trapezoid(confidences, fractions))
    except AttributeError:
        auc = float(np.trapz(confidences, fractions))

    return TabularInsertionResult(fractions, confidences, auc)