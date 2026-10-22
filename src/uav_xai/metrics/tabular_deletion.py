import numpy as np
from dataclasses import dataclass

@dataclass
class TabularDeletionResult:
    fractions: np.ndarray
    confidences: np.ndarray
    auc: float

def compute_tabular_deletion(model, input_data, attribution, target, baseline_value=None, steps=20):
    """
    Deletion AUC: удаляем самые важные признаки и смотрим, как падает confidence.
    """
    if baseline_value is None:
        baseline_value = 0.0

    input_data = np.asarray(input_data).ravel()
    attribution = np.asarray(attribution).ravel()
    n_features = len(input_data)

    # Сортируем признаки по убыванию модуля атрибуции
    order = np.argsort(np.abs(attribution))[::-1]
    fractions = np.linspace(0, 1, steps + 1)
    confidences = []

    for frac in fractions:
        num_to_mask = int(round(frac * n_features))
        masked_input = input_data.copy()
        
        if num_to_mask > 0:
            idx = order[:num_to_mask]
            # ИСПРАВЛЕНИЕ: Если baseline - массив, берем значения по индексам
            if isinstance(baseline_value, np.ndarray) and baseline_value.ndim > 0:
                masked_input[idx] = baseline_value[idx]
            else:
                masked_input[idx] = baseline_value

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([masked_input])[0]
            conf = float(proba[target])
        else:
            conf = float(model.predict([masked_input])[0] == target)
        confidences.append(conf)

    confidences = np.array(confidences)
    try:
        auc = float(np.trapezoid(confidences, fractions))
    except AttributeError:
        auc = float(np.trapz(confidences, fractions))

    return TabularDeletionResult(fractions, confidences, auc)