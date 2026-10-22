import time
import warnings
from typing import Any
import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
from sklearn.preprocessing import StandardScaler
from ..core.base import Explainer
from ..core.result import ExplanationResult

class CounterfactualTabularExplainer(Explainer):
    def __init__(self, feature_names: list[str] = None, continuous_features: list[str] = None):
        self.feature_names = feature_names or []
        self.continuous_features = continuous_features

    def explain(
        self,
        model: Any,
        input_data: np.ndarray,
        target: int | str | None = None,
        training_data: np.ndarray = None,
        feature_names: list[str] = None,
        **kwargs
    ) -> ExplanationResult:
        
        # Отключаем спам от sklearn
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if training_data is None:
            raise ValueError("CounterfactualTabularExplainer требует training_data.")
        
        fn = feature_names if feature_names is not None else self.feature_names
        if not fn:
            raise ValueError("CounterfactualTabularExplainer требует feature_names.")
        
        continuous = self.continuous_features if self.continuous_features else fn
        start_time = time.perf_counter()
        
        # 1. Стандартизация
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(training_data)
        input_scaled = scaler.transform([input_data])[0]
        df_train = pd.DataFrame(train_scaled, columns=fn)
        
        if target is None:
            if hasattr(model, "predict_proba"):
                target = int(np.argmax(model.predict_proba([input_data])[0]))
            else:
                target = int(model.predict([input_data])[0])
                
        desired_class = 1 - target if target in [0, 1] else (target + 1) % 2
        
        df_train['target'] = model.predict(training_data)
        
        d = dice_ml.Data(dataframe=df_train, continuous_features=continuous, outcome_name='target')
        m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
        exp = Dice(d, m, method="random")
        
        query_instance = pd.DataFrame([input_scaled], columns=fn)
        cf_array = []
        
        try:
            # Уменьшаем total_CFs до 2 для ускорения
            dice_exp = exp.generate_counterfactuals(
                query_instance, 
                total_CFs=2, 
                desired_class=desired_class,
                features_to_vary=fn
            )
            cf_list = dice_exp.cf_examples_list[0].final_cfs_df
            if cf_list is not None and len(cf_list) > 0:
                cf_scaled = cf_list[fn].values
                cf_array = scaler.inverse_transform(cf_scaled).tolist()
        except Exception as e:
            print(f"⚠️ DiCE не смог найти контрфактуалы: {e}")
            
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        # НОВОЕ: Усредняем модули изменений по всем контрфактуалам
        attribution = np.zeros(len(input_data))
        if cf_array:
            diffs = [np.abs(np.array(cf) - input_data) for cf in cf_array]
            attribution = np.mean(diffs, axis=0)
            
        return ExplanationResult(
            method="counterfactual",
            attribution=attribution,
            target=target,
            confidence=1.0,
            runtime_ms=runtime_ms,
            metadata={
                "feature_names": fn,
                "values": input_data.tolist(),
                "counterfactuals": cf_array,
                "desired_class": desired_class
            }
        )