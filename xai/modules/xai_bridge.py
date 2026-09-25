"""
XAI Bridge — связующий слой между flight_monitor.py и XAI-методами.
Версия 3.5 — оптимизированный + новые признаки
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import time

# Добавляем пути
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# ========================================
# ИМПОРТ XAI-БИБЛИОТЕК
# ========================================

# 1. LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME не установлен! Установи: pip install lime")

# 2. SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP не установлен! Установи: pip install shap")

# 3. LRP (через Captum)
try:
    import torch
    import torch.nn as nn
    from captum.attr import InputXGradient
    LRP_AVAILABLE = True
except ImportError:
    LRP_AVAILABLE = False
    print("⚠️ Captum не установлен! Установи: pip install captum")

# ========================================
# ОСНОВНОЙ КЛАСС XAI BRIDGE
# ========================================

class XAIBridge:
    """
    Мост между flight_monitor.py и XAI-методами.
    Поддерживает: LIME, SHAP, LRP (через Captum)
    """

    def __init__(
        self,
        log_manager=None,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.log = log_manager
        self.explanation_history = []
        self.feature_names = feature_names or [
            'altitude', 'speed', 'obstacle_dist', 'battery',
            'obstacle_density', 'yaw', 'roll', 'pitch',
            'distance_to_target', 'angle_to_target'
        ]
        self.class_names = ['Остановка', 'Продолжить']
        self.training_data = training_data

        # Модели для XAI
        self.sklearn_model = None
        self.torch_model = None

        # Счётчики методов
        self.method_counter = 0
        self.available_methods = []
        if LIME_AVAILABLE:
            self.available_methods.append('lime')
        if SHAP_AVAILABLE:
            self.available_methods.append('shap')
        if LRP_AVAILABLE:
            self.available_methods.append('lrp')
        if not self.available_methods:
            self.available_methods = ['fallback']
        print(f"✅ Доступные XAI методы: {self.available_methods}")

        # 👉 ИНИЦИАЛИЗАЦИЯ LIME
        self.lime_explainer = None
        if LIME_AVAILABLE and training_data is not None:
            try:
                self.lime_explainer = LimeTabularExplainer(
                    training_data=training_data,
                    mode='classification',
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    verbose=False,
                    discretize_continuous=True
                )
                self._log("✅ LIME Explainer инициализирован", level='system')
            except Exception as e:
                self._log(f"❌ Ошибка LIME: {e}", level='error')
                self.lime_explainer = None

        # 👉 ПОДГОТОВКА ДАННЫХ ДЛЯ SHAP
        self.shap_explainer = None
        self._shap_sample_data = None
        if SHAP_AVAILABLE and training_data is not None:
            try:
                if isinstance(training_data, np.ndarray) and len(training_data) > 0:
                    if len(training_data.shape) == 2 and training_data.shape[0] > 0:
                        if not np.any(np.isnan(training_data)):
                            self._shap_sample_data = training_data[:100] if len(training_data) > 100 else training_data
                            self._log("✅ SHAP данные подготовлены", level='system')
                        else:
                            self._log("⚠️ SHAP: данные содержат NaN", level='warn')
                    else:
                        self._log(f"⚠️ SHAP: неверная форма данных {training_data.shape}", level='warn')
                else:
                    self._log("⚠️ SHAP: данные пустые или неверного типа", level='warn')
            except Exception as e:
                self._log(f"❌ Ошибка подготовки данных SHAP: {e}", level='error')

        self._log(f"✅ XAI Bridge инициализирован. Методы: {self.available_methods}", level='system')

    def _log(self, message: str, level: str = 'info'):
        if self.log:
            if level == 'system':
                self.log.system(message)
            elif level == 'error':
                self.log.error(message)
            elif level == 'warn':
                self.log.warn(message)
            else:
                self.log.info(message)
        else:
            print(f"[{level.upper()}] {message}")

    def set_model(self, model):
        """Устанавливает sklearn-модель для LIME и SHAP."""
        self.sklearn_model = model

        # 👉 ИНИЦИАЛИЗИРУЕМ SHAP ТОЛЬКО КОГДА ЕСТЬ МОДЕЛЬ
        if SHAP_AVAILABLE and hasattr(self, '_shap_sample_data') and self._shap_sample_data is not None:
            try:
                import shap
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba,
                    self._shap_sample_data
                )
                self._log("✅ SHAP Explainer инициализирован с моделью", level='system')
            except Exception as e:
                self._log(f"❌ Ошибка SHAP при инициализации с моделью: {e}", level='error')
                self.shap_explainer = None
        elif self.shap_explainer is not None and model is not None:
            try:
                self.shap_explainer.model = model.predict_proba
                self._log("✅ SHAP модель обновлена", level='system')
            except Exception as e:
                self._log(f"⚠️ SHAP модель не обновлена: {e}", level='warn')

    def set_torch_model(self, model: nn.Module):
        """Устанавливает PyTorch-модель для LRP."""
        self.torch_model = model
        if model is not None:
            model.eval()
            self._log("✅ Torch модель установлена для LRP", level='system')

    def get_next_method(self) -> str:
        method = self.available_methods[self.method_counter % len(self.available_methods)]
        self.method_counter += 1
        return method

    def explain_decision(
        self,
        state: Dict[str, Any],
        action: int,
        confidence: float = 0.0,
        method: str = "auto"
    ) -> Optional[Dict[str, Any]]:
        if method == 'auto':
            method = self.get_next_method()

        try:
            features = self._extract_features(state)
        except Exception as e:
            self._log(f"❌ Ошибка извлечения признаков: {e}", level='error')
            result = self._fallback_explanation(state, action, confidence)
            self.explanation_history.append(result)
            return result

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            result = self._fallback_explanation(state, action, confidence)
            self.explanation_history.append(result)
            return result

        result = None
        if method == 'lime' and LIME_AVAILABLE and self.lime_explainer is not None and self.sklearn_model is not None:
            result = self._explain_lime(features, action, confidence)
        elif method == 'shap' and SHAP_AVAILABLE and self.shap_explainer is not None and self.sklearn_model is not None:
            result = self._explain_shap(features, action, confidence)
        elif method == 'lrp' and LRP_AVAILABLE and self.torch_model is not None:
            result = self._explain_lrp(features, action, confidence)
        else:
            self._log(f"⚠️ Метод {method} недоступен, использую fallback", level='warn')
            result = self._fallback_explanation(features, action, confidence)

        if result is not None:
            self.explanation_history.append(result)
        return result

    def _extract_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Извлекает признаки из состояния БПЛА (10 признаков)."""
        pos = state.get('position', {})
        vel = state.get('velocity', {})
        orient = state.get('orientation', {})
        sensors = state.get('sensors', {})
        
        # Вычисляем расстояние до цели
        target_x = state.get('target_x', 200)
        target_y = state.get('target_y', 300)
        dx = target_x - pos.get('x', 0)
        dy = target_y - pos.get('y', 0)
        distance_to_target = (dx*dx + dy*dy) ** 0.5
        
        # Вычисляем угол к цели
        angle_to_target = np.arctan2(dy, dx) - orient.get('yaw', 0)
        angle_to_target = np.degrees(angle_to_target)
        
        return np.array([
            abs(pos.get('z', 0)),                                    # высота
            np.sqrt(vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2),  # скорость
            sensors.get('obstacle_distance', 100.0),                # расстояние до препятствия
            sensors.get('battery', 100.0),                          # заряд батареи
            sensors.get('obstacle_density', 0.0),                   # плотность препятствий
            orient.get('yaw', 0),                                   # рыскание
            orient.get('roll', 0),                                  # крен
            orient.get('pitch', 0),                                 # тангаж
            distance_to_target,                                     # расстояние до цели
            angle_to_target,                                        # угол к цели
        ])

    # ========================================
    # LIME EXPLAINER (ОПТИМИЗИРОВАННЫЙ)
    # ========================================

    def _explain_lime(self, features: np.ndarray, action: int, confidence: float) -> Dict:
        start_time = datetime.now()
        try:
            exp = self.lime_explainer.explain_instance(
                data_row=features,
                predict_fn=self.sklearn_model.predict_proba,
                num_features=len(self.feature_names),
                num_samples=500,  # 👈 ОПТИМИЗАЦИЯ: было 1000
                labels=[0, 1]
            )
            runtime_ms = (datetime.now() - start_time).total_seconds() * 1000

            importance = np.zeros(len(self.feature_names))
            for feature, weight in exp.as_list(label=action):
                for i, name in enumerate(self.feature_names):
                    if name in feature or feature in name:
                        importance[i] = weight
                        break

            anomaly = self._check_anomaly(features, action, confidence)
            return {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'action_name': 'Остановка' if action == 0 else 'Продолжить',
                'confidence': confidence,
                'method': 'lime',
                'runtime_ms': runtime_ms,
                'attribution': importance.tolist(),
                'feature_names': self.feature_names,
                'is_anomaly': anomaly,
                'lime_details': exp.as_list(label=action),
                'explanation_text': self._generate_text(features, importance, action)
            }
        except Exception as e:
            self._log(f"❌ LIME ошибка: {e}", level='error')
            return self._fallback_explanation(features, action, confidence)

    # ========================================
    # SHAP EXPLAINER (ОПТИМИЗИРОВАННЫЙ)
    # ========================================

    def _explain_shap(self, features: np.ndarray, action: int, confidence: float) -> Dict:
        start_time = datetime.now()
        try:
            if self.sklearn_model is None:
                raise ValueError("Sklearn модель не установлена для SHAP")
            if self.shap_explainer is None:
                raise ValueError("SHAP Explainer не инициализирован")

            shap_values = self.shap_explainer.shap_values(
                features.reshape(1, -1),
                nsamples=50  # 👈 ОПТИМИЗАЦИЯ: было 100
            )
            runtime_ms = (datetime.now() - start_time).total_seconds() * 1000

            if isinstance(shap_values, list):
                if action < len(shap_values):
                    importance = shap_values[action][0]
                else:
                    importance = shap_values[0][0]
            else:
                if len(shap_values.shape) == 3:
                    importance = shap_values[0, :, action]
                else:
                    importance = shap_values[0]

            anomaly = self._check_anomaly(features, action, confidence)
            return {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'action_name': 'Остановка' if action == 0 else 'Продолжить',
                'confidence': confidence,
                'method': 'shap',
                'runtime_ms': runtime_ms,
                'attribution': importance.tolist() if isinstance(importance, np.ndarray) else [float(importance)] * len(self.feature_names),
                'feature_names': self.feature_names,
                'is_anomaly': anomaly,
                'shap_details': f"SHAP values for class {action}",
                'explanation_text': self._generate_text(features, importance, action)
            }
        except Exception as e:
            self._log(f"❌ SHAP ошибка: {e}", level='error')
            return self._fallback_explanation(features, action, confidence)

    # ========================================
    # LRP EXPLAINER
    # ========================================

    def _explain_lrp(self, features: np.ndarray, action: int, confidence: float) -> Dict:
        start_time = datetime.now()
        try:
            if self.torch_model is None:
                raise ValueError("PyTorch модель не установлена для LRP")

            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            input_tensor.requires_grad_(True)

            attr_method = InputXGradient(self.torch_model)
            attributions = attr_method.attribute(input_tensor, target=action)
            runtime_ms = (datetime.now() - start_time).total_seconds() * 1000

            importance = attributions.detach().numpy()[0]
            anomaly = self._check_anomaly(features, action, confidence)

            return {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'action_name': 'Остановка' if action == 0 else 'Продолжить',
                'confidence': confidence,
                'method': 'lrp',
                'runtime_ms': runtime_ms,
                'attribution': importance.tolist(),
                'feature_names': self.feature_names,
                'is_anomaly': anomaly,
                'lrp_details': 'Attributions via InputXGradient (Captum)',
                'explanation_text': self._generate_text(features, importance, action)
            }
        except Exception as e:
            self._log(f"❌ LRP ошибка: {e}", level='error')
            return self._fallback_explanation(features, action, confidence)

    # ========================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================

    def _check_anomaly(self, features: np.ndarray, action: int, confidence: float) -> Dict:
        anomaly_score = 0
        reasons = []
        altitude, speed, obstacle_dist, battery = features[0], features[1], features[2], features[3]

        if altitude < 2.0:
            anomaly_score += 0.3
            reasons.append("Критически низкая высота")
        if obstacle_dist < 3:
            anomaly_score += 0.4
            reasons.append("Критическое приближение к препятствию")
        if battery < 15:
            anomaly_score += 0.2
            reasons.append("Критический заряд батареи")
        if speed > 5 and obstacle_dist < 10:
            anomaly_score += 0.2
            reasons.append("Высокая скорость вблизи препятствия")
        if hasattr(self, '_last_features'):
            if np.max(np.abs(features[:3] - self._last_features[:3])) < 0.1:
                anomaly_score += 0.3
                reasons.append("Дрон завис на месте")
        self._last_features = features.copy()
        if action == 1 and obstacle_dist < 3:
            anomaly_score += 0.3
            reasons.append("Опасное продолжение движения")

        return {
            'score': min(anomaly_score, 1.0),
            'is_anomaly': anomaly_score > 0.4,
            'reasons': reasons,
            'level': 'critical' if anomaly_score > 0.7 else 'warning'
        }

    def _generate_text(self, features: np.ndarray, attribution: np.ndarray, action: int) -> str:
        if attribution is None or len(attribution) == 0:
            return "Объяснение недоступно"
        parts = [f"Решение: {'ОСТАНОВИТЬСЯ' if action == 0 else 'ПРОДОЛЖИТЬ'}"]
        parts.append("Влияющие факторы:")
        abs_attr = np.abs(attribution)
        top_indices = np.argsort(abs_attr)[-3:][::-1]
        for idx in top_indices:
            name = self.feature_names[idx]
            value = features[idx]
            importance = attribution[idx]
            direction = "способствует" if importance > 0 else "препятствует"
            parts.append(f" • {name}: {value:.2f} ({direction}, вес: {importance:+.3f})")
        return "\n".join(parts)

    def _fallback_explanation(self, features, action, confidence) -> Dict:
        return {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'action_name': 'Остановка' if action == 0 else 'Продолжить',
            'confidence': confidence,
            'method': 'fallback',
            'runtime_ms': 0,
            'attribution': [0.0] * len(self.feature_names),
            'feature_names': self.feature_names,
            'is_anomaly': {'is_anomaly': False, 'reasons': [], 'score': 0, 'level': 'warning'},
            'explanation_text': 'XAI не доступен (используется fallback)'
        }

    # ========================================
    # ОТЧЁТЫ И СОХРАНЕНИЕ
    # ========================================

    def get_summary(self) -> Dict:
        if not self.explanation_history:
            return {'count': 0}
        anomaly_count = sum(
            1 for e in self.explanation_history
            if e.get('is_anomaly', {}).get('is_anomaly', False)
        )
        methods_used = list(set(e.get('method') for e in self.explanation_history if e.get('method')))
        return {
            'count': len(self.explanation_history),
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_count / len(self.explanation_history) if self.explanation_history else 0,
            'methods_used': methods_used
        }

    def save_explanations(self, filepath: str) -> bool:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.explanation_history, f, indent=2, default=str, ensure_ascii=False)
            self._log(f"✅ XAI сохранены: {filepath} ({len(self.explanation_history)} объяснений)", level='system')
            return True
        except Exception as e:
            self._log(f"❌ Ошибка сохранения XAI: {e}", level='error')
            return False

    def clear_history(self):
        self.explanation_history = []
        self._last_features = None

    def get_last_explanation(self) -> Optional[Dict]:
        if self.explanation_history:
            return self.explanation_history[-1]
        return None