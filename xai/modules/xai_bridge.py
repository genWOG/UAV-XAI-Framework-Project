"""
XAI Bridge — связующий слой между flight_monitor.py и фреймворком uav_xai.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from uav_xai.core import create_default_engine
    HAS_UAV_XAI = True
except ImportError:
    HAS_UAV_XAI = False


class XAIBridge:
    def __init__(self, log_manager=None):
        self.log = log_manager
        self.explanation_history = []
        self.engine = None
        self._model = None
        self.feature_names = [
            'altitude', 'speed', 'obstacle_dist', 'battery',
            'obstacle_density', 'yaw', 'roll', 'pitch'
        ]
        
        if HAS_UAV_XAI:
            try:
                self.engine = create_default_engine()
                self._log("✅ XAIEngine инициализирован", level='system')
                self._log(f"   Доступные методы: {self.engine.list_methods()}", level='system')
            except Exception as e:
                self._log(f"❌ Ошибка XAIEngine: {e}", level='error')
                self.engine = None
    
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
    
    def _create_model(self):
        """Создаёт простую модель для XAI"""
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self, input_size=8, num_classes=2):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 16)
                self.fc2 = nn.Linear(16, num_classes)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)  # Логиты
        
        return SimpleModel()
    
    def explain_decision(
        self, 
        state: Dict[str, Any], 
        action: int,
        confidence: float = 0.0,
        method: str = "grad_cam"
    ) -> Optional[Dict[str, Any]]:
        """Объяснить решение БПЛА."""
        
        if not HAS_UAV_XAI or self.engine is None:
            return self._fallback_explanation(state, action, confidence)
        
        try:
            import torch
            
            # Извлекаем признаки
            pos = state.get('position', {})
            vel = state.get('velocity', {})
            orient = state.get('orientation', {})
            sensors = state.get('sensors', {})
            
            features = np.array([
                abs(pos.get('z', 0)),
                np.sqrt(vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2),
                sensors.get('obstacle_distance', 100.0),
                sensors.get('battery', 100.0),
                sensors.get('obstacle_density', 0.0),
                orient.get('yaw', 0),
                orient.get('roll', 0),
                orient.get('pitch', 0),
            ], dtype=np.float32)
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return self._fallback_explanation(state, action, confidence)
            
            # Создаём модель один раз
            if self._model is None:
                self._model = self._create_model()
                self._model.eval()
            
            model = self._model
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Для Grad-CAM используем fc1 как target_layer
            target_layer = model.fc1 if method == "grad_cam" else None
            
            start_time = datetime.now()
            
            result = self.engine.explain(
                method=method,
                model=model,
                input_data=input_tensor,
                target=action,
                target_layer=target_layer,
            )
            
            runtime_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            explanation = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'action_name': 'Остановка' if action == 0 else 'Продолжить',
                'confidence': confidence,
                'method': method,
                'runtime_ms': runtime_ms,
                'attribution': result.attribution.tolist() if hasattr(result, 'attribution') else None,
                'feature_names': self.feature_names,
                'feature_values': features.tolist(),
                'is_anomaly': self._check_anomaly(features, action, confidence)
            }
            
            explanation['explanation_text'] = self._generate_text(
                features, explanation.get('attribution'), action
            )
            
            self.explanation_history.append(explanation)
            return explanation
            
        except Exception as e:
            self._log(f"❌ XAI ошибка: {e}", level='error')
            return self._fallback_explanation(state, action, confidence)
    
    def _check_anomaly(self, features, action, confidence):
        anomaly_score = 0
        reasons = []
        
        altitude, speed, obstacle_dist, battery = features[0], features[1], features[2], features[3]
        
        if altitude < 5:
            anomaly_score += 0.3
            reasons.append("Опасная низкая высота")
        if obstacle_dist < 3:
            anomaly_score += 0.4
            reasons.append("Критическое приближение к препятствию")
        if battery < 15:
            anomaly_score += 0.2
            reasons.append("Критический заряд батареи")
        if speed > 5 and obstacle_dist < 10:
            anomaly_score += 0.2
            reasons.append("Высокая скорость вблизи препятствия")
        if action == 1 and obstacle_dist < 3:
            anomaly_score += 0.3
            reasons.append("Опасное продолжение движения")
        
        return {
            'score': min(anomaly_score, 1.0),
            'is_anomaly': anomaly_score > 0.4,
            'reasons': reasons,
            'level': 'critical' if anomaly_score > 0.7 else 'warning'
        }
    
    def _generate_text(self, features, attribution, action):
        if attribution is None:
            return "Объяснение недоступно"
        
        parts = [f"Решение: {'ОСТАНОВИТЬСЯ' if action == 0 else 'ПРОДОЛЖИТЬ'}"]
        parts.append("Влияющие факторы:")
        
        abs_attr = np.abs(attribution)
        top_indices = np.argsort(abs_attr)[-3:][::-1]
        
        for idx in top_indices:
            if idx < len(self.feature_names) and idx < len(features):
                name = self.feature_names[idx]
                value = features[idx]
                importance = attribution[idx]
                direction = "способствует" if importance > 0 else "препятствует"
                parts.append(f"  • {name}: {value:.2f} ({direction}, вес: {importance:+.3f})")
        
        return "\n".join(parts)
    
    def _fallback_explanation(self, state, action, confidence):
        return {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'action_name': 'Остановка' if action == 0 else 'Продолжить',
            'confidence': confidence,
            'method': 'fallback',
            'runtime_ms': 0,
            'attribution': None,
            'is_anomaly': {'is_anomaly': False, 'reasons': []},
            'explanation_text': 'XAI не доступен'
        }
    
    def get_summary(self):
        if not self.explanation_history:
            return {'count': 0, 'anomaly_count': 0, 'anomaly_rate': 0.0, 'methods_used': []}
        
        anomaly_count = sum(1 for e in self.explanation_history 
                           if e.get('is_anomaly', {}).get('is_anomaly', False))
        
        return {
            'count': len(self.explanation_history),
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_count / len(self.explanation_history) if self.explanation_history else 0.0,
            'methods_used': list(set(e.get('method') for e in self.explanation_history if e.get('method')))
        }
    
    def save_explanations(self, filepath: str):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.explanation_history, f, indent=2, default=str)