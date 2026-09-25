"""
Тест SHAP в XAI Bridge
"""
import sys
import os

# Добавляем путь к xai модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'xai', 'modules'))

from xai_bridge import XAIBridge
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("="*60)
print("🔬 ТЕСТ SHAP В XAI BRIDGE")
print("="*60)

# Генерируем данные
print("\n📊 Генерация данных...")
data = np.random.rand(100, 8)
labels = np.random.randint(0, 2, 100)
print(f"   Данные: {data.shape}")
print(f"   Метки: {labels.shape}")

# Обучаем модель
print("\n🤖 Обучение модели...")
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(data, labels)
print("   ✅ Модель обучена")

# Создаём XAI Bridge
print("\n🔧 Создаём XAI Bridge...")
xai = XAIBridge(training_data=data)
xai.set_model(model)

# Проверяем SHAP
print(f"\n✅ SHAP доступен: {xai.shap_explainer is not None}")

if xai.shap_explainer is not None:
    print("   ✅ SHAP успешно инициализирован!")
    
    # Проверяем, что можно получить объяснение
    print("\n🔬 Проверяем объяснение...")
    test_data = data[0]
    try:
        shap_values = xai.shap_explainer.shap_values(test_data.reshape(1, -1), nsamples=10)
        print(f"   ✅ SHAP работает! shap_values: {type(shap_values)}")
    except Exception as e:
        print(f"   ❌ Ошибка при вызове SHAP: {e}")
else:
    print("   ❌ SHAP НЕ инициализирован!")

print("\n" + "="*60)
print("🏁 ТЕСТ ЗАВЕРШЁН")
print("="*60)