"""
Тест SHAP для XAI Bridge
Проверяет, что SHAP правильно инициализируется и работает
"""

import sys
import os
import numpy as np

# Добавляем пути
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xai', 'modules'))

print("="*60)
print("🔬 ТЕСТ SHAP")
print("="*60 + "\n")

# ========================================
# 1. ГЕНЕРИРУЕМ ДАННЫЕ
# ========================================

def generate_training_data(n_samples=300):
    """Генерирует синтетические данные."""
    np.random.seed(42)
    data = []
    labels = []
    
    for _ in range(n_samples):
        altitude = np.random.uniform(2, 30)
        speed = np.random.uniform(1, 15)
        obstacle_dist = np.random.uniform(2, 100)
        battery = np.random.uniform(20, 100)
        density = np.random.uniform(0, 1)
        yaw = np.random.uniform(-180, 180)
        roll = np.random.uniform(-30, 30)
        pitch = np.random.uniform(-30, 30)
        
        if obstacle_dist < 10 or battery < 15:
            label = 0
        else:
            label = 1
            
        data.append([altitude, speed, obstacle_dist, battery, density, yaw, roll, pitch])
        labels.append(label)
    
    return np.array(data), np.array(labels)

print("📊 Генерация данных...")
training_data, training_labels = generate_training_data(300)
print(f"   Данные: {training_data.shape}")
print(f"   Метки: {training_labels.shape}")
print(f"   Тип данных: {type(training_data)}")
print(f"   NaN в данных: {np.any(np.isnan(training_data))}")

# ========================================
# 2. ОБУЧАЕМ МОДЕЛЬ
# ========================================

print("\n🤖 Обучение модели...")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(training_data, training_labels)
print("   ✅ Модель обучена")

# ========================================
# 3. ТЕСТ SHAP
# ========================================

print("\n🔬 Тестируем SHAP...")

try:
    import shap
    print("   ✅ SHAP импортирован")
    
    # Проверяем данные
    print(f"\n   Данные для SHAP:")
    print(f"   - Форма: {training_data.shape}")
    print(f"   - Тип: {type(training_data)}")
    print(f"   - NaN: {np.any(np.isnan(training_data))}")
    print(f"   - Min: {np.min(training_data)}")
    print(f"   - Max: {np.max(training_data)}")
    
    # Создаём подвыборку
    sample_data = training_data[:100] if len(training_data) > 100 else training_data
    print(f"\n   Подвыборка: {sample_data.shape}")
    
    # Пробуем создать KernelExplainer
    print("\n   Создаём KernelExplainer...")
    explainer = shap.KernelExplainer(
        model.predict_proba,
        sample_data
    )
    print("   ✅ KernelExplainer создан")
    
    # Выбираем тестовый пример
    test_instance = training_data[0].reshape(1, -1)
    print(f"\n   Тестовый пример: {test_instance.shape}")
    
    # Получаем SHAP значения
    print("\n   Вычисляем SHAP значения...")
    shap_values = explainer.shap_values(test_instance, nsamples=50)
    print(f"   ✅ SHAP значения получены")
    print(f"   Тип shap_values: {type(shap_values)}")
    
    if isinstance(shap_values, list):
        print(f"   Количество классов: {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            print(f"   - Класс {i}: {sv.shape}")
    else:
        print(f"   shap_values.shape: {shap_values.shape}")
    
    print("\n✅ SHAP РАБОТАЕТ!")

except ImportError as e:
    print(f"\n❌ SHAP не установлен: {e}")
    print("   Установи: pip install shap")

except Exception as e:
    print(f"\n❌ Ошибка SHAP: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🏁 ТЕСТ ЗАВЕРШЁН")
print("="*60)