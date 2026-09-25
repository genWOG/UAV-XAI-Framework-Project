"""
Тест прямого LIME (без OmniXAI)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Прямой LIME Test v1.0")
print("=" * 60)

# === 1. Генерация данных ===========================================
print("\n[1] Генерация синтетических данных...")
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

feature_names = [
    'Высота', 'Скорость', 'Расст_до_преп', 'Заряд_бат',
    'Видимость', 'Боковой_ветер', 'Плотн_преп', 'Угол_наклона'
]

X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='Целевая_переменная')

print(f"   Образцов: {len(X_df)}, Признаков: {len(feature_names)}")

# === 2. Разбиваем на train/test ====================================
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42
)

print(f"[2] Обучающая: {X_train.shape[0]}, Тестовая: {X_test.shape[0]}")

# === 3. Обучаем модель ============================================
print("[3] Обучение RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"   Точность: {accuracy:.3f}")

# === 4. Создаём LIME Explainer =====================================
print("[4] Создание LIME объяснителя...")
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    mode='classification',
    training_labels=y_train.values,
    feature_names=feature_names,
    class_names=['Остановка', 'Продолжить'],
    verbose=False,
    discretize_continuous=True
)

print("   ✅ LIME объяснитель создан")

# === 5. Выбираем тестовый пример ===================================
test_instance = X_test.iloc[0]
print(f"\n[5] Тестовый пример (первый из тестовой выборки):")
print(f"   Признаки: {test_instance.values}")

# Предсказание модели для этого примера
pred_proba = model.predict_proba([test_instance.values])[0]
pred_class = model.predict([test_instance.values])[0]
print(f"   Предсказание: {['Остановка', 'Продолжить'][pred_class]}")
print(f"   Вероятности: P(Остановка)={pred_proba[0]:.3f}, P(Продолжить)={pred_proba[1]:.3f}")

# === 6. Получаем LIME объяснение ===================================
print("\n[6] Получение LIME объяснения...")
start_time = time.time()

exp = lime_explainer.explain_instance(
    data_row=test_instance.values,
    predict_fn=model.predict_proba,
    num_features=8,
    num_samples=1000,
    labels=[0, 1]
)

elapsed_time = time.time() - start_time
print(f"   ✅ Время вычисления: {elapsed_time:.4f} сек")

# === 7. Анализ результатов =========================================
print("\n[7] Результаты LIME:")
print("-" * 60)

# Для класса 1 (Продолжить)
print("\n🔍 Топ-5 признаков для решения 'Продолжить':")
for feature, importance in exp.as_list(label=1)[:5]:
    print(f"   {feature:20s}: {importance:+.3f}")

print("\n🔍 Топ-5 признаков для решения 'Остановка':")
for feature, importance in exp.as_list(label=0)[:5]:
    print(f"   {feature:20s}: {importance:+.3f}")

# === 8. Визуализация LIME ==========================================
print("\n[8] Сохраняем визуализацию LIME...")
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    features_1, importance_1 = zip(*exp.as_list(label=1)[:8])
    ax1.barh(features_1, importance_1, color='green', alpha=0.7)
    ax1.set_title('Важность признаков для класса "Продолжить"')
    ax1.set_xlabel('Вклад в решение')
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    features_0, importance_0 = zip(*exp.as_list(label=0)[:8])
    ax2.barh(features_0, importance_0, color='red', alpha=0.7)
    ax2.set_title('Важность признаков для класса "Остановка"')
    ax2.set_xlabel('Вклад в решение')
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('lime_importance.png', dpi=150)
    print("   ✅ LIME график сохранён как 'lime_importance.png'")
except Exception as e:
    print(f"   ⚠️ Ошибка: {e}")

# === 9. SHAP визуализация =========================================
print("\n[9] SHAP визуализация...")
try:
    import shap
    
    print("   Создание KernelExplainer...")
    explainer_shap = shap.KernelExplainer(
        model.predict_proba, 
        X_train.values[:100]
    )
    
    print("   Вычисление SHAP значений...")
    shap_values = explainer_shap.shap_values(
        test_instance.values.reshape(1, -1),
        nsamples=100
    )
    
    print("   ✅ SHAP значения получены")
    
    # Получаем значения для класса 1 (Продолжить)
    if len(shap_values) > 1:
        shap_vals_class1 = shap_values[1][0]
        shap_vals_class0 = shap_values[0][0]
        expected_value = explainer_shap.expected_value[1]
    else:
        shap_vals_class1 = shap_values[0][0]
        shap_vals_class0 = shap_values[0][0]
        expected_value = explainer_shap.expected_value
    
    print(f"\n   SHAP значения для класса 'Продолжить':")
    for i, (feature, value) in enumerate(zip(feature_names, shap_vals_class1)):
        print(f"      {feature:20s}: {value:+.4f}")
    
    # Визуализация SHAP - простой барплот
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(np.abs(shap_vals_class1))[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = [shap_vals_class1[i] for i in sorted_idx]
    
    colors = ['green' if v > 0 else 'red' for v in sorted_values]
    plt.barh(sorted_features, sorted_values, color=colors, alpha=0.7)
    plt.title('SHAP значения для решения "Продолжить"')
    plt.xlabel('Вклад в решение (SHAP value)')
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('shap_importance.png', dpi=150)
    print("   ✅ SHAP график сохранён как 'shap_importance.png'")
    
    # Сравнение классов
    plt.figure(figsize=(10, 6))
    x = np.arange(len(feature_names))
    width = 0.35
        # Сравнение классов (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    try:
        plt.figure(figsize=(10, 6))
        
        # Сортируем по важности для класса 1
        sorted_idx = np.argsort(np.abs(shap_vals_class1))[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        vals_class1 = [shap_vals_class1[i] for i in sorted_idx]
        vals_class0 = [shap_vals_class0[i] for i in sorted_idx]
        
        # Создаём позиции для баров
        y_pos = np.arange(len(sorted_features))
        height = 0.35  # Высота баров (не ширина!)
        
        # Рисуем горизонтальные бары
        plt.barh(y_pos - height/2, vals_class1, height, 
                 label='Класс "Продолжить"', color='green', alpha=0.7)
        plt.barh(y_pos + height/2, vals_class0, height, 
                 label='Класс "Остановка"', color='red', alpha=0.7)
        
        # Настройка осей
        plt.yticks(y_pos, sorted_features)
        plt.xlabel('SHAP значение')
        plt.title('Сравнение SHAP значений для двух классов')
        plt.legend()
        plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('shap_comparison.png', dpi=150)
        print("   ✅ SHAP сравнение сохранено как 'shap_comparison.png'")
    except Exception as e:
        print(f"   ⚠️ Ошибка сравнения: {e}")
    
    print("\n   ✅ SHAP успешно выполнен!")
    
except Exception as e:
    print(f"   ❌ SHAP ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Тест LIME + SHAP завершён успешно! 🚀")
print("=" * 60)