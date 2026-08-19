"""
XAI для БПЛА - демонстрация на реальных данных
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("UAV XAI Демонстрация на реальных данных")
print("=" * 60)

# === 1. Генерируем реалистичные данные полёта ======================
print("\n[1] Генерация данных полёта БПЛА...")

np.random.seed(42)
n_samples = 2000

# Реалистичные диапазоны параметров
data = {
    'Высота': np.random.uniform(10, 120, n_samples),           # метры
    'Скорость': np.random.uniform(0, 25, n_samples),           # м/с
    'Расст_до_преп': np.random.uniform(5, 80, n_samples),      # метры
    'Заряд_бат': np.random.uniform(20, 100, n_samples),        # %
    'Видимость': np.random.uniform(0.3, 1.0, n_samples),       # 0-1
    'Боковой_ветер': np.random.uniform(0, 15, n_samples),      # м/с
    'Плотн_преп': np.random.uniform(0, 1, n_samples),          # 0-1
    'Угол_наклона': np.random.uniform(-30, 30, n_samples)      # градусы
}

df = pd.DataFrame(data)

# === 2. Создаём целевую переменную (решение: продолжать или нет) ===
print("[2] Создание целевой переменной...")

# Логика принятия решения (симулируем экспертную систему)
df['Решение'] = 1  # По умолчанию "Продолжить"

# Условия для остановки:
stop_conditions = (
    (df['Заряд_бат'] < 30) |           # Низкий заряд
    (df['Расст_до_преп'] < 15) |       # Близкое препятствие
    (df['Боковой_ветер'] > 10) |       # Сильный ветер
    (df['Плотн_преп'] > 0.7) |         # Высокая плотность препятствий
    (df['Угол_наклона'] > 20)          # Слишком большой угол
)

df.loc[stop_conditions, 'Решение'] = 0  # "Остановка"

print(f"   Продолжить: {(df['Решение'] == 1).sum()}")
print(f"   Остановка: {(df['Решение'] == 0).sum()}")

# === 3. Подготовка данных =========================================
print("[3] Подготовка данных...")

feature_names = [
    'Высота', 'Скорость', 'Расст_до_преп', 'Заряд_бат',
    'Видимость', 'Боковой_ветер', 'Плотн_преп', 'Угол_наклона'
]

X = df[feature_names].values
y = df['Решение'].values

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print(f"   Обучающая: {X_train.shape[0]}, Тестовая: {X_test.shape[0]}")

# === 4. Обучение модели ============================================
print("[4] Обучение модели...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"   Точность: {accuracy:.3f}")

# === 5. LIME объяснение ============================================
print("[5] LIME объяснение...")

lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    mode='classification',
    training_labels=y_train,
    feature_names=feature_names,
    class_names=['Остановка', 'Продолжить'],
    verbose=False,
    discretize_continuous=True
)

# Выбираем случайный пример из тестовой выборки
test_idx = np.random.randint(0, len(X_test))
test_instance = X_test[test_idx]
true_label = y_test[test_idx]
pred_label = model.predict([test_instance])[0]
pred_proba = model.predict_proba([test_instance])[0]

print(f"\n   Тестовый пример #{test_idx}:")
print(f"   Истинное решение: {'Продолжить' if true_label == 1 else 'Остановка'}")
print(f"   Предсказание: {'Продолжить' if pred_label == 1 else 'Остановка'}")
print(f"   Вероятности: P(Остановка)={pred_proba[0]:.3f}, P(Продолжить)={pred_proba[1]:.3f}")

# Получаем LIME объяснение
start_time = time.time()
exp = lime_explainer.explain_instance(
    data_row=test_instance,
    predict_fn=model.predict_proba,
    num_features=8,
    num_samples=1000,
    labels=[0, 1]
)
lime_time = time.time() - start_time

print(f"\n   LIME время: {lime_time:.4f} сек")

# Выводим топ-признаки
print("\n   Топ-5 признаков для решения 'Продолжить':")
for feature, importance in exp.as_list(label=1)[:5]:
    print(f"      {feature:20s}: {importance:+.3f}")

# === 6. Визуализация ===============================================
print("\n[6] Сохранение графиков...")

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LIME для класса "Продолжить"
    features_1, importance_1 = zip(*exp.as_list(label=1)[:8])
    axes[0].barh(features_1, importance_1, color='green', alpha=0.7)
    axes[0].set_title('LIME: Важность признаков для "Продолжить"')
    axes[0].set_xlabel('Вклад в решение')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # LIME для класса "Остановка"
    features_0, importance_0 = zip(*exp.as_list(label=0)[:8])
    axes[1].barh(features_0, importance_0, color='red', alpha=0.7)
    axes[1].set_title('LIME: Важность признаков для "Остановка"')
    axes[1].set_xlabel('Вклад в решение')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('uav_lime_explanation.png', dpi=150)
    print("   ✅ График сохранён как 'uav_lime_explanation.png'")
except Exception as e:
    print(f"   ⚠️ Ошибка: {e}")

# === 7. Статистика по всем тестовым примерам =======================
print("\n[7] Анализ важности признаков (глобальный)...")


# Собираем важности для всех примеров
all_importances = {feat: [] for feat in feature_names}

for i in range(min(50, len(X_test))):
    exp_i = lime_explainer.explain_instance(
        data_row=X_test[i],
        predict_fn=model.predict_proba,
        num_features=8,
        num_samples=500,
        labels=[1]
    )
    for feature, importance in exp_i.as_list(label=1):
        if feature in all_importances:
            all_importances[feature].append(importance)

# Средние важности
avg_importances = {feat: np.mean(vals) if vals else 0 
                   for feat, vals in all_importances.items()}

print("\n   Средняя важность признаков для решения 'Продолжить':")
for feature, importance in sorted(avg_importances.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"      {feature:20s}: {importance:+.3f}")

print("\n" + "=" * 60)
print("✅ UAV XAI Демонстрация завершена!")
print("=" * 60)