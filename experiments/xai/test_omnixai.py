"""
Первый тест OmniXAI на синтетических данных.
Проверяем LIME через OmniXAI.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

# === 1. Генерация синтетических данных ==============================
print("=" * 60)
print("UAV-XAI / OmniXAI Test v1.1 (LIME только)")
print("=" * 60)

# Генерируем 1000 образцов, 8 признаков, 2 класса
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

print(f"\n[1] Сгенерировано данных: {len(X_df)} образцов, {len(feature_names)} признаков")

# === 2. Разбиваем на train/test =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42
)

print(f"[2] Обучающая выборка: {X_train.shape[0]}")
print(f"    Тестовая выборка: {X_test.shape[0]}")

# === 3. Обучаем модель =============================================
print("[3] Обучение модели RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"    Точность модели: {accuracy:.3f}")

# === 4. Импорт OmniXAI =============================================
print("[4] Импорт OmniXAI...")
from omnixai.explainers.tabular import TabularExplainer
from omnixai.data.tabular import Tabular

# === 5. Создаём Tabular-объект для данных ==========================
print("[5] Создание Tabular-объекта...")
train_data = Tabular(
    data=X_train,
    feature_columns=feature_names
)

# === 6. Определяем функцию предсказания ============================
def predict_function(z):
    """Функция предсказания для OmniXAI"""
    # Преобразуем в numpy, если это необходимо
    if isinstance(z, pd.DataFrame):
        z = z.values
    return model.predict_proba(z)

# === 7. Создаём Explainer (ТОЛЬКО LIME) ============================
print("[6] Создание TabularExplainer (только LIME)...")
try:
    explainer = TabularExplainer(
        explainers=["lime"],  # Только LIME
        mode="classification",
        data=train_data,
        model=model,
        # params={"lime": {"kernel_width": 0.75, "feature_selection": "auto"}}
    )
    print("    ✅ TabularExplainer создан успешно")
except Exception as e:
    print(f"    ❌ Ошибка: {e}")
    exit(1)

# === 8. Выбираем тестовый пример для объяснения ====================
test_instance = X_test.iloc[:1]
print(f"\n[7] Объясняем решение для образца: {test_instance.values[0]}")

# Создаём Tabular-объект для тестового примера
test_data = Tabular(
    data=test_instance,
    feature_columns=feature_names
)

# === 9. Получаем объяснения ========================================
print("[8] Получение объяснений...")
start_time = time.time()

try:
    explanations = explainer.explain(X=test_data)
    elapsed_time = time.time() - start_time
    print(f"    ✅ Время вычисления: {elapsed_time:.3f} сек")
except Exception as e:
    print(f"    ❌ Ошибка при получении объяснений: {e}")
    exit(1)

# === 10. Анализ результатов ========================================
print("\n[9] Результаты:")
print("-" * 60)

if explanations is not None:
    print(f"    ✅ Объяснения получены!")
    print(f"    Тип объекта: {type(explanations)}")
    
    # Проверяем атрибуты
    attr_list = dir(explanations)
    print(f"    Доступные атрибуты: {', '.join([a for a in attr_list if not a.startswith('_')])}")
    
    # Проверяем наличие LIME
    if hasattr(explanations, 'lime'):
        lime_exp = explanations.lime
        print(f"\n    🔍 LIME объяснения:")
        print(f"       Тип: {type(lime_exp)}")
        
        # Проверяем содержимое
        if hasattr(lime_exp, 'local_explanations'):
            print(f"       Количество локальных объяснений: {len(lime_exp.local_explanations)}")
            # Берём первое объяснение
            if len(lime_exp.local_explanations) > 0:
                exp = lime_exp.local_explanations[0]
                print(f"       Первое объяснение (экземпляр):")
                if hasattr(exp, 'feature_names'):
                    print(f"          - Признаки: {exp.feature_names}")
                if hasattr(exp, 'scores'):
                    print(f"          - Важности: {exp.scores}")
                if hasattr(exp, 'predicted_label'):
                    print(f"          - Предсказанный класс: {exp.predicted_label}")
                if hasattr(exp, 'predicted_proba'):
                    print(f"          - Вероятность: {exp.predicted_proba}")
        
        # Альтернативный способ извлечения
        elif hasattr(lime_exp, 'explanations'):
            print(f"       Объяснения: {type(lime_exp.explanations)}")
            if hasattr(lime_exp.explanations, 'feature_names'):
                print(f"          - Признаки: {lime_exp.explanations.feature_names}")
            if hasattr(lime_exp.explanations, 'scores'):
                print(f"          - Важности: {lime_exp.explanations.scores}")
        
        # Если структура совсем другая, выводим содержимое
        else:
            print("       Структура LIME:")
            for attr in dir(lime_exp):
                if not attr.startswith('_'):
                    print(f"          - {attr}: {type(getattr(lime_exp, attr))}")
    else:
        print("\n    ⚠️ Нет LIME объяснений")
        print(f"    Доступные ключи: {list(explanations.__dict__.keys()) if hasattr(explanations, '__dict__') else 'нет'}")
else:
    print("    ❌ Объяснения не получены")

print("\n" + "=" * 60)
print("Тест завершён!")
print("=" * 60)

# === 11. Дополнительная проверка: прямой вызов LIME ===============
print("\n[Дополнительно] Пробуем прямой вызов LIME...")
try:
    from lime.lime_tabular import LimeTabularExplainer
    
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        mode='classification',
        training_labels=y_train.values,
        feature_names=feature_names,
        class_names=['class_0', 'class_1'],
        verbose=False
    )
    
    # Объясняем тот же пример
    exp = lime_explainer.explain_instance(
        data_row=test_instance.values[0],
        predict_fn=model.predict_proba,
        num_features=8,
        num_samples=1000
    )
    
    print("    ✅ Прямой LIME сработал!")
    print("    Топ-3 важных признака:")
    for feature, importance in exp.as_list()[:3]:
        print(f"       - {feature}: {importance:.3f}")
    
except Exception as e:
    print(f"    ❌ Прямой LIME упал: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Полный тест завершён!")
print("=" * 60)