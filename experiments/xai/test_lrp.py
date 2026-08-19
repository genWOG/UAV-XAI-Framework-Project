"""
Тест LRP на простой PyTorch нейросети
С поддержкой Zennit и Captum (исправленная версия)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LRP Test v3.0 (PyTorch нейросеть) - Исправленная версия")
print("=" * 60)

# === 1. Генерация данных ===========================================
print("\n[1] Генерация данных...")
X, y = make_classification(
    n_samples=2000,
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

# Стандартизация для нейросети
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print(f"   Обучающая: {X_train.shape[0]}, Тестовая: {X_test.shape[0]}")

# === 2. Простая нейросеть ==========================================
class SimpleNN(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

print("[2] Создание нейросети...")
model = SimpleNN(input_size=8, hidden_size=64, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 3. Обучение ===================================================
print("[3] Обучение модели...")
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)

epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"   Эпоха {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# === 4. Оценка модели ==============================================
print("[4] Оценка модели...")
X_test_t = torch.FloatTensor(X_test)
with torch.no_grad():
    predictions = model(X_test_t)
    pred_labels = (predictions > 0.5).numpy().flatten()
    accuracy = (pred_labels == y_test).mean()
    print(f"   Точность: {accuracy:.3f}")

# === 5. Выбираем тестовый пример ===================================
test_instance = X_test[0:1]
test_tensor = torch.FloatTensor(test_instance)
test_tensor.requires_grad = True

print(f"\n[5] Тестовый пример:")
print(f"   Признаки: {test_instance[0]}")

with torch.no_grad():
    pred = model(test_tensor)
    pred_class = 'Продолжить' if pred.item() > 0.5 else 'Остановка'
    print(f"   Предсказание: {pred.item():.4f} (вероятность класса 1)")
    print(f"   Класс: {pred_class}")

# === 6. Пробуем LRP через Zennit (исправленный импорт) =============
print("\n[6] Попытка LRP через Zennit...")
lrp_success = False

try:
    # Правильный импорт для Zennit 1.0.0
    from zennit.composites import EpsilonGammaBox
    from zennit.core import Composite
    from zennit.rules import Epsilon, Gamma, ZBox
    from zennit.types import Linear, Conv2d, Activation
    
    print("   ✅ Zennit найден (правильный импорт)")
    
    # Создаём композит для LRP
    composite = EpsilonGammaBox(
        low=0.0,
        high=1.0,
        epsilon=1e-6,
        gamma=0.1
    )
    
    # Применяем LRP
    from zennit.torch import modify
    
    with modify(model, composite) as modified_model:
        out = modified_model(test_tensor)
        out.backward()
        relevance = test_tensor.grad.numpy()[0]
    
    print("   ✅ LRP через Zennit выполнен успешно!")
    
    # Нормализуем релевантность
    relevance_norm = relevance / np.abs(relevance).sum()
    
    print(f"\n   LRP релевантность признаков:")
    for i, (name, rel) in enumerate(zip(feature_names, relevance_norm)):
        print(f"      {name:20s}: {rel:+.4f}")
    
    # Визуализация
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    colors = ['green' if r > 0 else 'red' for r in relevance]
    plt.barh(feature_names, relevance, color=colors, alpha=0.7)
    plt.title('LRP Релевантность признаков (Zennit)')
    plt.xlabel('Вклад в решение')
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('lrp_zennit.png', dpi=150)
    print("   ✅ График сохранён как 'lrp_zennit.png'")
    
    lrp_success = True
    
except ImportError as e:
    print(f"   ⚠️ Ошибка импорта Zennit: {e}")
    print("   Пробуем альтернативный импорт...")
    
    # Альтернативный импорт для старых версий
    try:
        import zennit
        print(f"   Zennit version: {zennit.__version__}")
        print("   Используйте: from zennit.torch import modify")
        print("   from zennit.composites import EpsilonGammaBox")
    except:
        pass
        
except Exception as e:
    print(f"   ⚠️ Ошибка Zennit: {e}")

# === 7. Если Zennit не сработал, пробуем Captum (исправленный) =====
if not lrp_success:
    print("\n[7] Попытка LRP через Captum...")
    try:
        from captum.attr import LayerGradientXActivation, LayerIntegratedGradients, InputXGradient
        
        print("   ✅ Captum найден")
        
        model.eval()
        
        # Используем InputXGradient как простой аналог LRP
        lrp_alike = InputXGradient(model)
        
        # Получаем атрибуции
        attributions = lrp_alike.attribute(test_tensor, target=0)  # target=0 для класса 0
        relevance = attributions.detach().numpy()[0]
        
        print("   ✅ Captum атрибуции получены")
        
        # Нормализуем
        relevance_norm = relevance / np.abs(relevance).sum()
        
        print(f"\n   InputXGradient атрибуции (аналог LRP):")
        for i, (name, rel) in enumerate(zip(feature_names, relevance_norm)):
            print(f"      {name:20s}: {rel:+.4f}")
        
        # Визуализация
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        colors = ['green' if r > 0 else 'red' for r in relevance]
        plt.barh(feature_names, relevance, color=colors, alpha=0.7)
        plt.title('InputXGradient (аналог LRP) через Captum')
        plt.xlabel('Вклад в решение')
        plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('lrp_captum.png', dpi=150)
        print("   ✅ График сохранён как 'lrp_captum.png'")
        
        lrp_success = True
        
    except ImportError as e:
        print(f"   ❌ Captum не установлен: {e}")
    except Exception as e:
        print(f"   ❌ Ошибка Captum: {e}")
        print("   Пробуем LayerGradientXActivation...")
        
        try:
            # Альтернативный метод
            lrp_alike = LayerGradientXActivation(model, model.fc1)
            attributions = lrp_alike.attribute(test_tensor, target=0)
            relevance = attributions.detach().numpy()[0]
            
            print("   ✅ LayerGradientXActivation сработал!")
            
            # Нормализуем
            relevance_norm = relevance / np.abs(relevance).sum()
            
            print(f"\n   LayerGradientXActivation атрибуции:")
            for i, (name, rel) in enumerate(zip(feature_names, relevance_norm)):
                print(f"      {name:20s}: {rel:+.4f}")
            
            # Визуализация
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            colors = ['green' if r > 0 else 'red' for r in relevance]
            plt.barh(feature_names, relevance, color=colors, alpha=0.7)
            plt.title('LayerGradientXActivation (аналог LRP)')
            plt.xlabel('Вклад в решение')
            plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.savefig('lrp_captum_layer.png', dpi=150)
            print("   ✅ График сохранён как 'lrp_captum_layer.png'")
            
            lrp_success = True
            
        except Exception as e2:
            print(f"   ❌ Ошибка LayerGradientXActivation: {e2}")

# === 8. Если ничего не сработало ===================================
if not lrp_success:
    print("\n[8] ⚠️ LRP не доступен. Используем упрощённый метод...")
    
    # Простой градиентный метод
    try:
        model.eval()
        test_tensor.requires_grad = True
        out = model(test_tensor)
        out.backward()
        
        relevance = test_tensor.grad.numpy()[0]
        relevance_norm = relevance / np.abs(relevance).sum()
        
        print(f"\n   Градиентный метод (упрощённый LRP):")
        for i, (name, rel) in enumerate(zip(feature_names, relevance_norm)):
            print(f"      {name:20s}: {rel:+.4f}")
        
        # Визуализация
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        colors = ['green' if r > 0 else 'red' for r in relevance]
        plt.barh(feature_names, relevance, color=colors, alpha=0.7)
        plt.title('Градиентный метод (упрощённый LRP)')
        plt.xlabel('Вклад в решение')
        plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('lrp_gradient.png', dpi=150)
        print("   ✅ График сохранён как 'lrp_gradient.png'")
        
        lrp_success = True
        
    except Exception as e:
        print(f"   ❌ Ошибка градиентного метода: {e}")

# === 9. Сравнение с LIME на той же модели =========================
print("\n[9] Сравнение с LIME на той же нейросети...")
try:
    from lime.lime_tabular import LimeTabularExplainer
    
    # Функция предсказания для LIME
    def predict_proba_lime(x):
        model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x)
            preds = model(x_t).numpy()
            # Преобразуем в [P(класс0), P(класс1)]
            return np.hstack([1 - preds, preds])
    
    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        mode='classification',
        training_labels=y_train,
        feature_names=feature_names,
        class_names=['Остановка', 'Продолжить'],
        verbose=False,
        discretize_continuous=True
    )
    
    exp = lime_explainer.explain_instance(
        data_row=test_instance[0],
        predict_fn=predict_proba_lime,
        num_features=8,
        num_samples=500,
        labels=[1]
    )
    
    print("   ✅ LIME объяснение получено")
    print("\n   Топ-5 признаков для 'Продолжить' (LIME):")
    for feature, importance in exp.as_list(label=1)[:5]:
        print(f"      {feature:20s}: {importance:+.3f}")
    
    # Визуализация LIME
    import matplotlib.pyplot as plt
    features, importance = zip(*exp.as_list(label=1)[:8])
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color='blue', alpha=0.7)
    plt.title('LIME для нейросети')
    plt.xlabel('Вклад в решение')
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('lime_nn_comparison.png', dpi=150)
    print("   ✅ График сохранён как 'lime_nn_comparison.png'")
    
except Exception as e:
    print(f"   ⚠️ LIME ошибка: {e}")

print("\n" + "=" * 60)
print("Тест LRP завершён!")
print("=" * 60)