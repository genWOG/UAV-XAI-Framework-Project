"""
Сравнительный анализ XAI методов: LIME vs SHAP vs LRP
Для UAV проекта
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import shap
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ XAI МЕТОДОВ")
print("LIME vs SHAP vs LRP (через Captum)")
print("=" * 70)

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print(f"   Обучающая: {X_train.shape[0]}, Тестовая: {X_test.shape[0]}")

# === 2. Модель 1: RandomForest =====================================
print("\n[2] Обучение RandomForest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"   Точность RF: {rf_accuracy:.3f}")

# === 3. Модель 2: PyTorch нейросеть ================================
print("[3] Обучение нейросети...")

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

nn_model = SimpleNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = nn_model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    preds = nn_model(torch.FloatTensor(X_test))
    nn_accuracy = ((preds.numpy().flatten() > 0.5) == y_test).mean()
print(f"   Точность NN: {nn_accuracy:.3f}")

# === 4. Выбираем тестовый пример ===================================
test_idx = 0
test_instance = X_test[test_idx:test_idx+1]
print(f"\n[4] Тестовый пример #{test_idx}:")
print(f"   Признаки: {test_instance[0]}")

rf_pred = rf_model.predict(test_instance)[0]
rf_proba = rf_model.predict_proba(test_instance)[0]
print(f"   RF предсказание: {'Продолжить' if rf_pred == 1 else 'Остановка'}")
print(f"   RF вероятности: P(0)={rf_proba[0]:.3f}, P(1)={rf_proba[1]:.3f}")

with torch.no_grad():
    nn_pred = nn_model(torch.FloatTensor(test_instance)).item()
print(f"   NN предсказание: {'Продолжить' if nn_pred > 0.5 else 'Остановка'}")
print(f"   NN вероятность: {nn_pred:.3f}")

# === 5. LIME для RandomForest ======================================
print("\n[5] LIME для RandomForest...")
lime_rf = LimeTabularExplainer(
    training_data=X_train,
    mode='classification',
    training_labels=y_train,
    feature_names=feature_names,
    class_names=['Остановка', 'Продолжить'],
    verbose=False,
    discretize_continuous=True
)

start_time = time.time()
exp_rf = lime_rf.explain_instance(
    data_row=test_instance[0],
    predict_fn=rf_model.predict_proba,
    num_features=8,
    num_samples=500,
    labels=[1]
)
lime_rf_time = time.time() - start_time

print(f"   Время: {lime_rf_time:.4f} сек")
print("   Топ-5 признаков для 'Продолжить':")
for feature, imp in exp_rf.as_list(label=1)[:5]:
    print(f"      {feature:20s}: {imp:+.3f}")

# === 6. SHAP для RandomForest ======================================
print("\n[6] SHAP для RandomForest...")
start_time = time.time()
explainer_shap = shap.KernelExplainer(rf_model.predict_proba, X_train[:100])
shap_values = explainer_shap.shap_values(test_instance, nsamples=100)
shap_time = time.time() - start_time

print(f"   Время: {shap_time:.4f} сек")
shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
print("   SHAP значения:")
for i, (name, val) in enumerate(zip(feature_names, shap_vals)):
    print(f"      {name:20s}: {val:+.3f}")

# === 7. LRP для нейросети ==========================================
print("\n[7] LRP (через Captum) для нейросети...")
try:
    from captum.attr import InputXGradient
    
    nn_model.eval()
    test_tensor = torch.FloatTensor(test_instance)
    test_tensor.requires_grad = True
    
    start_time = time.time()
    lrp_alike = InputXGradient(nn_model)
    attributions = lrp_alike.attribute(test_tensor, target=0)
    lrp_time = time.time() - start_time
    
    relevance = attributions.detach().numpy()[0]
    
    print(f"   Время: {lrp_time:.4f} сек")
    print("   LRP релевантность:")
    for i, (name, rel) in enumerate(zip(feature_names, relevance)):
        print(f"      {name:20s}: {rel:+.4f}")
    
    lrp_available = True
except Exception as e:
    print(f"   ❌ LRP не доступен: {e}")
    lrp_available = False

# === 8. Визуализация сравнения =====================================
print("\n[8] Создание сравнительных графиков...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# LIME
features, importance = zip(*exp_rf.as_list(label=1)[:8])
axes[0, 0].barh(features, importance, color='green', alpha=0.7)
axes[0, 0].set_title('LIME (RandomForest)\nВремя: {:.4f} сек'.format(lime_rf_time))
axes[0, 0].set_xlabel('Вклад в решение')
axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=0.5)

# SHAP
sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_shap = [shap_vals[i] for i in sorted_idx]
colors = ['green' if v > 0 else 'red' for v in sorted_shap]
axes[0, 1].barh(sorted_features, sorted_shap, color=colors, alpha=0.7)
axes[0, 1].set_title('SHAP (RandomForest)\nВремя: {:.4f} сек'.format(shap_time))
axes[0, 1].set_xlabel('SHAP значение')
axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=0.5)

# LRP
if lrp_available:
    sorted_idx = np.argsort(np.abs(relevance))[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_lrp = [relevance[i] for i in sorted_idx]
    colors = ['green' if v > 0 else 'red' for v in sorted_lrp]
    axes[1, 0].barh(sorted_features, sorted_lrp, color=colors, alpha=0.7)
    axes[1, 0].set_title('LRP (PyTorch NN)\nВремя: {:.4f} сек'.format(lrp_time))
    axes[1, 0].set_xlabel('Релевантность')
    axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=0.5)
else:
    axes[1, 0].text(0.5, 0.5, 'LRP не доступен', ha='center', va='center', fontsize=14)

# Сравнение методов (гистограмма)
methods = ['LIME', 'SHAP', 'LRP']
times = [lime_rf_time, shap_time, lrp_time if lrp_available else 0]
axes[1, 1].bar(methods, times, color=['green', 'blue', 'orange'], alpha=0.7)
axes[1, 1].set_title('Сравнение времени выполнения')
axes[1, 1].set_ylabel('Время (сек)')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('xai_comparison.png', dpi=150)
print("   ✅ График сохранён как 'xai_comparison.png'")

# === 9. Итоговая таблица ===========================================
print("\n[9] Итоговое сравнение:")
print("-" * 70)
print(f"{'Метод':<15} {'Модель':<15} {'Время (сек)':<15} {'Главный признак':<20}")
print("-" * 70)

lime_top = exp_rf.as_list(label=1)[0][0]
shap_top = feature_names[np.argmax(np.abs(shap_vals))]
lrp_top = feature_names[np.argmax(np.abs(relevance))] if lrp_available else 'N/A'

print(f"{'LIME':<15} {'RandomForest':<15} {lime_rf_time:<15.4f} {lime_top:<20}")
print(f"{'SHAP':<15} {'RandomForest':<15} {shap_time:<15.4f} {shap_top:<20}")
if lrp_available:
    print(f"{'LRP':<15} {'PyTorch NN':<15} {lrp_time:<15.4f} {lrp_top:<20}")

print("-" * 70)

print("\n" + "=" * 70)
print("✅ Сравнительный анализ завершён!")
print("=" * 70)