import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import shap
import warnings
warnings.filterwarnings('ignore')
import os
import shutil
from datetime import datetime

# ============ 0. НАСТРОЙКА ПУТЕЙ ============
PROJECT_ROOT = r"D:\Projects\UAV-XAI-Framework-Project"
RESULTS_DIR = r"D:\Projects\UAV-XAI-Framework-Project\results\1stTest"

# Создаём папку, если её нет
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============ 1. ЗАГРУЗКА ЛОГОВ ============
print("="*60)
print("XAI АНАЛИЗ ПОЛЁТНЫХ ДАННЫХ")
print("="*60)

# Ищем последнюю папку с логами в experiments/xai
logs_base = os.path.join(PROJECT_ROOT, "experiments", "xai")
log_dirs = [d for d in os.listdir(logs_base) if d.startswith("logs_flight_")]

if log_dirs:
    latest_log = sorted(log_dirs)[-1]
    log_dir = os.path.join(logs_base, latest_log)
    print(f"📂 Найдена последняя папка с логами: {latest_log}")
else:
    print("⚠️ Папка с логами не найдена! Укажи путь вручную.")
    exit()

# Загружаем логи
df = pd.read_csv(os.path.join(log_dir, "flight_log.csv"))
print(f"\n✅ Загружено {len(df)} записей")

# ============ 2. ПОДГОТОВКА ДАННЫХ ============
# Добавляем полезные признаки
df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
df['altitude'] = -df['z']
df['distance'] = np.sqrt(df['x']**2 + df['y']**2)

# Целевая переменная: столкновение (1) или безопасный полёт (0)
threshold_speed = 0.5
if df['speed'].iloc[-1] < threshold_speed and df['distance'].iloc[-1] > 10:
    df['crash'] = 1
else:
    df['crash'] = 0

# Дополнительная проверка: не долетел до цели
target_x, target_y = 50, 100
final_x, final_y = df['x'].iloc[-1], df['y'].iloc[-1]
distance_to_target = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)
if distance_to_target > 5:
    df['crash'] = 1

# === ДОБАВЛЯЕМ ИСКУССТВЕННЫЙ "БЕЗОПАСНЫЙ" КЛАСС ===
num_safe = int(len(df) * 0.2)
df_safe = df.copy()
df_safe['crash'] = 0
df_safe['x'] = df_safe['x'] * 0.5
df_safe['y'] = df_safe['y'] * 0.5

df_combined = pd.concat([df, df_safe], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n🎯 Статус полёта: {'СТОЛКНОВЕНИЕ' if df['crash'].iloc[-1] == 1 else 'БЕЗОПАСНЫЙ'}")
print(f"📊 Всего записей для анализа: {len(df_combined)}")

# ============ 3. ПРИЗНАКИ ДЛЯ МОДЕЛИ ============
feature_names = ['speed', 'altitude', 'distance', 'vx', 'vy', 'vz']
X = df_combined[feature_names].values
y = df_combined['crash'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============ 4. ОБУЧЕНИЕ МОДЕЛИ ============
print("\n🧠 Обучение модели RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
accuracy = model.score(X_scaled, y)
print(f"   Точность модели: {accuracy:.3f}")

# ============ 5. LIME ============
print("\n🔍 LIME анализ...")
lime_explainer = LimeTabularExplainer(
    X_scaled,
    feature_names=feature_names,
    class_names=['Безопасно', 'Столкновение'],
    mode='classification',
    discretize_continuous=True
)

lime_results = []
for i in range(-5, 0):
    idx = len(X_scaled) + i - len(df_safe)
    if idx >= 0 and idx < len(X_scaled):
        try:
            exp = lime_explainer.explain_instance(
                X_scaled[idx],
                model.predict_proba,
                num_features=6,
                num_samples=1000
            )
            lime_results.append(exp.as_list(label=1))
            print(f"\n   Момент {i}:")
            for feature, importance in exp.as_list(label=1)[:3]:
                print(f"      {feature:20s}: {importance:+.3f}")
        except Exception as e:
            print(f"   ⚠️ Ошибка для момента {i}: {e}")

# ============ 6. SHAP (ИСПРАВЛЕННЫЙ) ============
print("\n🔮 SHAP анализ...")
try:
    sample_size = min(30, len(X_scaled))
    background_data = X_scaled[:sample_size]
    
    explainer_shap = shap.KernelExplainer(model.predict_proba, background_data)
    explain_data = X_scaled[-10:]
    shap_values = explainer_shap.shap_values(explain_data, nsamples=50)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, explain_data, feature_names=feature_names, show=False)
    plt.tight_layout()
    shap_path = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(shap_path, dpi=150)
    plt.close()
    print(f"   ✅ SHAP график сохранён: {shap_path}")
    
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df.to_csv(os.path.join(RESULTS_DIR, "shap_values.csv"), index=False)
    print(f"   ✅ SHAP значения сохранены: {RESULTS_DIR}/shap_values.csv")
    
except Exception as e:
    print(f"   ⚠️ SHAP ошибка: {e}")
    shap_path = None
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f'SHAP не доступен\nОшибка: {str(e)}', 
             ha='center', va='center', fontsize=14)
    plt.title('SHAP анализ')
    plt.tight_layout()
    shap_path = os.path.join(RESULTS_DIR, "shap_error.png")
    plt.savefig(shap_path, dpi=150)
    plt.close()
    print(f"   ⚠️ Сохранён график с ошибкой: {shap_path}")

# ============ 7. ВИЗУАЛИЗАЦИЯ ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Траектория
axes[0, 0].plot(df['x'], df['y'], 'b-', linewidth=2, label='Траектория')
axes[0, 0].plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=12, label='Старт')
axes[0, 0].plot(df['x'].iloc[-1], df['y'].iloc[-1], 'rx', markersize=15, label='Финиш')
axes[0, 0].set_title('Траектория полёта')
axes[0, 0].set_xlabel('X (м)')
axes[0, 0].set_ylabel('Y (м)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].axis('equal')

# Скорость
axes[0, 1].plot(df['speed'], 'g-', linewidth=2)
axes[0, 1].axhline(y=threshold_speed, color='r', linestyle='--', label='Порог скорости')
axes[0, 1].set_title('Скорость')
axes[0, 1].set_xlabel('Время (кадр)')
axes[0, 1].set_ylabel('Скорость (м/с)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Высота
axes[1, 0].plot(df['altitude'], 'b-', linewidth=2)
axes[1, 0].axhline(y=1, color='r', linestyle='--', label='Целевая высота')
axes[1, 0].set_title('Высота над землёй')
axes[1, 0].set_xlabel('Время (кадр)')
axes[1, 0].set_ylabel('Высота (м)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Расстояние
axes[1, 1].plot(df['distance'], 'purple', linewidth=2)
axes[1, 1].axhline(y=np.sqrt(target_x**2 + target_y**2), color='r', linestyle='--', label='Целевое расстояние')
axes[1, 1].set_title('Расстояние от старта')
axes[1, 1].set_xlabel('Время (кадр)')
axes[1, 1].set_ylabel('Расстояние (м)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
analysis_path = os.path.join(RESULTS_DIR, "flight_analysis.png")
plt.savefig(analysis_path, dpi=150)
print(f"   ✅ Аналитический график сохранён: {analysis_path}")

# ============ 8. ВАЖНОСТЬ ПРИЗНАКОВ ============
importance = model.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(feature_names, importance, color='skyblue')
plt.title('Важность признаков (RandomForest)')
plt.xlabel('Важность')
plt.tight_layout()
importance_path = os.path.join(RESULTS_DIR, "feature_importance.png")
plt.savefig(importance_path, dpi=150)
print(f"   ✅ Важность признаков сохранена: {importance_path}")

# ============ 9. КОПИРУЕМ ЛОГИ И КАДРЫ ============
print("\n📂 Копирование логов и кадров в results...")
logs_dest = os.path.join(RESULTS_DIR, os.path.basename(log_dir))
if os.path.exists(logs_dest):
    shutil.rmtree(logs_dest)
shutil.copytree(log_dir, logs_dest)
print(f"   ✅ Логи и кадры скопированы в: {logs_dest}")

# ============ 10. ИТОГОВЫЙ ОТЧЁТ ============
print("\n" + "="*60)
print("📊 ИТОГОВЫЙ ОТЧЁТ ПО XAI АНАЛИЗУ")
print("="*60)
print(f"📂 Результаты сохранены в: {RESULTS_DIR}")
print(f"📈 Анализ полёта: {analysis_path}")
if shap_path:
    print(f"🔮 SHAP график: {shap_path}")
print(f"📊 Важность признаков: {importance_path}")
print(f"📂 Логи и кадры: {logs_dest}")
print("\n🏁 Статус полёта:", "СТОЛКНОВЕНИЕ" if df['crash'].iloc[-1] == 1 else "БЕЗОПАСНЫЙ")
print(f"📊 Точность модели: {accuracy:.3f}")
print("="*60)