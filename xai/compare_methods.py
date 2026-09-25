"""
Сравнение XAI методов по типам событий
Версия 2.0 — графики на русском языке
Запуск: python xai/compare_methods.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Настройка русского шрифта
plt.rcParams['font.family'] = 'DejaVu Sans'
# Если не работает, раскомментируй:
# plt.rcParams['font.family'] = 'Arial Unicode MS'

def load_xai_data(filepath):
    """Загружает XAI объяснения из JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_flight_data(filepath):
    """Загружает лог полёта из JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_explanation(entry, flight_log=None):
    """
    Классифицирует объяснение по типу события.
    
    Возвращает:
    - 'движение': обычный полёт
    - 'коррекция': коррекция маршрута
    - 'остановка': прекращение полёта
    - 'препятствие': обнаружение препятствия
    """
    method = entry.get('method', 'unknown')
    action = entry.get('action', 1)
    is_anomaly = entry.get('is_anomaly', {}).get('is_anomaly', False)
    
    # Проверяем, есть ли в логе информация о препятствии
    has_obstacle = False
    if flight_log:
        timestamp = entry.get('timestamp', '')
        for log_entry in flight_log.get('logs', []):
            if log_entry.get('timestamp', '') == timestamp:
                if 'ПРЕПЯТСТВИЕ' in log_entry.get('message', '') or 'препятствие' in log_entry.get('message', ''):
                    has_obstacle = True
                break
    
    # Классификация
    if action == 0 or is_anomaly:
        return 'остановка'
    elif has_obstacle:
        return 'препятствие'
    elif 'коррекция' in str(entry.get('explanation_text', '')).lower():
        return 'коррекция'
    else:
        return 'движение'

def translate_feature_names(features):
    """Переводит названия признаков на русский."""
    translation = {
        'altitude': 'Высота',
        'speed': 'Скорость',
        'obstacle_dist': 'Расстояние до препятствия',
        'battery': 'Заряд батареи',
        'obstacle_density': 'Плотность препятствий',
        'yaw': 'Рыскание (Yaw)',
        'roll': 'Крен (Roll)',
        'pitch': 'Тангаж (Pitch)',
        'distance_to_target': 'Расстояние до цели',
        'angle_to_target': 'Угол к цели'
    }
    return [translation.get(f, f) for f in features]

def plot_by_event(xai_data, flight_log, save_dir="results/xai_plots"):
    """Строит графики по типам событий."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Классифицируем объяснения
    classified = defaultdict(list)
    for entry in xai_data:
        event_type = classify_explanation(entry, flight_log)
        classified[event_type].append(entry)
    
    # Названия событий на русском
    event_names = {
        'движение': 'Движение (обычный полёт)',
        'коррекция': 'Коррекция маршрута',
        'остановка': 'Прекращение полёта (остановка)',
        'препятствие': 'Обнаружение препятствия'
    }
    
    # Цвета для методов
    method_colors = {
        'lime': '#2E86AB',
        'shap': '#A23B72',
        'lrp': '#F18F01',
        'fallback': '#888888'
    }
    
    # Собираем все признаки
    all_features = xai_data[0].get('feature_names', []) if xai_data else []
    ru_features = translate_feature_names(all_features)
    
    # Для каждого типа события строим график
    for event_type, entries in classified.items():
        if not entries:
            print(f"⚠️ Нет данных для события: {event_type}")
            continue
        
        # Группируем по методам
        methods_data = defaultdict(list)
        for entry in entries:
            method = entry.get('method', 'fallback')
            attribution = np.array(entry.get('attribution', []))
            if len(attribution) == len(all_features):
                methods_data[method].append(attribution)
        
        if not methods_data:
            continue
        
        # Создаём график
        fig, axes = plt.subplots(1, len(methods_data), figsize=(6 * len(methods_data), 7))
        if len(methods_data) == 1:
            axes = [axes]
        
        fig.suptitle(f'Важность признаков при событии: {event_names[event_type]}', 
                     fontsize=14, fontweight='bold')
        
        for idx, (method, attributions) in enumerate(methods_data.items()):
            ax = axes[idx]
            
            # Усредняем
            avg = np.mean(attributions, axis=0)
            std = np.std(attributions, axis=0)
            
            # Строим бары
            y_pos = np.arange(len(ru_features))
            bars = ax.barh(y_pos, avg, color=method_colors.get(method, '#888'), alpha=0.7)
            
            # Цвета баров: зелёный (+), красный (-)
            for i, (bar, val) in enumerate(zip(bars, avg)):
                if val > 0:
                    bar.set_color('#2ECC71')  # зелёный
                else:
                    bar.set_color('#E74C3C')   # красный
            
            # Оси
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ru_features, fontsize=10)
            ax.set_xlabel('Вклад в решение', fontsize=11)
            ax.set_title(f'{method.upper()}\n({len(attributions)} объяснений)', fontsize=12)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Значения на барах
            for i, v in enumerate(avg):
                ax.text(v + (0.01 if v > 0 else -0.01), i, f'{v:.3f}',
                        va='center', ha='left' if v > 0 else 'right', fontsize=8)
            
            # Error bars
            ax.errorbar(avg, y_pos, xerr=std, fmt='none', color='black', 
                       capsize=3, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'comparison_{event_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ График сохранён: {save_path}")

def plot_summary_stats(xai_data, save_dir="results/xai_plots"):
    """Строит сводную статистику по методам."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Считаем количество вызовов
    methods = {}
    for entry in xai_data:
        method = entry.get('method', 'fallback')
        methods[method] = methods.get(method, 0) + 1
    
    # Время выполнения
    times = {}
    for entry in xai_data:
        method = entry.get('method', 'fallback')
        runtime = entry.get('runtime_ms', 0)
        times.setdefault(method, []).append(runtime)
    
    # График 1: Количество вызовов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Круговая диаграмма
    if methods:
        labels = [m.upper() for m in methods.keys()]
        values = list(methods.values())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#888888'][:len(methods)]
        ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Распределение вызовов методов', fontsize=12, fontweight='bold')
    
    # Время выполнения
    if times:
        box_data = []
        box_labels = []
        for method, time_list in times.items():
            if time_list:
                box_data.append(time_list)
                box_labels.append(method.upper())
        if box_data:
            bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#2E86AB', '#A23B72', '#F18F01', '#888888']):
                patch.set_facecolor(color)
            ax2.set_ylabel('Время выполнения (мс)', fontsize=11)
            ax2.set_title('Время выполнения методов', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'summary_stats.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Сводная статистика: {save_path}")

def plot_confusion_matrix(xai_data, flight_log, save_dir="results/xai_plots"):
    """Строит матрицу: метод → тип события."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Классифицируем
    events = ['движение', 'коррекция', 'остановка', 'препятствие']
    methods = ['lime', 'shap', 'lrp', 'fallback']
    
    matrix = np.zeros((len(methods), len(events)))
    
    for entry in xai_data:
        method = entry.get('method', 'fallback')
        event = classify_explanation(entry, flight_log)
        if method in methods and event in events:
            i = methods.index(method)
            j = events.index(event)
            matrix[i, j] += 1
    
    # Нормализуем по строкам
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums
    
    # Строим тепловую карту
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(matrix_norm, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(events)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(events)
    ax.set_yticklabels([m.upper() for m in methods])
    
    # Поворачиваем подписи
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Добавляем значения
    for i in range(len(methods)):
        for j in range(len(events)):
            if matrix[i, j] > 0:
                ax.text(j, i, f'{int(matrix[i, j])}\n({matrix_norm[i, j]:.0%})',
                       ha='center', va='center', fontsize=9)
    
    ax.set_title('Какие методы используются для каких событий', fontsize=13, fontweight='bold')
    ax.set_xlabel('Тип события', fontsize=11)
    ax.set_ylabel('Метод XAI', fontsize=11)
    
    plt.colorbar(im, ax=ax, label='Доля вызовов метода')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'method_event_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Матрица методов: {save_path}")

def main():
    print("="*60)
    print("📊 АНАЛИЗ XAI ПО ТИПАМ СОБЫТИЙ")
    print("="*60)
    
    # Загружаем последний XAI файл
    log_dir = "data/logs"
    xai_files = [f for f in os.listdir(log_dir) if f.startswith('xai_') and f.endswith('.json') and 'local' not in f]
    flight_files = [f for f in os.listdir(log_dir) if f.startswith('flight_') and f.endswith('.json')]
    
    if not xai_files:
        print("❌ Нет XAI файлов")
        return
    
    if not flight_files:
        print("⚠️ Нет логов полёта")
        flight_log = None
    else:
        latest_flight = sorted(flight_files)[-1]
        flight_log = load_flight_data(os.path.join(log_dir, latest_flight))
        print(f"📂 Лог полёта: {latest_flight}")
    
    latest_file = sorted(xai_files)[-1]
    filepath = os.path.join(log_dir, latest_file)
    print(f"📂 XAI файл: {latest_file}")
    
    xai_data = load_xai_data(filepath)
    if not xai_data:
        print("❌ Нет данных")
        return
    
    print(f"📊 Всего объяснений: {len(xai_data)}")
    
    # Классифицируем
    classified = defaultdict(int)
    for entry in xai_data:
        event = classify_explanation(entry, flight_log)
        classified[event] += 1
    
    print("\n📊 Распределение по событиям:")
    for event, count in classified.items():
        print(f"  {event}: {count} ({count/len(xai_data)*100:.1f}%)")
    
    # Строим графики
    print("\n🎨 Строим графики...")
    plot_by_event(xai_data, flight_log)
    plot_summary_stats(xai_data)
    plot_confusion_matrix(xai_data, flight_log)
    
    print("\n✅ Готово! Графики в папке results/xai_plots/")

if __name__ == "__main__":
    main()