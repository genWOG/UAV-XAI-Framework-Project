import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def get_latest_files():
    # Ищем самый свежий session.json из папки data/incidents
    session_files = glob.glob('data/incidents/*/session.json')
    if not session_files:
        raise FileNotFoundError("Не найден session.json (полетные логи)")
    latest_session = max(session_files, key=os.path.getmtime)

    # Ищем самый свежий xai_session_*.json
    xai_files = glob.glob('data/logs/xai_session_*.json')
    if not xai_files:
        raise FileNotFoundError("Не найден xai_session.json (логи XAI)")
    latest_xai = max(xai_files, key=os.path.getmtime)
    
    return latest_session, latest_xai

def plot_trajectory(session_data, output_dir):
    telemetry = session_data.get('telemetry', [])
    events = session_data.get('events', [])
    
    if not telemetry: return

    xs = [t['position']['x'] for t in telemetry]
    ys = [t['position']['y'] for t in telemetry]
    
    plt.figure(figsize=(10, 8))
    plt.plot(xs, ys, '-', color='gray', alpha=0.6, label='Траектория полета')
    
    # Отмечаем события
    avoid_x, avoid_y = [], []
    target_x, target_y = [], []
    
    for ev in events:
        pos = ev.get('position', [0, 0, 0])
        if ev['kind'] == 'obstacle_avoid':
            avoid_x.append(pos[0])
            avoid_y.append(pos[1])
        elif ev['kind'] == 'target_reached':
            target_x.append(pos[0])
            target_y.append(pos[1])

    if avoid_x:
        plt.scatter(avoid_x, avoid_y, c='red', marker='x', s=100, label='Обход препятствия', zorder=5)
    if target_x:
        plt.scatter(target_x, target_y, c='green', marker='*', s=150, label='Достигнута цель', zorder=5)

    plt.title('Траектория полета дрона (X-Y)')
    plt.xlabel('Координата X (Север, м)')
    plt.ylabel('Координата Y (Восток, м)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal') # Чтобы круги не были овалами
    
    out_path = output_dir / 'trajectory.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График траектории сохранен: {out_path}")

def plot_feature_importance(xai_data, output_dir):
    if not xai_data: return
    
    feature_names = xai_data[0].get('feature_names', [])
    if not feature_names: return

    # Группируем атрибуции по действиям (0=Стоп, 1=Продолжить)
    attr_by_action = {0: [], 1: []}
    
    for exp in xai_data:
        action = exp.get('action')
        attr = exp.get('attribution')
        if action in attr_by_action and attr:
            # Берем модуль атрибуции (важность признака)
            attr_by_action[action].append(np.abs(attr))

    # Усредняем
    mean_attr = {}
    for action, attrs in attr_by_action.items():
        if attrs:
            mean_attr[action] = np.mean(attrs, axis=0)

    if not mean_attr: return

    # Строим график
    x = np.arange(len(feature_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 0 in mean_attr:
        ax.bar(x - width/2, mean_attr[0], width, label='Решение: Остановка', color='crimson', alpha=0.8)
    if 1 in mean_attr:
        ax.bar(x + width/2, mean_attr[1], width, label='Решение: Продолжить', color='forestgreen', alpha=0.8)

    ax.set_ylabel('Средняя важность (|Attribution|)')
    ax.set_title('Важность признаков для принятия решений (LIME/SHAP/LRP)')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    
    out_path = output_dir / 'feature_importance.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График важности признаков сохранен: {out_path}")

def plot_metrics(xai_data, output_dir):
    if not xai_data: return

    metrics_by_method = defaultdict(lambda: {'del': [], 'ins': []})
    
    for exp in xai_data:
        method = exp.get('method')
        metrics = exp.get('metrics')
        if method and metrics:
            del_auc = metrics.get('deletion_auc')
            ins_auc = metrics.get('insertion_auc')
            if del_auc is not None and ins_auc is not None:
                metrics_by_method[method]['del'].append(del_auc)
                metrics_by_method[method]['ins'].append(ins_auc)

    if not metrics_by_method: return

    methods = list(metrics_by_method.keys())
    mean_del = [np.mean(metrics_by_method[m]['del']) for m in methods]
    mean_ins = [np.mean(metrics_by_method[m]['ins']) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, mean_del, width, label='Deletion AUC (меньше = лучше)', color='royalblue')
    ax.bar(x + width/2, mean_ins, width, label='Insertion AUC (больше = лучше)', color='orange')

    ax.set_ylabel('AUC (площадь под кривой)')
    ax.set_title('Сравнение метрик достоверности XAI-методов')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    plt.tight_layout()

    out_path = output_dir / 'metrics_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График метрик сохранен: {out_path}")

def main():
    print("📊 Анализ логов полета и XAI...")
    
    try:
        session_path, xai_path = get_latest_files()
        print(f"📂 Файл сессии: {session_path}")
        print(f"📂 Файл XAI: {xai_path}")
        
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
            
        with open(xai_path, 'r', encoding='utf-8') as f:
            xai_data = json.load(f)
            
        output_dir = Path('results/analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_trajectory(session_data, output_dir)
        plot_feature_importance(xai_data, output_dir)
        plot_metrics(xai_data, output_dir)
        
        print("\n🎉 Все графики успешно построены! Ищи их в папке results/analysis/")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()