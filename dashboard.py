import streamlit as st
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

st.set_page_config(page_title="UAV-XAI Dashboard", layout="wide")
st.title("🚁 UAV-XAI: Анализ инцидентов")

FEATURE_NAMES_RU = {
    'altitude': 'Высота (м)', 'speed': 'Скорость (м/с)', 'obstacle_dist': 'Расст. до препятствия (м)',
    'battery': 'Батарея (%)', 'obstacle_density': 'Плотность препятствий', 'yaw': 'Курс (Yaw)',
    'roll': 'Крен (Roll)', 'pitch': 'Тангаж (Pitch)', 'distance_to_target': 'Дист. до цели (м)',
    'angle_to_target': 'Угол на цель (град)'
}

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("📁 История отчетов")
report_files = sorted(glob.glob('data/reports/*_report_*.json'), key=os.path.getmtime, reverse=True)

if not report_files:
    st.sidebar.warning("Отчеты не найдены.")
    st.stop()

selected_report = st.sidebar.selectbox("Выберите отчет:", report_files, format_func=lambda x: os.path.basename(x))

@st.cache_data
def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_report(selected_report)
flight_type = data.get('flight_type', 'unknown')

st.sidebar.success(f"Тип: **{flight_type.upper()}**")
st.sidebar.info(f"Время: {data['timestamp'][:19]}")

# Загружаем сырые логи для графиков
try:
    with open(data['session_file'], 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    with open(data['xai_file'], 'r', encoding='utf-8') as f:
        xai_data = json.load(f)
except Exception as e:
    st.error(f"Не удалось загрузить сырые логи: {e}")
    session_data, xai_data = None, None

# Создаем вкладки
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Сводка + CF", "🗺️ Траектория", "📊 Важность признаков", 
    "📈 Метрики XAI", "⚖️ Сравнение"
])

# === ВКЛАДКА 1: Сводка и Counterfactual ===
with tab1:
    st.header(f"Анализ полета: {flight_type.upper()}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Целей достигнуто", data['summary'].get('targets_reached', 0))
    col2.metric("Обходов препятствий", data['summary'].get('obstacle_events', 0))
    col3.metric("Застреваний", data['summary'].get('stuck_events', 0))
    
    st.subheader("🧠 Counterfactual (Что если?)")
    cf = data.get('counterfactual', {})
    if cf.get('conclusion'):
        st.warning(cf['conclusion'])
    if cf.get('changes'):
        for ch in cf['changes']:
            st.write(f"- {ch['ru_name']}: с {ch['old']} на {ch['new']}")

# === ВКЛАДКА 2: Траектория ===
with tab2:
    st.header("Траектория полета")
    if session_data:
        telemetry = session_data.get('telemetry', [])
        events = session_data.get('events', [])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        xs = [t['position']['x'] for t in telemetry]
        ys = [t['position']['y'] for t in telemetry]
        ax.plot(xs, ys, '-', color='gray', alpha=0.6, label='Траектория')
        
        avoid_x, avoid_y, target_x, target_y = [], [], [], []
        for ev in events:
            pos = ev.get('position', [0, 0, 0])
            if ev['kind'] == 'obstacle_avoid':
                avoid_x.append(pos[0]); avoid_y.append(pos[1])
            elif ev['kind'] == 'target_reached':
                target_x.append(pos[0]); target_y.append(pos[1])
        
        if avoid_x: ax.scatter(avoid_x, avoid_y, c='red', marker='x', s=80, label='Обход', zorder=5)
        if target_x: ax.scatter(target_x, target_y, c='green', marker='*', s=150, label='Цель', zorder=5)
        
        ax.set_xlabel('X (Север, м)')
        ax.set_ylabel('Y (Восток, м)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis('equal')
        st.pyplot(fig)

# === ВКЛАДКА 3: Важность признаков ===
with tab3:
    st.header("Важность признаков (LIME/SHAP/LRP)")
    if xai_data:
        feature_names = xai_data[0].get('feature_names', [])
        ru_names = [FEATURE_NAMES_RU.get(f, f) for f in feature_names]
        
        attr_by_action = {0: [], 1: []}
        for exp in xai_data:
            action = exp.get('action')
            attr = exp.get('attribution')
            if action in attr_by_action and attr:
                attr_by_action[action].append(np.abs(attr))
        
        mean_attr = {a: np.mean(attrs, axis=0) for a, attrs in attr_by_action.items() if attrs}
        
        if mean_attr:
            x = np.arange(len(ru_names))
            width = 0.35
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            if 0 in mean_attr: ax2.bar(x - width/2, mean_attr[0], width, label='Остановка', color='crimson', alpha=0.8)
            if 1 in mean_attr: ax2.bar(x + width/2, mean_attr[1], width, label='Продолжить', color='forestgreen', alpha=0.8)
            ax2.set_ylabel('Средняя важность')
            ax2.set_xticks(x)
            ax2.set_xticklabels(ru_names, rotation=45, ha='right')
            ax2.legend()
            st.pyplot(fig2)

# === ВКЛАДКА 4: Метрики XAI ===
with tab4:
    st.header("Метрики достоверности (Deletion / Insertion AUC)")
    if xai_data:
        metrics_by_method = defaultdict(lambda: {'del': [], 'ins': []})
        for exp in xai_data:
            method = exp.get('method')
            m = exp.get('metrics')
            if method and m:
                if 'deletion_auc' in m: metrics_by_method[method]['del'].append(m['deletion_auc'])
                if 'insertion_auc' in m: metrics_by_method[method]['ins'].append(m['insertion_auc'])
        
        if metrics_by_method:
            methods = list(metrics_by_method.keys())
            mean_del = [np.mean(metrics_by_method[m]['del']) for m in methods]
            mean_ins = [np.mean(metrics_by_method[m]['ins']) for m in methods]
            
            x = np.arange(len(methods))
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.bar(x - 0.2, mean_del, 0.4, label='Deletion AUC (меньше = лучше)', color='royalblue')
            ax3.bar(x + 0.2, mean_ins, 0.4, label='Insertion AUC (больше = лучше)', color='orange')
            ax3.set_xticks(x)
            ax3.set_xticklabels([m.upper() for m in methods])
            ax3.legend()
            st.pyplot(fig3)

# === ВКЛАДКА 5: Сравнение ===
with tab5:
    st.header("⚖️ Сравнение LiDAR vs Camera")
    comp_path = 'data/reports/comparative_analysis.json'
    if os.path.exists(comp_path):
        with open(comp_path, 'r', encoding='utf-8') as f:
            comp_data = json.load(f)
        
        st.info(f"**Сравниваются:**\n- LiDAR: `{comp_data.get('lidar_report_file', 'N/A')}`\n- Camera: `{comp_data.get('camera_report_file', 'N/A')}`")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📡 LiDAR")
            st.metric("Обходы", comp_data['lidar_metrics'].get('obstacle_events', 0))
            st.metric("Застревания", comp_data['lidar_metrics'].get('stuck_events', 0))
        with col2:
            st.subheader("📷 Camera")
            st.metric("Обходы", comp_data['camera_metrics'].get('obstacle_events', 0))
            st.metric("Застревания", comp_data['camera_metrics'].get('stuck_events', 0))

        # График сравнения
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        labels = ['Обходы', 'Застревания']
        lidar_vals = [comp_data['lidar_metrics'].get('obstacle_events', 0), comp_data['lidar_metrics'].get('stuck_events', 0)]
        cam_vals = [comp_data['camera_metrics'].get('obstacle_events', 0), comp_data['camera_metrics'].get('stuck_events', 0)]
        
        x = np.arange(len(labels))
        ax4.bar(x - 0.2, lidar_vals, 0.4, label='LiDAR', color='blue')
        ax4.bar(x + 0.2, cam_vals, 0.4, label='Camera', color='red')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        st.pyplot(fig4)
    else:
        st.warning("Запустите compare_reports.py для сравнения.")