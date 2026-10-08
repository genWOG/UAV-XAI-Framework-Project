"""
XAI Дашборд для UAV проекта
Версия 5.0 — три отдельных графика в ряд для каждого события
Запуск: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import defaultdict

# Настройка страницы
st.set_page_config(
    page_title="UAV XAI Дашборд",
    page_icon="🚁",
    layout="wide"
)

st.title("🚁 UAV XAI Дашборд")
st.markdown("---")

# ========================================
# ЗАГРУЗКА ДАННЫХ
# ========================================

@st.cache_data
def load_xai_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_flight_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Поиск последних файлов
log_dir = "data/logs"
xai_files = [f for f in os.listdir(log_dir) if f.startswith('xai_') and f.endswith('.json') and 'local' not in f]
flight_files = [f for f in os.listdir(log_dir) if f.startswith('flight_') and f.endswith('.json')]

if not xai_files:
    st.error("❌ Нет XAI файлов!")
    st.stop()

# Выбор файлов
col1, col2 = st.columns(2)
with col1:
    selected_xai = st.selectbox("Выберите XAI файл:", sorted(xai_files, reverse=True))
with col2:
    if flight_files:
        selected_flight = st.selectbox("Выберите лог полёта:", sorted(flight_files, reverse=True))
    else:
        selected_flight = None
        st.warning("⚠️ Логов полёта не найдено")

# Загрузка
xai_data = load_xai_data(os.path.join(log_dir, selected_xai))
flight_data = load_flight_data(os.path.join(log_dir, selected_flight)) if selected_flight else None

st.info(f"📊 Загружено {len(xai_data)} XAI объяснений")

# ========================================
# ФУНКЦИИ ДЛЯ КЛАССИФИКАЦИИ
# ========================================

def classify_explanation(entry, flight_log=None):
    """Классифицирует объяснение по типу события."""
    action = entry.get('action', 1)
    is_anomaly = entry.get('is_anomaly', {}).get('is_anomaly', False)
    
    has_obstacle = False
    if flight_log:
        timestamp = entry.get('timestamp', '')
        for log_entry in flight_log.get('logs', []):
            if log_entry.get('timestamp', '') == timestamp:
                if 'ПРЕПЯТСТВИЕ' in log_entry.get('message', '') or 'препятствие' in log_entry.get('message', ''):
                    has_obstacle = True
                break
    
    if action == 0 or is_anomaly:
        return 'остановка'
    elif has_obstacle:
        return 'препятствие'
    elif 'коррекция' in str(entry.get('explanation_text', '')).lower():
        return 'коррекция'
    else:
        return 'движение'

def translate_feature_names(features):
    translation = {
        'altitude': 'Высота',
        'speed': 'Скорость',
        'obstacle_dist': 'Расстояние до препятствия',
        'battery': 'Заряд батареи',
        'obstacle_density': 'Плотность препятствий',
        'yaw': 'Рыскание',
        'roll': 'Крен',
        'pitch': 'Тангаж',
        'distance_to_target': 'Расстояние до цели',
        'angle_to_target': 'Угол к цели'
    }
    return [translation.get(f, f) for f in features]

# ========================================
# СТАТИСТИКА
# ========================================

st.subheader("📊 Общая статистика")

col1, col2, col3, col4, col5 = st.columns(5)

methods = {}
for entry in xai_data:
    method = entry.get('method', 'fallback')
    methods[method] = methods.get(method, 0) + 1

anomalies = sum(1 for e in xai_data if e.get('is_anomaly', {}).get('is_anomaly', False))
avg_time = np.mean([e.get('runtime_ms', 0) for e in xai_data if e.get('runtime_ms', 0) > 0])

# Классифицируем по событиям
classified = defaultdict(int)
for entry in xai_data:
    event = classify_explanation(entry, flight_data)
    classified[event] += 1

with col1:
    st.metric("Всего объяснений", len(xai_data))
with col2:
    st.metric("Аномалий", anomalies, f"{anomalies/len(xai_data)*100:.1f}%")
with col3:
    st.metric("Методов", len(methods))
with col4:
    st.metric("Ср. время", f"{avg_time:.1f} мс")
with col5:
    st.metric("Событий", len(classified))

# ========================================
# РАСПРЕДЕЛЕНИЕ ПО СОБЫТИЯМ
# ========================================

st.subheader("📊 Распределение по типам событий")

col1, col2 = st.columns(2)

with col1:
    if classified:
        event_names_ru = {
            'движение': 'Движение',
            'коррекция': 'Коррекция',
            'остановка': 'Остановка',
            'препятствие': 'Препятствие'
        }
        fig = px.pie(
            values=list(classified.values()),
            names=[event_names_ru.get(k, k) for k in classified.keys()],
            title="Типы событий"
        )
        st.plotly_chart(fig, width='stretch')

with col2:
    if methods:
        fig = px.pie(
            values=list(methods.values()),
            names=[m.upper() for m in methods.keys()],
            title="Вызовы XAI методов"
        )
        st.plotly_chart(fig, width='stretch')

# ========================================
# ГРАФИКИ СРАВНЕНИЯ МЕТОДОВ ПО СОБЫТИЯМ
# ========================================

st.subheader("📈 Сравнение XAI методов по типам событий")

# Собираем данные по событиям и методам
all_features = xai_data[0].get('feature_names', []) if xai_data else []
ru_features = translate_feature_names(all_features)

# Группируем: событие → метод → список attribution
event_method_data = defaultdict(lambda: defaultdict(list))
for entry in xai_data:
    event = classify_explanation(entry, flight_data)
    method = entry.get('method', 'fallback')
    attr = np.array(entry.get('attribution', []))
    if len(attr) == len(all_features):
        event_method_data[event][method].append(attr)

event_names_ru = {
    'движение': 'Движение (обычный полёт)',
    'коррекция': 'Коррекция маршрута',
    'остановка': 'Прекращение полёта',
    'препятствие': 'Обнаружение препятствия'
}

method_colors = {
    'lime': '#2E86AB',
    'shap': '#A23B72',
    'lrp': '#F18F01',
    'fallback': '#888888'
}

method_labels = {
    'lime': 'LIME',
    'shap': 'SHAP',
    'lrp': 'LRP',
    'fallback': 'FALLBACK'
}

# Создаём вкладки для каждого события
if event_method_data:
    event_order = ['движение', 'коррекция', 'остановка', 'препятствие']
    available_events = [e for e in event_order if e in event_method_data]
    
    if not available_events:
        available_events = list(event_method_data.keys())
    
    tabs = st.tabs([event_names_ru.get(e, e) for e in available_events])
    
    for tab, event in zip(tabs, available_events):
        with tab:
            methods_data = event_method_data.get(event, {})
            if not methods_data:
                st.info("Нет данных для этого события")
                continue
            
            # 🔥 ГЛАВНОЕ ИЗМЕНЕНИЕ: три отдельных графика в ряд
            method_keys = list(methods_data.keys())
            # Сортируем методы для порядка: LIME, SHAP, LRP
            method_order = ['lime', 'shap', 'lrp', 'fallback']
            method_keys_sorted = [m for m in method_order if m in method_keys]
            
            # Создаём 3 колонки для 3-х графиков
            cols = st.columns(len(method_keys_sorted))
            
            for idx, method in enumerate(method_keys_sorted):
                with cols[idx]:
                    attributions = methods_data.get(method, [])
                    if not attributions:
                        st.write(f"Нет данных для {method_labels.get(method, method.upper())}")
                        continue
                    
                    # Усредняем
                    avg = np.mean(attributions, axis=0)
                    std = np.std(attributions, axis=0)
                    
                    # Создаём отдельный график для каждого метода
                    fig, ax = plt.subplots(figsize=(5, max(5, len(all_features) * 0.4)))
                    
                    y_pos = np.arange(len(ru_features))
                    bars = ax.barh(y_pos, avg, color=method_colors.get(method, '#888'), alpha=0.8)
                    
                    # Цвета баров: зелёный (+), красный (-)
                    for i, (bar, val) in enumerate(zip(bars, avg)):
                        if val > 0:
                            bar.set_color('#2ECC71')  # зелёный
                        else:
                            bar.set_color('#E74C3C')   # красный
                    
                    # Добавляем значения на бары
                    for i, v in enumerate(avg):
                        ax.text(
                            v + (0.01 if v > 0 else -0.01), 
                            i, 
                            f'{v:.3f}',
                            va='center', 
                            ha='left' if v > 0 else 'right', 
                            fontsize=8
                        )
                    
                    # Добавляем error bars
                    ax.errorbar(
                        avg, 
                        y_pos, 
                        xerr=std, 
                        fmt='none', 
                        color='black', 
                        capsize=2, 
                        alpha=0.3, 
                        elinewidth=1
                    )
                    
                    # Оси
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(ru_features, fontsize=9)
                    ax.set_xlabel('Вклад в решение', fontsize=10)
                    ax.set_title(f'{method_labels.get(method, method.upper())}\n({len(attributions)} объяснений)', fontsize=11)
                    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

# ========================================
# ВРЕМЕННАЯ ШКАЛА
# ========================================

st.subheader("📈 Временная шкала объяснений")

if xai_data:
    df = pd.DataFrame([
        {
            'index': i,
            'method': e.get('method', 'fallback').upper(),
            'runtime_ms': e.get('runtime_ms', 0),
            'action': 'Остановка' if e.get('action', 1) == 0 else 'Продолжить',
            'is_anomaly': e.get('is_anomaly', {}).get('is_anomaly', False),
            'event': classify_explanation(e, flight_data)
        }
        for i, e in enumerate(xai_data)
    ])
    
    fig = px.scatter(
        df,
        x='index',
        y='runtime_ms',
        color='method',
        symbol='event',
        hover_data=['action', 'is_anomaly'],
        title='Время выполнения XAI по порядку'
    )
    st.plotly_chart(fig, width='stretch')

# ========================================
# МАТРИЦА МЕТОДОВ И СОБЫТИЙ
# ========================================

st.subheader("📊 Матрица: Методы → События")

if xai_data:
    events = ['движение', 'коррекция', 'остановка', 'препятствие']
    methods_list = ['lime', 'shap', 'lrp', 'fallback']
    
    matrix = np.zeros((len(methods_list), len(events)))
    for entry in xai_data:
        method = entry.get('method', 'fallback')
        event = classify_explanation(entry, flight_data)
        if method in methods_list and event in events:
            i = methods_list.index(method)
            j = events.index(event)
            matrix[i, j] += 1
    
    # Нормализуем по строкам
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix_norm, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(events)))
    ax.set_yticks(np.arange(len(methods_list)))
    ax.set_xticklabels([event_names_ru.get(e, e) for e in events])
    ax.set_yticklabels([m.upper() for m in methods_list])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    for i in range(len(methods_list)):
        for j in range(len(events)):
            if matrix[i, j] > 0:
                ax.text(j, i, f'{int(matrix[i, j])}\n({matrix_norm[i, j]:.0%})',
                       ha='center', va='center', fontsize=9)
    
    ax.set_title('Какие методы используются для каких событий', fontsize=13)
    plt.colorbar(im, ax=ax, label='Доля вызовов метода')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========================================
# ВРЕМЯ ВЫПОЛНЕНИЯ МЕТОДОВ
# ========================================

st.subheader("⏱️ Время выполнения методов")

if xai_data:
    times_by_method = {}
    for entry in xai_data:
        method = entry.get('method', 'fallback')
        runtime = entry.get('runtime_ms', 0)
        if runtime > 0:
            times_by_method.setdefault(method, []).append(runtime)
    
    if times_by_method:
        fig = go.Figure()
        for method, times in times_by_method.items():
            fig.add_trace(go.Box(
                y=times,
                name=method.upper(),
                boxmean='sd'
            ))
        fig.update_layout(
            title="Разброс времени выполнения",
            yaxis_title="Время (мс)"
        )
        st.plotly_chart(fig, width='stretch')

# ========================================
# ЛОГ ПОЛЁТА
# ========================================

if flight_data:
    st.subheader("✈️ Лог полёта")
    
    logs = flight_data.get('logs', [])
    df_log = pd.DataFrame([
        {
            'time': l.get('time', ''),
            'level': l.get('level', ''),
            'message': l.get('message', '')[:100]
        }
        for l in logs
    ])
    
    levels = st.multiselect(
        "Уровень логирования",
        options=['INFO', 'WARN', 'ERROR', 'DRONE', 'OK', 'SYSTEM'],
        default=['WARN', 'ERROR', 'DRONE']
    )
    
    if levels:
        df_filtered = df_log[df_log['level'].isin(levels)]
        st.dataframe(df_filtered, use_container_width=True, height=300)

# ========================================
# ПОСЛЕДНЕЕ ОБЪЯСНЕНИЕ
# ========================================

st.subheader("📝 Последнее объяснение")

if xai_data:
    last = xai_data[-1]
    with st.expander("Показать последнее объяснение"):
        st.json(last)

st.markdown("---")
st.caption(f"🚁 UAV XAI Дашборд | Обновлено: {datetime.now().strftime('%H:%M:%S')}")