import json
import glob
import os
import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
from sklearn.ensemble import RandomForestClassifier

# Словарь для перевода
FEATURE_NAMES_RU = {
    'altitude': 'Высота', 'speed': 'Скорость', 'obstacle_dist': 'Расстояние до препятствия',
    'battery': 'Батарея', 'obstacle_density': 'Плотность препятствий',
    'yaw': 'Курс', 'roll': 'Крен', 'pitch': 'Тангаж',
    'distance_to_target': 'Дистанция до цели', 'angle_to_target': 'Угол на цель'
}

def generate_training_data(n_samples=500, feature_names=None):
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        data.append([
            np.random.uniform(2, 30), np.random.uniform(1, 15), np.random.uniform(2, 100),
            np.random.uniform(20, 100), np.random.uniform(0, 1), np.random.uniform(-180, 180),
            np.random.uniform(-30, 30), np.random.uniform(-30, 30), np.random.uniform(10, 500), np.random.uniform(-180, 180)
        ])
    return pd.DataFrame(data, columns=feature_names)

def main():
    print("🔍 Поиск инцидентов...")
    session_files = glob.glob('data/incidents/*/session.json')
    xai_files = glob.glob('data/logs/xai_session_*.json')
    
    if not session_files or not xai_files:
        print("❌ Логи не найдены.")
        return
        
    latest_session = max(session_files, key=os.path.getmtime)
    latest_xai = max(xai_files, key=os.path.getmtime)
    
    with open(latest_session, 'r') as f: session_data = json.load(f)
    with open(latest_xai, 'r') as f: xai_data = json.load(f)
    
    # Ищем интересные события
    events = [e for e in session_data.get('events', []) if e['kind'] in ['stuck', 'obstacle_avoid']]
    
    if not events:
        print("✅ Инцидентов не найдено.")
        return
        
    print(f"Найдено {len(events)} инцидентов. Анализируем первый...")
    incident = events[0]
    print(f"📌 Тип события: {incident['kind']} в {incident['t']}")
    
    # Ищем XAI-объяснение, ближайшее по времени к этому событию (погрешность 2 сек)
    best_xai = None
    min_dt = 2.0
    for xai in xai_data:
        # XAI лог использует строку времени, конвертируем
        import datetime
        xai_t = datetime.datetime.fromisoformat(xai['timestamp']).timestamp()
        dt = abs(xai_t - incident['t'])
        if dt < min_dt:
            min_dt = dt
            best_xai = xai
            
    if not best_xai:
        print("⚠️ Не нашли XAI-объяснение для этого инцидента.")
        return
        
    features = np.array(best_xai['raw_state'])
    feature_names = best_xai['feature_names']
    
    print(f"\nТекущее состояние (action={best_xai['action_name']}):")
    for i, name in enumerate(feature_names):
        print(f"  - {FEATURE_NAMES_RU[name]}: {features[i]:.2f}")
        
    print("\n🧠 Обучаем модель (без масштабирования) для DiCE...")
    
    # 1. Создаем DataFrame для обучения (чтобы убрать ворнинги sklearn)
    X_train = generate_training_data(500, feature_names)
    
    # Эвристика для меток: 0 если близко к препятствию или мало батареи, иначе 1
    y_train = np.array([0 if (x['obstacle_dist'] < 10 or x['battery'] < 15) else 1 for _, x in X_train.iterrows()])
    
    # 2. Обучаем модель
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    print("🧪 Генерируем Counterfactual (может занять ~30-60 секунд)...")
    
    # 3. Готовим данные для DiCE
    df_train = X_train.copy()
    df_train['target'] = y_train
    query_instance = pd.DataFrame([features], columns=feature_names)
    
    d = dice_ml.Data(dataframe=df_train, continuous_features=feature_names, outcome_name='target')
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    exp = Dice(d, m, method="random")
    
    try:
        # Ищем, что нужно изменить, чтобы модель сказала "Лететь" (1)
        # Указываем, какие признаки можно менять (исключаем курс, крен, тангаж)
        dice_exp = exp.generate_counterfactuals(
            query_instance, 
            total_CFs=1, 
            desired_class=1,
            features_to_vary=['obstacle_dist', 'battery', 'speed', 'distance_to_target'],
            permitted_range={'obstacle_dist': [2, 100], 'battery': [20, 100]}
        )
        
        cf_list = dice_exp.cf_examples_list[0].final_cfs_df
        
        if cf_list is not None and len(cf_list) > 0:
            cf_original = cf_list[feature_names].values[0]
            
            print("\n" + "="*60)
            print("📄 ПОДРОБНЫЙ ОТЧЕТ COUNTERFACTUAL")
            print("="*60)
            print("Дрон принял решение ОСТАНОВИТЬСЯ.")
            print("Чтобы модель передумала и сказала ПРОДОЛЖИТЬ, необходимо:")
            
            changes_list = []
            original_state = {}
            cf_state = {}
            
            for i, name in enumerate(feature_names):
                ru_name = FEATURE_NAMES_RU[name]
                original_state[ru_name] = round(float(features[i]), 2)
                cf_state[ru_name] = round(float(cf_original[i]), 2)
                
                diff = cf_original[i] - features[i]
                if abs(diff) > 0.1: # Если изменение существенное
                    print(f"  🔹 Изменить {ru_name}: с {features[i]:.1f} на {cf_original[i]:.1f}")
                    changes_list.append({
                        "ru_name": ru_name,
                        "old_value": round(float(features[i]), 2),
                        "new_value": round(float(cf_original[i]), 2),
                        "delta": round(float(diff), 2)
                    })
                    
            # Формируем текстовое заключение
            if changes_list:
                main_change = max(changes_list, key=lambda x: abs(x['delta']))
                conclusion = (
                    f"Модель приняла решение «Остановка», так как {main_change['ru_name']} "
                    f"({main_change['old_value']}) было критическим. Чтобы модель приняла решение «Продолжить», "
                    f"необходимо изменить {main_change['ru_name']} до {main_change['new_value']}."
                )
            else:
                conclusion = "Модель не нашла существенных изменений для смены решения."
                
            print("="*60)
            
            # === СОХРАНЕНИЕ ПОДРОБНОГО ОТЧЕТА В JSON ===
            report_data = {
                "timestamp": incident['timestamp'],
                "action_original": "Остановка",
                "action_desired": "Продолжить",
                "original_state": original_state,
                "counterfactual_state": cf_state,
                "changes": changes_list,
                "conclusion": conclusion
            }
            os.makedirs('data/logs', exist_ok=True)
            with open('data/logs/counterfactual_report.json', 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print("💾 Подробный отчет сохранен в data/logs/counterfactual_report.json")
            
    except Exception as e:
        print(f"❌ Ошибка DiCE: {e}")

if __name__ == "__main__":
    main()