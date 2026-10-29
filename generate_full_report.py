import json
import glob
import os
import sys
import argparse
import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

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

def run_counterfactual(features, feature_names):
    X_train = generate_training_data(500, feature_names)
    y_train = np.array([0 if (x['obstacle_dist'] < 10 or x['battery'] < 15) else 1 for _, x in X_train.iterrows()])
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    df_train = X_train.copy()
    df_train['target'] = y_train
    query_instance = pd.DataFrame([features], columns=feature_names)
    
    d = dice_ml.Data(dataframe=df_train, continuous_features=feature_names, outcome_name='target')
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    exp = Dice(d, m, method="random")
    
    try:
        dice_exp = exp.generate_counterfactuals(
            query_instance, total_CFs=1, desired_class=1,
            features_to_vary=['obstacle_dist', 'battery', 'speed', 'distance_to_target']
        )
        cf_list = dice_exp.cf_examples_list[0].final_cfs_df
        if cf_list is not None and len(cf_list) > 0:
            cf_orig = cf_list[feature_names].values[0]
            changes = []
            for i, name in enumerate(feature_names):
                diff = cf_orig[i] - features[i]
                if abs(diff) > 0.1:
                    changes.append({"ru_name": FEATURE_NAMES_RU[name], "old": round(float(features[i]), 2), "new": round(float(cf_orig[i]), 2)})
            
            if changes:
                main = max(changes, key=lambda x: abs(x['new'] - x['old']))
                concl = f"Модель остановилась, так как {main['ru_name']} ({main['old']}) было критическим. Чтобы лететь, нужно изменить {main['ru_name']} до {main['new']}."
            else:
                concl = "Модель не нашла существенных изменений."
            return {"changes": changes, "conclusion": concl}
    except Exception as e:
        print(f"Ошибка DiCE: {e}")
    return {"changes": [], "conclusion": "Контрфактуал не сгенерирован."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["lidar", "camera"], required=True, help="Тип полета")
    args = parser.parse_args()

    print(f"📦 Генерация отчета для контура: {args.type.upper()}")
    
    session_files = glob.glob('data/incidents/*/session.json')
    xai_files = glob.glob('data/logs/xai_session_*.json')
    
    if not session_files or not xai_files:
        print("❌ Логи не найдены.")
        return

    latest_session = max(session_files, key=os.path.getmtime)
    latest_xai = max(xai_files, key=os.path.getmtime)

    with open(latest_session, 'r') as f: session = json.load(f)
    with open(latest_xai, 'r') as f: xai = json.load(f)

    stops = [e for e in xai if e.get('action') == 0]
    cf_result = {"changes": [], "conclusion": "Инцидентов 'Остановка' не было."}
    
    if stops:
        incident = stops[0]
        print("🧪 Генерация Counterfactual...")
        cf_result = run_counterfactual(np.array(incident['raw_state']), incident['feature_names'])

    # Сводный отчет
    report_data = {
        "flight_type": args.type,
        "timestamp": datetime.now().isoformat(),
        "session_file": latest_session,  # <--- ДОБАВИТЬ
        "xai_file": latest_xai,          # <--- ДОБАВИТЬ
        "summary": session.get('summary', {}),
        "counterfactual": cf_result,
        "xai_data_summary": {
            "total_explanations": len(xai),
            "methods_used": list(set([e['method'] for e in xai]))
        }
    }

    os.makedirs('data/reports', exist_ok=True)
    out_name = f"data/reports/{args.type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Отчет сохранен: {out_name}")

if __name__ == "__main__":
    main()