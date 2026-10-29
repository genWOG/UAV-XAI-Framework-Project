import json
import glob
import os
from datetime import datetime

def get_latest_report(flight_type):
    files = glob.glob(f'data/reports/{flight_type}_report_*.json')
    if not files: return None
    return max(files, key=os.path.getmtime)

def main():
    lidar_file = get_latest_report('lidar')
    camera_file = get_latest_report('camera')

    if not lidar_file or not camera_file:
        print("❌ Не хватает отчетов. Сделайте полеты и запустите generate_full_report.py для обоих типов.")
        return

    with open(lidar_file, 'r') as f: lidar_data = json.load(f)
    with open(camera_file, 'r') as f: camera_data = json.load(f)

    print("📊 Сравнение контуров LiDAR vs Camera")
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "lidar_report_file": os.path.basename(lidar_file),   # <--- ДОБАВЛЕНО
        "camera_report_file": os.path.basename(camera_file), # <--- ДОБАВЛЕНО
        "lidar_metrics": lidar_data['summary'],
        "camera_metrics": camera_data['summary'],
        "lidar_xai_methods": lidar_data['xai_data_summary']['methods_used'],
        "camera_xai_methods": camera_data['xai_data_summary']['methods_used'],
        "lidar_cf": lidar_data['counterfactual']['conclusion'],
        "camera_cf": camera_data['counterfactual']['conclusion']
    }

    os.makedirs('data/reports', exist_ok=True)
    out_name = 'data/reports/comparative_analysis.json'
    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Сравнительный анализ сохранен: {out_name}")

if __name__ == "__main__":
    main()