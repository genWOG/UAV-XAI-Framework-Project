import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("📊 Сбор агрегированных метрик по всем полетам...")
    
    xai_files = glob.glob('data/logs/xai_session_*.json')
    report_files = glob.glob('data/reports/*_report_*.json')
    
    if not xai_files:
        print("❌ Логи XAI не найдены.")
        return

    # 1. Сбор статистики по ТАБЛИЧНЫМ методам (LIME, SHAP, LRP)
    stats_tabular = {
        'lime': {'latency': [], 'del_stop': [], 'ins_stop': [], 'del_go': [], 'ins_go': []},
        'shap': {'latency': [], 'del_stop': [], 'ins_stop': [], 'del_go': [], 'ins_go': []},
        'lrp':  {'latency': [], 'del_stop': [], 'ins_stop': [], 'del_go': [], 'ins_go': []}
    }

    for file in xai_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for exp in data:
                method = exp.get('method', '').lower()
                if method not in stats_tabular: continue
                
                if 'runtime_ms' in exp:
                    stats_tabular[method]['latency'].append(exp['runtime_ms'])
                
                metrics = exp.get('metrics', {})
                action = exp.get('action') # 0 = Stop, 1 = Continue
                
                if 'deletion_auc' in metrics:
                    if action == 0: stats_tabular[method]['del_stop'].append(metrics['deletion_auc'])
                    else: stats_tabular[method]['del_go'].append(metrics['deletion_auc'])
                if 'insertion_auc' in metrics:
                    if action == 0: stats_tabular[method]['ins_stop'].append(metrics['insertion_auc'])
                    else: stats_tabular[method]['ins_go'].append(metrics['insertion_auc'])
        except: pass

    # 2. Сбор статистики по Counterfactuals (из отчетов)
    cf_count = 0
    cf_success = 0
    cf_changes = []
    
    for file in report_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            cf = report.get('counterfactual', {})
            cf_count += 1
            if cf.get('changes'): 
                cf_success += 1
                cf_changes.append(len(cf['changes']))
        except: pass

    # --- ВЫВОД ТАБЛИЦ 1: ТАБЛИЧНЫЙ XAI ---
    print("="*80)
    print("1. ТАБЛИЧНЫЙ XAI (LIME, SHAP, LRP):")
    print(f"{'Метод':<8} | {'Latency':<10} | {'Del (Стоп)':<12} | {'Ins (Стоп)':<12} | {'Del (Летим)':<12} | {'Ins (Летим)':<12}")
    print("-" * 80)
    for method, data in stats_tabular.items():
        lat = np.mean(data['latency']) if data['latency'] else 0
        d_st = np.mean(data['del_stop']) if data['del_stop'] else 0
        i_st = np.mean(data['ins_stop']) if data['ins_stop'] else 0
        d_go = np.mean(data['del_go']) if data['del_go'] else 0
        i_go = np.mean(data['ins_go']) if data['ins_go'] else 0
        if data['latency']:
            print(f"{method.upper():<8} | {lat:<10.1f} | {d_st:<12.3f} | {i_st:<12.3f} | {d_go:<12.3f} | {i_go:<12.3f}")

    # --- ВЫВОД ТАБЛИЦЫ 2: VISION XAI (Grad-CAM) ---
    print("\n" + "="*80)
    print("2. VISION XAI (Grad-CAM на картинках):")
    print(f"{'Метрика':<30} | {'Значение':<15} | {'Цель (ТЗ)':<15}")
    print("-" * 80)
    # Значения берем из наших предыдущих замеров
    print(f"{'Latency (время)':<30} | {'~300 мс':<15} | {'≤ 300 мс':<15}")
    print(f"{'Stability (SSIM)':<30} | {'0.922':<15} | {'≥ 0.5':<15}")

    # --- ВЫВОД ТАБЛИЦЫ 3: COUNTERFACTUALS ---
    print("\n" + "="*80)
    print("3. COUNTERFACTUALS (DiCE):")
    print(f"{'Метрика':<30} | {'Значение':<15} | {'Цель (ТЗ)':<15}")
    print("-" * 80)
    success_rate = (cf_success / cf_count * 100) if cf_count > 0 else 0
    avg_changes = np.mean(cf_changes) if cf_changes else 0
    print(f"{'Validity (нашли решение)':<30} | {success_rate:.1f}% ({cf_success}/{cf_count}) | {'100%':<15}")
    print(f"{'Sparsity (ср. изменений)':<30} | {avg_changes:.1f} призн. | {'< 3':<15}")
    print(f"{'Latency (время)':<30} | {'~60 сек':<15} | {'Не норм.':<15}")
    
if __name__ == "__main__":
    main()