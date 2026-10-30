import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    os.makedirs('results/analysis', exist_ok=True)
    print("📊 Генерация финальных схем для диплома...")

    # ==========================================
    # ГРАФИК 1: Latency vs Требования ТЗ
    # ==========================================
    methods_lat = ['LRP', 'LIME', 'SHAP', 'Grad-CAM', 'Counterfactuals']
    actual_lat = [2.2, 38.5, 43.6, 300.0, 60000.0]
    target_lat = [10000, None, 30000, 300, None] # None, если ТЗ не нормирует

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods_lat))
    
    # Столбцы фактического времени
    bars = ax.bar(x, actual_lat, color=['#2ca02c', '#1f77b4', '#1f77b4', '#ff7f0e', '#d62728'], alpha=0.7, label='Фактическое время (мс)')
    
    # Точки требований ТЗ
    for i, (actual, target) in enumerate(zip(actual_lat, target_lat)):
        if target is not None:
            ax.plot(i, target, 'r*', markersize=15, label='Требование ТЗ' if i == 0 else "")
            # Рисуем стрелочку, показывающую, что мы влезли в лимит
            ax.annotate(f"Цель ТЗ: {target} мс", (i, target), textcoords="offset points", xytext=(0,10), ha='center', color='red')

    ax.set_yscale('log')
    ax.set_ylabel('Время генерации (мс), лог. шкала', fontsize=12)
    ax.set_title('Сравнение Latency XAI-методов с требованиями ТЗ (п. 3.7.1)', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_lat, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/analysis/tz_latency_comparison.png', dpi=300)
    plt.close()

    # ==========================================
    # ГРАФИК 2: Faithfulness (Deletion/Insertion) для табличных методов
    # ==========================================
    methods_faith = ['LIME', 'SHAP', 'LRP']
    del_auc = [0.169, 0.174, 0.186] # Для решений "Летим"
    ins_auc = [0.850, 0.839, 0.860]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods_faith))
    width = 0.35
    
    ax.bar(x - width/2, del_auc, width, label='Deletion AUC (цель ≤ 0.25)', color='royalblue')
    ax.bar(x + width/2, ins_auc, width, label='Insertion AUC (цель ≥ 0.65)', color='orange')
    
    # Линии целевых значений
    ax.axhline(y=0.25, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(y=0.65, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Faithfulness табличных XAI-методов (решения «Продолжить»)', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_faith, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/analysis/tz_faithfulness_comparison.png', dpi=300)
    plt.close()

    # ==========================================
    # ГРАФИК 3: Сводный график соответствия ТЗ (Compliance Score)
    # ==========================================
    # Считаем, во сколько раз мы лучше (или хуже) цели.
    # Формула: (Факт / Цель) для тех, где больше = лучше. (Цель / Факт) для тех, где меньше = лучше.
    
    metrics = [
        'LRP Latency', 'SHAP Latency', 'Grad-CAM Latency', 
        'Deletion AUC', 'Insertion AUC', 'SSIM (Stability)'
    ]
    
    scores = [
        10000 / 2.2,      # LRP Latency (цель 10000, факт 2.2)
        30000 / 43.6,     # SHAP Latency (цель 30000, факт 43.6)
        300 / 300,        # Grad-CAM Latency (цель 300, факт 300)
        0.25 / 0.169,     # Deletion (цель 0.25, факт 0.169 - чем меньше, тем лучше)
        0.850 / 0.65,     # Insertion (цель 0.65, факт 0.850 - чем больше, тем лучше)
        0.922 / 0.5       # SSIM (цель 0.5, факт 0.922 - чем больше, тем лучше)
    ]
    
    colors = ['#2ca02c' if s >= 1.0 else '#d62728' for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8)
    
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Порог ТЗ (1.0)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=12)
    ax.set_xlabel('Коэффициент соответствия (Факт / Цель). Чем больше, тем лучше.', fontsize=12)
    ax.set_title('Сводный график выполнения требований ТЗ (п. 3.7)', fontsize=14, weight='bold')
    ax.set_xscale('log') # Лог шкала, так как LRP перевыполнен в 4500 раз
    
    # Подписываем значения на столбцах
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width * 1.2, bar.get_y() + bar.get_height()/2, 
                f'x{score:.1f}', va='center', fontsize=11, fontweight='bold')
        
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/analysis/tz_compliance_summary.png', dpi=300)
    plt.close()

    print("🎉 Все 3 схемы сохранены в папку results/analysis/")
    print("1. tz_latency_comparison.png")
    print("2. tz_faithfulness_comparison.png")
    print("3. tz_compliance_summary.png")

if __name__ == "__main__":
    main()