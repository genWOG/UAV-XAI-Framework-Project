Студенческий проект 
# UAV-XAI-Framework-Project

## 🚁 Объяснимый ИИ для автономных БПЛА

Проект направлен на исследование post-hoc методов объяснимого искусственного интеллекта (XAI) для повышения безопасности, надежности и доверия к автономным беспилотным летательным аппаратам (БПЛА) в гражданских и коммерческих сценариях.

## 📋 Цели исследования

- Усовершенствование комплекса пост-аналитических методов XAI
- Формирование доказательной базы при расследовании инцидентов с БПЛА
- Сертификация и внедрение БПЛА в гражданские сценарии

## 🛠 Используемые методы

| Метод | Описание | Статус |
|-------|----------|--------|
| **LIME** | Локальные интерпретируемые модельно-агностические объяснения | ✅ |
| **SHAP** | SHapley Additive exPlanations | ✅ |
| **LRP** | Layer-wise Relevance Propagation (через Captum) | ✅ |

## 📊 Результаты

### Сравнительный анализ

| Метод | Модель | Время (сек) | Главный признак |
|-------|--------|-------------|-----------------|
| LIME | RandomForest | 0.023 | Скорость |
| SHAP | RandomForest | 0.102 | Скорость |
| LRP | PyTorch NN | 0.0015 | Скорость |

### Ключевые выводы

- **Скорость** БПЛА является главным фактором, влияющим на решение
- **LIME** — лучший для быстрых объяснений в реальном времени
- **LRP** — идеален для глубокого анализа нейросетей
- **SHAP** — даёт наиболее детальное теоретическое обоснование

UAV-XAI-Framework-Project/
├── experiments/
│ └── xai/
│ ├── test_omnixai.py # Тест OmniXAI
│ ├── test_lime_direct.py # LIME + SHAP
│ ├── test_lrp.py # LRP через Captum
│ ├── uav_xai_demo.py # Демонстрация на UAV данных
│ └── xai_comparison.py # Сравнительный анализ
├── results/ # Результаты экспериментов
├── README.md
└── .gitignore

## 🚀 Установка

```bash
# Создание окружения
conda create -n uav_xai python=3.11
conda activate uav_xai

# Установка зависимостей
pip install numpy pandas scikit-learn matplotlib
pip install torch torchvision
pip install lime shap captum zennit
pip install omnixai

📈 Примеры визуализаций

Графики сохраняются в папке results/:

    lime_importance.png — важность признаков (LIME)

    shap_importance.png — SHAP значения

    lrp_captum.png — LRP релевантность

    xai_comparison.png — сравнение всех методов
	
Источники
https://github.com/salesforce/OmniXAI
https://arxiv.org/abs/1602.04938
https://arxiv.org/abs/1705.07874
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140