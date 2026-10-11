# CHANGELOG — UAV-XAI Image XAI for YOLO

**Обновлено:** 12 октября 2026 г.
**Область:** только image-XAI для YOLO — Grad-CAM, LIME, SHAP, LRP.

## Git и рабочее состояние

- Репозиторий: `E:\UAV-XAI-Framework-Project-review`
- Ветка: `feature/yolo-detection-metrics`
- HEAD: `e4ce2bc feat: improve YOLO image explainers and experiments`
- Предыдущие коммиты: `53d503e` (detection faithfulness metrics), `e4ce2bc` (YOLO image explainers и эксперименты).
- Текущая итерация **не закоммичена**. Изменены 7 файлов (см. карту ниже); staged-файлов нет. Merge/push не выполнялись. Основной репозиторий `E:\UAV-XAI-Framework-Project` не трогать.

## Карта текущих изменений: файл → зачем

| Файл | Изменение и назначение |
|---|---|
| `src/uav_xai/metrics/detection.py` | Пошаговая диагностика deletion/insertion: confidence класса, confidence same-class кандидата с максимальным IoU, IoU исходной цели, факт наличия объекта и факт успешного сопоставления. Старый thresholded detection score/AUC сохранён. Исправлена передача `iou_threshold` в deletion. Добавлена отдельная localization-weighted кривая/AUC на основе `confidence × IoU`. |
| `src/uav_xai/core/runner.py` | Сохранение `deletion_curve` и `insertion_curve` в `result.json`; диагностический график `<method>_detection_diagnostics.png`; запись localization AUC рядом со старой метрикой. |
| `src/uav_xai/methods/lime_image.py` | YOLO LIME: векторизация формирования superpixel perturbations, пакетный inference с `prediction_batch_size` (по умолчанию 32), локальная взвешенная регрессия LIME сохранена. Для detection target целевая функция переключена на непрерывный `confidence × IoU`; добавлены метаданные local fit/prediction и режима target score. |
| `src/uav_xai/methods/shap_image.py` | YOLO SHAP: векторизация проекции masks в изображения и конфигурируемый пакетный inference (по умолчанию 32 вместо прежних 8). Бюджет samples не уменьшался. Detection target использует непрерывный `confidence × IoU`; прежняя сегментация/baseline конфигурации сохранена. |
| `tests/test_detection_metrics.py` | Тесты различают падение confidence, сдвиг bbox, исчезновение детекции, наличие другого объекта того же класса, настроенный IoU threshold и граничный IoU=0. Проверяется сериализация диагностических кривых для deletion и insertion. |
| `tests/test_lime_yolo.py` | Проверка пакетного/векторизованного LIME и ограничения размера inference batch. |
| `tests/test_shap_yolo.py` | Проверка конфигурируемого batch size SHAP. |

Конфигурации эксперимента и фабрика в этой итерации не изменялись. Новые файлы исходников в репозитории не добавлялись; временные YAML для двух отдельных запусков удалены после использования.

## Применённые методы

1. **Detection-aware faithfulness diagnostics.** Старый score остаётся confidence обнаружения того же класса, только если bbox проходит существующий IoU threshold; иначе score равен нулю. Диагностические кривые показывают confidence и локализацию отдельно, чтобы падение AUC не трактовалось автоматически как падение confidence.
2. **Непрерывная цель LIME/SHAP.** Для локальной аппроксимации XAI используется `confidence × IoU`, поэтому bbox с ненулевым, но недостаточным IoU не сразу теряет весь сигнал. Старые thresholded deletion/insertion AUC при этом сохранены для сопоставимости; localization AUC публикуется отдельно. Это отдельная целевая функция объяснителя, поэтому карты после её введения не следует напрямую сравнивать с картами прежней версии как контролируемый эксперимент.
3. **Векторизация superpixel perturbations.** Массивы возмущённых изображений формируются пакетно, а не вложенными циклами по каждому сегменту каждого образца. Количество samples в этой итерации не снижалось.
4. **Batch inference.** LIME/SHAP используют настраиваемый размер пакета; default — 32. Это уменьшает накладные расходы Python, но не устраняет стоимость самого YOLO inference и NMS.

## Эксперименты и результаты

Изображение: `data/images/modern_ship.jpg` (1199×674), модель YOLO26n на CPU, вход модели 640×640. Цель: крупная детекция `boat`, исходные confidence около `0.8431`, IoU сопоставления около `0.9882`.

### Исходный единый запуск до текущей итерации

`results/yolo_detection_20261012_011815/`

| Метод | Runtime | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---:|---:|---:|
| Grad-CAM | 230.13 ms | 0.0212 | 0.7819 |
| LIME | 42,490 ms | 0.0343 | 0.1631 |
| SHAP | 37,013 ms | 0.0815 | 0.7094 |
| LRP | 1,015.94 ms | 0.0704 | 0.6553 |

### После векторизации, до непрерывной цели

- LIME: `results/yolo_detection_20261012_021736/` — 36,207 ms, deletion `0.0212`, insertion `0.2049`.
- SHAP: `results/yolo_detection_20261012_021815/` — 22,722 ms, deletion `0.0592`, insertion `0.5608`.

### Последние запуски с непрерывной целью `confidence × IoU`

| Метод | Каталог результата | Runtime | Старый deletion AUC ↓ | Старый insertion AUC ↑ | Localization deletion AUC | Localization insertion AUC |
|---|---|---:|---:|---:|---:|---:|
| LIME | `results/yolo_detection_20261012_022906/` | 36,127.72 ms | 0.0683 | 0.4412 | 0.10394 | 0.49300 |
| SHAP | `results/yolo_detection_20261012_022945/` | 22,577.07 ms | 0.1016 | 0.6196 | 0.12037 | 0.62254 |

LIME local fit score в последнем запуске: `0.5383` — качество локальной аппроксимации пока ограничено. В deletion на 25% perturbation LIME имел confidence `0.5461`, но IoU только `0.1571`; SHAP — confidence `0.5396`, IoU `0.0785`. Это подтверждает, что confidence и сохранение локализации — разные аспекты faithfulness.

**Осторожность при сравнении:** последние карты LIME/SHAP обучались на другой (непрерывной) целевой функции. Изменения AUC относительно более раннего прогона нельзя приписывать только векторизации или трактовать как чистое сравнение алгоритмов. Все замеры сделаны на CPU одного изображения; они не подтверждают временные требования для Jetson.

## Проверки

После текущих изменений:
- целевые тесты метрик/LIME YOLO/SHAP YOLO: `17 passed`;
- полный `pytest`: `41 passed, 3 warnings`;
- `git diff --check`: ошибок whitespace нет; остаются только предупреждения Git о преобразовании LF/CRLF.
- Предупреждения pytest приходят из кода зависимости SHAP (`PendingDeprecationWarning`).

## Следующие действия

1. Продолжать оптимизацию времени LIME/SHAP, отдельно измеряя runtime explainer и полный pipeline. Текущие ~36.1 s и ~22.6 s на CPU всё ещё далеко от целевых задержек.
2. Не менять samples, baseline и сегментацию вслепую. Сначала сравнить диагностические confidence/IoU-кривые и проверить качество локальной модели LIME.
3. После стабилизации протокола проверить Grad-CAM prior и знаковую релевантность LRP, затем оценивать стабильность на нескольких изображениях.
4. Любые новые изменения — только в review-worktree; тесты и diff review перед обсуждением коммита. **Никаких коммитов без явного согласия пользователя.**
