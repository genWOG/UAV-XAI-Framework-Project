# UAV-XAI Framework — YOLO XAI Progress

Дата контрольной точки: 2026-10-11

## Git и рабочая директория

Рабочая директория: `E:\UAV-XAI-Framework-Project-review`
Рабочая ветка: `feature/yolo-detection-metrics`
Исходная база ветки: `origin/main`, commit `7abc65a`
Последний известный commit: `53d503e`
Commit message: `feat: add YOLO detection faithfulness metrics`

Все дальнейшие изменения делать в review worktree и feature-ветке.
Не изменять основную директорию `E:\UAV-XAI-Framework-Project`.
Не выполнять merge, push или reset без явного разрешения.

## Что уже сделано

### Detection-aware faithfulness metrics

Добавлены:
- `src/uav_xai/metrics/detection.py`
- обновление `src/uav_xai/metrics/__init__.py`
- `tests/test_detection_metrics.py`

Реализованы detection-aware deletion/insertion metrics.
Сопоставление использует класс и IoU bbox; порог по умолчанию — 0.5.
Метрики возвращают IoU и confidence целевого обнаружения.

В `53d503e` также вошли:
- интеграция detection metrics в `src/uav_xai/core/runner.py`;
- `config/yolo_experiment.yaml`;
- тесты detection metrics.

### YOLO + Grad-CAM

Конфигурация: `config/yolo_experiment.yaml`

Успешный запуск:
`results/yolo_detection_20261011_214407/`

Результаты:
- Deletion AUC: `0.0212178`
- Insertion AUC: `0.781869`
- IoU сопоставления целевого обнаружения при запуске: примерно `0.9882`
- Runtime Grad-CAM: примерно `199.03 ms`

В каталоге результата сохранены визуализации, `result.json`, `comparison.csv` и копия конфигурации.

### YOLO + LIME

Добавлены, пока не закоммичены:
- изменения `src/uav_xai/core/factory.py`;
- изменения `src/uav_xai/core/runner.py`;
- изменения `src/uav_xai/methods/lime_image.py`;
- `config/yolo_lime_experiment.yaml`;
- `tests/test_lime_yolo.py`.

LIME теперь поддерживает detection target типа `XAITarget`, используя класс и IoU целевого bbox. Путь для классификационного LIME сохранён.

В runner учитываются различия координат:
- исходное изображение;
- letterboxed вход модели;
- attribution map LIME в оригинальном размере;
- attribution map Grad-CAM в размере модели.

Успешный запуск:
`results/yolo_detection_20261011_224640/`

Результаты:
- Deletion AUC: `0.2185209`
- Insertion AUC: `0.2149836`
- Runtime LIME: примерно `12194.15 ms`
- Число сегментов: `703`
- Число samples: `50`

В каталоге результата сохранены `lime.png`, `lime_faithfulness.png`, общие визуализации сравнения, `comparison.csv`, `result.json` и конфигурация.

### Проверки

После добавления YOLO LIME:
- полный тестовый набор: `29 passed, 3 warnings`;
- detection metrics tests: `4 passed`;
- LIME classification и YOLO tests: `3 passed`;
- Python compile checks прошли;
- последний известный `git diff --check` не выявил ошибок пробелов.

## Что делать дальше

1. Сначала проверить, что changelog создан и текущее состояние Git соответствует этому описанию.
2. Довести и протестировать SHAP для YOLO detection targets.
3. Довести и протестировать LRP для YOLO detection targets.
4. Для SHAP и LRP обеспечить поддержку detection target с сопоставлением по классу и IoU.
5. Подключить их к runner и провести реальные YOLO-запуски с faithfulness metrics.
6. После изменений снова выполнить полный набор тестов и `git diff --check`.
7. Обновить этот changelog с результатами и состоянием коммитов.

## Важные оговорки

- SHAP и LRP для YOLO на момент этой контрольной точки ещё не реализованы и не подтверждены запуском.
- Сравнение Grad-CAM и LIME пока основано на одном изображении и одном запуске каждого метода. Это предварительное наблюдение, а не статистически подтверждённый вывод.
- LIME работает заметно медленнее Grad-CAM в проведённом запуске.
- Перед новыми изменениями проверить `git status --short` и не потерять незакоммиченные изменения LIME.

## Final checkpoint and merge map for Gena

Checkpoint date: 2026-10-11
Branch: `feature/yolo-detection-metrics`
HEAD commit: `53d503e feat: add YOLO detection faithfulness metrics`
Base: `origin/main` at `7abc65a`

### Real YOLO26n experiments

All four runs used CPU, the image `data/images/modern_ship.jpg`
(original size 1199x674), and a 640x640 model input.
The selected target was the boat detection with confidence approximately
0.8431. Target matching in the LRP run reached IoU 0.9882.

| Method | Results directory | Runtime | Deletion AUC | Insertion AUC |
|---|---|---:|---:|---:|
| Grad-CAM | `results/yolo_detection_20261011_214407/` | 199.03 ms | 0.0212178 | 0.781869 |
| LIME | `results/yolo_detection_20261011_224640/` | 12194.15 ms | 0.2185209 | 0.2149836 |
| SHAP | `results/yolo_detection_20261011_233344/` | 3226.55 ms | 0.4292 | 0.3576 |
| LRP | `results/yolo_detection_20261011_233504/` | 948.11 ms | 0.0677 | 0.3338 |

Each successful run saved `result.json`, `comparison.csv`, and faithfulness
and runtime comparison plots. SHAP and LRP have real-run evidence, not just
unit tests. The comparison uses one image and is preliminary, not statistical
evidence that one method is universally better.

### Test checkpoint

The full test suite passed after adding SHAP and LRP support:
`31 passed, 3 warnings in 5.53s`.

Warnings came from SHAP dependency code (`PendingDeprecationWarning`).
`python -m compileall -q src/uav_xai` passed.
`git diff --check` reported only Git LF-to-CRLF conversion warnings, not
whitespace errors.

Rerun compile and tests after any further edits before integration.

### File map: committed foundation

These files are in feature commit `53d503e`:

- `src/uav_xai/metrics/detection.py` — detection-aware deletion/insertion
  metrics and target detection matching.
- `src/uav_xai/metrics/__init__.py` — exports the detection metric functions.
- `tests/test_detection_metrics.py` — detection metric tests.
- `config/yolo_experiment.yaml` — Grad-CAM YOLO experiment configuration.
- `src/uav_xai/core/runner.py` — committed detection-target and
  faithfulness-metric integration. IMPORTANT: this file also has additional
  uncommitted changes described below; do not treat the whole current file
  as identical to its committed version.

### File map: uncommitted implementation changes

These need to be preserved and included when integrating the completed work:

- `src/uav_xai/core/factory.py`
  - Builds LIME, SHAP and LRP image explainers from experiment configuration.
  - Handles detection task configuration without requiring the ordinary
    classification preprocess callback.
  - Integrate method by method; do not overwrite it with an older copy.

- `src/uav_xai/core/runner.py`
  - Has the committed detection-metric integration from `53d503e`.
  - Also has uncommitted coordinate conversion for attribution maps and
    target boxes, including LIME maps in original-image coordinates.
  - Detection metrics must receive attribution in 640x640 letterboxed model
    coordinates and a target bbox transformed into the same coordinate space.
  - Preserve both the committed integration and these uncommitted changes.

- `src/uav_xai/methods/lime_image.py`
  - Adds detection-target LIME using class and IoU matching.
  - Keeps the classification LIME path.

- `src/uav_xai/methods/shap_image.py`
  - Adds SHAP for detection targets by scoring the target detection across
    superpixel perturbations.
  - Keeps classification SHAP.

- `src/uav_xai/methods/lrp_image.py`
  - Adds LRP for detection targets using a differentiable YOLO target.
  - Keeps classification LRP.

- `config/yolo_lime_experiment.yaml`
- `config/yolo_shap_experiment.yaml`
- `config/yolo_lrp_experiment.yaml`
  - Separate experiment configurations for each method.

- `tests/test_lime_yolo.py`
- `tests/test_shap_yolo.py`
- `tests/test_lrp_yolo.py`
  - Detection-specific unit tests.

- `CHANGELOG_YOLO_XAI_PROGRESS.md`
  - Progress, experiment results, and this merge map.

### Existing shared infrastructure: preserve and reuse

- `src/uav_xai/core/target.py` — normalized `XAITarget` representation.
- `src/uav_xai/core/target_builder.py` — differentiable target builder for
  classification and YOLO detection.
- `src/uav_xai/methods/gradcam.py` — existing structured-target integration
  through `TargetBuilder`.
- `tests/test_shap_image.py` — SHAP classification regression test.
- `tests/test_lrp_image.py` — LRP classification regression test.
- Existing classification LIME tests must also continue passing.

### Safe integration order for Gena

1. Start from the correct shared base and preserve the detection-metric
   implementation from commit `53d503e`.
2. Integrate `factory.py` and `runner.py` carefully. They are shared files:
   do not replace them wholesale with an older version. Retain both the
   detection metrics and the additional method/coordinate-space changes.
3. Include `lime_image.py`, `shap_image.py`, and `lrp_image.py`.
4. Include all three method-specific configs and tests listed above.
5. Keep `target.py`, `target_builder.py`, and existing Grad-CAM behavior
   working; do not duplicate the target-matching logic unnecessarily.
6. Run the full test suite and `git diff --check` after integration.
7. Keep generated experiment outputs as validation evidence. Do not blindly
   copy the entire `results/` directory into the source change.

### Git and merge restrictions

At this checkpoint, `53d503e` is the known committed feature change.
The later LIME, SHAP, LRP and changelog edits are uncommitted working-tree
changes. Do not assume they are included in `53d503e`.

Before merging or handing off, inspect `git status --short`, `git diff`,
and `git diff --stat`, and ensure new untracked configs and tests are included.
Do not merge into main, push, reset or discard working-tree changes without
explicit approval.


## Latest multi-method quality iteration (2026-10-12)

Unified configuration: `config/yolo_multi_method_one_image.yaml`
Input: `data/images/modern_ship.jpg`; model: YOLO26n; device: CPU.
Target: `boat`, confidence 0.8431.
Results: `results/yolo_detection_20261012_011815/`.

| Method | Runtime (ms) | Deletion AUC | Insertion AUC |
|---|---:|---:|---:|
| Grad-CAM | 230.13 | 0.0212 | 0.7819 |
| LIME | 42490.22 | 0.0343 | 0.1631 |
| SHAP | 37013.25 | 0.0815 | 0.7094 |
| LRP | 1015.94 | 0.0704 | 0.6553 |

These results are from one image and are preliminary.
LIME and SHAP remain too slow for the intended real-time goals.
The current comparison needs diagnostic confidence and IoU curves
before interpreting AUC differences as attribution quality.

The YOLO LRP inference-mode issue was fixed sufficiently for the
four-method experiment to complete successfully.
