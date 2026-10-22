"""
Автоматическая разметка новых кадров с помощью обученной YOLO v2.
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

# Пути
PROJECT_DIR = Path('/mnt/d/Projects/UAV-XAI-Framework-Project')
FRAMES_DIR = PROJECT_DIR / 'results/frames'
IMAGES_TRAIN = PROJECT_DIR / 'data/datasets/uav_obstacles/images/train'
LABELS_TRAIN = PROJECT_DIR / 'data/datasets/uav_obstacles/labels/train'
MODEL_PATH = PROJECT_DIR / 'runs/detect/uav_obstacles_v2/weights/best.pt'

# Загружаем модель
model = YOLO(str(MODEL_PATH))

# Получаем список уже размеченных картинок (по имени)
existing_labels = {p.stem for p in LABELS_TRAIN.glob('*.txt')}
print(f"Уже размечено: {len(existing_labels)}")

# Ищем новые кадры
all_frames = list(FRAMES_DIR.glob('*.jpg')) + list(FRAMES_DIR.glob('*.png'))
new_frames = [f for f in all_frames if f.stem not in existing_labels]
print(f"Новых кадров для разметки: {len(new_frames)}")

if not new_frames:
    print("Нет новых кадров. Всё размечено!")
    exit()

# Прогоняем YOLO
print("Запуск YOLO для автолейблинга (это может занять пару минут)...")
results = model.predict(
    source=new_frames,
    conf=0.25,        # Порог уверенности (можешь поменять на 0.3 для строгости)
    save_txt=True,    # Сохраняем результаты в txt
    save_conf=False,  # Не пишем confidence в txt (YOLO формат)
    save=False,       # Не сохраняем картинки с рамками (нам нужны только txt)
    project='runs/detect',
    name='auto_label',
    exist_ok=True,
    device='cpu',
    stream=True,      # Для экономии памяти
    verbose=False
)

# Копируем результаты
count = 0
for r in results:
    img_path = Path(r.path)
    # Копируем изображение в train
    shutil.copy(img_path, IMAGES_TRAIN / img_path.name)
    
    # Ищем сгенерированный txt
    txt_path = Path('runs/detect/auto_label/labels') / (img_path.stem + '.txt')
    if txt_path.exists():
        shutil.copy(txt_path, LABELS_TRAIN / txt_path.name)
        count += 1
    else:
        # Если YOLO ничего не нашел, создаем пустой файл (значит кадр пустой)
        (LABELS_TRAIN / (img_path.stem + '.txt')).touch()

print(f"✅ Автоматически размечено и скопировано: {count} кадров")