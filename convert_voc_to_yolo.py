"""
Конвертер PascalVOC (XML) -> YOLO (TXT)
Ищет XML в annotations/annotations и сохраняет TXT в labels/train.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# Классы
CLASSES = ['house', 'wall', 'tree', 'pole']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

# Пути
base_dir = Path('/mnt/d/Projects/UAV-XAI-Framework-Project/data/datasets/uav_obstacles/labels/train')
src_dir = base_dir / 'annotations' / 'annotations'
dst_dir = base_dir

converted = 0
skipped = 0

if not src_dir.exists():
    print(f"❌ Папка не найдена: {src_dir}")
    exit(1)

for xml_file in src_dir.glob('*.xml'):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            skipped += 1
            continue

        w = int(size.find('width').text)
        h = int(size.find('height').text)

        lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            
            if name == 'small wall':
                name = 'wall'

            if name not in CLASS_MAP:
                continue

            cls_id = CLASS_MAP[name]

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if lines:
            out_file = dst_dir / (xml_file.stem + '.txt')
            with open(out_file, 'w') as f:
                f.write('\n'.join(lines))
            converted += 1
        else:
            pass # Пустые файлы не сохраняем

    except Exception as e:
        print(f"❌ Ошибка в {xml_file.name}: {e}")
        skipped += 1

print(f"✅ Конвертировано: {converted}")
print(f"⚠️  Пропущено: {skipped}")

# Удаляем папку annotations со всем содержимым, чтобы не мешала
if src_dir.exists():
    shutil.rmtree(base_dir / 'annotations')
    print("🧹 Папка annotations удалена")