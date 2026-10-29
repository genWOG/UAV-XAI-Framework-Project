import os
import glob
import cv2
import numpy as np
import shutil
from ultralytics import YOLO
from datetime import datetime

def main():
    print("🎭 Симуляция кадра инцидента (обход препятствия)...")
    
    # 1. Берем любой старый кадр (где точно есть препятствие)
    old_frames = glob.glob('results/frames/frame_20261009_*.jpg')
    if not old_frames:
        print("❌ Нет старых кадров для симуляции. Нужен хотя бы один.")
        return
        
    source_frame = old_frames[0]
    
    # 2. Сохраняем его под именем свежего инцидента
    os.makedirs('results/frames', exist_ok=True)
    new_frame_name = 'frame_obstacle_avoid_MOCK.jpg'
    new_frame_path = os.path.join('results/frames', new_frame_name)
    shutil.copy(source_frame, new_frame_path)
    print(f"✅ Создан кадр: {new_frame_name}")
    
    # 3. Запускаем YOLO
    print("🧠 Генерация YOLO детекции...")
    model = YOLO('runs/detect/uav_obstacles_v2/weights/best.pt')
    
    img = cv2.imread(new_frame_path)
    
    results = model(new_frame_path, device='cpu', verbose=False)[0]
    
    if len(results.boxes) > 0:
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Рисуем bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{model.names[cls]} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Сохраняем визуализацию
        os.makedirs('results/vision_xai', exist_ok=True)
        out_path = 'results/vision_xai/cam_' + new_frame_name
        cv2.imwrite(out_path, img)
        print(f"✅ Визуализация YOLO сохранена: {out_path}")
    else:
        print("⚠️ YOLO ничего не нашла на этом кадре.")

if __name__ == "__main__":
    main()