import glob
import os
import shutil
import subprocess
from pathlib import Path

def main():
    print("👁️ Подготовка кадров и генерация Grad-CAM...")
    
    # 1. Ищем ВСЕ кадры инцидентов
    all_frames = glob.glob('results/frames/frame_*.jpg')
    if not all_frames:
        print("❌ Кадры инцидентов не найдены.")
        return

    # Сортируем по времени модификации (от старых к новым)
    all_frames.sort(key=os.path.getmtime)
    
    # Берем последние 20 кадров (или все, если их меньше)
    frames_to_process = all_frames[-20:] 
    
    print(f"Найдено {len(all_frames)} кадров. Обрабатываем последние {len(frames_to_process)}...")
    out_dir = Path('results/vision_xai')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Временный конфиг для uav_xai
    temp_config = "config/yolo_temp_cam.yaml"
    
    for frame_path in frames_to_process:
        filename = os.path.basename(frame_path)
        print(f"Обработка: {filename}")
        
        # Создаем временный YAML конфиг для нашего CLI
        with open(temp_config, 'w') as f:
            f.write(f"""
experiment:
  name: temp_vision_xai
model:
  name: yolo
  task: detection
  weights: runs/detect/uav_obstacles_v2/weights/best.pt
  prediction_confidence_threshold: 0.05
image:
  path: {frame_path}
xai:
  input_type: image
  method: grad_cam
  image_size: 640
  target_layer: "21"
  detection_iou_threshold: 0.2
  detection_score_threshold: 0.05
  detection_top_k: 1
methods:
  - grad_cam
output:
  root: results/vision_xai/temp
""")
        
        # Запускаем наш рабочий CLI фреймворка
        try:
            # Запускаем наш рабочий CLI фреймворка
            cmd = f"CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 -m uav_xai experiment --config {temp_config}"
            # Добавляем stdout=subprocess.DEVNULL, чтобы не засорять консоль логами фреймворка
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Ищем сгенерированную картинку
            generated_cams = glob.glob('results/vision_xai/temp/*/grad_cam*.png')
            if generated_cams:
                shutil.copy(generated_cams[0], out_dir / f"cam_{filename}")
                print(f"  ✅ Grad-CAM сохранен: cam_{filename}")
            else:
                print(f"  ⚠️ CAM не сгенерирован для {filename} (возможно, нет детекций)")
                # Копируем оригинал как fallback
                shutil.copy(frame_path, out_dir / filename)
                
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Пропуск {filename}: YOLO не нашла объектов для объяснения.")
            # Копируем оригинал, чтобы дашборд не был пустым
            shutil.copy(frame_path, out_dir / filename)
        except Exception as e:
            print(f"  ❌ Ошибка при вызове фреймворка: {e}")
            shutil.copy(frame_path, out_dir / filename)
            
    # Удаляем временный конфиг
    if os.path.exists(temp_config):
        os.remove(temp_config)
        
    print("\n🎉 Готово! Все тепловые карты сохранены в results/vision_xai/")

if __name__ == "__main__":
    main()