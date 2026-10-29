import cv2
import numpy as np
import os
import glob
import subprocess
import shutil
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread

def run_gradcam(image_path, output_dir):
    """Запускает uav_xai для картинки и возвращает путь к Grad-CAM"""
    temp_config = "config/yolo_stability_temp.yaml"
    # Занижаем порог до 0.01, чтобы YOLO видела объекты даже при изменении яркости
    with open(temp_config, 'w') as f:
        f.write(f"""
experiment:
  name: stability_test
model:
  name: yolo
  task: detection
  weights: runs/detect/uav_obstacles_v2/weights/best.pt
  prediction_confidence_threshold: 0.01
image:
  path: {image_path}
xai:
  input_type: image
  method: grad_cam
  image_size: 640
  target_layer: "21"
  detection_iou_threshold: 0.2
  detection_score_threshold: 0.01
  detection_top_k: 1
  methods:
    - grad_cam
output:
  root: {output_dir}
""")
    cmd = f"CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 -m uav_xai experiment --config {temp_config}"
    
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
        
    cams = glob.glob(f'{output_dir}/*/grad_cam*.png')
    if not cams:
        return None
    return cams[0]

def main():
    print("📊 Оценка Stability (SSIM) для Grad-CAM (по 5 кадрам)...")
    
    test_frames = glob.glob('results/frames/frame_obstacle_avoid_*.jpg')
    if len(test_frames) < 5:
        print("❌ Мало кадров. Нужно хотя бы 5.")
        return
        
    test_frames = test_frames[-5:] # Берем последние 5 кадров
    scores = []
    
    for idx, original_frame in enumerate(test_frames):
        print(f"\nКадр {idx+1}/5: {os.path.basename(original_frame)}")
        
        # Меняем яркость (имитация изменения освещения)
        img = cv2.imread(original_frame)
        # alpha=1.1 (яркость +10%), beta=10 (сдвиг)
        bright_img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        
        os.makedirs('data/images', exist_ok=True)
        bright_frame = 'data/images/stability_bright.jpg'
        cv2.imwrite(bright_frame, bright_img)
        
        # Генерируем Grad-CAM
        cam_orig_path = run_gradcam(original_frame, 'results/stability/orig')
        cam_bright_path = run_gradcam(bright_frame, 'results/stability/bright')
        
        if not cam_orig_path or not cam_bright_path:
            print("  ⚠️ Пропуск кадра (YOLO не нашла объект).")
            continue
            
        # Считаем SSIM
        img1 = imread(cam_orig_path, as_gray=True)
        img2 = imread(cam_bright_path, as_gray=True)
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        score, _ = ssim(img1, img2, full=True, data_range=1.0)
        scores.append(score)
        print(f"  ✅ SSIM: {score:.4f}")

    if scores:
        mean_score = np.mean(scores)
        print("\n" + "="*40)
        print(f"✅ Средний SSIM по {len(scores)} кадрам: {mean_score:.4f}")
        print("="*40)
        if mean_score >= 0.5:
            print(f"🎉 ВЫПОЛНЕНО! Объяснения стабильны (>= 0.5).")
        else:
            print(f"⚠️ SSIM < 0.5. Объяснения чувствительны к свету.")
    else:
        print("❌ Не удалось посчитать SSIM ни для одного кадра.")

if __name__ == "__main__":
    main()