"""
Тест интеграции фреймворка uav_xai.
Проверяет базовую работу XAIEngine с Grad-CAM.
"""

import sys
import os
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# Добавляем путь к src
project_root = "D:/Projects/UAV-XAI-Framework-Project"
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from uav_xai.core import create_default_engine, ModelPredictor


def test_gradcam():
    """
    Тестируем Grad-CAM через XAIEngine.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ GRAD-CAM")
    print("=" * 60)
    
    # Проверяем доступные методы
    engine = create_default_engine()
    methods = engine.list_methods()
    print(f"Зарегистрированные методы: {methods}")
    
    # Загружаем модель
    print("\n[1] Загрузка модели ResNet-18...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    
    # Загружаем изображение
    print("[2] Загрузка изображения...")
    # Создаем тестовое изображение, если нет реального
    test_image_path = os.path.join(project_root, "data", "test_image.jpg")
    
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path).convert("RGB")
    else:
        # Создаем синтетическое изображение
        print("   ⚠️ Тестовое изображение не найдено, создаем синтетическое...")
        import numpy as np
        array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(array)
    
    print(f"   Размер изображения: {image.size}")
    
    # Предобработка
    print("[3] Предобработка изображения...")
    transform = weights.transforms()
    input_tensor = transform(image).unsqueeze(0)
    
    # Предсказание
    print("[4] Предсказание модели...")
    predictor = ModelPredictor(model)
    prediction = predictor.predict(input_tensor)
    
    categories = weights.meta["categories"]
    predicted_class = categories[prediction.class_id]
    
    print(f"   Класс: {predicted_class}")
    print(f"   ID: {prediction.class_id}")
    print(f"   Уверенность: {prediction.confidence:.4f}")
    
    # Grad-CAM
    print("\n[5] Генерация Grad-CAM объяснения...")
    
    # Целевой слой для ResNet-18
    target_layer = model.layer4[-1].conv2
    
    result = engine.explain(
        method="grad_cam",
        model=model,
        input_data=input_tensor,
        target=prediction.class_id,
        target_layer=target_layer,
    )
    
    print(f"   Метод: {result.method}")
    print(f"   Время: {result.runtime_ms:.2f} мс")
    print(f"   Размер attribution: {result.attribution.shape}")
    print(f"   Уверенность: {result.confidence:.4f}")
    
    # Сохраняем визуализацию
    print("\n[6] Сохранение визуализации...")
    from uav_xai.utils import overlay_heatmap, save_image
    
    visualization = overlay_heatmap(
        image=image,
        heatmap=result.attribution,
        alpha=0.45,
    )
    
    output_dir = os.path.join(project_root, "results", "gradcam_test")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "gradcam_output.png")
    save_image(visualization, output_path)
    
    print(f"   Визуализация сохранена: {output_path}")
    
    # Выводим метаданные
    print("\n[7] Метаданные:")
    for key, value in result.metadata.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ ТЕСТ GRAD-CAM ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 60)


if __name__ == "__main__":
    test_gradcam()