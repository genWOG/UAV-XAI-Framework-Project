import cosysairsim as airsim
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ============ 1. ПОДКЛЮЧЕНИЕ ============
print("Подключение к симулятору...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ============ 2. ПАПКА ДЛЯ ЛОГОВ ============
log_dir = f"logs_flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
print(f"Логи будут сохранены в: {log_dir}")

# ============ 3. ЗАПИСЬ КАДРОВ ============
def capture_image(client, name):
    try:
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
        ])
        for i, img in enumerate(responses):
            if img.pixels_as_float:
                data = np.frombuffer(img.image_data_float, dtype=np.float32).reshape(img.height, img.width, -1)
                plt.imsave(f"{log_dir}/{name}_type_{i}.png", data, cmap='hot')
            else:
                data = np.frombuffer(img.image_data_uint8, dtype=np.uint8).reshape(img.height, img.width, -1)
                plt.imsave(f"{log_dir}/{name}_type_{i}.png", data)
    except Exception as e:
        print(f"⚠️ Ошибка захвата кадра: {e}")

# ============ 4. ПОЛЁТ НА 1 МЕТРЕ ============
print("Взлёт на 1 метр...")
client.takeoffAsync().join()
time.sleep(2)

log_data = []

# Целевые координаты: вперёд (x) 50 м, вправо (y) 100 м
target_x = 50
target_y = 100
altitude = -1  # 1 МЕТР ВЫСОТЫ

print(f"Полёт по диагонали: x={target_x} м, y={target_y} м, высота={abs(altitude)} м...")
print("⚠️ Внимание: полёт на высоте 1 метр — будут препятствия!")

# Запускаем движение
client.moveToPositionAsync(target_x, target_y, altitude, 10).join()

# ============ 5. ЗАПИСЬ ДАННЫХ ============
print("Запись данных...")
for i in range(100):
    try:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        log_data.append({
            'time': time.time(),
            'x': pos.x_val,
            'y': pos.y_val,
            'z': pos.z_val,
            'vx': vel.x_val,
            'vy': vel.y_val,
            'vz': vel.z_val
        })
        
        if i % 5 == 0:
            capture_image(client, f"frame_{i:04d}")
            
        time.sleep(0.1)
    except Exception as e:
        print(f"Ошибка записи: {e}")
        break

print("Полёт завершён! Фиксируем результат...")
time.sleep(2)

# ============ 6. ФИКСАЦИЯ ============
client.simPause(True)
capture_image(client, "final_position")

# ============ 7. СОХРАНЕНИЕ ЛОГОВ ============
df = pd.DataFrame(log_data)
df.to_csv(f"{log_dir}/flight_log.csv", index=False)

print(f"\n✅ Логи сохранены: {log_dir}/flight_log.csv")
print(f"✅ Количество записей: {len(df)}")

# ============ 8. ГРАФИК ТРАЕКТОРИИ ============
try:
    plt.figure(figsize=(12, 10))
    plt.plot(df['x'], df['y'], 'b-', linewidth=2, label='Траектория полёта')
    plt.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=12, label='Старт')
    plt.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'rx', markersize=15, label='Финиш')
    plt.title('Траектория полёта (высота 1 м, вперёд 50 м, вправо 100 м)')
    plt.xlabel('X (метры)')
    plt.ylabel('Y (метры)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.savefig(f"{log_dir}/trajectory.png", dpi=150)
    print(f"✅ График траектории: {log_dir}/trajectory.png")
except Exception as e:
    print(f"⚠Ошибка построения графика: {e}")

# ============ 9. СТАТИСТИКА ============
if len(df) > 0:
    max_speed = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2).max()
    final_x = df['x'].iloc[-1]
    final_y = df['y'].iloc[-1]
    
    print(f"\nСтатистика полёта:")
    print(f"   Максимальная скорость: {max_speed:.2f} м/с")
    print(f"   Длительность: {len(df) * 0.1:.1f} сек")
    print(f"   Конечная позиция: x={final_x:.2f}, y={final_y:.2f}")

# ============ 10. ЗАВЕРШЕНИЕ ============
client.simPause(False)
client.armDisarm(False)
client.enableApiControl(False)

print(f"\nВсе данные сохранены в папку: {log_dir}")
print("\nДля создания видео из кадров выполни:")
print(f"   ffmpeg -framerate 10 -i {log_dir}/frame_%04d_type_0.png -c:v libx264 -pix_fmt yuv420p {log_dir}/flight_video.mp4")