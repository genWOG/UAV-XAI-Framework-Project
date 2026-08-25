import cosysairsim as airsim
import time

# Подключаемся к симулятору
client = airsim.MultirotorClient()
client.confirmConnection()  # Ждём, пока появится "Connected!"

# Берём управление на себя
client.enableApiControl(True)
client.armDisarm(True)  # "Заводим" моторы

print("Взлетаем!")
client.takeoffAsync().join()  # Взлетаем и ждём завершения

time.sleep(2)

print("Летим вперёд на 10 метров")
client.moveToPositionAsync(10, 0, -5, 5).join()  # x=10, y=0, высота 5 метров

time.sleep(2)

print("Садимся")
client.landAsync().join()  # Садимся

# Отключаем управление
client.armDisarm(False)
client.enableApiControl(False)

print("Полёт завершён!")