import airsim
import time

# Создаём клиента для мультироторного дрона
client = airsim.MultirotorClient()
client.confirmConnection()  # Ждём подтверждения соединения

# Включаем управление и взлетаем
client.enableApiControl(True)
client.armDisarm(True)

print("Взлетаем...")
client.takeoffAsync().join()

time.sleep(2)

print("Летим вперёд на 5 метров...")
client.moveToPositionAsync(5, 0, -3, 5).join()  # x=5, y=0, z=-3 (высота 3 метра)

time.sleep(2)

print("Садимся...")
client.landAsync().join()

# Отключаем управление
client.armDisarm(False)
client.enableApiControl(False)

print("Тест завершён!")