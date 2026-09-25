# test_sensors.py
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Проверяем DistanceSensor
data = client.getDistanceSensorData()
print(f"DistanceSensor: {data.distance} м")

# Проверяем LiDAR
lidar = client.getLidarData()
print(f"LiDAR точек: {len(lidar.point_cloud)}")

client.landAsync().join()