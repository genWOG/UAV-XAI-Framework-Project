#!/bin/bash
# Принудительно удаляем все ROS-переменные
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset AMENT_PREFIX_PATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset ROS_LOCALHOST_ONLY
unset COLCON_PREFIX_PATH

source ~/mavsdk_env/bin/activate
cd /mnt/d/Projects/UAV-XAI-Framework-Project

# Жесткая проверка: если rclpy есть - прерываем запуск
if python3 -c "import rclpy" 2>/dev/null; then
    echo "❌ ОШИБКА: rclpy всё ещё загружен! Закрой старые терминалы."
    exit 1
else
    echo "✅ Окружение чистое. rclpy не найден. Запускаем..."
fi

python3 flight_monitor.py