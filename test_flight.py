"""
Тестовый автономный полёт через MAVSDK + PX4.
"""
import asyncio
import mavsdk_grpc as mavsdk
from mavsdk_grpc import System
from mavsdk_grpc.offboard import OffboardError, PositionNedYaw


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    print("🔌 Подключение к PX4...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ PX4 подключён!")
            break

    print("⏳ Ожидание готовности дрона...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ Дрон готов к полёту!")
            break

    print("🔓 Arm...")
    await drone.action.arm()

    print("🛫 Взлёт...")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    print("\n📊 ТЕЛЕМЕТРИЯ:")
    async for position in drone.telemetry.position():
        print(f"   Широта: {position.latitude_deg:.6f}")
        print(f"   Долгота: {position.longitude_deg:.6f}")
        print(f"   Высота: {position.relative_altitude_m:.1f} м")
        break

    # 👇 НАЧАЛЬНАЯ OFFBOARD-ТОЧКА (текущая позиция)
    print("\n🎯 Инициализация offboard...")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -5.0, 0.0))

    # 👇 ЗАПУСК OFFBOARD
    try:
        await drone.offboard.start()
        print("✅ Offboard запущен")
    except OffboardError as e:
        print(f"❌ Ошибка запуска offboard: {e._result.result}")
        await drone.action.disarm()
        return

    # 👇 ДВИЖЕНИЕ ВПЕРЁД
    print("\n🚁 Движение вперёд на 10 м...")
    await drone.offboard.set_position_ned(PositionNedYaw(10.0, 0.0, -5.0, 0.0))
    await asyncio.sleep(8)

    # 👇 ОСТАНОВКА OFFBOARD
    print("⏹ Остановка offboard...")
    try:
        await drone.offboard.stop()
    except OffboardError as e:
        print(f"⚠️ Ошибка остановки offboard: {e._result.result}")

    print("🛬 Посадка...")
    await drone.action.land()
    await asyncio.sleep(8)

    print("\n✅ Полёт завершён!")


if __name__ == "__main__":
    asyncio.run(run())