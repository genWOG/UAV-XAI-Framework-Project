"""
GPS навигатор для БПЛА.
Использует координаты для построения маршрута.
"""
import math

class GPSNavigator:
    def __init__(self, start_x=0, start_y=0, target_x=200, target_y=300):
        self.start_x = start_x
        self.start_y = start_y
        self.target_x = target_x
        self.target_y = target_y
        self.current_x = start_x
        self.current_y = start_y
        self.route = []
        self.waypoint_index = 0
        self.waypoint_distance = 20.0  # метров между точками

    def update_position(self, x, y):
        """Обновляет текущую позицию."""
        self.current_x = x
        self.current_y = y

    def build_route(self):
        """Строит маршрут от текущей позиции до цели."""
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist == 0:
            self.route = [(self.target_x, self.target_y)]
            self.waypoint_index = 0
            return

        # Количество waypoint'ов
        num_waypoints = max(2, int(dist / self.waypoint_distance))
        route = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            wx = self.current_x + dx * t
            wy = self.current_y + dy * t
            route.append((wx, wy))

        # Добавляем финальную цель
        if route[-1] != (self.target_x, self.target_y):
            route.append((self.target_x, self.target_y))

        self.route = route
        self.waypoint_index = 0

    def get_next_waypoint(self):
        """Возвращает следующую точку маршрута."""
        if not self.route or self.waypoint_index >= len(self.route):
            return (self.target_x, self.target_y)

        waypoint = self.route[self.waypoint_index]
        dx = waypoint[0] - self.current_x
        dy = waypoint[1] - self.current_y
        dist = math.sqrt(dx*dx + dy*dy)

        # Если достигли waypoint'а, переходим к следующему
        if dist < 3.0:
            self.waypoint_index += 1
            if self.waypoint_index < len(self.route):
                return self.route[self.waypoint_index]
            else:
                return (self.target_x, self.target_y)

        return waypoint

    def get_distance_to_target(self):
        """Расстояние до цели."""
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        return math.sqrt(dx*dx + dy*dy)

    def is_target_reached(self, threshold=5.0):
        """Проверяет, достигнута ли цель."""
        return self.get_distance_to_target() < threshold