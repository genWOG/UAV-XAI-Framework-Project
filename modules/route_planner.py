"""
Планирование маршрута и обход препятствий
"""

class RoutePlanner:
    def __init__(self, log_manager):
        self.log = log_manager
        self.current_route = []
        self.alternative_routes = []
        self.is_rerouting = False

    def plan(self, start_x, start_y, target_x, target_y, altitude=-1):
        self.log.route(f"Планирование маршрута: ({start_x},{start_y}) → ({target_x},{target_y})")
        self.log.route(f"Высота: {abs(altitude)} м")
        
        self.current_route = [
            {'x': start_x, 'y': start_y, 'z': altitude},
            {'x': target_x, 'y': target_y, 'z': altitude}
        ]
        
        length = self._calculate_length()
        self.log.ok(f"Маршрут построен. Длина: {length:.1f} м")
        return self.current_route

    def avoid_obstacle(self, obstacle, current_x, current_y):
        self.log.warn(f"Обнаружено препятствие: {obstacle['object']}")
        self.log.route(f"Поиск обходного пути...")
        
        offset = 5 if obstacle['type'] == 'large' else 3
        new_x = current_x + offset
        new_y = current_y + offset / 2
        
        self.log.route(f"Обход: смещение +{offset} м по X, +{offset/2} м по Y")
        self.is_rerouting = True
        
        return {'x': new_x, 'y': new_y, 'z': -1}

    def _calculate_length(self):
        if len(self.current_route) < 2:
            return 0
        dx = self.current_route[-1]['x'] - self.current_route[0]['x']
        dy = self.current_route[-1]['y'] - self.current_route[0]['y']
        return (dx**2 + dy**2) ** 0.5

    def get_status(self):
        if self.is_rerouting:
            return "🔄 Перестроение маршрута"
        return "✅ Маршрут активен"