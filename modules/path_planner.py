"""
Планировщик пути для БПЛА.
Использует простой алгоритм обхода препятствий.
"""

import math
import random
from typing import List, Tuple, Optional

class PathPlanner:
    """Планировщик пути на основе сетки."""
    
    def __init__(self, grid_size=10, max_x=500, max_y=500):
        self.grid_size = grid_size
        self.max_x = max_x
        self.max_y = max_y
        self.grid = {}  # (x, y) -> obstacle_type
        self.path = []
        self.waypoints = []
        
    def add_obstacle(self, x, y, obstacle_type="building"):
        """Добавляет препятствие в карту."""
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        self.grid[(grid_x, grid_y)] = obstacle_type
        
    def is_obstacle(self, x, y):
        """Проверяет, есть ли препятствие в клетке."""
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        return (grid_x, grid_y) in self.grid
    
    def get_neighbors(self, pos, target_x, target_y):
        """Возвращает соседние клетки с приоритетом к цели."""
        x, y = pos
        neighbors = []
        
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.max_x / self.grid_size and 0 <= ny < self.max_y / self.grid_size:
                if not self.is_obstacle(nx * self.grid_size, ny * self.grid_size):
                    dist_to_target = ((nx * self.grid_size - target_x)**2 + 
                                     (ny * self.grid_size - target_y)**2)**0.5
                    neighbors.append((nx, ny, dist_to_target))
        
        neighbors.sort(key=lambda n: n[2])
        return [(n[0], n[1]) for n in neighbors[:4]]
    
    def find_path(self, start_x, start_y, target_x, target_y):
        """Находит путь от старта к цели, обходя препятствия."""
        start_grid = (int(start_x / self.grid_size), int(start_y / self.grid_size))
        target_grid = (int(target_x / self.grid_size), int(target_y / self.grid_size))
        
        if self.is_obstacle(target_x, target_y):
            target_grid = self._find_nearest_free(target_x, target_y)
            if target_grid is None:
                return []
        
        path = [start_grid]
        current = start_grid
        visited = set()
        max_steps = 100
        
        for _ in range(max_steps):
            if current == target_grid:
                break
            
            visited.add(current)
            neighbors = self.get_neighbors(current, target_x, target_y)
            
            found = False
            for nx, ny in neighbors:
                if (nx, ny) not in visited:
                    current = (nx, ny)
                    path.append(current)
                    found = True
                    break
            
            if not found:
                for _ in range(10):
                    nx = current[0] + random.randint(-1, 1)
                    ny = current[1] + random.randint(-1, 1)
                    if (nx, ny) not in visited and not self.is_obstacle(nx * self.grid_size, ny * self.grid_size):
                        current = (nx, ny)
                        path.append(current)
                        found = True
                        break
                if not found:
                    break
        
        waypoints = []
        for gx, gy in path:
            waypoints.append((gx * self.grid_size + self.grid_size/2, 
                              gy * self.grid_size + self.grid_size/2))
        
        self.path = path
        self.waypoints = waypoints
        return waypoints
    
    def _find_nearest_free(self, x, y):
        """Находит ближайшую свободную клетку."""
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        
        for radius in range(1, 20):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.max_x / self.grid_size and 0 <= ny < self.max_y / self.grid_size:
                        if not self.is_obstacle(nx * self.grid_size, ny * self.grid_size):
                            return (nx, ny)
        return None
    
    def get_next_waypoint(self, current_x, current_y):
        """Возвращает следующую точку маршрута."""
        if not self.waypoints:
            return None
        
        min_dist = float('inf')
        best_idx = 0
        for i, (wx, wy) in enumerate(self.waypoints):
            dist = ((wx - current_x)**2 + (wy - current_y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if best_idx < len(self.waypoints) - 1:
            return self.waypoints[best_idx + 1]
        return self.waypoints[-1]
    
    def clear(self):
        """Очищает карту и путь."""
        self.grid.clear()
        self.path = []
        self.waypoints = []