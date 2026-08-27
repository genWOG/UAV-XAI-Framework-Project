"""
Управление логами: цветной вывод, временные метки, сохранение в JSON
"""

from datetime import datetime
import json
import os

class LogManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logs = []
        self.start_time = datetime.now()
        
        # Цвета для терминала
        self.colors = {
            'INFO': '\033[92m',    # Зелёный
            'WARN': '\033[93m',    # Жёлтый
            'ERROR': '\033[91m',   # Красный
            'OK': '\033[92m',      # Зелёный
            'SYSTEM': '\033[96m',  # Голубой
            'OBSTACLE': '\033[95m',# Фиолетовый
            'DRONE': '\033[94m',   # Синий
            'ROUTE': '\033[93m',   # Жёлтый
            'SENSOR': '\033[96m',  # Голубой
            'RESET': '\033[0m'     # Сброс
        }
        
        self.icons = {
            'INFO': 'ℹ️',
            'WARN': '⚠️',
            'ERROR': '🚨',
            'OK': '✅',
            'SYSTEM': '🔧',
            'OBSTACLE': '🛑',
            'DRONE': '🚁',
            'ROUTE': '🗺️',
            'SENSOR': '📡',
            'TIME': '⏱️'
        }

    def _timestamp(self):
        """Возвращает время в формате MM:SS"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins:02d}:{secs:02d}"

    def _log(self, level, message):
        """Внутренний метод для логирования"""
        ts = self._timestamp()
        color = self.colors.get(level, self.colors['INFO'])
        icon = self.icons.get(level, '📌')
        
        # Цветной вывод в терминал
        print(f"{color}[{ts}] {icon} {message}{self.colors['RESET']}")
        
        # Сохраняем в список
        self.logs.append({
            'time': ts,
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    def info(self, message):
        self._log('INFO', message)

    def warn(self, message):
        self._log('WARN', message)

    def error(self, message):
        self._log('ERROR', message)

    def ok(self, message):
        self._log('OK', message)

    def system(self, message):
        self._log('SYSTEM', message)

    def obstacle(self, message):
        self._log('OBSTACLE', message)

    def drone(self, message):
        self._log('DRONE', f"{self.icons['DRONE']} {message}")

    def route(self, message):
        self._log('ROUTE', f"{self.icons['ROUTE']} {message}")

    def sensor(self, message):
        self._log('SENSOR', f"{self.icons['SENSOR']} {message}")

    def save(self, filename=None):
        """Сохраняет логи в JSON"""
        if filename is None:
            filename = f"flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'start_time': self.start_time.isoformat(),
                'logs': self.logs,
                'total': len(self.logs)
            }, f, ensure_ascii=False, indent=2)
        
        print(f"{self.colors['OK']}✅ Логи сохранены: {filepath}{self.colors['RESET']}")
        return filepath