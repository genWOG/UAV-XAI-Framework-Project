"""
Модуль логирования для БПЛА.
Поддерживает вывод в файл и в консоль.
"""

import os
import json
import time
from datetime import datetime

class LogManager:
    def __init__(self, log_dir="data/logs", console_output=True):
        self.log_dir = log_dir
        self.console_output = console_output
        self.logs = []
        self.start_time = time.time()
        self.flight_time = 0
        
        # Создаём папку для логов
        os.makedirs(log_dir, exist_ok=True)
        
        # Запоминаем время старта
        self.start_datetime = datetime.now()
        
    def _log(self, level, message, force_console=False):
        """Внутренний метод логирования"""
        # Вычисляем время полёта
        self.flight_time = int(time.time() - self.start_time)
        time_str = f"{self.flight_time // 60:02d}:{self.flight_time % 60:02d}"
        
        # Создаём запись
        entry = {
            "time": time_str,
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.logs.append(entry)
        
        # Вывод в консоль (если включено)
        if self.console_output or force_console:
            print(f"[{time_str}] {message}")
    
    def info(self, message):
        self._log("INFO", message)
    
    def warn(self, message):
        self._log("WARN", message)
    
    def error(self, message):
        self._log("ERROR", message, force_console=True)
    
    def system(self, message):
        self._log("SYSTEM", message, force_console=True)
    
    def ok(self, message):
        self._log("OK", message, force_console=True)
    
    def drone(self, message):
        self._log("DRONE", message)
    
    def obstacle(self, message):
        self._log("OBSTACLE", message)
    
    def route(self, message):
        self._log("ROUTE", message)
    
    def save(self):
        """Сохраняет логи в файл"""
        if not self.logs:
            return
        
        # Имя файла с временем
        filename = f"flight_{self.start_datetime.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Сохраняем
        data = {
            "start_time": self.start_datetime.isoformat(),
            "logs": self.logs,
            "total": len(self.logs)
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[SYSTEM] ✅ Логи сохранены: {filepath}")
            return filepath
        except Exception as e:
            print(f"[ERROR] ❌ Ошибка сохранения логов: {e}")
            return None
    
    def get_logs(self):
        """Возвращает все логи"""
        return self.logs
    
    def clear(self):
        """Очищает логи"""
        self.logs = []
        self.start_time = time.time()
        self.flight_time = 0