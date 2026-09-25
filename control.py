"""
Панель управления симуляцией UAV-XAI

Запуск: python control.py
"""

import sys
import os
import subprocess
import time
import signal
import threading
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext

# ========================================
# ФИКС: Используем Python из окружения uav_xai
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Ищем Python из окружения uav_xai
def get_uav_xai_python():
    """Находит Python из окружения uav_xai"""
    # Проверяем, активна ли среда
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env == 'uav_xai':
        return sys.executable
    
    # Проверяем пути к Anaconda
    possible_paths = [
        r"C:\Users\CP\anaconda3\envs\uav_xai\python.exe",
        r"C:\ProgramData\Anaconda3\envs\uav_xai\python.exe",
        r"C:\Anaconda3\envs\uav_xai\python.exe",
        r"C:\Users\CP\miniconda3\envs\uav_xai\python.exe",
    ]
    
    # Ищем в переменной окружения
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        possible_paths.append(os.path.join(conda_prefix, 'python.exe'))
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Если не нашли — используем текущий Python
    return sys.executable

PYTHON_EXE = get_uav_xai_python()
IS_UAV_XAI = 'uav_xai' in PYTHON_EXE

# Глобальные переменные
process = None
is_running = False


class UAVControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚁 UAV-XAI Панель управления")
        self.root.geometry("1000x750")
        self.root.resizable(True, True)

        # Основной фрейм
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== Верхняя панель =====
        top_frame = ttk.LabelFrame(main_frame, text="Управление симуляцией", padding="10")
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Кнопки
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_start = ttk.Button(btn_frame, text="▶ Запустить полёт", 
                                    command=self.start_flight, width=20)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = ttk.Button(btn_frame, text="⏹ Остановить полёт", 
                                   command=self.stop_flight, width=20, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_graphs = ttk.Button(btn_frame, text="📊 Сравнить методы", 
                                     command=self.run_comparison, width=20)
        self.btn_graphs.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_dashboard = ttk.Button(btn_frame, text="📈 Открыть дашборд", 
                                        command=self.open_dashboard, width=20)
        self.btn_dashboard.pack(side=tk.LEFT)

        # Информация об окружении
        env_frame = ttk.Frame(top_frame)
        env_frame.pack(fill=tk.X, pady=2)
        
        self.env_label = ttk.Label(env_frame, 
                                   text=f"🐍 Python: {os.path.basename(PYTHON_EXE)} | {'✅ uav_xai' if IS_UAV_XAI else '⚠️ Другое окружение'}",
                                   font=("Arial", 9))
        self.env_label.pack(side=tk.LEFT)

        # Статус
        status_frame = ttk.Frame(top_frame)
        status_frame.pack(fill=tk.X, pady=5)

        ttk.Label(status_frame, text="Статус:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="⏸ Остановлен", 
                                      foreground="gray", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(status_frame, text="Время:").pack(side=tk.LEFT)
        self.time_label = ttk.Label(status_frame, text="--:--:--")
        self.time_label.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(status_frame, text="Объяснений:").pack(side=tk.LEFT)
        self.count_label = ttk.Label(status_frame, text="0")
        self.count_label.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(status_frame, text="Препятствий:").pack(side=tk.LEFT)
        self.obstacle_label = ttk.Label(status_frame, text="0")
        self.obstacle_label.pack(side=tk.LEFT, padx=(5, 20))

        # ===== Лог =====
        log_frame = ttk.LabelFrame(main_frame, text="Лог выполнения", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, 
                                                   font=("Courier New", 9),
                                                   wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Настройка цветов для лога
        self.log_text.tag_configure("INFO", foreground="#0066CC")
        self.log_text.tag_configure("OK", foreground="#008000")
        self.log_text.tag_configure("WARN", foreground="#FF8C00")
        self.log_text.tag_configure("ERROR", foreground="#CC0000", font=("Courier New", 9, "bold"))
        self.log_text.tag_configure("DRONE", foreground="#8B008B")
        self.log_text.tag_configure("SYSTEM", foreground="#000000", font=("Courier New", 9, "bold"))
        self.log_text.tag_configure("OBSTACLE", foreground="#CC5500")

        # ===== Нижняя панель =====
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(bottom_frame, text="🗑 Очистить лог", 
                   command=self.clear_log, width=15).pack(side=tk.RIGHT, padx=(0, 10))
        ttk.Button(bottom_frame, text="📂 Открыть папку логов", 
                   command=self.open_log_folder, width=20).pack(side=tk.RIGHT, padx=(0, 10))
        ttk.Button(bottom_frame, text="🔄 Проверить пути", 
                   command=self.check_paths, width=15).pack(side=tk.RIGHT, padx=(0, 10))

        # Обновляем время и статус
        self.update_time()
        self.check_status()
        
        # Логируем информацию о путях
        self.log(f"📂 Рабочая папка: {BASE_DIR}", "SYSTEM")
        self.log(f"🐍 Python: {PYTHON_EXE}", "SYSTEM")
        self.log(f"📦 Окружение: {'uav_xai (✅)' if IS_UAV_XAI else 'другое (⚠️)'}", "SYSTEM")
        self.log("✅ Панель управления запущена", "OK")

    def log(self, message, level="INFO"):
        """Добавление сообщения в лог с цветом"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        message = message.strip()
        if not message:
            return
        
        # Обновляем счётчики
        if "Всего объяснений:" in message:
            try:
                count = message.split(":")[1].strip()
                self.count_label.config(text=count)
            except:
                pass
        
        if "Всего препятствий:" in message:
            try:
                count = message.split(":")[1].strip()
                self.obstacle_label.config(text=count)
            except:
                pass

        self.log_text.insert(tk.END, f"[{timestamp}] ", "SYSTEM")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.log("[Лог очищен]", "SYSTEM")

    def open_log_folder(self):
        """Открывает папку с логами в проводнике"""
        log_dir = os.path.join(BASE_DIR, "data", "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(log_dir)
        else:
            subprocess.Popen(["open", log_dir])

    def check_paths(self):
        """Проверяет наличие всех необходимых файлов"""
        self.log("=" * 60, "SYSTEM")
        self.log("🔍 ПРОВЕРКА ПУТЕЙ", "SYSTEM")
        self.log("=" * 60, "SYSTEM")
        
        files_to_check = [
            "flight_monitor.py",
            os.path.join("modules", "log_manager.py"),
            os.path.join("modules", "airsim_adapter.py"),
            os.path.join("xai", "modules", "xai_bridge.py"),
        ]
        
        for f in files_to_check:
            full_path = os.path.join(BASE_DIR, f)
            if os.path.exists(full_path):
                self.log(f"✅ {f} — найден", "OK")
            else:
                self.log(f"❌ {f} — НЕ НАЙДЕН!", "ERROR")
        
        # Проверяем пакеты
        self.log("", "SYSTEM")
        self.log("📦 ПРОВЕРКА ПАКЕТОВ", "SYSTEM")
        try:
            import sklearn
            self.log("✅ scikit-learn — установлен", "OK")
        except ImportError:
            self.log("❌ scikit-learn — НЕ УСТАНОВЛЕН!", "ERROR")
        
        try:
            import lime
            self.log("✅ lime — установлен", "OK")
        except ImportError:
            self.log("❌ lime — НЕ УСТАНОВЛЕН!", "ERROR")
        
        try:
            import shap
            self.log("✅ shap — установлен", "OK")
        except ImportError:
            self.log("❌ shap — НЕ УСТАНОВЛЕН!", "ERROR")
        
        try:
            import airsim
            self.log("✅ airsim — установлен", "OK")
        except ImportError:
            self.log("❌ airsim — НЕ УСТАНОВЛЕН!", "ERROR")
        
        self.log("=" * 60, "SYSTEM")

    def update_time(self):
        self.time_label.config(text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self.update_time)

    def check_status(self):
        """Проверка статуса процесса"""
        global process, is_running
        if process is not None:
            poll = process.poll()
            if poll is not None:
                is_running = False
                self.btn_start.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
                self.status_label.config(text="⏸ Остановлен", foreground="gray")
                if poll == 0:
                    self.log("✅ Полёт завершён успешно", "OK")
                else:
                    self.log(f"⚠️ Полёт завершён с кодом {poll}", "WARN")
                process = None
        self.root.after(1000, self.check_status)

    def start_flight(self):
        """Запуск полёта"""
        global process, is_running
        if is_running:
            self.log("⚠️ Полёт уже запущен", "WARN")
            return

        self.log("=" * 60, "SYSTEM")
        self.log("🚁 ЗАПУСК ПОЛЁТА", "SYSTEM")
        self.log("=" * 60, "SYSTEM")

        try:
            flight_script = os.path.join(BASE_DIR, "flight_monitor.py")
            
            if not os.path.exists(flight_script):
                self.log(f"❌ Файл не найден: {flight_script}", "ERROR")
                return
            
            self.log(f"📂 Скрипт: {flight_script}", "SYSTEM")
            self.log(f"🐍 Python: {PYTHON_EXE}", "SYSTEM")

            # Запускаем с правильным Python
            process = subprocess.Popen(
                [PYTHON_EXE, "-u", flight_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
                cwd=BASE_DIR,
                env=os.environ.copy()
            )
            is_running = True

            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.status_label.config(text="▶ Выполняется", foreground="green")

            def read_output():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            self.log(line.strip())
                    process.stdout.close()
                except Exception as e:
                    self.log(f"⚠️ Ошибка чтения вывода: {e}", "WARN")

            threading.Thread(target=read_output, daemon=True).start()

        except Exception as e:
            self.log(f"❌ Ошибка запуска: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.status_label.config(text="❌ Ошибка", foreground="red")

    def stop_flight(self):
        """Остановка полёта"""
        global process, is_running
        if not is_running or process is None:
            self.log("⚠️ Полёт не запущен", "WARN")
            return

        self.log("=" * 60, "SYSTEM")
        self.log("⏹ ОСТАНОВКА ПОЛЁТА", "SYSTEM")
        self.log("=" * 60, "SYSTEM")

        try:
            if sys.platform == "win32":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)

            try:
                process.wait(timeout=5)
                self.log("✅ Полёт остановлен корректно", "OK")
            except subprocess.TimeoutExpired:
                process.kill()
                self.log("⚠️ Полёт остановлен принудительно", "WARN")

        except Exception as e:
            self.log(f"❌ Ошибка остановки: {e}", "ERROR")
            if process:
                process.kill()

        finally:
            is_running = False
            process = None
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.status_label.config(text="⏸ Остановлен", foreground="gray")

    def run_comparison(self):
        """Запуск сравнения методов"""
        self.log("=" * 60, "SYSTEM")
        self.log("📊 ЗАПУСК СРАВНЕНИЯ МЕТОДОВ", "SYSTEM")
        self.log("=" * 60, "SYSTEM")

        try:
            compare_script = os.path.join(BASE_DIR, "xai", "compare_methods.py")
            
            if not os.path.exists(compare_script):
                self.log(f"❌ Файл не найден: {compare_script}", "ERROR")
                return

            result = subprocess.run(
                [PYTHON_EXE, compare_script],
                capture_output=True,
                text=True,
                cwd=BASE_DIR,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode == 0:
                self.log("✅ Сравнение завершено успешно", "OK")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log(line.strip(), "INFO")
            else:
                self.log(f"⚠️ Ошибка при сравнении (код {result.returncode})", "WARN")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        self.log(line.strip(), "ERROR")

        except Exception as e:
            self.log(f"❌ Ошибка запуска сравнения: {e}", "ERROR")

        self.log("=" * 60, "SYSTEM")

    def open_dashboard(self):
        """Открытие дашборда"""
        self.log("📈 ОТКРЫТИЕ ДАШБОРДА", "SYSTEM")

        try:
            dashboard_script = os.path.join(BASE_DIR, "dashboard.py")
            
            if not os.path.exists(dashboard_script):
                self.log(f"❌ Файл не найден: {dashboard_script}", "ERROR")
                return

            if sys.platform == "win32":
                subprocess.Popen(["start", "streamlit", "run", dashboard_script], shell=True, cwd=BASE_DIR)
            else:
                subprocess.Popen(["streamlit", "run", dashboard_script], cwd=BASE_DIR)
            self.log("✅ Дашборд открыт", "OK")
        except Exception as e:
            self.log(f"❌ Ошибка открытия дашборда: {e}", "ERROR")


def main():
    if sys.platform == "win32":
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        except:
            pass
    
    root = tk.Tk()
    app = UAVControlApp(root)

    def on_closing():
        if app.btn_stop["state"] != tk.DISABLED:
            app.stop_flight()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()