@echo off
chcp 65001 > nul
echo ========================================
echo Активация окружения uav_xai...
echo ========================================
call C:\Users\CP\anaconda3\Scripts\activate.bat uav_xai
echo.
echo ========================================
echo Запуск полёта
echo ========================================
echo.
echo [ИНСТРУКЦИЯ]
echo   Для остановки нажмите Ctrl+C
echo   Логи сохранятся автоматически
echo.
echo ========================================
python flight_monitor.py
pause