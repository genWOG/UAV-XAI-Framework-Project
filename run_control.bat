@echo off
chcp 65001 > nul
echo ========================================
echo 🔄 Активация окружения uav_xai...
echo ========================================

REM Активируем conda
call C:\Users\CP\anaconda3\Scripts\activate.bat uav_xai

REM Проверяем, что окружение активировано
echo ✅ Окружение: %CONDA_DEFAULT_ENV%

REM Проверяем, что виден cosysairsim
python -c "import cosysairsim; print('✅ cosysairsim виден')" || (
    echo ❌ cosysairsim НЕ НАЙДЕН!
    echo Добавьте путь в PYTHONPATH или установите pip install cosysairsim
    pause
    exit /b 1
)

echo ========================================
echo 🚁 Запуск панели управления...
echo ========================================

python control.py
pause