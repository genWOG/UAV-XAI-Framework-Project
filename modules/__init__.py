"""
Основные модули мониторинга полёта БПЛА.

Примечание: AirSimAdapter больше не импортируется автоматически,
т.к. мы перешли на PX4Adapter. Если понадобится AirSim — импортируй
его явно:
    from modules.airsim_adapter import AirSimAdapter
"""

from .log_manager import LogManager
from .obstacle_detector import ObstacleDetector
from .route_planner import RoutePlanner
from .telemetry_parser import TelemetryParser
from .px4_adapter import PX4Adapter

__all__ = [
    "LogManager",
    "ObstacleDetector",
    "RoutePlanner",
    "TelemetryParser",
    "PX4Adapter",
]