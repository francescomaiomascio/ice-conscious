from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from datetime import datetime


class SystemHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


@dataclass
class SystemAwareness:
    """
    Stato percepito del sistema dall'interno.

    NON è monitoring tecnico.
    """

    timestamp: datetime

    health: SystemHealth
    load: float                     # 0.0 - 1.0
    cognitive_pressure: float       # saturazione awareness

    active_workspaces: int
    active_tasks: int

    notes: Optional[Dict[str, str]] = None
