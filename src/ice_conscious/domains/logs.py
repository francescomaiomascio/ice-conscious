from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class LogSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CognitiveLogEvent:
    """
    Evento log visto dalla coscienza, non dal sistema.

    Non è un record tecnico.
    È un segnale semantico.
    """

    log_id: str
    timestamp: datetime
    severity: LogSeverity

    message: str
    source: Optional[str] = None        # agent, runtime, user, system
    context: Optional[Dict[str, Any]] = None

    confidence: float = 1.0
    relevance: float = 0.5
