from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ============================================================
# MODELLI
# ============================================================

@dataclass
class SeverityScore:
    """
    Valutazione finale di severità cognitiva.
    """
    severity: float            # 0.0 → 1.0
    level: str                 # info, warning, critical
    explanation: Optional[str] = None


# ============================================================
# LOGICA
# ============================================================

def compute_severity(
    *,
    relevance: float,
    confidence: float,
    anomaly_severity: float = 0.0,
) -> SeverityScore:
    """
    Combina segnali cognitivi in una severità interpretabile.
    """
    score = relevance * 0.5
    score += (1.0 - confidence) * 0.3
    score += anomaly_severity * 0.7

    score = min(max(score, 0.0), 1.0)

    if score >= 0.75:
        level = "critical"
    elif score >= 0.4:
        level = "warning"
    else:
        level = "info"

    return SeverityScore(
        severity=score,
        level=level,
        explanation=(
            f"relevance={relevance:.2f}, "
            f"confidence={confidence:.2f}, "
            f"anomaly={anomaly_severity:.2f}"
        ),
    )
