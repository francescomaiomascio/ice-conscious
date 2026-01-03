from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime


# ============================================================
# MODELLI
# ============================================================

@dataclass
class AnomalySignal:
    """
    Segnale di anomalia cognitiva.

    Non è un errore di sistema.
    È una deviazione dal comportamento atteso.
    """
    kind: str                  # confidence_drop, score_spike, temporal_gap, incoherence
    severity: float            # 0.0 → 1.0
    message: str
    metadata: Dict[str, Any]


@dataclass
class AnomalyReport:
    """
    Report aggregato di anomalie.
    """
    signals: List[AnomalySignal]

    @property
    def is_anomalous(self) -> bool:
        return any(s.severity >= 0.6 for s in self.signals)

    @property
    def max_severity(self) -> float:
        return max((s.severity for s in self.signals), default=0.0)


# ============================================================
# LOGICA
# ============================================================

def detect_anomalies(
    *,
    base_score: float,
    confidence: float,
    previous_score: Optional[float] = None,
    previous_confidence: Optional[float] = None,
    last_seen_at: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> AnomalyReport:
    """
    Rileva anomalie cognitive leggere.

    Nessuna statistica pesante:
    solo segnali interpretabili.
    """
    now = now or datetime.utcnow()
    signals: List[AnomalySignal] = []

    # ----------------------------------------------------------
    # Confidence drop
    # ----------------------------------------------------------
    if previous_confidence is not None:
        delta = previous_confidence - confidence
        if delta > 0.3:
            signals.append(
                AnomalySignal(
                    kind="confidence_drop",
                    severity=min(delta, 1.0),
                    message="Confidence dropped significantly",
                    metadata={"delta": delta},
                )
            )

    # ----------------------------------------------------------
    # Score spike
    # ----------------------------------------------------------
    if previous_score is not None:
        if base_score > previous_score * 2.5:
            signals.append(
                AnomalySignal(
                    kind="score_spike",
                    severity=0.7,
                    message="Unusual relevance spike",
                    metadata={
                        "previous": previous_score,
                        "current": base_score,
                    },
                )
            )

    # ----------------------------------------------------------
    # Temporal gap
    # ----------------------------------------------------------
    if last_seen_at is not None:
        delta_sec = (now - last_seen_at).total_seconds()
        if delta_sec > 3600 * 24:
            signals.append(
                AnomalySignal(
                    kind="temporal_gap",
                    severity=0.4,
                    message="Entity reappeared after long inactivity",
                    metadata={"seconds": delta_sec},
                )
            )

    # ----------------------------------------------------------
    # Low confidence baseline
    # ----------------------------------------------------------
    if confidence < 0.3:
        signals.append(
            AnomalySignal(
                kind="low_confidence",
                severity=0.5,
                message="Persistently low confidence",
                metadata={"confidence": confidence},
            )
        )

    return AnomalyReport(signals=signals)
