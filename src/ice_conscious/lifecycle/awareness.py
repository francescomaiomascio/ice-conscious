from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from ice_conscious.memory.working import WorkingMemory
from ice_conscious.memory.contracts import MemoryRecord


# ============================================================
# AWARENESS STATE
# ============================================================

class AwarenessState(str, Enum):
    """
    Stati del ciclo di consapevolezza.
    """

    DORMANT = "dormant"        # nessun contesto attivo
    FOCUSING = "focusing"      # costruzione contesto
    ACTIVE = "active"          # decisione / ragionamento
    SATURATED = "saturated"    # overload cognitivo
    COOLING = "cooling"        # rilascio contesto
    TERMINATED = "terminated"  # fine ciclo


# ============================================================
# AWARENESS SNAPSHOT
# ============================================================

@dataclass
class AwarenessSnapshot:
    """
    Fotografia dello stato cognitivo in un istante.
    """

    state: AwarenessState
    timestamp: datetime

    focus_items: int
    total_items: int

    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# AWARENESS CORE
# ============================================================

class Awareness:
    """
    Ciclo di consapevolezza cognitiva.

    NON è un orchestrator.
    NON prende decisioni operative.
    NON invoca agenti.

    Definisce:
    - quando esiste un contesto
    - quando è attivo
    - quando è saturo
    - quando deve dissolversi
    """

    def __init__(
        self,
        *,
        working_memory: WorkingMemory,
        max_focus_items: int = 10,
        saturation_threshold: int = 128,
    ) -> None:
        self._wm = working_memory
        self._max_focus = max_focus_items
        self._saturation_threshold = saturation_threshold

        self._state: AwarenessState = AwarenessState.DORMANT
        self._created_at: datetime = datetime.utcnow()
        self._last_transition: datetime = self._created_at

    # ----------------------------------------------------------
    # STATE
    # ----------------------------------------------------------

    @property
    def state(self) -> AwarenessState:
        return self._state

    def _transition(self, new_state: AwarenessState) -> None:
        self._state = new_state
        self._last_transition = datetime.utcnow()

    # ----------------------------------------------------------
    # LIFECYCLE
    # ----------------------------------------------------------

    def awaken(self) -> None:
        """
        Inizio del ciclo cognitivo.
        """
        if self._state != AwarenessState.DORMANT:
            return
        self._transition(AwarenessState.FOCUSING)

    def activate(self) -> None:
        """
        Contesto pronto per ragionare.
        """
        if self._state not in (AwarenessState.FOCUSING, AwarenessState.COOLING):
            return
        self._transition(AwarenessState.ACTIVE)

    def cool_down(self) -> None:
        """
        Rilascio progressivo del contesto.
        """
        if self._state != AwarenessState.ACTIVE:
            return
        self._transition(AwarenessState.COOLING)

    def terminate(self) -> None:
        """
        Fine del ciclo di consapevolezza.
        """
        self._transition(AwarenessState.TERMINATED)
        self._wm.clear()

    # ----------------------------------------------------------
    # COGNITIVE CHECKS
    # ----------------------------------------------------------

    def assess(self) -> None:
        """
        Valuta lo stato cognitivo corrente e aggiorna lo stato.
        """
        total = len(self._wm.items)

        if total == 0:
            self._transition(AwarenessState.DORMANT)
            return

        if total > self._saturation_threshold:
            self._transition(AwarenessState.SATURATED)
            return

        if self._state == AwarenessState.FOCUSING and total > 0:
            self._transition(AwarenessState.ACTIVE)

    # ----------------------------------------------------------
    # FOCUS
    # ----------------------------------------------------------

    def focus(self) -> list[MemoryRecord | Any]:
        """
        Restituisce il contenuto cognitivo attualmente in focus.
        """
        items = self._wm.focus(self._max_focus)
        return [i.content for i in items]

    # ----------------------------------------------------------
    # SNAPSHOT
    # ----------------------------------------------------------

    def snapshot(self) -> AwarenessSnapshot:
        """
        Stato osservabile della consapevolezza.
        """
        return AwarenessSnapshot(
            state=self._state,
            timestamp=datetime.utcnow(),
            focus_items=len(self._wm.focus(self._max_focus)),
            total_items=len(self._wm.items),
            metadata={
                "created_at": self._created_at.isoformat(),
                "last_transition": self._last_transition.isoformat(),
            },
        )
