from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# EVENT KINDS
# ============================================================

class EpisodicEventKind(str, Enum):
    """
    Tipologie semantiche di eventi episodici.

    NON rappresentano componenti tecnici,
    ma ciò che la coscienza percepisce come accadimento.
    """

    RAG_QUERY = "rag.query"
    RAG_SESSION = "rag.session"

    LLM_INTERACTION = "llm.interaction"

    CODE_CHANGE = "code.change"
    FILE_INGEST = "file.ingest"

    PLAN_STEP = "planner.step"
    PLAN_DONE = "planner.done"

    USER_ACTION = "user.action"
    SYSTEM_EVENT = "system.event"


# ============================================================
# EPISODIC EVENT
# ============================================================

@dataclass
class EpisodicEvent:
    """
    Evento atomico nella memoria episodica.

    È il più piccolo frammento di esperienza
    che la coscienza può ricordare.
    """

    event_id: str
    timestamp: datetime

    kind: EpisodicEventKind
    summary: str

    payload: Dict[str, Any] = field(default_factory=dict)

    # attributi cognitivi
    confidence: float = 1.0
    relevance: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "kind": self.kind.value,
            "summary": self.summary,
            "payload": self.payload,
            "confidence": self.confidence,
            "relevance": self.relevance,
        }


# ============================================================
# EPISODIC TRACE
# ============================================================

@dataclass
class EpisodicTrace:
    """
    Sequenza coerente di eventi episodici.

    Una trace rappresenta:
    - un task
    - un workflow
    - una conversazione
    - una sessione cognitiva
    """

    trace_id: str
    events: List[EpisodicEvent] = field(default_factory=list)

    def add_event(self, event: EpisodicEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)

    @property
    def start_time(self) -> Optional[datetime]:
        if not self.events:
            return None
        return self.events[0].timestamp

    @property
    def end_time(self) -> Optional[datetime]:
        if not self.events:
            return None
        return self.events[-1].timestamp

    @property
    def duration_seconds(self) -> Optional[float]:
        if not self.start_time or not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def is_completed(self) -> bool:
        """
        Una trace è considerata completata se contiene
        almeno un evento di tipo PLAN_DONE.
        """
        return any(e.kind == EpisodicEventKind.PLAN_DONE for e in self.events)

    def summarize(self, max_events: int = 5) -> str:
        """
        Sintesi narrativa della trace.
        """
        if not self.events:
            return "empty trace"

        selected = self.events[:max_events]
        parts = [e.summary for e in selected]

        if len(self.events) > max_events:
            parts.append("...")

        return " → ".join(parts)


# ============================================================
# EPISODIC TIMELINE
# ============================================================

@dataclass
class EpisodicTimeline:
    """
    Vista temporale globale della memoria episodica.

    NON è uno storage.
    NON è un repository.
    È una proiezione cognitiva.
    """

    events: List[EpisodicEvent] = field(default_factory=list)

    def ingest(self, events: Iterable[EpisodicEvent]) -> None:
        for e in events:
            self.events.append(e)
        self.events.sort(key=lambda e: e.timestamp)

    def window(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        kinds: Optional[List[EpisodicEventKind]] = None,
    ) -> List[EpisodicEvent]:
        """
        Estrae una finestra cognitiva dalla timeline.
        """
        result: List[EpisodicEvent] = []

        for e in self.events:
            if since and e.timestamp < since:
                continue
            if until and e.timestamp > until:
                continue
            if kinds and e.kind not in kinds:
                continue
            result.append(e)

        return result

    def density(self, window_seconds: Optional[int] = None) -> float:
        """
        Densità di eventi.
        Usata per valutare contesto e carico cognitivo.
        """
        if not self.events:
            return 0.0

        if not window_seconds:
            return float(len(self.events))

        start = self.events[-1].timestamp
        cutoff = start.timestamp() - window_seconds

        count = sum(1 for e in self.events if e.timestamp.timestamp() >= cutoff)
        return count / window_seconds
