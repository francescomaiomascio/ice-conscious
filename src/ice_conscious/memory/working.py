from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta


# ============================================================
# WORKING MEMORY ITEM
# ============================================================

@dataclass
class WorkingMemoryItem:
    """
    Unità atomica di memoria di lavoro.

    Esiste SOLO nel contesto attivo:
    - una sessione
    - un task
    - una finestra di attenzione
    """

    item_id: str
    kind: str                 # entity, concept, query, plan_step, code_chunk, ...
    content: Any

    relevance: float = 1.0
    confidence: float = 1.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None

    ttl: Optional[timedelta] = None   # tempo di vita cognitivo

    def touch(self) -> None:
        self.last_accessed_at = datetime.utcnow()

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.ttl is None:
            return False
        now = now or datetime.utcnow()
        return self.created_at + self.ttl < now


# ============================================================
# WORKING MEMORY
# ============================================================

@dataclass
class WorkingMemory:
    """
    Memoria di lavoro del sistema.

    Contiene SOLO ciò che è:
    - rilevante ora
    - utile per la decisione corrente
    - temporaneo per definizione

    Non è persistente.
    Non è storica.
    """

    items: Dict[str, WorkingMemoryItem] = field(default_factory=dict)

    # limiti cognitivi (soft)
    max_items: int = 128
    min_relevance: float = 0.1

    # ----------------------------------------------------------
    # INSERIMENTO
    # ----------------------------------------------------------

    def add(self, item: WorkingMemoryItem) -> None:
        """
        Inserisce un item nel contesto attivo.
        """
        self.items[item.item_id] = item
        self._enforce_limits()

    def upsert(
        self,
        *,
        item_id: str,
        kind: str,
        content: Any,
        relevance: float = 1.0,
        confidence: float = 1.0,
        ttl: Optional[timedelta] = None,
    ) -> WorkingMemoryItem:
        """
        Inserisce o aggiorna un item.
        """
        if item_id in self.items:
            item = self.items[item_id]
            item.content = content
            item.relevance = relevance
            item.confidence = confidence
            item.ttl = ttl
            item.touch()
        else:
            item = WorkingMemoryItem(
                item_id=item_id,
                kind=kind,
                content=content,
                relevance=relevance,
                confidence=confidence,
                ttl=ttl,
            )
            self.items[item_id] = item

        self._enforce_limits()
        return item

    # ----------------------------------------------------------
    # ACCESSO
    # ----------------------------------------------------------

    def get(self, item_id: str) -> Optional[WorkingMemoryItem]:
        item = self.items.get(item_id)
        if item:
            item.touch()
        return item

    def all(self) -> List[WorkingMemoryItem]:
        return list(self.items.values())

    def by_kind(self, kind: str) -> List[WorkingMemoryItem]:
        return [i for i in self.items.values() if i.kind == kind]

    # ----------------------------------------------------------
    # PULIZIA COGNITIVA
    # ----------------------------------------------------------

    def prune(self, now: Optional[datetime] = None) -> None:
        """
        Rimuove item:
        - scaduti
        - con rilevanza troppo bassa
        """
        now = now or datetime.utcnow()
        to_remove: List[str] = []

        for item_id, item in self.items.items():
            if item.is_expired(now):
                to_remove.append(item_id)
            elif item.relevance < self.min_relevance:
                to_remove.append(item_id)

        for item_id in to_remove:
            self.items.pop(item_id, None)

    def clear(self) -> None:
        """
        Reset completo del contesto attivo.
        """
        self.items.clear()

    # ----------------------------------------------------------
    # FOCUS & ATTENZIONE
    # ----------------------------------------------------------

    def focus(self, top_k: int = 10) -> List[WorkingMemoryItem]:
        """
        Restituisce gli item più rilevanti,
        simulando il focus attentivo.
        """
        ordered = sorted(
            self.items.values(),
            key=lambda i: (i.relevance, i.confidence),
            reverse=True,
        )
        return ordered[:top_k]

    # ----------------------------------------------------------
    # INTERNAL
    # ----------------------------------------------------------

    def _enforce_limits(self) -> None:
        """
        Mantiene la memoria entro i limiti cognitivi.
        """
        if len(self.items) <= self.max_items:
            return

        # ordina per utilità cognitiva
        ordered = sorted(
            self.items.values(),
            key=lambda i: (i.relevance, i.confidence),
            reverse=True,
        )

        # tieni solo i più rilevanti
        keep = ordered[: self.max_items]
        self.items = {i.item_id: i for i in keep}
