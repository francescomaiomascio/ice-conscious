from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, Any, List
from datetime import datetime


# ============================================================================
# MEMORY RECORDS (COGNITIVE)
# ============================================================================

@dataclass
class EpisodicMemoryRecord:
    """
    Evento cognitivo persistibile.

    NON è un log tecnico.
    NON è una sessione runtime.
    NON è una trace.

    È un evento significativo per la coscienza:
    - decisione
    - errore rilevante
    - insight
    - azione deliberata
    """

    episode_id: str
    workspace_id: str

    kind: str                   # decision, error, insight, action, observation
    summary: str
    details: Optional[str] = None

    related_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    confidence: float = 1.0
    importance: float = 1.0

    occurred_at: Optional[datetime] = None
    recorded_at: Optional[datetime] = None


@dataclass
class SemanticMemoryRecord:
    """
    Memoria semantica persistente.

    NON è una KnowledgeEntity.
    NON è un embedding.
    NON è un concetto atomico.

    È una *regola*, *astrazione*, *pattern consolidato*.
    """

    memory_id: str
    workspace_id: str

    label: str
    description: Optional[str] = None

    scope: Optional[str] = None          # global, workspace, domain, agent
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    confidence: float = 1.0
    stability: float = 1.0               # quanto è consolidata nel tempo

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# MEMORY REPOSITORY CONTRACT
# ============================================================================

class MemoryRepository(Protocol):
    """
    Repository cognitivo della memoria persistente.

    NON gestisce:
    - working memory
    - contesto runtime
    - TTL
    - eviction

    Gestisce SOLO:
    - memoria episodica
    - memoria semantica
    """

    # ------------------------------------------------------------------
    # EPISODIC MEMORY
    # ------------------------------------------------------------------

    def save_episode(self, record: EpisodicMemoryRecord) -> EpisodicMemoryRecord:
        """
        Registra un evento cognitivo.
        """
        ...

    def get_episode(self, episode_id: str) -> Optional[EpisodicMemoryRecord]:
        """
        Recupera un evento per ID.
        """
        ...

    def list_episodes(
        self,
        workspace_id: str,
        *,
        kind: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[EpisodicMemoryRecord]:
        """
        Elenco eventi cognitivi.
        """
        ...

    def delete_episode(self, episode_id: str) -> None:
        """
        Rimuove un evento.
        """
        ...

    # ------------------------------------------------------------------
    # SEMANTIC MEMORY
    # ------------------------------------------------------------------

    def save_semantic(self, record: SemanticMemoryRecord) -> SemanticMemoryRecord:
        """
        Registra o aggiorna una memoria semantica.
        """
        ...

    def get_semantic(self, memory_id: str) -> Optional[SemanticMemoryRecord]:
        """
        Recupera una memoria semantica.
        """
        ...

    def list_semantic(
        self,
        workspace_id: str,
        *,
        scope: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[SemanticMemoryRecord]:
        """
        Elenco memorie semantiche.
        """
        ...

    def delete_semantic(self, memory_id: str) -> None:
        """
        Rimuove una memoria semantica.
        """
        ...

    # ------------------------------------------------------------------
    # INTROSPECTION
    # ------------------------------------------------------------------

    def count_episodes(self, workspace_id: Optional[str] = None) -> int:
        """
        Conta eventi episodici.
        """
        ...

    def count_semantic(self, workspace_id: Optional[str] = None) -> int:
        """
        Conta memorie semantiche.
        """
        ...
