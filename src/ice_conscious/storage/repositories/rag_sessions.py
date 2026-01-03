from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List
from datetime import datetime


# ============================================================================
# COGNITIVE MODEL
# ============================================================================

@dataclass
class RAGSessionRecord:
    """
    Rappresentazione cognitiva di una sessione RAG.

    NON è una sessione runtime.
    NON è una chat.
    NON è un trace LLM.

    È la memoria strutturata di:
    - una domanda
    - un contesto costruito
    - un insieme di evidenze recuperate
    - un risultato prodotto
    """

    session_id: str
    workspace_id: str

    query_text: str

    # contesto cognitivo
    context_text: Optional[str] = None
    context_metadata: Optional[Dict[str, Any]] = None

    # evidenze (embedding / entità / fonti)
    retrieved_items: Optional[List[Dict[str, Any]]] = None

    # risultato semantico (non LLM-specific)
    answer: Optional[str] = None
    confidence: Optional[float] = None

    # metriche cognitive (non performance)
    relevance_score: Optional[float] = None

    created_at: Optional[datetime] = None


# ============================================================================
# REPOSITORY CONTRACT
# ============================================================================

class RAGSessionRepository(Protocol):
    """
    Contratto astratto per la memoria delle sessioni RAG.

    Questo repository:
    - NON esegue retrieval
    - NON chiama LLM
    - NON costruisce prompt

    Serve solo a ricordare *cosa è successo*,
    in forma interrogabile.
    """

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------

    def save(self, session: RAGSessionRecord) -> RAGSessionRecord:
        """
        Registra una nuova sessione RAG.
        """
        ...

    def update(self, session_id: str, fields: Dict[str, Any]) -> None:
        """
        Aggiorna campi semantici della sessione
        (confidence, relevance, answer, ecc.).
        """
        ...

    def delete(self, session_id: str) -> None:
        """
        Rimuove una sessione RAG dalla memoria.
        """
        ...

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> Optional[RAGSessionRecord]:
        """
        Recupera una sessione RAG per ID.
        """
        ...

    def list_by_workspace(
        self,
        workspace_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[RAGSessionRecord]:
        """
        Tutte le sessioni RAG di un workspace.
        """
        ...

    def list_recent(
        self,
        workspace_id: str,
        *,
        since: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[RAGSessionRecord]:
        """
        Sessioni recenti (memoria a breve termine).
        """
        ...

    # ------------------------------------------------------------------
    # INTROSPECTION
    # ------------------------------------------------------------------

    def exists(self, session_id: str) -> bool:
        """
        Verifica se la sessione esiste.
        """
        ...

    def count(self, workspace_id: Optional[str] = None) -> int:
        """
        Conta sessioni RAG.
        """
        ...
