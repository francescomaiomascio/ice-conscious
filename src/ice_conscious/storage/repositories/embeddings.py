from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Iterable, Dict, Any, List
from datetime import datetime


# ============================================================================
# COGNITIVE MODEL
# ============================================================================

@dataclass
class EmbeddingRecord:
    """
    Rappresentazione cognitiva di un embedding.

    NON è un vettore tecnico.
    NON è legato a FAISS / Chroma / DB.
    È un artefatto semantico:
    - derivato da un testo
    - associato a un contesto
    - potenzialmente collegato a una entità
    """

    embedding_id: str
    workspace_id: str

    text: str
    entity_id: Optional[str] = None

    metadata: Dict[str, Any] | None = None

    # informazioni descrittive (non operative)
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None

    created_at: Optional[datetime] = None


# ============================================================================
# REPOSITORY CONTRACT
# ============================================================================

class EmbeddingRepository(Protocol):
    """
    Contratto astratto per repository di embedding.

    Questo livello:
    - NON sa dove vivono i vettori
    - NON sa come vengono indicizzati
    - NON fa similarity search

    Gestisce SOLO la memoria semantica degli embedding.
    """

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------

    def save(self, record: EmbeddingRecord) -> EmbeddingRecord:
        """
        Registra un embedding semantico.

        Deve essere idempotente rispetto a embedding_id.
        """
        ...

    def delete(self, embedding_id: str) -> None:
        """
        Rimuove un embedding dalla memoria semantica.
        """
        ...

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def get(self, embedding_id: str) -> Optional[EmbeddingRecord]:
        """
        Recupera un embedding per ID.
        """
        ...

    def list_by_workspace(self, workspace_id: str) -> List[EmbeddingRecord]:
        """
        Tutti gli embedding di un workspace.
        """
        ...

    def list_by_entity(
        self,
        workspace_id: str,
        entity_id: str,
    ) -> List[EmbeddingRecord]:
        """
        Embedding associati a una specifica entità.
        """
        ...

    # ------------------------------------------------------------------
    # INTROSPECTION
    # ------------------------------------------------------------------

    def exists(self, embedding_id: str) -> bool:
        """
        Verifica esistenza embedding.
        """
        ...

    def count(self, workspace_id: Optional[str] = None) -> int:
        """
        Conta embedding (globalmente o per workspace).
        """
        ...
