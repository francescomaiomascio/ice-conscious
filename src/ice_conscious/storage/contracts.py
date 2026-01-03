# ice_conscious/storage/contracts.py

from typing import Protocol, Iterable, Optional, Dict, Any
from ice_conscious.knowledge.entities import KnowledgeEntity


# ============================================================
# KNOWLEDGE
# ============================================================

class KnowledgeStore(Protocol):
    """Persistenza delle entità di conoscenza."""

    def get(self, entity_id: str) -> Optional[KnowledgeEntity]: ...
    def save(self, entity: KnowledgeEntity) -> None: ...
    def list(self, workspace_id: str) -> Iterable[KnowledgeEntity]: ...


class KnowledgeRelationshipStore(Protocol):
    """Relazioni semantiche tra entità."""

    def create(self, data: Dict[str, Any]) -> None: ...
    def for_entity(self, workspace_id: str, entity_id: str) -> Iterable[Dict[str, Any]]: ...
    def by_type(self, workspace_id: str, rel_type: str) -> Iterable[Dict[str, Any]]: ...


# ============================================================
# EMBEDDINGS
# ============================================================

class EmbeddingStore(Protocol):
    """Associazione testo → embedding → riferimento."""

    def save(
        self,
        *,
        embedding_id: str,
        workspace_id: str,
        text: str,
        vector_ref: Optional[str],
        metadata: Optional[dict] = None,
        entity_id: Optional[str] = None,
    ) -> None: ...

    def for_entity(self, entity_id: str) -> Iterable[Dict[str, Any]]: ...
    def get(self, embedding_id: str) -> Optional[Dict[str, Any]]: ...


# ============================================================
# RAG
# ============================================================

class RAGSessionStore(Protocol):
    """Sessioni RAG come artefatti cognitivi."""

    def create(
        self,
        *,
        session_id: str,
        workspace_id: str,
        query_text: str,
        query_intent: Optional[str],
        context_text: str,
        retrieved_embeddings: Optional[list[str]],
        metadata: Optional[dict] = None,
    ) -> None: ...

    def get(self, session_id: str) -> Optional[Dict[str, Any]]: ...
