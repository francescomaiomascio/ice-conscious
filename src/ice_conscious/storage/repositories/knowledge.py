from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, Any, List
from datetime import datetime


# ============================================================================
# COGNITIVE MODELS
# ============================================================================

@dataclass
class KnowledgeRecord:
    """
    Rappresentazione cognitiva di una entità di conoscenza.

    NON è un record DB.
    NON è un nodo di grafo.
    NON è un embedding.

    È un fatto/contenuto che il sistema considera esistente,
    interrogabile e valutabile.
    """

    entity_id: str
    workspace_id: str

    kind: str                # concept, code, log, document, pattern, rule, ecc.
    name: str
    description: Optional[str] = None

    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    confidence: float = 1.0
    relevance: float = 1.0

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class KnowledgeRelationRecord:
    """
    Relazione cognitiva tra due entità.

    NON è un edge computazionale.
    NON implica traversabilità automatica.

    Serve a esprimere:
    - dipendenze concettuali
    - correlazioni
    - causalità dichiarata
    """

    relation_id: str
    workspace_id: str

    source_id: str
    target_id: str

    relation_type: str       # depends_on, similar_to, causes, explains, ecc.
    strength: float = 1.0
    confidence: float = 1.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: Optional[datetime] = None


# ============================================================================
# REPOSITORY CONTRACT
# ============================================================================

class KnowledgeRepository(Protocol):
    """
    Repository cognitivo della conoscenza.

    NON costruisce grafi.
    NON calcola centralità.
    NON fa embedding.
    NON fa retrieval semantico.

    Espone solo:
    - memoria
    - accesso
    - interrogazione strutturata
    """

    # ------------------------------------------------------------------
    # ENTITY WRITE
    # ------------------------------------------------------------------

    def save_entity(self, record: KnowledgeRecord) -> KnowledgeRecord:
        """
        Registra o aggiorna una entità di conoscenza.
        """
        ...

    def delete_entity(self, entity_id: str) -> None:
        """
        Rimuove una entità di conoscenza.
        """
        ...

    # ------------------------------------------------------------------
    # ENTITY READ
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[KnowledgeRecord]:
        """
        Recupera una entità per ID.
        """
        ...

    def list_entities(
        self,
        workspace_id: str,
        *,
        kind: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[KnowledgeRecord]:
        """
        Elenco entità di un workspace, con filtri semplici.
        """
        ...

    # ------------------------------------------------------------------
    # RELATION WRITE
    # ------------------------------------------------------------------

    def save_relation(self, relation: KnowledgeRelationRecord) -> KnowledgeRelationRecord:
        """
        Registra una relazione cognitiva.
        """
        ...

    def delete_relation(self, relation_id: str) -> None:
        """
        Rimuove una relazione.
        """
        ...

    # ------------------------------------------------------------------
    # RELATION READ
    # ------------------------------------------------------------------

    def list_relations(
        self,
        workspace_id: str,
        *,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[KnowledgeRelationRecord]:
        """
        Elenco relazioni cognitive con filtri.
        """
        ...

    # ------------------------------------------------------------------
    # INTROSPECTION
    # ------------------------------------------------------------------

    def exists_entity(self, entity_id: str) -> bool:
        """
        Verifica esistenza entità.
        """
        ...

    def count_entities(self, workspace_id: Optional[str] = None) -> int:
        """
        Conta entità memorizzate.
        """
        ...

    def count_relations(self, workspace_id: Optional[str] = None) -> int:
        """
        Conta relazioni memorizzate.
        """
        ...
