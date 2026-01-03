from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


# ============================================================
# VIEW MODELS (DOMINIO)
# ============================================================

@dataclass
class KnowledgeViewEntity:
    """
    Rappresentazione 'vista' di un'entità di conoscenza.

    NON è un'entità persistente.
    NON è un record DB.
    È ciò che la coscienza decide di mostrare o usare.
    """
    entity_id: str
    entity_type: str
    name: str
    description: Optional[str] = None

    # punteggi cognitivi
    relevance_score: float = 0.0
    confidence_score: float = 1.0

    # attributi opzionali
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeViewContext:
    """
    Contesto cognitivo aggregato.

    Serve per:
    - RAG
    - reasoning
    - spiegazioni
    - prompt assembly
    """
    entities: List[KnowledgeViewEntity] = field(default_factory=list)

    summary: Optional[str] = None
    tokens_estimate: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeViewHit:
    """
    Singolo risultato di vista.

    Collega:
    - entità
    - punteggio
    - origine cognitiva
    """
    entity: KnowledgeViewEntity
    score: float

    source: Literal["vector", "keyword", "hybrid", "inferred"] = "hybrid"

    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeViewResult:
    """
    Risultato finale di una interrogazione di knowledge.

    Questo è ciò che ESCE dal dominio conscious
    verso engine / api / ui.
    """
    hits: List[KnowledgeViewHit] = field(default_factory=list)

    context: Optional[KnowledgeViewContext] = None

    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------
    # Helper cognitivi
    # ----------------------------------------------------------

    def top_entities(self, n: int = 5) -> List[KnowledgeViewEntity]:
        return [h.entity for h in self.hits[:n]]

    def has_results(self) -> bool:
        return bool(self.hits)

    def total_hits(self) -> int:
        return len(self.hits)
