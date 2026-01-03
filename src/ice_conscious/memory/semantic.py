from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from enum import Enum
from datetime import datetime


# ============================================================
# SEMANTIC TYPES
# ============================================================

class SemanticKind(str, Enum):
    """
    Tipologie semantiche di conoscenza stabile.

    NON descrivono file o record,
    ma categorie cognitive persistenti.
    """

    CONCEPT = "concept"
    FACT = "fact"
    RULE = "rule"
    DEFINITION = "definition"
    PATTERN = "pattern"
    ENTITY = "entity"


# ============================================================
# SEMANTIC MEMORY ITEM
# ============================================================

@dataclass
class SemanticItem:
    """
    Unità atomica di memoria semantica.

    Rappresenta qualcosa che il sistema considera
    *vero*, *stabile* o *riutilizzabile nel tempo*.
    """

    semantic_id: str
    kind: SemanticKind

    name: str
    description: Optional[str] = None

    attributes: Dict[str, Any] = field(default_factory=dict)

    # metriche cognitive
    confidence: float = 1.0
    relevance: float = 1.0

    # metadati temporali (non storici, solo tracciamento)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated_at: Optional[datetime] = None

    def update(
        self,
        *,
        description: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        relevance: Optional[float] = None,
    ) -> None:
        """
        Aggiorna la conoscenza semantica.
        """
        if description is not None:
            self.description = description
        if attributes is not None:
            self.attributes.update(attributes)
        if confidence is not None:
            self.confidence = confidence
        if relevance is not None:
            self.relevance = relevance

        self.last_updated_at = datetime.utcnow()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "kind": self.kind.value,
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat() if self.last_updated_at else None,
        }


# ============================================================
# SEMANTIC MEMORY
# ============================================================

@dataclass
class SemanticMemory:
    """
    Collezione coerente di conoscenza stabile.

    NON è:
    - un database
    - un repository
    - un knowledge graph

    È la memoria di ciò che il sistema *sa*.
    """

    items: Dict[str, SemanticItem] = field(default_factory=dict)

    # ----------------------------------------------------------
    # CRUD COGNITIVO
    # ----------------------------------------------------------

    def add(self, item: SemanticItem) -> None:
        self.items[item.semantic_id] = item

    def get(self, semantic_id: str) -> Optional[SemanticItem]:
        return self.items.get(semantic_id)

    def remove(self, semantic_id: str) -> None:
        self.items.pop(semantic_id, None)

    # ----------------------------------------------------------
    # QUERY SEMANTICHE
    # ----------------------------------------------------------

    def find_by_kind(self, kind: SemanticKind) -> List[SemanticItem]:
        return [i for i in self.items.values() if i.kind == kind]

    def find_by_name(self, name: str) -> List[SemanticItem]:
        q = name.lower()
        return [i for i in self.items.values() if q in i.name.lower()]

    def filter(
        self,
        *,
        min_confidence: Optional[float] = None,
        min_relevance: Optional[float] = None,
    ) -> List[SemanticItem]:
        result: List[SemanticItem] = []

        for item in self.items.values():
            if min_confidence is not None and item.confidence < min_confidence:
                continue
            if min_relevance is not None and item.relevance < min_relevance:
                continue
            result.append(item)

        return result

    # ----------------------------------------------------------
    # CONSOLIDAMENTO
    # ----------------------------------------------------------

    def consolidate(self) -> None:
        """
        Rafforza la memoria semantica rimuovendo
        conoscenza debole o obsoleta.

        Regole semplici, estendibili.
        """
        to_remove: List[str] = []

        for sid, item in self.items.items():
            if item.confidence < 0.2:
                to_remove.append(sid)

        for sid in to_remove:
            del self.items[sid]

    # ----------------------------------------------------------
    # EXPORT
    # ----------------------------------------------------------

    def snapshot(self) -> List[Dict[str, Any]]:
        """
        Snapshot serializzabile della memoria semantica.
        """
        return [i.as_dict() for i in self.items.values()]
