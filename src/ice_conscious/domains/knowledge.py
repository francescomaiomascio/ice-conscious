from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class KnowledgeScope(str, Enum):
    """
    Ambito della conoscenza.
    """
    USER = "user"           # conoscenza prodotta dall'utente
    PROJECT = "project"     # conoscenza specifica del workspace
    SYSTEM = "system"       # conoscenza interna ICE
    GLOBAL = "global"       # conoscenza condivisa / documentale


class KnowledgeKind(str, Enum):
    """
    Natura semantica della conoscenza.
    """
    FACT = "fact"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    DECISION = "decision"
    OBSERVATION = "observation"
    ASSUMPTION = "assumption"


@dataclass
class KnowledgeItem:
    """
    Unità cognitiva di conoscenza.

    NON è un'entità DB.
    NON è un embedding.
    NON è una view.

    È ciò che la coscienza *ritiene vero o utile*.
    """

    knowledge_id: str

    scope: KnowledgeScope
    kind: KnowledgeKind

    title: str
    content: Any

    source: Optional[str] = None          # agent, user, system, ingest
    related_entity_id: Optional[str] = None

    confidence: float = 1.0               # fiducia cognitiva
    relevance: float = 1.0                # rilevanza percepita

    created_at: datetime = datetime.utcnow()
    updated_at: Optional[datetime] = None

    metadata: Optional[Dict[str, Any]] = None
