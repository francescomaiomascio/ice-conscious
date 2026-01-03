# ice_conscious/knowledge/entities.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime


@dataclass
class KnowledgeEntity:
    """
    Entità di conoscenza pura.

    NON conosce:
    - database
    - workspace
    - backend
    - storage
    """

    # Identità semantica
    entity_id: str
    entity_type: str
    name: str

    # Contenuto
    description: Optional[str] = None
    properties: Dict[str, object] = field(default_factory=dict)

    # Semantica estesa
    metadata: Dict[str, object] = field(default_factory=dict)

    # Scoring cognitivo
    confidence_score: float = 1.0
    centrality_score: Optional[float] = None
    connection_count: int = 0
    verified_count: int = 0

    # Tempo cognitivo
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
