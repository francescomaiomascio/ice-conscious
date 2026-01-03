from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .filters import KnowledgeFilter
from .scoring import RankingConfig


@dataclass
class KnowledgeSearchQuery:
    """
    Query semantica sul Knowledge domain.

    NON esegue nulla.
    Descrive cosa si vuole cercare e con quali criteri cognitivi.
    """
    workspace_id: str

    # contenuto della query
    text: Optional[str] = None

    # filtri strutturati (tipo entità, tag, semantica, ecc.)
    filters: List[KnowledgeFilter] = field(default_factory=list)

    # strategie di ricerca
    use_vector: bool = True
    use_keyword: bool = True

    # limiti e ranking
    top_k: int = 10
    ranking: RankingConfig = field(default_factory=RankingConfig)
