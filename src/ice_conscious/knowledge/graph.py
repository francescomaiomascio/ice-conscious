from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .scoring import RankingConfig


# ============================================================
# FILTER
# ============================================================

@dataclass(frozen=True)
class KnowledgeFilter:
    """
    Filtro semantico applicabile a una query di knowledge.

    È intenzionalmente astratto:
    - NON sa come viene tradotto (SQL, vector, ecc.)
    - descrive solo cosa filtrare
    """
    field: str
    op: str = "eq"   # eq, in, ne, lt, gt, contains, ecc.
    value: Any = None


# ============================================================
# QUERY
# ============================================================

@dataclass
class KnowledgeSearchQuery:
    """
    Query concettuale per interrogare la conoscenza.

    Questa classe NON esegue nulla.
    Descrive soltanto l'intento cognitivo della ricerca.
    """

    # Contesto
    workspace_id: str

    # Contenuto della query
    text: Optional[str] = None

    # Modalità di ricerca
    use_vector: bool = True
    use_keyword: bool = True

    # Filtri strutturati
    filters: List[KnowledgeFilter] = field(default_factory=list)

    # Limiti e ranking
    top_k: int = 10
    ranking: RankingConfig = field(default_factory=RankingConfig)

    # Metadata opzionali (future estensioni)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------
    # Helper semantici (NON esecutivi)
    # ----------------------------------------------------------

    def has_text(self) -> bool:
        """Indica se la query contiene testo ricercabile."""
        return bool(self.text and self.text.strip())

    def has_filters(self) -> bool:
        """Indica se la query contiene filtri strutturati."""
        return bool(self.filters)

    def add_filter(self, field: str, value: Any, op: str = "eq") -> None:
        """Aggiunge un filtro in modo fluente."""
        self.filters.append(KnowledgeFilter(field=field, op=op, value=value))
