from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass(frozen=True)
class RAGSession:
    """
    Rappresenta una singola esperienza RAG.
    NON è persistente.
    """
    session_id: str
    workspace_id: str
    query_text: str

    context_text: str
    hits: List[Dict[str, Any]]

    created_at: datetime
