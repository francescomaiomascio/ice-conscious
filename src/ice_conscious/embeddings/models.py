from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional


@dataclass(frozen=True)
class EmbeddingVector:
    """
    Rappresentazione vettoriale di un significato.

    NON contiene informazioni di storage.
    NON conosce backend.
    """
    values: Sequence[float]
    dim: int


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Risultato cognitivo di una operazione di embedding.
    """
    vector: EmbeddingVector

    # informazioni semantiche
    text: Optional[str] = None
    semantic_hash: Optional[str] = None
    model_name: Optional[str] = None
