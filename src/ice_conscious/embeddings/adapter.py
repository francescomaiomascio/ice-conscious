from __future__ import annotations

from typing import Protocol, Sequence
from .models import EmbeddingResult, EmbeddingVector


class EmbeddingAdapter(Protocol):
    """
    Contratto cognitivo per produrre embeddings.

    NON specifica:
    - come viene calcolato
    - dove viene salvato
    """

    model_name: str

    def embed_one(self, text: str) -> EmbeddingResult:
        ...

    def embed_many(self, texts: Sequence[str]) -> Sequence[EmbeddingResult]:
        ...
