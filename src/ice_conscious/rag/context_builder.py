from __future__ import annotations
from typing import Iterable, List


class RAGContextBuilder:
    """
    Costruisce un contesto testuale a partire dagli hit del retrieval.
    Dominio cognitivo puro.
    """

    def build(self, hits: Iterable[object]) -> str:
        lines: List[str] = []

        for idx, h in enumerate(hits, start=1):
            entity = getattr(h, "entity", None)
            name = getattr(entity, "name", None) if entity else None
            desc = getattr(entity, "description", None) if entity else None

            line = f"[{idx}] {name or 'entity'}"
            if desc:
                line += f": {desc}"
            lines.append(line)

        return "\n".join(lines) if lines else "(no context)"
