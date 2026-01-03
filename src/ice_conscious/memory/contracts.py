from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional


# ============================================================
# MEMORY RECORD (ASTRAZIONE BASE)
# ============================================================

@dataclass
class MemoryRecord:
    """
    Record cognitivo astratto.

    È la forma minima comune a:
    - memoria episodica
    - memoria semantica
    - memoria di lavoro (quando serializzabile)

    NON implica persistenza.
    NON implica storage.
    """

    record_id: str
    kind: str                  # episodic | semantic | working | custom
    content: Any

    confidence: float = 1.0
    created_at: datetime = datetime.utcnow()


# ============================================================
# MEMORY INTERFACE
# ============================================================

class Memory(ABC):
    """
    Contratto astratto per una memoria cognitiva.

    Questo NON è un repository.
    NON conosce database.
    NON conosce filesystem.

    Definisce SOLO:
    - cosa significa scrivere
    - cosa significa leggere
    - cosa significa dimenticare
    """

    @abstractmethod
    def write(self, record: MemoryRecord) -> None:
        """
        Scrive (o apprende) un record nella memoria.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, record_id: str) -> Optional[MemoryRecord]:
        """
        Recupera un record per id.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, **filters) -> Iterable[MemoryRecord]:
        """
        Query semantica sulla memoria.

        Esempi:
        - kind="episodic"
        - confidence__gt=0.8
        """
        raise NotImplementedError

    @abstractmethod
    def forget(self, record_id: str) -> None:
        """
        Rimuove un record dalla memoria.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Cancella completamente la memoria.
        """
        raise NotImplementedError


# ============================================================
# SPECIALIZZAZIONI (MARKER INTERFACES)
# ============================================================

class EpisodicMemory(Memory):
    """
    Memoria degli eventi.
    """
    pass


class SemanticMemory(Memory):
    """
    Memoria delle conoscenze stabili.
    """
    pass


class WorkingMemoryContract(Memory):
    """
    Contratto formale della memoria di lavoro.

    NOTA:
    - spesso NON viene serializzata
    - può vivere solo in RAM
    """
    pass
