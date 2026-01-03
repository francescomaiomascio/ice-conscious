from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class CodeArtifactType(str, Enum):
    FILE = "file"
    MODULE = "module"
    FUNCTION = "function"
    CLASS = "class"


@dataclass
class CodeArtifact:
    """
    Rappresentazione cognitiva di un artefatto di codice.

    NON è AST.
    NON è filesystem.
    """

    artifact_id: str
    artifact_type: CodeArtifactType

    name: str
    path: Optional[str]

    language: Optional[str] = None
    summary: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    confidence: float = 1.0
    relevance: float = 1.0
