from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURAZIONE SCORING COGNITIVO
# ============================================================

@dataclass
class KnowledgeScoringConfig:
    """
    Configurazione dello scoring cognitivo.

    Qui NON stiamo valutando similarità numerica pura,
    ma rilevanza percepita dalla coscienza.
    """

    # pesi base
    relevance_weight: float = 1.0
    confidence_weight: float = 0.5

    # bonus contestuali
    boost_code: float = 1.2
    boost_log: float = 1.1
    boost_doc: float = 1.0

    # penalità
    penalty_low_confidence: float = 0.7
    penalty_sparse_context: float = 0.85


# ============================================================
# SCORE MODEL
# ============================================================

@dataclass
class KnowledgeScore:
    """
    Risultato dettagliato di uno scoring.
    """
    final_score: float

    relevance: float
    confidence: float

    boosts: Dict[str, float]
    penalties: Dict[str, float]

    explanation: Optional[str] = None


# ============================================================
# LOGICA DI SCORING
# ============================================================

def score_entity(
    *,
    entity_type: str,
    base_relevance: float,
    confidence: float,
    context_density: Optional[float] = None,
    cfg: KnowledgeScoringConfig | None = None,
) -> KnowledgeScore:
    """
    Calcola uno score cognitivo finale per una entità.

    Parametri:
    - base_relevance: score grezzo (vector / keyword / inference)
    - confidence: quanto il sistema si fida di questa entità
    - context_density: quanto è connessa nel contesto attuale (0-1)
    """
    cfg = cfg or KnowledgeScoringConfig()

    boosts: Dict[str, float] = {}
    penalties: Dict[str, float] = {}

    # ----------------------------------------------------------
    # Base
    # ----------------------------------------------------------

    score = base_relevance * cfg.relevance_weight
    score *= confidence * cfg.confidence_weight + 1.0

    # ----------------------------------------------------------
    # Boost per tipo entità
    # ----------------------------------------------------------

    et = (entity_type or "").lower()

    if "code" in et:
        score *= cfg.boost_code
        boosts["type:code"] = cfg.boost_code
    elif "log" in et:
        score *= cfg.boost_log
        boosts["type:log"] = cfg.boost_log
    elif "doc" in et or "documentation" in et:
        score *= cfg.boost_doc
        boosts["type:doc"] = cfg.boost_doc

    # ----------------------------------------------------------
    # Penalità
    # ----------------------------------------------------------

    if confidence < 0.5:
        score *= cfg.penalty_low_confidence
        penalties["low_confidence"] = cfg.penalty_low_confidence

    if context_density is not None and context_density < 0.3:
        score *= cfg.penalty_sparse_context
        penalties["sparse_context"] = cfg.penalty_sparse_context

    # ----------------------------------------------------------
    # Normalizzazione minima
    # ----------------------------------------------------------

    score = max(score, 0.0)

    explanation = (
        f"base={base_relevance:.3f}, "
        f"confidence={confidence:.2f}, "
        f"boosts={list(boosts.keys())}, "
        f"penalties={list(penalties.keys())}"
    )

    return KnowledgeScore(
        final_score=score,
        relevance=base_relevance,
        confidence=confidence,
        boosts=boosts,
        penalties=penalties,
        explanation=explanation,
    )


# ============================================================
# UTILS DI BATCH
# ============================================================

def score_entities(
    items: List[Dict[str, Any]],
    cfg: KnowledgeScoringConfig | None = None,
) -> List[KnowledgeScore]:
    """
    Utility per applicare scoring su una lista generica di elementi.

    Ogni item DEVE contenere:
    - entity_type
    - base_relevance
    - confidence
    """
    results: List[KnowledgeScore] = []

    for item in items:
        results.append(
            score_entity(
                entity_type=item.get("entity_type", ""),
                base_relevance=item.get("base_relevance", 0.0),
                confidence=item.get("confidence", 1.0),
                context_density=item.get("context_density"),
                cfg=cfg,
            )
        )

    return results
