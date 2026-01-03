from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Iterable
from collections import defaultdict


# ============================================================
# MODELLI
# ============================================================

@dataclass
class Cluster:
    """
    Cluster concettuale leggero.
    """
    cluster_id: str
    label: str
    items: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# LOGICA
# ============================================================

def cluster_by_entity_type(
    items: Iterable[Dict[str, Any]],
    *,
    type_field: str = "entity_type",
) -> List[Cluster]:
    """
    Clustering deterministico per tipo entità.

    Usato per:
    - raggruppare concetti
    - semplificare vista cognitiva
    """
    buckets: Dict[str, List[Any]] = defaultdict(list)

    for item in items:
        et = (item.get(type_field) or "unknown").lower()
        buckets[et].append(item)

    clusters: List[Cluster] = []
    for idx, (etype, values) in enumerate(buckets.items(), start=1):
        clusters.append(
            Cluster(
                cluster_id=f"cluster_{idx}",
                label=etype,
                items=values,
                metadata={"count": len(values)},
            )
        )

    return clusters


def cluster_by_relevance_band(
    items: Iterable[Dict[str, Any]],
    *,
    score_field: str = "score",
) -> List[Cluster]:
    """
    Clustering per bande di rilevanza.
    """
    bands = {
        "high": [],
        "medium": [],
        "low": [],
    }

    for item in items:
        score = item.get(score_field, 0.0)
        if score >= 0.75:
            bands["high"].append(item)
        elif score >= 0.4:
            bands["medium"].append(item)
        else:
            bands["low"].append(item)

    clusters: List[Cluster] = []
    for label, values in bands.items():
        if not values:
            continue
        clusters.append(
            Cluster(
                cluster_id=f"cluster_{label}",
                label=label,
                items=values,
                metadata={"count": len(values)},
            )
        )

    return clusters
