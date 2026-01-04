from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ice_conscious.embeddings.adapter import UnifiedEmbeddingAdapter
from ice_engine.storage.backends.vector.base import VectorBackend
from ice_engine.storage.base import StorageBackend


class RAGStorageAdapter:
    """
    Adapter di storage per il dominio RAG (ice_conscious).

    Responsabilità:
    - persistenza documenti + metadata
    - gestione embeddings
    - interazione con vector backend
    - retrieval raw (NO ranking, NO intent, NO scoring)

    Non prende decisioni semantiche.
    """

    def __init__(
        self,
        relational_backend: StorageBackend,
        embeddings: UnifiedEmbeddingAdapter,
        vector_backend: Optional[VectorBackend] = None,
        workspace_id: str = "default",
    ):
        self.rel = relational_backend
        self.vec = vector_backend
        self.embed = embeddings
        self.workspace_id = workspace_id

    # ------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Indicizza un documento testuale.

        - calcola embedding
        - salva su relational
        - salva su vector backend (se presente)
        """
        metadata = metadata or {}
        metadata["workspace_id"] = self.workspace_id

        embedding = self.embed.embed_one(text)

        self._store_relational(
            doc_id=doc_id,
            text=text,
            embedding=embedding.vector,
            dim=embedding.dim,
            metadata=metadata,
        )

        if self.vec:
            self.vec.add_embedding(
                id=doc_id,
                embedding=embedding.vector,
                text=text,
                metadata=metadata,
            )

    def ingest_file(
        self,
        path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        text = path.read_text(errors="ignore")
        doc_id = path.as_posix()

        meta = metadata or {}
        meta["path"] = doc_id

        self.ingest_text(doc_id, text, meta)
        return doc_id

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Ricerca per similarità vettoriale.

        Ritorna documenti grezzi.
        Nessun ranking cognitivo.
        """
        if not self.vec:
            return []

        q = self.embed.embed_one(query)

        results = self.vec.search_similar(
            q.vector,
            top_k=top_k,
            filter_metadata={"workspace_id": self.workspace_id},
        )

        return self._hydrate_results(results)

    def fetch_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self.rel.fetch_one(
            """
            SELECT content_text, content_metadata
            FROM knowledge_embeddings
            WHERE embedding_id = ?
            """,
            (doc_id,),
        )
        if not row:
            return None

        return {
            "doc_id": doc_id,
            "text": row["content_text"],
            "metadata": json.loads(row["content_metadata"] or "{}"),
        }

    # ------------------------------------------------------------------
    # DELETE / MAINTENANCE
    # ------------------------------------------------------------------

    def delete(self, doc_id: str) -> None:
        self.rel.execute(
            "DELETE FROM knowledge_embeddings WHERE embedding_id = ?",
            (doc_id,),
        )
        if hasattr(self.rel, "commit"):
            self.rel.commit()

        if self.vec:
            self.vec.delete(doc_id)

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _store_relational(
        self,
        doc_id: str,
        text: str,
        embedding: List[float],
        dim: int,
        metadata: Dict[str, Any],
    ) -> None:
        self.rel.execute(
            """
            INSERT OR REPLACE INTO knowledge_embeddings (
                embedding_id,
                workspace_id,
                content_type,
                content_text,
                embedding_vector,
                embedding_dimensions,
                content_metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                self.workspace_id,
                metadata.get("content_type", "text"),
                text,
                self._pack_vector(embedding),
                dim,
                json.dumps(metadata),
            ),
        )

        if hasattr(self.rel, "commit"):
            self.rel.commit()

    def _hydrate_results(self, results) -> List[Dict[str, Any]]:
        hydrated = []

        for r in results:
            row = self.rel.fetch_one(
                """
                SELECT content_text, content_metadata
                FROM knowledge_embeddings
                WHERE embedding_id = ?
                """,
                (r.id,),
            )
            if not row:
                continue

            hydrated.append(
                {
                    "doc_id": r.id,
                    "score": r.score,
                    "distance": getattr(r, "distance", None),
                    "text": row["content_text"],
                    "metadata": json.loads(row["content_metadata"] or "{}"),
                }
            )

        return hydrated

    @staticmethod
    def _pack_vector(vec: List[float]) -> bytes:
        import struct
        return struct.pack(f"{len(vec)}f", *vec)
