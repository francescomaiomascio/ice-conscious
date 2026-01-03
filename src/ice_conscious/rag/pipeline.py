from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
from datetime import datetime
from .context_builder import RAGContextBuilder
from .sessions import RAGSession
from ..knowledge.queries import KnowledgeSearchQuery, KnowledgeSearchResult


@dataclass
class RAGPipelineResult:
    session: RAGSession
    search_result: KnowledgeSearchResult
    context_text: str


class RAGPipeline:
    """
    Pipeline RAG cognitiva pura.

    NON chiama LLM.
    NON salva nulla.
    """

    def __init__(
        self,
        search_service,
        context_builder: Optional[RAGContextBuilder] = None,
    ) -> None:
        self.search_service = search_service
        self.context_builder = context_builder or RAGContextBuilder()

    def run(
        self,
        *,
        workspace_id: str,
        query_text: str,
        filters: Optional[List[Any]] = None,
        top_k: int = 8,
    ) -> RAGPipelineResult:

        ks_query = KnowledgeSearchQuery(
            workspace_id=workspace_id,
            text=query_text,
            filters=filters or [],
            top_k=top_k,
        )

        search_result = self.search_service.search(ks_query)
        context_text = self.context_builder.build(search_result.hits)

        session = RAGSession(
            session_id="rag-session",
            workspace_id=workspace_id,
            query_text=query_text,
            context_text=context_text,
            hits=[
                {
                    "entity_id": h.entity.entity_id,
                    "entity_type": h.entity.entity_type,
                    "name": h.entity.name,
                    "score": h.score,
                    "source": h.source,
                }
                for h in search_result.hits
            ],
            created_at=datetime.utcnow(),
        )

        return RAGPipelineResult(
            session=session,
            search_result=search_result,
            context_text=context_text,
        )
