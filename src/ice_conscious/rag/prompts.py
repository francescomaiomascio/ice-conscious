from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


# ============================================================
# PROMPT TEMPLATE BASE
# ============================================================

@dataclass
class PromptTemplate:
    """
    Template di prompt cognitivo.

    NON è un prompt tecnico per il modello,
    ma una rappresentazione semantica dell'intento.
    """
    name: str
    system: str
    user: str
    constraints: Optional[str] = None

    def render(
        self,
        *,
        query: str,
        context: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Renderizza il prompt finale pronto per l'LLM.
        """
        parts = []

        if context:
            parts.append("CONTEXT:\n" + context.strip())

        parts.append("QUESTION:\n" + query.strip())

        if extra_instructions:
            parts.append("INSTRUCTIONS:\n" + extra_instructions.strip())

        user_prompt = "\n\n".join(parts)

        return {
            "system": self.system.strip(),
            "user": user_prompt.strip(),
        }


# ============================================================
# PROMPT RAG STANDARD
# ============================================================

RAG_BASE_PROMPT = PromptTemplate(
    name="rag.base",
    system="""
You are a reasoning assistant with access to a retrieved knowledge context.

Your task is NOT to invent information.
Your task is to:
- reason over the provided context
- explain uncertainties explicitly
- answer concisely and accurately

If the context is insufficient, say so clearly.
""",
    user="""
Answer the question using ONLY the provided context.
""",
    constraints="""
- Do not hallucinate.
- Prefer stating uncertainty over guessing.
- Keep the answer grounded in the context.
""",
)


# ============================================================
# PROMPT RAG ANALITICO
# ============================================================

RAG_ANALYTICAL_PROMPT = PromptTemplate(
    name="rag.analytical",
    system="""
You are an analytical reasoning agent.

You must:
- decompose the problem
- reason step by step internally
- produce a clear final answer

Hidden reasoning is allowed internally,
but the final answer must be concise.
""",
    user="""
Use the context to analyze the question.
""",
    constraints="""
- Explicitly mention missing or weak evidence.
- Avoid speculative language.
""",
)


# ============================================================
# PROMPT RAG EXPLAINABLE
# ============================================================

RAG_EXPLAINABLE_PROMPT = PromptTemplate(
    name="rag.explainable",
    system="""
You are an explainability-oriented assistant.

Your goal is not only to answer,
but to explain WHY the answer follows from the context.
""",
    user="""
Answer the question and justify your answer using the context.
""",
    constraints="""
- Cite which parts of the context support each claim.
- If multiple interpretations exist, mention them.
""",
)


# ============================================================
# REGISTRO TEMPLATE
# ============================================================

PROMPT_REGISTRY: Dict[str, PromptTemplate] = {
    RAG_BASE_PROMPT.name: RAG_BASE_PROMPT,
    RAG_ANALYTICAL_PROMPT.name: RAG_ANALYTICAL_PROMPT,
    RAG_EXPLAINABLE_PROMPT.name: RAG_EXPLAINABLE_PROMPT,
}


def get_prompt_template(name: str = "rag.base") -> PromptTemplate:
    """
    Recupera un prompt template dal registro.
    """
    if name not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt template: {name}")
    return PROMPT_REGISTRY[name]
