from app.llm.base import LLMProvider
from app.llm.prompts import REFUSAL_MESSAGE, build_system_prompt, build_user_prompt
from app.models.schemas import QueryResponse
from app.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMProvider) -> None:
        self._retriever = retriever
        self._llm = llm

    def answer(self, tenant_id: str, question: str, top_k: int | None = None) -> QueryResponse:
        result = self._retriever.retrieve(tenant_id, question, top_k)

        if not result.is_grounded:
            return QueryResponse(
                tenant_id=tenant_id,
                question=question,
                answer=REFUSAL_MESSAGE,
                grounded=False,
                sources=[],
                refusal_reason="top match score fell below the similarity threshold",
            )

        answer_text = self._llm.generate(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(question, result.matches),
        )
        return QueryResponse(
            tenant_id=tenant_id,
            question=question,
            answer=answer_text,
            grounded=True,
            sources=result.matches,
        )
