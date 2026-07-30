from app.models.schemas import SourceChunk

REFUSAL_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that question confidently."
)

SYSTEM_PROMPT = (
    "You are a support assistant that answers questions using ONLY the provided context excerpts. "
    "If the context does not contain the answer, say so plainly instead of guessing or using "
    "outside knowledge. Always be concise. When you use a fact from an excerpt, you may refer to "
    "its source label in parentheses."
)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(question: str, sources: list[SourceChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[Source {i + 1} | {source.filename} | {source.locator}]\n{source.text}" for i, source in enumerate(sources)
    )
    return (
        f"Context excerpts:\n{context_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If it does not answer the question, say you don't know."
    )
