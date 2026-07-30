from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider


class BGEEmbeddingProvider(EmbeddingProvider):
    """BAAI/bge-* embeddings. Per BGE's own usage notes: the query instruction prefix is required
    at retrieval time but must NOT be added when embedding documents at indexing time.
    """

    def __init__(self, model_name: str, device: str, query_instruction: str) -> None:
        self._model = SentenceTransformer(model_name, device=device)
        self._query_instruction = query_instruction

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        return self._model.get_max_seq_length()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{self._query_instruction}{text}"
        embedding = self._model.encode([prefixed], normalize_embeddings=True, convert_to_numpy=True)
        return embedding[0].tolist()
