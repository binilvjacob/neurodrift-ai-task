from functools import lru_cache

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.bge import BGEEmbeddingProvider
from app.ingestion.chunker import Chunker
from app.ingestion.pipeline import IngestionPipeline
from app.llm.base import LLMProvider
from app.llm.hf_inference import HFInferenceLLM
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.base import VectorStore
from app.vectorstore.pinecone_store import PineconeVectorStore


@lru_cache
def get_embedder() -> EmbeddingProvider:
    settings = get_settings()
    return BGEEmbeddingProvider(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        query_instruction=settings.embedding_query_instruction,
    )


@lru_cache
def get_chunker() -> Chunker:
    settings = get_settings()
    return Chunker(
        tokenizer_name=settings.embedding_model_name,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    embedder = get_embedder()
    return PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        dimension=embedder.dimension,
    )


@lru_cache
def get_llm() -> LLMProvider:
    settings = get_settings()
    return HFInferenceLLM(
        model=settings.hf_llm_model,
        hf_token=settings.hf_token,
        base_url=settings.hf_base_url,
        provider=settings.hf_llm_provider,
        timeout=settings.hf_request_timeout_s,
    )


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(chunker=get_chunker(), embedder=get_embedder(), vector_store=get_vector_store())


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    settings = get_settings()
    retriever = Retriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        top_k=settings.top_k,
        score_threshold=settings.similarity_threshold,
    )
    return RAGPipeline(retriever=retriever, llm=get_llm())
