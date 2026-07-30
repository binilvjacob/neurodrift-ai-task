from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

TENANT_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # embeddings
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "

    # pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # llm (HF Inference Providers, OpenAI-compatible)
    hf_token: str
    hf_llm_model: str = "openai/gpt-oss-20b"
    hf_llm_provider: str | None = None
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_request_timeout_s: float = 30.0

    # retrieval / grounding
    top_k: int = 5
    similarity_threshold: float = 0.5

    # chunking (tokens, per the embedding model's own tokenizer)
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 64


@lru_cache
def get_settings() -> Settings:
    return Settings()
