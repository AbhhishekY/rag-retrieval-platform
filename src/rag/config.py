"""Config loaded from environment (.env file supported)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("./data")
    index_dir: Path = Path("./indices")
    output_dir: Path = Path("./outputs")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_api_key: str = ""
    azure_openai_chat_deployment: str = ""

    chunk_size: int = 512
    chunk_overlap: int = 51
    chunk_strategy: str = "recursive"

    top_k_retrieve: int = 100
    top_k_rerank: int = 20
    top_k_final: int = 5
    fusion_method: str = "rrf"
    hybrid_alpha: float = 0.5
    rrf_k: int = 60


def get_settings() -> Settings:
    return Settings()
