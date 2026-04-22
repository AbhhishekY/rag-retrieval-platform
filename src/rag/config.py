"""Runtime config: constants + optional .env overrides.

Defaults live in `rag.constants`. This file is only responsible for layering
environment-variable overrides on top (via pydantic-settings).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import constants as C


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path(C.DATA_DIR)
    index_dir: Path = Path(C.INDEX_DIR)
    output_dir: Path = Path(C.OUTPUT_DIR)

    embedding_model: str = C.EMBEDDING_MODEL
    reranker_model: str = C.RERANKER_MODEL

    chunk_size: int = C.CHUNK_SIZE
    chunk_overlap: int = C.CHUNK_OVERLAP
    chunk_strategy: str = C.CHUNK_STRATEGY

    top_k_retrieve: int = C.TOP_K_RETRIEVE
    top_k_rerank: int = C.TOP_K_RERANK
    top_k_final: int = C.TOP_K_FINAL

    fusion_method: str = C.FUSION_METHOD
    hybrid_alpha: float = C.HYBRID_ALPHA
    rrf_k: int = C.RRF_K

    use_rerank_default: bool = C.USE_RERANK_DEFAULT

    embed_batch_size: int = C.EMBED_BATCH_SIZE
    rerank_batch_size: int = C.RERANK_BATCH_SIZE


def get_settings() -> Settings:
    return Settings()
