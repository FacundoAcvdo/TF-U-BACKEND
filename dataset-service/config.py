from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Dataset Service"
    app_version: str = "1.0.0"

    internal_service_token: str = ""

    max_upload_size_mb: int = 50

    supabase_url: str = ""
    supabase_secret_key: str = ""
    supabase_anon_key: str = ""
    supabase_datasets_table: str = "datasets"

    chroma_host: str = ""
    chroma_port: int = 0
    chroma_token: str = ""
    chroma_collection_name: str = "documents_e5"

    embedding_provider: str = "none"
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_batch_size: int = 64
    embedding_device: str = "auto"
    embedding_fp16: bool = True

    ingest_chunk_size: int = 2000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()