from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Dataset Service"
    app_version: str = "1.0.0"

    supabase_url: str = ""
    supabase_secret_key: str = ""
    supabase_anon_key: str = ""
    supabase_datasets_table: str = "datasets"

    chroma_host: str = ""
    chroma_port: int = 0
    chroma_token: str = ""
    chroma_collection_name: str = "documents_e5"

    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    title_enabled: bool = True
    title_batch_size: int = 20
    title_max_chars_per_doc: int = 400

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()