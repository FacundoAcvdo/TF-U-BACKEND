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
    chroma_collection_name: str = "documents"

    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:8b"
    ollama_keep_alive: str = "5m"
    ollama_timeout_seconds: int = 180
    ollama_num_ctx: int = 4096
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
