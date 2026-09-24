from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    cleaning_download_path: str = "./data/temp_downloads"
    supabase_url: str = ""
    supabase_secret_key: str = ""
    supabase_anon_key: str = ""
    supabase_datasets_table: str = "datasets"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()