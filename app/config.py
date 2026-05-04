from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_ENV: str = "development"
    APP_PORT: int = 8000

    USE_LOCAL_STORAGE: bool = True
    LOCAL_STORAGE_DIR: str = "./static/images"
    LOCAL_BASE_URL: str = "http://localhost:8000/static/images"

    REPLICATE_API_TOKEN: str = ""

    DATABASE_URL: str = "sqlite+aiosqlite:///./aesthetic.db"

    SD_DEVICE: str = "cuda"
    SD_PRECISION: str = "fp16"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()