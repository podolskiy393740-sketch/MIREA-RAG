from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    # Заглушка до выбора модели эмбеддингов — см. .env.example.
    embedding_dim: int = 768
