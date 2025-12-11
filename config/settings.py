from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_api_key: str | None = Field(default=None)
    llm_model_main: str = Field(default="gpt-4o-mini")
    llm_model_observer: str = Field(default="gpt-4o-mini")
    data_dir: str = Field(default="./.devagent_data")
    event_db_path: str = Field(default="./.devagent_data/events.db")
    memory_db_path: str = Field(default="./.devagent_data/memory.db")
    trace_db_path: str = Field(default="./.devagent_data/trace.db")
    vector_dim: int = Field(default=768)
    coarse_k: int = Field(default=50)
    rerank_k: int = Field(default=20)

    class Config:
        env_file = ".env"


settings = Settings()

__all__ = ["Settings", "settings"]
