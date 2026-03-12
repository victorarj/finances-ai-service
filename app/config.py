from functools import lru_cache
from urllib.parse import urlparse

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    ai_service_url: str = "http://localhost:8001"
    internal_api_secret: str = "replace-me"
    openai_api_key: str = ""
    llm_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"
    node_backend_url: str | None = None

    @computed_field
    @property
    def normalized_minio_endpoint(self) -> str:
        endpoint = self.minio_endpoint.strip()
        if endpoint.startswith(("http://", "https://")):
            return endpoint.rstrip("/")
        return f"http://{endpoint}".rstrip("/")

    @computed_field
    @property
    def minio_secure(self) -> bool:
        return self.normalized_minio_endpoint.startswith("https://")

    @computed_field
    @property
    def minio_host(self) -> str:
        parsed = urlparse(self.normalized_minio_endpoint)
        return parsed.netloc or parsed.path

    @computed_field
    @property
    def normalized_ollama_base_url(self) -> str:
        base_url = self.ollama_base_url.strip()
        if base_url.startswith(("http://", "https://")):
            return base_url.rstrip("/")
        return f"http://{base_url}".rstrip("/")

    @computed_field
    @property
    def normalized_ai_service_url(self) -> str:
        service_url = self.ai_service_url.strip()
        if service_url.startswith(("http://", "https://")):
            return service_url.rstrip("/")
        return f"http://{service_url}".rstrip("/")

    @computed_field
    @property
    def normalized_node_backend_url(self) -> str:
        if self.node_backend_url:
            base_url = self.node_backend_url.strip()
            if base_url.startswith(("http://", "https://")):
                return base_url.rstrip("/")
            return f"http://{base_url}".rstrip("/")

        parsed = urlparse(self.normalized_ai_service_url)
        if parsed.hostname in {"localhost", "127.0.0.1"}:
            return "http://localhost:3000"
        return "http://backend:3000"

    @computed_field
    @property
    def validated_llm_provider(self) -> str:
        provider = self.llm_provider.strip().lower()
        if provider not in {"openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be 'openai' or 'ollama'")
        return provider

    @computed_field
    @property
    def embedding_model(self) -> str:
        if self.validated_llm_provider == "openai":
            return "text-embedding-3-small"
        return "nomic-embed-text"

    @computed_field
    @property
    def chat_model(self) -> str:
        if self.validated_llm_provider == "openai":
            return "gpt-4o"
        return "llama3.1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
