from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"
    groq_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "acadamate_docs"
    rag_api_key: str
    postgres_url: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
