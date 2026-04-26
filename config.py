from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str = ""
    groq_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "acadamate_docs"
    rag_api_key: str
    postgres_url: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
