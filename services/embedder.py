import requests
from langchain_core.embeddings import Embeddings
from typing import List
from config import settings

EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-embedding-001:embedContent"
)


def _embed_single(text: str) -> List[float]:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": settings.google_api_key,
    }
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768,
    }
    response = requests.post(EMBED_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["embedding"]["values"]


class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_embed_single(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return _embed_single(text)
