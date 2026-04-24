import time
import requests
from langchain_core.embeddings import Embeddings
from typing import List
from config import settings

EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-embedding-001:embedContent"
)


def _embed_single(text: str, retries: int = 5) -> List[float]:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": settings.google_api_key,
    }
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768,
    }

    for attempt in range(retries):
        response = requests.post(EMBED_URL, json=payload, headers=headers)
        if response.status_code == 429:
            wait = 2 ** attempt
            print(f"Rate limit — {wait}s gözlənilir...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()["embedding"]["values"]

    raise RuntimeError("Embedding API rate limit — bütün cəhdlər uğursuz oldu.")


class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(_embed_single(text))
            if (i + 1) % 10 == 0:
                time.sleep(1)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return _embed_single(text)
