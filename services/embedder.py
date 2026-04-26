from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 8


class LocalEmbeddings(Embeddings):
    _model: SentenceTransformer | None = None

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            print(f"Embedding modeli yüklənir: {MODEL_NAME} (ilk dəfə 2-5 dəq çəkə bilər)")
            cls._model = SentenceTransformer(MODEL_NAME)
            print("Model hazırdır.")
        return cls._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


GeminiEmbeddings = LocalEmbeddings
