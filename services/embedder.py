import logging
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 8


class LocalEmbeddings(Embeddings):
    _model: SentenceTransformer | None = None

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            logger.info("Embedding modeli yüklənir", extra={"model": MODEL_NAME})
            cls._model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding modeli hazırdır", extra={"model": MODEL_NAME})
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
