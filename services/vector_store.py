import logging
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from services.embedder import LocalEmbeddings
from config import settings

logger = logging.getLogger(__name__)

VECTOR_SIZE = 1024
SPARSE_MODEL = "Qdrant/bm25"


def get_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()


_sparse_embeddings: FastEmbedSparse | None = None


def get_sparse_embeddings() -> FastEmbedSparse:
    global _sparse_embeddings
    if _sparse_embeddings is None:
        logger.info("Sparse embedding modeli yüklənir", extra={"model": SPARSE_MODEL})
        _sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)
        logger.info("Sparse model hazırdır")
    return _sparse_embeddings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
    )


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in existing:
        logger.info("Kolleksiya yaradılır", extra={"collection": settings.collection_name})
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
