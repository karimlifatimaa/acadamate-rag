from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from services.embedder import LocalEmbeddings
from config import settings

VECTOR_SIZE = 1024
SPARSE_MODEL = "Qdrant/bm25"


def get_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()


_sparse_embeddings: FastEmbedSparse | None = None


def get_sparse_embeddings() -> FastEmbedSparse:
    global _sparse_embeddings
    if _sparse_embeddings is None:
        print(f"Sparse embedding modeli yüklənir: {SPARSE_MODEL}")
        _sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)
        print("Sparse model hazırdır.")
    return _sparse_embeddings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in existing:
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
    client = get_qdrant_client()
    ensure_collection(client)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
