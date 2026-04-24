from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Tuple
from services.vector_store import get_vector_store

SIMILARITY_THRESHOLD = 0.3
TOP_K = 5


def retrieve(question: str, subject: str, grade: int) -> List[Tuple[Document, float]]:
    store: QdrantVectorStore = get_vector_store()

    metadata_filter = Filter(
        must=[
            FieldCondition(key="metadata.subject", match=MatchValue(value=subject)),
            FieldCondition(key="metadata.grade", match=MatchValue(value=grade)),
        ]
    )

    results: List[Tuple[Document, float]] = store.similarity_search_with_score(
        query=question,
        k=TOP_K,
        filter=metadata_filter,
    )

    return [(doc, score) for doc, score in results if score >= SIMILARITY_THRESHOLD]
