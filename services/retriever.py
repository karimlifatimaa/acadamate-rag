from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Tuple
from services.vector_store import get_vector_store
from services.query_augmenter import augment_query

SIMILARITY_THRESHOLD = 0.3
FETCH_K = 20
TOP_K = 5
DEDUP_PREFIX = 120


def _dedupe(results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    seen = set()
    unique: List[Tuple[Document, float]] = []
    for doc, score in results:
        key = doc.page_content.strip()[:DEDUP_PREFIX]
        if key in seen:
            continue
        seen.add(key)
        unique.append((doc, score))
    return unique


def retrieve(question: str, subject: str, grade: int) -> List[Tuple[Document, float]]:
    store: QdrantVectorStore = get_vector_store()

    augmented_query = augment_query(question)

    metadata_filter = Filter(
        must=[
            FieldCondition(key="metadata.subject", match=MatchValue(value=subject)),
            FieldCondition(key="metadata.grade", match=MatchValue(value=grade)),
        ]
    )

    raw: List[Tuple[Document, float]] = store.similarity_search_with_score(
        query=augmented_query,
        k=FETCH_K,
        filter=metadata_filter,
    )

    filtered = [(doc, score) for doc, score in raw if score >= SIMILARITY_THRESHOLD]
    unique = _dedupe(filtered)
    return unique[:TOP_K]
