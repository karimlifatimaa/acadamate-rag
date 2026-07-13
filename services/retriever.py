import re
import logging
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from typing import List, Tuple, Optional
from services.vector_store import get_vector_store
from services.query_augmenter import augment_query
from services.reranker import rerank

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.3
FETCH_K = 20
TOP_K = 5
DEDUP_PREFIX = 120


def _extract_page(question: str) -> Optional[int]:
    match = re.search(r'səhifə\s*(\d+)|səh\.?\s*(\d+)|page\s*(\d+)', question, re.IGNORECASE)
    if match:
        return int(next(g for g in match.groups() if g))
    return None


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

    base_filter = Filter(
        must=[
            FieldCondition(key="metadata.subject", match=MatchValue(value=subject)),
            FieldCondition(key="metadata.grade", match=MatchValue(value=grade)),
        ]
    )

    raw: List[Tuple[Document, float]] = store.similarity_search_with_score(
        query=augmented_query,
        k=FETCH_K,
        filter=base_filter,
    )

    # Şagird "səhifə X" dedisə həmin səhifənin chunk-larını yüksək prioritetlə əlavə et
    page = _extract_page(question)
    if page is not None:
        page_filter = Filter(
            must=[
                FieldCondition(key="metadata.subject", match=MatchValue(value=subject)),
                FieldCondition(key="metadata.grade", match=MatchValue(value=grade)),
                FieldCondition(key="metadata.page", match=MatchValue(value=page)),
            ]
        )
        page_results: List[Tuple[Document, float]] = store.similarity_search_with_score(
            query=augmented_query,
            k=5,
            filter=page_filter,
        )
        logger.info("Səhifə filter əlavə edildi", extra={"page": page, "found": len(page_results)})
        # Səhifə chunk-larını əvvələ qoy, sonra ümumi nəticələr
        raw = page_results + [r for r in raw if r not in page_results]

    filtered = [(doc, score) for doc, score in raw if score >= SIMILARITY_THRESHOLD]
    unique = _dedupe(filtered)

    # Reranking: namizədləri Groq LLM ilə yenidən sırala, əlaqəsizləri at
    reranked = rerank(question, unique, TOP_K)
    return reranked
