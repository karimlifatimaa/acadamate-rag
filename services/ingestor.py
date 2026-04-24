from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from typing import List
from utils.pdf_loader import load_and_split
from services.vector_store import get_vector_store


def ingest_pdf(file_path: str, subject: str, grade: int) -> int:
    chunks: List[Document] = load_and_split(file_path, subject, grade)

    store: QdrantVectorStore = get_vector_store()
    store.add_documents(chunks)

    return len(chunks)
