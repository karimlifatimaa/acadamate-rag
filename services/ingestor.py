import logging
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from typing import List
from utils.pdf_loader import load_and_split
from services.vector_store import get_vector_store
from database import get_db, BookModel

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str, subject: str, grade: int, original_name: str = "") -> int:
    logger.info("Ingest başladı", extra={"original_name": original_name, "subject": subject, "grade": grade})

    # Mənalı açar: "subject/grade/filename.pdf"  — temp path deyil
    source_key = f"{subject}/{grade}/{original_name}" if original_name else file_path

    chunks: List[Document] = load_and_split(file_path, subject, grade, source_key=source_key)

    store: QdrantVectorStore = get_vector_store()
    store.add_documents(chunks)

    _upsert_book(
        source_file=source_key,
        subject=subject,
        grade=grade,
        original_name=original_name,
        total_chunks=len(chunks),
    )

    logger.info("Ingest tamamlandı", extra={"file": file_path, "chunks_indexed": len(chunks)})
    return len(chunks)


def _upsert_book(source_file: str, subject: str, grade: int, original_name: str, total_chunks: int) -> None:
    with get_db() as db:
        existing = db.query(BookModel).filter(BookModel.source_file == source_file).first()
        if existing:
            existing.subject = subject
            existing.grade = grade
            existing.original_name = original_name
            existing.total_chunks = total_chunks
            existing.status = "READY"
            existing.error_message = None
        else:
            db.add(BookModel(
                subject=subject,
                grade=grade,
                source_file=source_file,
                original_name=original_name,
                total_chunks=total_chunks,
                status="READY",
            ))
