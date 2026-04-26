import logging
from typing import List, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
from database import get_db, BookModel
from models.schemas import BookInfo
from services.vector_store import get_qdrant_client
from config import settings

logger = logging.getLogger(__name__)


def list_books() -> List[BookInfo]:
    with get_db() as db:
        rows = db.query(BookModel).order_by(BookModel.uploaded_at.desc()).all()
        return [
            BookInfo(
                id=str(row.id),
                subject=row.subject,
                grade=row.grade,
                source_file=row.source_file,
                original_name=row.original_name or "",
                uploaded_at=row.uploaded_at.isoformat() if row.uploaded_at else "",
                status=row.status,
                total_chunks=row.total_chunks or 0,
            )
            for row in rows
        ]


def delete_book_by_id(book_id: str) -> int:
    with get_db() as db:
        row = db.query(BookModel).filter(BookModel.id == book_id).first()
        if row is None:
            return 0

        deleted = _delete_qdrant_chunks(source_file=row.source_file)
        db.delete(row)

    logger.info("Kitab silindi", extra={"book_id": book_id, "deleted_chunks": deleted})
    return deleted


def delete_books(
    subject: Optional[str] = None,
    grade: Optional[int] = None,
    source_file: Optional[str] = None,
) -> int:
    if subject is None and grade is None and source_file is None:
        raise ValueError("Ən azı bir filter təyin et")

    with get_db() as db:
        q = db.query(BookModel)
        if subject is not None:
            q = q.filter(BookModel.subject == subject)
        if grade is not None:
            q = q.filter(BookModel.grade == grade)
        if source_file is not None:
            q = q.filter(BookModel.source_file == source_file)

        rows = q.all()
        total_deleted = 0
        for row in rows:
            total_deleted += _delete_qdrant_chunks(source_file=row.source_file)
            db.delete(row)

    logger.info(
        "Kitablar silindi",
        extra={"subject": subject, "grade": grade, "source_file": source_file, "deleted_chunks": total_deleted},
    )
    return total_deleted


def _delete_qdrant_chunks(source_file: str) -> int:
    client = get_qdrant_client()
    qfilter = Filter(
        must=[FieldCondition(key="metadata.source_file", match=MatchValue(value=source_file))]
    )
    count = client.count(
        collection_name=settings.collection_name,
        count_filter=qfilter,
        exact=True,
    ).count
    client.delete(
        collection_name=settings.collection_name,
        points_selector=FilterSelector(filter=qfilter),
    )
    return count
