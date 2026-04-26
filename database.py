import uuid
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker
from contextlib import contextmanager
from config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class BookModel(Base):
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject = Column(String(50), nullable=False)
    grade = Column(Integer, nullable=False)
    source_file = Column(String(500), nullable=False, unique=True)
    original_name = Column(String(255))
    uploaded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), nullable=False, default="READY")
    total_chunks = Column(Integer)
    error_message = Column(Text)


_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(settings.postgres_url, pool_pre_ping=True, pool_size=5)
        _SessionLocal = sessionmaker(bind=_engine)
    return _engine


def init_db() -> None:
    engine = _get_engine()
    Base.metadata.create_all(engine)
    logger.info("Postgres books cədvəli hazırdır")


@contextmanager
def get_db():
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
