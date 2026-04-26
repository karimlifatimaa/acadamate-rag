import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from routers import ask, ingest, health, books
from services.vector_store import (
    ensure_collection,
    get_qdrant_client,
    get_embeddings,
    get_sparse_embeddings,
)
from utils.logger import setup_logging
from database import init_db
from config import settings

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup başladı")

    client = get_qdrant_client()
    ensure_collection(client)

    if settings.postgres_url:
        init_db()
    else:
        logger.warning("POSTGRES_URL təyin edilməyib — kitab kataloqu deaktivdir")

    logger.info("Embedding modelləri preload edilir")
    get_embeddings().embed_query("warmup")
    get_sparse_embeddings()
    logger.info("Startup tamamlandı")

    yield

    logger.info("Shutdown")


limiter = Limiter(key_func=get_remote_address, default_limits=["5/second"])

app = FastAPI(
    title="Acadamate RAG API",
    description="Məktəblilər üçün dərslik əsaslı sual-cavab sistemi",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(ask.router)
app.include_router(ingest.router)
app.include_router(health.router)
app.include_router(books.router)
