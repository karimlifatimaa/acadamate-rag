from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from routers import ask, ingest, health

limiter = Limiter(key_func=get_remote_address, default_limits=["5/second"])

app = FastAPI(
    title="Acadamate RAG API",
    description="Məktəblilər üçün dərslik əsaslı sual-cavab sistemi",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(ask.router)
app.include_router(ingest.router)
app.include_router(health.router)
