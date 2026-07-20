# Acadamate RAG API

A Retrieval-Augmented Generation (RAG) API for Azerbaijani school students. It answers questions strictly based on uploaded textbooks, ensuring accurate and grade-appropriate responses in Azerbaijani.

## Architecture

```
Spring Boot  →  POST /ask  →  FastAPI RAG Service
                                      │
   ┌──────────────────────────────────┼──────────────────────────────────┐
   │              │                    │                 │                 │
 HyDE        Hybrid Search         Rerank            Answer          Azure OpenAI
(Groq)     (Qdrant: dense+BM25)    (Groq)            (Groq)         (ada-002 embed)
```

Pipeline (per question):
1. **HyDE** — Groq Llama-3.3 generates a hypothetical textbook answer to enrich the query.
2. **Dual hybrid retrieval** — the store is queried with **both** the raw question (exact keywords/titles) **and** the HyDE-augmented query (semantic), via Dense (Azure `ada-002`) + Sparse (BM25) fused with **RRF**. Results are merged for higher recall.
3. **Rerank** — Groq Llama-3.3 re-scores all candidates and keeps only the chunks genuinely relevant to the question (drops the rest). If none are relevant, it returns empty → the model answers *"Bu mövzu dərslikdə izah olunmayıb."* instead of hallucinating.
4. **Generate** — Groq Llama-3.3 produces a structured Azerbaijani answer grounded in the retrieved chunks only.

Scanned (image-only) PDFs are supported: if a PDF has no text layer, ingestion automatically falls back to **OCR** (Tesseract, Azerbaijani).

## Tech Stack

| Component | Choice |
|---|---|
| Framework | FastAPI |
| Vector DB | Qdrant (self-hosted via Docker, named vectors: `dense` + `sparse`) |
| Dense Embedding | Azure OpenAI `text-embedding-ada-002` (1536-dim) |
| Sparse Embedding | `Qdrant/bm25` via `fastembed` (local) |
| Retrieval | Dual hybrid (raw + HyDE) with Reciprocal Rank Fusion |
| Reranking | Groq `llama-3.3-70b-versatile` (LLM-based relevance filter) |
| Chunking | `SemanticChunker` (langchain-experimental, percentile 95) |
| OCR (scanned PDFs) | Tesseract (`aze`) + poppler |
| Query Augmentation | HyDE via Groq Llama-3.3-70b-versatile |
| LLM | `llama-3.3-70b-versatile` (Groq) |
| Orchestration | LangChain |

> Dense embeddings run on **Azure OpenAI** — no heavy local model, so no CPU/RAM cost on the server. Only the small BM25 sparse model runs locally.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- **Tesseract + poppler** for OCR of scanned PDFs:
  ```bash
  # macOS
  brew install tesseract tesseract-lang poppler
  # Debian/Ubuntu
  sudo apt-get install tesseract-ocr tesseract-ocr-aze poppler-utils
  ```
- An **Azure OpenAI** resource with a `text-embedding-ada-002` deployment (for embeddings)
- A **Groq** API key (for the LLM)

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd acadamate-rag

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Azure OpenAI (embeddings)
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-02-01

# Groq (LLM: answer + rerank + HyDE)
GROQ_API_KEY=your-groq-api-key

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=acadamate_docs

# API auth (Spring Boot → this service)
RAG_API_KEY=your-secret-key-for-spring-boot

# Postgres (book catalog)
POSTGRES_URL=postgresql://user:password@localhost:5432/acadamate
```

> Get your Groq API key from [console.groq.com](https://console.groq.com/keys). Get Azure OpenAI keys from the Azure Portal → your resource → *Keys and Endpoint*.

### 3. Start Qdrant

```bash
docker compose up -d qdrant
```

### 4. Run the API

```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

### 5. Or run everything with Docker Compose

```bash
docker compose up
```

---

## API Endpoints

All endpoints (except `/health`) require an `X-API-Key` header matching `RAG_API_KEY` in your `.env`.

### `POST /ingest`

Upload a textbook PDF (multipart form-data). Each chunk is indexed with both a dense Azure `ada-002` vector and a BM25 sparse vector. Scanned PDFs are OCR-ed automatically (slower).

**Request** (`multipart/form-data`):

| Field | Type | Notes |
|---|---|---|
| `file` | file | the PDF |
| `subject` | text | e.g. `Az dili` |
| `grade` | text | 1–11 |

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: your-secret-key" \
  -F "file=@azdili7.pdf" \
  -F "subject=Az dili" \
  -F "grade=7"
```

**Response:**
```json
{
  "chunks_indexed": 462,
  "subject": "Az dili",
  "grade": 7
}
```

---

### `POST /ask`

Ask a question based on the uploaded textbooks.

**Request:**
```json
{
  "question": "Son yarpaq mətni haqqında məlumat ver",
  "subject": "Az dili",
  "grade": 7
}
```

> `subject` and `grade` must match **exactly** what was used at ingest time (exact-match filter).

**Response:**
```json
{
  "answer": "Son yarpaq mətni iki rəssam qızın — Syu və Consinin həyatından...",
  "sources": [
    { "page": 18, "subject": "Az dili", "grade": 7, "excerpt": "..." },
    { "page": 19, "subject": "Az dili", "grade": 7, "excerpt": "..." }
  ],
  "confidence": 0.37
}
```

If the question is unrelated to the textbook content:
```json
{
  "answer": "Bu mövzu dərslikdə izah olunmayıb.",
  "sources": [],
  "confidence": 0.0
}
```

If the LLM provider is rate-limited or unavailable, the API returns **HTTP 503** with a clear message instead of a generic 500:
```json
{ "detail": "AI xidməti hazırda məşğuldur (sorğu limiti). Bir neçə saniyədən sonra yenidən cəhd edin." }
```

---

### `GET /books`

List ingested books (from the Postgres catalog).

### `DELETE /books`

Delete a book's chunks. Filter by query params: `?subject=&grade=&source_file=`.

### `GET /health`

Health check for Spring Boot Actuator integration.

**Response:**
```json
{ "status": "ok", "qdrant": "up" }
```

---

## Spring Boot Integration

Add to `application.yml`:

```yaml
rag:
  base-url: http://localhost:8000
  api-key: ${RAG_API_KEY}
```

Use `RestClient` or `WebClient` with the `X-API-Key` header to call `/ask`, `/ingest`, `/books`, and `/health`.

---

## Project Structure

```
acadamate-rag/
├── main.py                       # FastAPI app + lifespan startup
├── config.py                     # Settings from .env
├── database.py                   # Postgres book catalog
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── routers/
│   ├── ask.py                    # POST /ask (503 on LLM unavailable)
│   ├── ingest.py                 # POST /ingest (multipart upload)
│   ├── books.py                  # GET/DELETE /books
│   └── health.py                 # GET /health
├── services/
│   ├── embedder.py               # Azure OpenAI dense embeddings
│   ├── vector_store.py           # Qdrant hybrid (dense + BM25 sparse)
│   ├── ingestor.py               # PDF → chunks → Qdrant
│   ├── query_augmenter.py        # HyDE via Groq Llama-3.3
│   ├── retriever.py              # Dual hybrid retrieval + rerank → top-5
│   ├── reranker.py               # Groq LLM relevance rerank
│   ├── books.py                  # Book catalog service
│   └── generator.py              # Groq Llama-3.3 answer generation
├── models/
│   └── schemas.py                # Pydantic request/response models
└── utils/
    ├── pdf_loader.py             # PDF → (OCR fallback) → SemanticChunker
    └── logger.py                 # JSON structured logging
```


```
