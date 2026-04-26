# Acadamate RAG API

A Retrieval-Augmented Generation (RAG) API for Azerbaijani school students. It answers questions strictly based on uploaded textbooks, ensuring accurate and grade-appropriate responses in Azerbaijani.

## Architecture

```
Spring Boot  →  POST /ask  →  FastAPI RAG Service
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
         Qdrant DB              Local BGE-M3              Groq API
       (hybrid index)         (dense + sparse)      llama-3.3-70b-versatile
                                                    (HyDE + answer gen)
```

Pipeline (per question):
1. **HyDE** — Groq Llama-3.3 generates a hypothetical textbook answer to enrich the query.
2. **Hybrid retrieval** — Qdrant fetches top-K via Dense (BGE-M3) + Sparse (BM25), fused with **RRF**.
3. **Filter & dedup** — score threshold + content-prefix dedup → top 5 chunks.
4. **Generate** — Groq Llama-3.3 produces a structured Azerbaijani answer grounded in the retrieved chunks only.

## Tech Stack

| Component | Choice |
|---|---|
| Framework | FastAPI |
| Vector DB | Qdrant (self-hosted via Docker, named vectors: `dense` + `sparse`) |
| Dense Embedding | `BAAI/bge-m3` (local, 1024-dim, multilingual incl. Azerbaijani) |
| Sparse Embedding | `Qdrant/bm25` via `fastembed` |
| Retrieval | Hybrid (Dense + BM25) with Reciprocal Rank Fusion |
| Chunking | `SemanticChunker` (langchain-experimental, percentile 95) |
| Query Augmentation | HyDE via Groq Llama-3.3-70b-versatile |
| LLM | `llama-3.3-70b-versatile` (Groq) |
| Orchestration | LangChain |

> Embeddings run **locally on CPU** — no embedding API costs and no rate limits during ingestion. First model download is ~4.3 GB.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- ~5 GB free disk for the BGE-M3 model + Qdrant storage

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
GROQ_API_KEY=your-groq-api-key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=acadamate_docs
RAG_API_KEY=your-secret-key-for-spring-boot
```

> Get your Groq API key from [console.groq.com](https://console.groq.com/keys) (free tier is sufficient for development).

### 3. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

### 4. Run the API

```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

> On the first `/ingest` or `/ask` call, BGE-M3 (~4.3 GB) and BM25 models are downloaded and cached. Initial load takes 2–5 minutes.

### 5. Or run everything with Docker Compose

```bash
docker-compose up
```

---

## API Endpoints

All endpoints require `X-API-Key` header matching `RAG_API_KEY` in your `.env`.

### `POST /ingest`

Upload a textbook PDF to the vector store. Each chunk is indexed with both a dense BGE-M3 vector and a BM25 sparse vector.

**Request:**
```json
{
  "file_path": "/data/math_grade8.pdf",
  "subject": "riyaziyyat",
  "grade": 8
}
```

**Response:**
```json
{
  "chunks_indexed": 142,
  "subject": "riyaziyyat",
  "grade": 8
}
```

---

### `POST /ask`

Ask a question based on the uploaded textbooks.

**Request:**
```json
{
  "question": "Pifaqor teoremi nədir?",
  "subject": "riyaziyyat",
  "grade": 8
}
```

**Response:**
```json
{
  "answer": "Pifaqor teoremi...",
  "sources": [
    {
      "page": 45,
      "subject": "riyaziyyat",
      "grade": 8,
      "excerpt": "..."
    }
  ],
  "confidence": 0.87
}
```

If the question is unrelated to the textbook content, the API returns:
```json
{
  "answer": "Bu mövzu dərslikdə izah olunmayıb.",
  "sources": [],
  "confidence": 0.0
}
```

---

### `GET /health`

Health check for Spring Boot Actuator integration.

**Response:**
```json
{
  "status": "ok",
  "qdrant": "up"
}
```

---

## Spring Boot Integration

Add to `application.yml`:

```yaml
rag:
  base-url: http://localhost:8000
  api-key: ${RAG_API_KEY}
```

Use `RestClient` or `WebClient` with `X-API-Key` header to call `/ask`, `/ingest`, and `/health`.

---

## Project Structure

```
acadamate-rag/
├── main.py                       # FastAPI app + rate limiter
├── config.py                     # Settings from .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── IMPROVEMENTS.md               # Roadmap and applied improvements
├── routers/
│   ├── ask.py                    # POST /ask
│   ├── ingest.py                 # POST /ingest
│   └── health.py                 # GET /health
├── services/
│   ├── embedder.py               # Local BGE-M3 dense embeddings
│   ├── vector_store.py           # Qdrant hybrid (dense + BM25 sparse)
│   ├── ingestor.py               # PDF → chunks → Qdrant
│   ├── query_augmenter.py        # HyDE via Groq Llama-3.3
│   ├── retriever.py              # Hybrid retrieval + dedup → top-5
│   └── generator.py              # Groq Llama-3.3 answer generation
├── models/
│   └── schemas.py                # Pydantic request/response models
└── utils/
    └── pdf_loader.py             # PDF → SemanticChunker → documents
```

---

## Notes on the Stack

- **Why local embeddings?** Hosted embedding APIs hit free-tier rate limits during full-book ingestion and require prepaid billing on Tier 1+ in some regions. BGE-M3 runs on CPU, has strong multilingual support including Azerbaijani, and removes the embedding API as a bottleneck.
- **Why Groq for the LLM?** Free tier provides high RPM with `llama-3.3-70b-versatile` and OpenAI-compatible endpoints, which is enough for development and testing without prepaid credits.
- **Why hybrid search?** Dense vectors are strong for paraphrased / semantic queries; BM25 is strong for specific terms, names, and numbers (e.g. "misal №3", "Pifaqor"). RRF combines both rankings without manual weight tuning.
