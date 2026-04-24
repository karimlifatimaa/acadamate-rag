# Acadamate RAG API

A Retrieval-Augmented Generation (RAG) API for Azerbaijani school students. It answers questions strictly based on uploaded textbooks, ensuring accurate and grade-appropriate responses in Azerbaijani.

## Architecture

```
Spring Boot  →  POST /ask  →  FastAPI RAG Service
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                     Qdrant DB            Google AI API
                   (vector store)    gemini-embedding-001 + gemini-flash
```

## Tech Stack

| Component | Choice |
|---|---|
| Framework | FastAPI |
| Vector DB | Qdrant (self-hosted via Docker) |
| Embedding | `gemini-embedding-001` (Google AI) |
| LLM | `gemini-flash-latest` (Google AI) |
| Orchestration | LangChain |

## Getting Started

### Prerequisites

- Python 3.11+
- Docker

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
GOOGLE_API_KEY=your-google-ai-studio-api-key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=acadamate_docs
RAG_API_KEY=your-secret-key-for-spring-boot
```

> Get your Google API key from [Google AI Studio](https://aistudio.google.com/apikey)

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

### 5. Or run everything with Docker Compose

```bash
docker-compose up
```

---

## API Endpoints

All endpoints require `X-API-Key` header matching `RAG_API_KEY` in your `.env`.

### `POST /ingest`

Upload a textbook PDF to the vector store.

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
├── main.py                   # FastAPI app + rate limiter
├── config.py                 # Settings from .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── routers/
│   ├── ask.py                # POST /ask
│   ├── ingest.py             # POST /ingest
│   └── health.py             # GET /health
├── services/
│   ├── embedder.py           # Google gemini-embedding-001
│   ├── vector_store.py       # Qdrant connection
│   ├── ingestor.py           # PDF → chunks → Qdrant
│   ├── retriever.py          # Question → top-5 chunks
│   └── generator.py          # Gemini response generation
├── models/
│   └── schemas.py            # Pydantic request/response models
└── utils/
    └── pdf_loader.py         # PDF → LangChain documents
```
