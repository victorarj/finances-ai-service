# finances-ai-service

`finances-ai-service` is a standalone FastAPI service that acts as the AI layer for the existing `finances` Node/Express backend. It ingests stored documents, writes chunk embeddings into PostgreSQL with pgvector, and answers user questions with retrieved context.

## Endpoints

- `GET /health`
- `POST /ingest`
- `POST /query`
- `GET /documents/{id}/chunks?user_id=<id>`

## Environment

Copy `.env.example` to `.env` and set:

- `DATABASE_URL`
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET`
- `AI_SERVICE_URL`
- `INTERNAL_API_SECRET`
- `OPENAI_API_KEY`
- `LLM_PROVIDER`
- `OLLAMA_BASE_URL`

Optional:

- `NODE_BACKEND_URL`
  Defaults to `http://localhost:3000` for local runs and `http://backend:3000` in Docker.

## Local run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## Docker run

This repo is designed to be combined with the existing `finances` stack:

```bash
docker compose -f ../finances/docker-compose.yml -f docker-compose.override.yml up --build
```

The override assumes the base stack already provides `postgres` and `minio` services.

## Request and response shapes

### `POST /ingest`

Request:

```json
{
  "document_id": "12",
  "user_id": "7",
  "storage_key": "documents/7/payslip.pdf",
  "source_type": "payslip"
}
```

Success response:

```json
{
  "document_id": 12,
  "user_id": 7,
  "storage_key": "documents/7/payslip.pdf",
  "source_type": "payslip",
  "chunk_count": 4,
  "status": "ready"
}
```

Failure response:

```json
{
  "error": "ingestion_failed",
  "detail": "No text could be extracted from document",
  "document_id": 12,
  "status": "failed"
}
```

### `POST /query`

Request:

```json
{
  "question": "What changed this month?",
  "user_id": "7",
  "document_ids": ["12", "13"]
}
```

Response:

```json
{
  "answer": "Income increased by 10%.",
  "sources": [
    {
      "document_id": 12,
      "chunk_index": 0,
      "content_preview": "Income increased by 10%"
    }
  ]
}
```

## Notes

- Ingestion always calls the Node backend status route with the `x-internal-api-secret` header.
- Callback failures are logged but do not crash the ingestion pipeline.
- PostgreSQL retrieval always filters by `user_id`.
- OCR support is limited to `png`, `jpg`, and `jpeg` in v1.
- Ollama embedding models must return 1536-dimensional vectors to match the existing `document_chunks.embedding` schema.
