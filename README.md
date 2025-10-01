# LawDoc SaaS

Multi-tenant RAG API for law firms.

## Endpoints
- GET /stats?client=acme
- GET /search?client=acme&q=...
- POST /ingest_json
- POST /ask

## Ingest JSON shape
{
  "client": "acme",
  "filename": "SomeDoc.txt",
  "content_b64": "<base64 text file content>"
}
