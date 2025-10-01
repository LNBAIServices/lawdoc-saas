import os, base64, json
from typing import List, Optional
import requests

from fastapi import FastAPI, Header, HTTPException, Query, Body
from pydantic import BaseModel

# ------------------
# Config from env
# ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DATA_DIR       = os.getenv("DATA_DIR", "/var/data/chroma")
ACTION_KEY     = os.getenv("ACTION_KEY")  # optional API key for this service (x-api-key)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

# ------------------
# Init Chroma
# ------------------
import chromadb
client = chromadb.PersistentClient(path=DATA_DIR)

def collection_name(client_id: str) -> str:
    # One collection per client; simple and safe
    return f"docs_{client_id}".lower()

def ensure_collection(client_id: str):
    name = collection_name(client_id)
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)

# ------------------
# Models
# ------------------
class IngestItem(BaseModel):
    client: str
    filename: str
    content_b64: str

class AskRequest(BaseModel):
    client: str
    q: str
    top_k: int = 6

app = FastAPI(title="LawDoc SaaS", version="1.0.0")

# ------------------
# Helpers
# ------------------
def check_action_key(x_api_key: Optional[str]):
    if ACTION_KEY and x_api_key != ACTION_KEY:
        raise HTTPException(status_code=401, detail="Invalid action key")

def embed_texts(texts: List[str]) -> List[List[float]]:
    # OpenAI Embeddings
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"model": EMBED_MODEL, "input": texts}, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embed failed: {resp.text}")
    data = resp.json()
    return [d["embedding"] for d in data["data"]]

def chat_answer(question: str, context: str) -> str:
    # OpenAI Chat Completion (Responses API compatible shape)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are a legal document assistant. Cite when possible."},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
    ]
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Chat failed: {r.text}")
    j = r.json()
    return j["choices"][0]["message"]["content"]

# ------------------
# Routes
# ------------------
@app.get("/")
def home():
    return {"ok": True, "msg": "RAG API is running. See /docs."}

@app.get("/stats")
def stats(client: str = Query(...)):
    col = ensure_collection(client)
    return {"client": client, "count": col.count()}

@app.get("/search")
def search(client: str = Query(...), q: str = Query(...), top_k: int = 6):
    col = ensure_collection(client)
    results = col.query(query_texts=[q], n_results=top_k, include=["metadatas", "documents"])
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"filename": meta.get("filename"), "preview": doc[:240]})
    return {"client": client, "q": q, "hits": hits}

@app.post("/ingest_json")
def ingest_json(item: IngestItem, x_api_key: Optional[str] = Header(None)):
    check_action_key(x_api_key)
    # Decode
    try:
        content_bytes = base64.b64decode(item.content_b64)
        text = content_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    # Split simple paragraphs
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks")

    # Embed + upsert
    vecs = embed_texts(chunks)
    col = ensure_collection(item.client)
    ids = [f"{item.filename}:{i}" for i in range(len(chunks))]
    metas = [{"filename": item.filename} for _ in chunks]
    col.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)
    return {"ok": True, "client": item.client, "filename": item.filename, "added": len(chunks)}

@app.post("/ask")
def ask(req: AskRequest, x_api_key: Optional[str] = Header(None)):
    check_action_key(x_api_key)
    col = ensure_collection(req.client)
    r = col.query(query_texts=[req.q], n_results=req.top_k, include=["metadatas", "documents"])
    if not r["documents"] or not r["documents"][0]:
        return {"answer": "Not found in the provided documents.", "sources": []}

    docs = r["documents"][0]
    metas = r["metadatas"][0]
    context = ""
    sources = []
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        fname = m.get("filename", "unknown")
        context += f"[{i}] {fname}\n{d}\n\n"
        sources.append({"id": i, "filename": fname, "snippet": d[:240]})

    answer = chat_answer(req.q, context)
    return {"answer": answer, "sources": sources}
