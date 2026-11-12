import base64
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from APP.rag import (
    DEFAULT_PDF_PATH,
    extract_text_from_pdf_bytes,
    extract_text_from_pdf_path,
    generate_answer,
    retrieve_chunks,
    split_text,
)

load_dotenv()

UI_PATH = Path(__file__).resolve().parent / "APP" / "ui.html"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

app = FastAPI(title="AI RAG Chatbot", version="1.0.0")


class ChatRequest(BaseModel):
    question: str
    provider: str | None = "auto"
    api_key: str | None = None
    chunks: list[str] | None = None


class UploadRequest(BaseModel):
    filename: str
    content_base64: str


@lru_cache(maxsize=1)
def get_default_document() -> dict:
    """Load and chunk the bundled sample PDF once per process."""
    text = extract_text_from_pdf_path(DEFAULT_PDF_PATH)
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    return {
        "filename": DEFAULT_PDF_PATH.name,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve the RAG Chatbot web interface."""
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/chunks")
def list_chunks() -> dict:
    """Return extracted text statistics and chunks for the default PDF."""
    return get_default_document()


@app.post("/api/upload")
def upload_pdf(payload: UploadRequest) -> dict:
    """Extract text and chunks from an uploaded base64-encoded PDF."""
    try:
        raw = base64.b64decode(payload.content_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF content.") from exc

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 4 MB limit.")

    try:
        text = extract_text_from_pdf_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc

    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    return {
        "filename": payload.filename,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict:
    """Retrieve matching chunks and generate a RAG answer."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = payload.chunks if payload.chunks else get_default_document()["chunks"]
    matches = retrieve_chunks(question, chunks, top_k=3)
    return await generate_answer(
        query=question,
        matches=matches,
        provider=payload.provider,
        api_key=payload.api_key,
    )
