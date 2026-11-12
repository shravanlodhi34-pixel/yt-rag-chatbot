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
    summarize_content,
)
from APP.youtube import extract_video_id, fetch_transcript, fetch_video_metadata

load_dotenv()

UI_PATH = Path(__file__).resolve().parent / "APP" / "ui.html"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

app = FastAPI(title="YT-RAG: YouTube Video Knowledge Assistant", version="2.0.0")


class ChatRequest(BaseModel):
    question: str
    provider: str | None = "auto"
    api_key: str | None = None
    chunks: list[str] | None = None
    source_title: str | None = None


class YouTubeRequest(BaseModel):
    url: str
    provider: str | None = "auto"
    api_key: str | None = None


class TextRequest(BaseModel):
    title: str = "Custom Transcript / Notes"
    text: str
    provider: str | None = "auto"
    api_key: str | None = None


class UploadRequest(BaseModel):
    filename: str
    content_base64: str


@lru_cache(maxsize=1)
def get_default_document() -> dict:
    """Load and chunk the bundled sample PDF once per process."""
    text = extract_text_from_pdf_path(DEFAULT_PDF_PATH)
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    return {
        "source_type": "pdf",
        "filename": DEFAULT_PDF_PATH.name,
        "title": DEFAULT_PDF_PATH.name,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve the YT-RAG web interface."""
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/chunks")
def list_chunks() -> dict:
    """Return extracted text statistics and chunks for the default document."""
    return get_default_document()


@app.post("/api/youtube")
async def load_youtube(payload: YouTubeRequest) -> dict:
    """Fetch a YouTube video's title, transcript, short summary, and RAG chunks."""
    try:
        video_id = extract_video_id(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    meta = await fetch_video_metadata(video_id)
    transcript = fetch_transcript(video_id)

    warning = None
    if transcript["available"]:
        source_text = transcript["text"]
        language = transcript["language"]
    elif meta["description"]:
        source_text = meta["description"]
        language = "description fallback"
        warning = f"Transcript fetch nahi ho paya ({transcript['error']}); video description use kiya gaya hai."
    else:
        source_text = meta["title"]
        language = "title only"
        warning = f"Transcript aur description unavailable hain ({transcript['error']}). Aap niche transcript manually paste kar sakte hain."

    chunks = split_text(source_text, chunk_size=500, chunk_overlap=50)
    summary_result = await summarize_content(
        title=meta["title"],
        text=source_text,
        provider=payload.provider,
        api_key=payload.api_key,
    )

    return {
        "source_type": "youtube",
        "video_id": video_id,
        "url": meta["watch_url"],
        "title": meta["title"],
        "filename": meta["title"],
        "author": meta["author"],
        "thumbnail_url": meta["thumbnail_url"],
        "language": language,
        "transcript_available": transcript["available"],
        "warning": warning,
        "total_characters": len(source_text),
        "total_chunks": len(chunks),
        "summary": summary_result["summary"],
        "mode": summary_result["mode"],
        "chunks": chunks,
    }


@app.post("/api/text")
async def load_text(payload: TextRequest) -> dict:
    """Chunk and summarize manually pasted transcript or text."""
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    title = payload.title.strip() or "Custom Transcript"
    chunks = split_text(clean_text, chunk_size=500, chunk_overlap=50)
    summary_result = await summarize_content(
        title=title,
        text=clean_text,
        provider=payload.provider,
        api_key=payload.api_key,
    )
    return {
        "source_type": "text",
        "title": title,
        "filename": title,
        "total_characters": len(clean_text),
        "total_chunks": len(chunks),
        "summary": summary_result["summary"],
        "mode": summary_result["mode"],
        "chunks": chunks,
    }


@app.post("/api/upload")
async def upload_pdf(payload: UploadRequest) -> dict:
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
        "source_type": "pdf",
        "title": payload.filename,
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

    default_doc = get_default_document()
    chunks = payload.chunks if payload.chunks else default_doc["chunks"]
    source_title = payload.source_title or default_doc["title"]
    matches = retrieve_chunks(question, chunks, top_k=3)
    return await generate_answer(
        query=question,
        matches=matches,
        provider=payload.provider,
        api_key=payload.api_key,
        source_title=source_title,
    )
