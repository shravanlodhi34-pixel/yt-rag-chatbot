import io
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from pypdf import PdfReader

from APP.youtube import summarize_short

DEFAULT_PDF_PATH = Path(__file__).resolve().parent.parent / "DOCUMENTS" / "AI_RAG_Sample_Document.pdf"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "what", "when", "where", "which", "who", "why", "how",
    "this", "these", "those", "can", "could", "should", "would", "do", "does",
    "did", "about", "into", "or", "if", "your", "you", "me", "my", "we", "our",
}

OPENAI_COMPAT_PROVIDERS = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.1-8b-instant",
    },
}


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract concatenated text from all pages of an in-memory PDF."""
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n".join(p for p in pages if p)


def extract_text_from_pdf_path(pdf_path: Path = DEFAULT_PDF_PATH) -> str:
    """Read a PDF file from disk and extract its text."""
    return extract_text_from_pdf_bytes(pdf_path.read_bytes())


def _split_segment(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text into pieces smaller than chunk_size."""
    if len(text) <= chunk_size or not separators:
        return [text] if text else []

    sep = separators[0]
    remaining = separators[1:]
    parts = text.split(sep) if sep else list(text)

    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current and sep else f"{current}{part}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(part) > chunk_size:
            pieces.extend(_split_segment(part, remaining, chunk_size))
        else:
            current = part
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks without heavy external dependencies."""
    cleaned = text.strip()
    if not cleaned:
        return []

    raw_pieces = _split_segment(cleaned, ["\n\n", "\n", ". ", " ", ""], chunk_size)
    if chunk_overlap <= 0 or len(raw_pieces) <= 1:
        return [p.strip() for p in raw_pieces if p.strip()]

    chunks: list[str] = []
    for idx, piece in enumerate(raw_pieces):
        piece_clean = piece.strip()
        if not piece_clean:
            continue
        if idx > 0 and chunks:
            prefix = chunks[-1][-chunk_overlap:].lstrip()
            merged = f"{prefix} {piece_clean}".strip()
            chunks.append(merged if len(merged) <= chunk_size + chunk_overlap else piece_clean)
        else:
            chunks.append(piece_clean)
    return chunks


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return filtered if filtered else tokens


def retrieve_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    """Rank chunks with TF-IDF cosine similarity plus phrase overlap boost."""
    if not chunks or not query.strip():
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return [{"index": 0, "score": 0.0, "text": chunks[0]}]

    doc_tokens = [_tokenize(chunk) for chunk in chunks]
    num_docs = len(chunks)
    df = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    idf = {term: math.log((num_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    query_tf = Counter(query_tokens)
    query_vec = {t: (c / len(query_tokens)) * idf.get(t, 1.0) for t, c in query_tf.items()}
    query_norm = math.sqrt(sum(v * v for v in query_vec.values())) or 1.0
    query_lower = query.lower().strip()

    scored: list[dict[str, Any]] = []
    for idx, (chunk, tokens) in enumerate(zip(chunks, doc_tokens)):
        if not tokens:
            continue
        tf = Counter(tokens)
        doc_vec = {t: (c / len(tokens)) * idf.get(t, 1.0) for t, c in tf.items()}
        doc_norm = math.sqrt(sum(v * v for v in doc_vec.values())) or 1.0
        dot = sum(query_vec[t] * doc_vec.get(t, 0.0) for t in query_vec)
        cosine = dot / (query_norm * doc_norm)
        phrase_boost = 0.25 if query_lower in chunk.lower() else 0.0
        score = round(min(1.0, cosine + phrase_boost), 4)
        scored.append({"index": idx, "score": score, "text": chunk})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _build_extractive_answer(query: str, matches: list[dict[str, Any]]) -> str:
    """Create a grounded answer directly from top chunks when no LLM key is used."""
    if not matches:
        return "Loaded content mein is sawal se milta-julta koi hissa nahi mila."

    if matches[0]["score"] <= 0.0:
        preview = matches[0]["text"]
        return (
            "Is query ka exact keyword match nahi mila, lekin content ka sabse kareebi hissa yeh hai:\n\n"
            f"{preview}\n\n"
            "*(Tip: Full conversational AI summary & Q&A ke liye upar Groq, Gemini, ya OpenAI API key select kar sakte hain.)*"
        )

    query_terms = set(_tokenize(query))
    best_sentences: list[str] = []
    for match in matches:
        if match["score"] <= 0.05 and best_sentences:
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n+", match["text"]):
            clean_sent = sent.strip()
            if not clean_sent:
                continue
            overlap = len(query_terms & set(_tokenize(clean_sent)))
            if overlap > 0 or len(best_sentences) < 2:
                if clean_sent not in best_sentences:
                    best_sentences.append(clean_sent)
            if len(best_sentences) >= 5:
                break
        if len(best_sentences) >= 5:
            break

    body = " ".join(best_sentences) if best_sentences else matches[0]["text"]
    return f"{body}\n\n*(Source: Local Extractive RAG — top {len(matches)} matching chunk(s) below)*"


async def _call_openai_compat(provider: str, api_key: str, prompt: str) -> str:
    cfg = OPENAI_COMPAT_PROVIDERS[provider]
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "Answer accurately and concisely using only the provided content."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            cfg["url"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def _call_gemini(api_key: str, prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _resolve_provider(provider: str | None, api_key: str | None) -> tuple[str, str | None]:
    selected = (provider or "").strip().lower()
    explicit_key = (api_key or "").strip() or None

    if selected in {"", "auto"}:
        for name, env_name in (("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY")):
            env_val = os.getenv(env_name, "").strip()
            if env_val:
                return name, env_val
        return "local", None

    if selected == "local":
        return "local", None

    if explicit_key:
        return selected, explicit_key

    env_map = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
    env_val = os.getenv(env_map.get(selected, ""), "").strip() or None
    return (selected, env_val) if env_val else ("local", None)


async def _call_llm(provider: str, api_key: str, prompt: str) -> str:
    if provider == "gemini":
        return await _call_gemini(api_key, prompt)
    if provider in OPENAI_COMPAT_PROVIDERS:
        return await _call_openai_compat(provider, api_key, prompt)
    raise ValueError(f"Unsupported provider: {provider}")


async def summarize_content(
    title: str,
    text: str,
    provider: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Summarize a YouTube video's title and transcript in short."""
    resolved_provider, resolved_key = _resolve_provider(provider, api_key)
    if resolved_provider == "local" or not resolved_key:
        return {"summary": summarize_short(title, text), "mode": "local"}

    clipped = text[:12000]
    prompt = (
        f"YouTube Video Title: {title}\n\n"
        "Explain this video in short (1-line overview + 3 to 5 crisp bullet points) "
        "based ONLY on the title and content below:\n\n"
        f"{clipped}"
    )
    try:
        summary = await _call_llm(resolved_provider, resolved_key, prompt)
        return {"summary": summary, "mode": resolved_provider}
    except Exception as exc:
        fallback = summarize_short(title, text)
        return {
            "summary": f"{fallback}\n\n*(LLM summary failed: {exc}; showed local short summary instead.)*",
            "mode": f"local (fallback from {resolved_provider})",
        }


async def generate_answer(
    query: str,
    matches: list[dict[str, Any]],
    provider: str | None = None,
    api_key: str | None = None,
    source_title: str | None = None,
) -> dict[str, Any]:
    """Generate a RAG response using the selected LLM provider or local extractive fallback."""
    resolved_provider, resolved_key = _resolve_provider(provider, api_key)
    if resolved_provider == "local" or not resolved_key:
        return {
            "answer": _build_extractive_answer(query, matches),
            "mode": "local",
            "sources": matches,
        }

    title_line = f"Source Title: {source_title}\n\n" if source_title else ""
    context_block = "\n\n".join(f"[Chunk {m['index'] + 1} | score={m['score']}]\n{m['text']}" for m in matches)
    prompt = (
        f"{title_line}"
        "Answer the user question concisely based ONLY on the context below. "
        "If the context does not contain the answer, say so clearly.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}"
    )

    try:
        answer = await _call_llm(resolved_provider, resolved_key, prompt)
        return {"answer": answer, "mode": resolved_provider, "sources": matches}
    except Exception as exc:
        fallback = _build_extractive_answer(query, matches)
        return {
            "answer": f"{fallback}\n\n*(LLM call failed: {exc}; showed local extractive answer instead.)*",
            "mode": f"local (fallback from {resolved_provider})",
            "sources": matches,
        }
