import html
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from youtube_transcript_api import YouTubeTranscriptApi

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "what", "when", "where", "which", "who", "why", "how",
    "this", "these", "those", "can", "could", "should", "would", "do", "does",
    "did", "about", "into", "or", "if", "your", "you", "me", "my", "we", "our",
    "so", "just", "like", "know", "going", "actually", "really", "yeah", "uh", "um",
}


def extract_video_id(url_or_id: str) -> str:
    """Extract an 11-character YouTube video ID from a URL or bare ID."""
    raw = url_or_id.strip()
    if not raw:
        raise ValueError("YouTube URL ya Video ID empty hai.")
    if VIDEO_ID_RE.match(raw):
        return raw

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.lower().lstrip("www.")

    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                candidate = parts[1]

    if not VIDEO_ID_RE.match(candidate):
        raise ValueError("Valid YouTube video URL ya 11-character video ID dein.")
    return candidate


async def fetch_video_metadata(video_id: str) -> dict[str, str]:
    """Fetch video title, channel name, thumbnail, and description from YouTube."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    meta = {
        "title": f"YouTube Video ({video_id})",
        "author": "YouTube Creator",
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "description": "",
        "watch_url": watch_url,
    }

    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        try:
            oembed_resp = await client.get(
                "https://www.youtube.com/oembed",
                params={"url": watch_url, "format": "json"},
            )
            if oembed_resp.status_code == 200:
                data = oembed_resp.json()
                meta["title"] = data.get("title") or meta["title"]
                meta["author"] = data.get("author_name") or meta["author"]
                meta["thumbnail_url"] = data.get("thumbnail_url") or meta["thumbnail_url"]
        except Exception:
            pass

        try:
            page_resp = await client.get(
                watch_url,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
            )
            if page_resp.status_code == 200:
                body = page_resp.text
                if meta["title"].startswith("YouTube Video ("):
                    title_match = re.search(r"<title>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
                    if title_match:
                        cleaned = html.unescape(title_match.group(1)).replace(" - YouTube", "").strip()
                        if cleaned:
                            meta["title"] = cleaned
                desc_match = re.search(r'"shortDescription":"((?:\\.|[^"\\])*)"', body)
                if desc_match:
                    meta["description"] = json.loads(f'"{desc_match.group(1)}"').strip()
        except Exception:
            pass

    return meta


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"


def fetch_transcript(video_id: str) -> dict[str, Any]:
    """Fetch transcript snippets for a YouTube video in English, Hindi, or any available language."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB", "en-IN", "hi"])
        except Exception:
            transcript = next(iter(transcript_list))

        fetched = transcript.fetch()
        lines = []
        plain_parts = []
        for snippet in fetched:
            text = re.sub(r"\s+", " ", snippet.text).strip()
            if not text:
                continue
            plain_parts.append(text)
            lines.append(f"[{_format_timestamp(snippet.start)}] {text}")

        plain_text = " ".join(plain_parts).strip()
        if not plain_text:
            return {"available": False, "text": "", "timestamped": "", "language": "", "error": "Transcript empty hai."}

        return {
            "available": True,
            "text": plain_text,
            "timestamped": "\n".join(lines),
            "language": f"{fetched.language} ({'auto' if fetched.is_generated else 'manual'})",
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "text": "",
            "timestamped": "",
            "language": "",
            "error": str(exc),
        }


def summarize_short(title: str, text: str, max_points: int = 4) -> str:
    """Build a short extractive summary using sentence importance across the transcript."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return f"Video Title: {title}\n\nIs video ka text/transcript available nahi ho paya."

    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", clean) if len(s.strip()) > 25]
    if len(sentences) <= 1:
        words = clean.split()
        sentences = [" ".join(words[i : i + 28]) for i in range(0, min(len(words), 140), 28)]

    words_all = [w for w in re.findall(r"[a-z0-9]+", clean.lower()) if w not in STOPWORDS and len(w) > 2]
    freq = Counter(words_all)
    if not freq:
        preview = clean[:400] + ("..." if len(clean) > 400 else "")
        return f"Video Title: {title}\n\nShort Summary:\n- {preview}"

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        tokens = [w for w in re.findall(r"[a-z0-9]+", sent.lower()) if w not in STOPWORDS and len(w) > 2]
        if not tokens:
            continue
        score = sum(freq[t] for t in tokens) / (len(tokens) ** 0.6)
        scored.append((score, idx, sent))

    top = sorted(scored, key=lambda item: item[0], reverse=True)[:max_points]
    top_in_order = sorted(top, key=lambda item: item[1])

    bullets = "\n".join(f"- {sent}" for _, _, sent in top_in_order)
    return f"Video Title: {title}\n\nShort Summary:\n{bullets}"
