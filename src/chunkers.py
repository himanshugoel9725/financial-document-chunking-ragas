from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List

@dataclass
class Chunk:
    text: str
    start: int | None = None
    end: int | None = None

def chunk_fixed_chars(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """
    Baseline #1: Fixed-size character windows with overlap.
    - chunk_size: how big each chunk is (in characters)
    - overlap: how many characters repeat between adjacent chunks
    Why overlap exists: prevents cutting important context at chunk boundaries.
    """
    text = text or ""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: List[Chunk] = []
    i = 0
    n = len(text)
    step = chunk_size - overlap

    while i < n:
        j = min(i + chunk_size, n)
        chunk_text = text[i:j].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, start=i, end=j))
        i += step

    return chunks

_SECTION_BREAK_RE = re.compile(r"\n\s*\n+|\n(?=[A-Z][A-Z \t&/-]{3,}\n)")

def chunk_by_layout_breaks(text: str, max_chars: int = 1200) -> List[Chunk]:
    """
    Baseline #2: Layout/structure-ish chunking.
    Idea: split on "natural" boundaries:
    - blank lines (paragraph boundaries)
    - section-like headers (ALL CAPS-ish lines)
    Then merge blocks until each chunk is <= max_chars.
    """
    text = (text or "").strip()
    if not text:
        return []

    blocks = [b.strip() for b in _SECTION_BREAK_RE.split(text) if b and b.strip()]

    merged: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            merged.append(buf.strip())
        buf = ""

    for b in blocks:
        if not buf:
            buf = b
        elif len(buf) + 2 + len(b) <= max_chars:
            buf = buf + "\n\n" + b
        else:
            flush()
            buf = b
    flush()

    return [Chunk(text=m) for m in merged]
