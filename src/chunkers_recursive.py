from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable
import re

@dataclass
class Chunk:
    text: str
    meta: dict | None = None


def _split_paragraphs(text: str) -> List[str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    return parts


def split_sentences_rule(text: str) -> List[str]:
    t = " ".join(text.split())
    if not t:
        return []
    sents = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in sents if s.strip()]


def split_sentences_nltk(text: str) -> List[str]:
    t = text.strip()
    if not t:
        return []
    from nltk.tokenize import sent_tokenize
    return [s.strip() for s in sent_tokenize(t) if s.strip()]


def _looks_table_like(paragraph: str) -> bool:
    # If it contains many short lines with numbers/currency, treat as table-ish
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False

    numericish = 0
    for ln in lines:
        has_num = bool(re.search(r"\d", ln))
        has_money = "$" in ln
        has_commas = bool(re.search(r"\d{1,3}(,\d{3})+", ln))
        has_pipe = "|" in ln
        if has_pipe or (has_num and (has_money or has_commas)):
            numericish += 1

    return (numericish / max(1, len(lines))) >= 0.4


def chunk_recursive(
    text: str,
    max_chars: int = 1200,
    min_chars: int = 300,
    overlap_chars: int = 0,
    sentence_splitter: Callable[[str], List[str]] = split_sentences_rule,
) -> List[Chunk]:
    """
    Meaning-first chunker with guards:
    - Prefer paragraph splits
    - Sentence split only for prose paragraphs (not tables)
    - Merge tiny chunks to satisfy min_chars
    """
    if not text or not text.strip():
        return []

    paras = _split_paragraphs(text)
    raw_chunks: List[str] = []

    def pack_units(units: List[str]) -> None:
        buf = ""
        for u in units:
            u = u.strip()
            if not u:
                continue

            # Hard fallback if unit itself is too big
            if len(u) > max_chars:
                if buf.strip():
                    raw_chunks.append(buf.strip())
                    buf = ""
                for i in range(0, len(u), max_chars):
                    piece = u[i : i + max_chars].strip()
                    if piece:
                        raw_chunks.append(piece)
                continue

            if not buf:
                buf = u
            elif len(buf) + 1 + len(u) <= max_chars:
                buf = buf + " " + u
            else:
                raw_chunks.append(buf.strip())
                buf = u

        if buf.strip():
            raw_chunks.append(buf.strip())

    for p in paras:
        if len(p) <= max_chars:
            raw_chunks.append(p)
            continue

        # If table-like, do NOT sentence split — just size-pack by lines
        if _looks_table_like(p):
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            pack_units(lines)
        else:
            sents = sentence_splitter(p) or [p]
            pack_units(sents)

    # Merge tiny chunks forward to satisfy min_chars (without exceeding max_chars)
    merged: List[str] = []
    buf = ""
    for c in raw_chunks:
        c = c.strip()
        if not c:
            continue
        if not buf:
            buf = c
            continue

        # If current buffer is too small, try to merge
        if len(buf) < min_chars and len(buf) + 2 + len(c) <= max_chars:
            buf = buf + "\n\n" + c
        else:
            merged.append(buf)
            buf = c

    if buf.strip():
        merged.append(buf)

    # Optional overlap
    if overlap_chars > 0 and len(merged) > 1:
        out: List[str] = []
        prev = ""
        for c in merged:
            if prev:
                prefix = prev[-overlap_chars:]
                out.append((prefix + " " + c).strip())
            else:
                out.append(c)
            prev = c
        merged = out

    return [Chunk(text=c, meta={"chunker": "recursive"}) for c in merged]
