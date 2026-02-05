from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import re

@dataclass
class Chunk:
    text: str
    meta: dict | None = None


def _split_paragraphs(text: str) -> List[str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    return parts


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def chunk_semantic_adjacent(
    text: str,
    max_chars: int = 1200,
    min_chars: int = 300,
    similarity_threshold: float = 0.78,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> List[Chunk]:
    """
    Baseline semantic chunking:
    1) Split into paragraph units
    2) Embed each unit
    3) Merge adjacent units while they remain semantically similar
       and chunk size stays within [min_chars, max_chars]
    """
    if not text or not text.strip():
        return []

    units = _split_paragraphs(text)
    if not units:
        return []

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    embs = model.encode(units, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

    chunks: List[str] = []
    buf_text = units[0]
    buf_emb = embs[0]

    for i in range(1, len(units)):
        cand_text = units[i]
        cand_emb = embs[i]

        sim = _cosine(buf_emb, cand_emb)

        # Decision: merge if semantically close OR buffer is still too small,
        # and we won't exceed max_chars
        should_merge = (sim >= similarity_threshold) or (len(buf_text) < min_chars)

        if should_merge and (len(buf_text) + 2 + len(cand_text) <= max_chars):
            buf_text = buf_text + "\n\n" + cand_text
            # Update embedding as mean (simple)
            buf_emb = (buf_emb + cand_emb) / 2.0
            # normalize again (keep cosine stable)
            buf_emb = buf_emb / (np.linalg.norm(buf_emb) + 1e-12)
        else:
            chunks.append(buf_text.strip())
            buf_text = cand_text
            buf_emb = cand_emb

    if buf_text.strip():
        chunks.append(buf_text.strip())

    return [Chunk(text=c, meta={"chunker": "semantic_adjacent", "threshold": similarity_threshold}) for c in chunks]
