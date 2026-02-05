from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.chunker_registry import CHUNKERS

IN_PATH = Path("artifacts/eval_financebench.jsonl")
OUT_PATH = Path("artifacts/retrieval_financebench.jsonl")

TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_rows() -> List[Dict]:
    rows = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    We use cosine similarity by normalizing vectors and using inner product index.
    """
    vecs = vectors.astype("float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def retrieve_top_k(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunks: List[str],
    k: int,
) -> List[Tuple[int, float, str]]:
    q = query_vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    index = build_faiss_index(chunk_vecs)
    scores, idxs = index.search(q, k)

    out = []
    for rank, (i, s) in enumerate(zip(idxs[0].tolist(), scores[0].tolist()), start=1):
        out.append((i, float(s), chunks[i]))
    return out


def main():
    rows = load_rows()
    model = SentenceTransformer(EMBED_MODEL)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for r in tqdm(rows, desc="Retrieval (FinanceBench)"):
            q = r["question"]
            doc_text = r["doc_text"]

            for chunker_name, chunker_fn in CHUNKERS.items():
                # 1) chunk doc
                chunk_objs = chunker_fn(doc_text)
                chunks = [c.text for c in chunk_objs if c.text.strip()]

                if not chunks:
                    continue

                # 2) embed chunks + query
                chunk_vecs = model.encode(
                    chunks,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
                q_vec = model.encode(
                    [q],
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )[0]

                # 3) retrieve
                top = retrieve_top_k(q_vec, chunk_vecs, chunks, TOP_K)

                out_row = {
                    "id": r["id"],
                    "dataset": r["dataset"],
                    "chunker": chunker_name,
                    "question": q,
                    "ground_truth": r["ground_truth"],
                    "retrieved_contexts": [t[2] for t in top],
                    "retrieved_scores": [t[1] for t in top],
                    "n_chunks_total": len(chunks),
                }
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Saved retrieval results to {OUT_PATH}")
    print("Next step: call LLM to generate answers using retrieved_contexts.")


if __name__ == "__main__":
    main()
