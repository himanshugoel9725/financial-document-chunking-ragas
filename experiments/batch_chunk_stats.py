from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_from_disk

from src.chunker_registry import CHUNKERS


DATA_DIR = Path("data")
OUT_CSV = Path("artifacts/chunk_stats.csv")

# how many docs to sample from each dataset
N_DOCS = 200

# which datasets to include
DATASETS = {
    "financebench": DATA_DIR / "financebench",
    # tatqa is raw json; we read that separately below
    "tatqa_raw": DATA_DIR / "tatqa_raw" / "tatqa_dataset_train.json",
}


def financebench_docs(n: int) -> List[str]:
    ds = load_from_disk(str(DATASETS["financebench"]))["train"]
    docs: List[str] = []
    for row in ds.select(range(min(n, len(ds)))):
        evidence = row.get("evidence", [])
        parts = []
        for item in evidence:
            if isinstance(item, dict):
                txt = (item.get("evidence_text") or item.get("evidence_text_full_page") or "").strip()
                if txt:
                    parts.append(txt)
        doc = "\n\n".join(parts).strip()
        if doc:
            docs.append(doc)
    return docs


def tatqa_docs(n: int) -> List[str]:
    path = DATASETS["tatqa_raw"]
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: List[str] = []
    # each item has table + paragraphs; we convert to a single “doc” string
    for item in data[: min(n, len(data))]:
        table = item.get("table", {}).get("table", [])
        paragraphs = item.get("paragraphs", [])

        table_str = ""
        if isinstance(table, list) and table:
            # simple row join; keeps numbers and structure
            rows = [" | ".join(map(str, r)) for r in table if isinstance(r, list)]
            table_str = "\n".join(rows)

        para_str = "\n".join([p.get("text", "") for p in paragraphs if isinstance(p, dict) and p.get("text")])

        doc = "\n\n".join([s for s in [table_str, para_str] if s]).strip()
        if doc:
            docs.append(doc)

    return docs


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(1, len(text))


def run_one(dataset_name: str, docs: List[str]) -> List[Dict]:
    rows: List[Dict] = []

    for chunker_name, chunker in CHUNKERS.items():
        for i, doc in enumerate(docs):
            chunks = chunker(doc)
            sizes = [len(c.text) for c in chunks] if chunks else [0]

            # numeric heaviness proxy (helps your “numbers matter” argument)
            dr = [digit_ratio(c.text) for c in chunks] if chunks else [0.0]

            rows.append(
                {
                    "dataset": dataset_name,
                    "doc_index": i,
                    "chunker": chunker_name,
                    "doc_chars": len(doc),
                    "num_chunks": len(chunks),
                    "min_chunk_chars": min(sizes),
                    "avg_chunk_chars": sum(sizes) / max(1, len(sizes)),
                    "max_chunk_chars": max(sizes),
                    "avg_digit_ratio": sum(dr) / max(1, len(dr)),
                    "max_digit_ratio": max(dr),
                }
            )

    return rows


def main():
    print("Loading docs...")
    fb = financebench_docs(N_DOCS)
    tq = tatqa_docs(N_DOCS)

    print(f"FinanceBench docs: {len(fb)}")
    print(f"TAT-QA docs: {len(tq)}")

    all_rows: List[Dict] = []
    all_rows += run_one("financebench", fb)
    all_rows += run_one("tatqa_raw", tq)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved: {OUT_CSV} (rows={len(all_rows)})")
    print("Next: open artifacts/chunk_stats.csv and compare chunkers.")


if __name__ == "__main__":
    main()
