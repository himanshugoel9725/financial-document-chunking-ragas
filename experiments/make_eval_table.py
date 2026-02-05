from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_from_disk

FB_PATH = Path("data/financebench")
OUT_PATH = Path("artifacts/eval_financebench.jsonl")

N = 50  # start small for cost control

def financebench_rows(n: int) -> List[Dict]:
    ds = load_from_disk(str(FB_PATH))["train"]
    rows = []
    for i in range(min(n, len(ds))):
        r = ds[i]
        evidence = r.get("evidence", [])
        parts = []
        for item in evidence:
            if isinstance(item, dict):
                parts.append((item.get("evidence_text") or item.get("evidence_text_full_page") or "").strip())
        doc_text = "\n\n".join([p for p in parts if p]).strip()

        rows.append({
            "id": r.get("financebench_id", f"fb_{i}"),
            "dataset": "financebench",
            "question": r.get("question", ""),
            "ground_truth": r.get("answer", ""),
            "doc_text": doc_text,
        })
    return rows

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = financebench_rows(N)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rows to {OUT_PATH}")
    print("Sample:")
    print(rows[0]["id"])
    print(rows[0]["question"][:120])
    print("doc_text chars:", len(rows[0]["doc_text"]))

if __name__ == "__main__":
    main()
