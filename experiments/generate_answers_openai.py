from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

IN_PATH = Path("artifacts/retrieval_financebench.jsonl")
OUT_PATH = Path("artifacts/answers_openai_financebench.jsonl")

SYSTEM_PROMPT = """You are a careful financial QA assistant.
Use ONLY the provided context to answer the user's question.
If the answer is not in the context, say: "Not enough information in the provided context."
Return a short, direct answer (no extra commentary)."""

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_context(row: Dict[str, Any]) -> str:
    # Most retrievers store contexts like: row["retrieved_contexts"] = [{"text": "...", "score": ...}, ...]
    ctx_items = row.get("retrieved_contexts", [])
    parts = []
    for i, item in enumerate(ctx_items, start=1):
        if isinstance(item, dict):
            t = (item.get("text") or item.get("chunk") or item.get("content") or "").strip()
        else:
            t = str(item).strip()
        if t:
            parts.append(f"[Context {i}]\n{t}")
    return "\n\n".join(parts).strip()

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key)

    rows = load_jsonl(IN_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            question = row.get("question", "").strip()
            context = build_context(row)

            user_prompt = f"""Question:
{question}

Context:
{context}
"""

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )

            answer = resp.choices[0].message.content.strip()

            out_row = dict(row)
            out_row["model_provider"] = "openai"
            out_row["model_name"] = model
            out_row["model_answer"] = answer

            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(rows)}")

    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
