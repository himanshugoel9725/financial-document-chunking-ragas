from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

# ---- CONFIG (tune these, but these defaults are "fast and worth it") ----
ANSWERS_PATH = Path("artifacts/answers_financebench_ragas_ready_v3_merged.jsonl")
EVAL_PATH    = Path("artifacts/eval_financebench_ragas_ready.jsonl")
OUT_PATH     = Path("artifacts/ragas_results_openai_fast.csv")

N_PER_CHUNKER = 25     # 25 = meaningful, 50 = stronger but slower/costlier
TOP_K_CTX     = 3      # only top 3 contexts
CTX_CHARS     = 900    # truncate each context to 900 chars
MAX_ANSWER_CHARS = 1200

# ---- OpenAI via LangChain (works with older ragas + wrappers) ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

lc_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=512,          # short judge outputs
    timeout=60,              # per request timeout
    max_retries=2,
)
lc_emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    timeout=60,
    max_retries=2,
)

ragas_llm = LangchainLLMWrapper(lc_llm)
ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

METRICS = [context_precision, context_recall, faithfulness, answer_relevancy]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_id(row: Dict[str, Any]) -> str:
    return str(row.get("financebench_id") or row.get("id") or "noid")


def clip_contexts(ctxs: Any) -> List[str]:
    if not isinstance(ctxs, list):
        return []
    out = []
    for c in ctxs[:TOP_K_CTX]:
        s = str(c).strip()
        if s:
            out.append(s[:CTX_CHARS])
    return out


def main():
    answers = load_jsonl(ANSWERS_PATH)
    gold_rows = load_jsonl(EVAL_PATH)
    gold = {pick_id(r): r for r in gold_rows}

    by_chunker = defaultdict(list)

    for a in answers:
        qid = pick_id(a)
        g = gold.get(qid, {})

        question = a.get("question") or g.get("question") or ""
        answer = (a.get("answer") or a.get("generated_answer") or "")[:MAX_ANSWER_CHARS]
        reference = g.get("answer") or g.get("ground_truth") or a.get("ground_truth") or ""

        ctxs = clip_contexts(a.get("retrieved_contexts") or a.get("contexts"))

        chunker = a.get("chunker", "unknown")

        # Provide BOTH naming schemes (ragas versions differ)
        row = {
            # common names
            "question": question,
            "answer": answer,
            "contexts": ctxs,
            "ground_truth": reference,

            # newer/alt names some setups expect
            "user_input": question,
            "response": answer,
            "retrieved_contexts": ctxs,
            "reference": reference,

            "chunker": chunker,
        }
        by_chunker[chunker].append(row)

    results = []
    for chunker, rows in by_chunker.items():
        rows = rows[:N_PER_CHUNKER]
        missing = sum(1 for r in rows if not r["contexts"] and not r["retrieved_contexts"])
        print(f"\nEvaluating chunker: {chunker} | examples={len(rows)} | missing_contexts={missing}")

        ds = Dataset.from_list(rows)

        # Try both evaluate() signatures (ragas versions differ)
        try:
            scores = evaluate(
                ds,
                metrics=METRICS,
                llm=ragas_llm,
                embeddings=ragas_emb,
                raise_exceptions=False,
            )
        except TypeError:
            scores = evaluate(
                ds,
                metrics=METRICS,
                raise_exceptions=False,
            )

        df = scores.to_pandas()

        metric_cols = [c for c in df.columns if c in ["context_precision","context_recall","faithfulness","answer_relevancy"]]
        means = (
            df[metric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean(numeric_only=True)
            .to_dict()
        )
        means["chunker"] = chunker
        means["n_examples"] = len(rows)
        means["n_missing_contexts"] = missing
        results.append(means)

    out_df = pd.DataFrame(results).sort_values("chunker")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print(out_df)


if __name__ == "__main__":
    main()
