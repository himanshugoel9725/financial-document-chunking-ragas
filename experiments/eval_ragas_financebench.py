from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _clip_ctx(xs, k=3, n=1500):
    if not isinstance(xs, list):
        return []
    out = []
    for x in xs[:k]:
        if x is None:
            continue
        out.append(str(x)[:n])
    return out

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ---------- SMALL + SAFE ----------
N_PER_CHUNKER = 5
TOP_K_CTX = 1
MAX_CTX_CHARS = 350
MAX_Q_CHARS = 250
MAX_A_CHARS = 250
MAX_REF_CHARS = 250

# Force serial execution
os.environ["RAGAS_MAX_WORKERS"] = "1"
os.environ["RAGAS_NUM_WORKERS"] = "1"

EVAL_PATH = Path("artifacts/eval_financebench_ragas_ready.jsonl")
ANSWERS_PATH = Path("artifacts/answers_financebench_ragas_ready_v3_merged.jsonl")
OUT_PATH = Path("artifacts/ragas_results.csv")


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


def clip(x: Any, n: int) -> str:
    s = "" if x is None else str(x)
    s = " ".join(s.split())
    return s[:n]


def clip_contexts(ctxs: Any) -> List[str]:
    if not isinstance(ctxs, list):
        return []
    out = []
    for c in ctxs:
        c = clip(c, MAX_CTX_CHARS)
        if c.strip():
            out.append(c)
        if len(out) >= TOP_K_CTX:
            break
    return out


def get_run_config():
    """
    Ragas has moved RunConfig between versions.
    This tries multiple import locations.
    """
    for mod, name in [
        ("ragas", "RunConfig"),
        ("ragas.run_config", "RunConfig"),
        ("ragas.config", "RunConfig"),
    ]:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception:
            continue
    return None


def main():
    # Local judge + embeddings
    lc_llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.0,
        max_tokens=128,
        num_predict=96,
        request_timeout=1200,  # ollama call can take longer
    )
    lc_emb = OllamaEmbeddings(model="nomic-embed-text")

    ragas_llm = LangchainLLMWrapper(lc_llm)
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    METRICS = [context_precision, context_recall]

    answers = load_jsonl(ANSWERS_PATH)
    gold_rows = load_jsonl(EVAL_PATH)
    gold = {pick_id(r): r for r in gold_rows}

    by_chunker = defaultdict(list)
    for a in answers:
        qid = pick_id(a)
        g = gold.get(qid, {})

        # IMPORTANT: use Ragas-v0.x column names
        row = {
            "user_input": clip(a.get("question") or g.get("question"), MAX_Q_CHARS),
            "response": clip(a.get("answer") or a.get("generated_answer"), MAX_A_CHARS),
            "reference": clip(g.get("answer") or g.get("ground_truth") or a.get("ground_truth"), MAX_REF_CHARS),
            "retrieved_contexts": _clip_ctx(a.get("retrieved_contexts")),
            "chunker": a.get("chunker", "unknown"),
        }
        by_chunker[row["chunker"]].append(row)

    RunConfig = get_run_config()
    run_config = None
    if RunConfig is not None:
        # This is the key fix: stop the 180s timeout
        try:
            run_config = RunConfig(timeout=1200, max_workers=1)
        except TypeError:
            # some versions use different field names
            run_config = RunConfig(timeout=1200)

    results = []
    for chunker, rows in by_chunker.items():
        rows = rows[:N_PER_CHUNKER]
        missing = sum(1 for r in rows if not r["retrieved_contexts"])
        print(f"Evaluating chunker: {chunker} | examples={len(rows)} | missing_contexts={missing}")

        ds = Dataset.from_list([{k: v for k, v in r.items() if k != "chunker"} for r in rows])

        kwargs = dict(
            dataset=ds,
            metrics=METRICS,
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False,
        )
        if run_config is not None:
            kwargs["run_config"] = run_config

        scores = evaluate(**kwargs)

        df = scores.to_pandas()
        # only average numeric metric columns
        means = (
            df.apply(pd.to_numeric, errors="coerce")
            .mean(numeric_only=True)
            .to_dict()
        )
        means["chunker"] = chunker
        means["n_examples"] = len(rows)
        means["n_missing_contexts"] = missing
        results.append(means)

    out_df = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print("\nSaved RAGAS results to:", OUT_PATH)
    print(out_df)


if __name__ == "__main__":
    main()
