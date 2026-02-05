import json
from pathlib import Path
import requests
from tqdm import tqdm

IN_PATH = Path("artifacts/retrieval_financebench.jsonl")
OUT_PATH = Path("artifacts/answers_financebench_ollama.jsonl")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"   # << faster than llama3.1:8b on CPU

# SPEED CONTROLS
TOP_K = 3               # only use top 3 retrieved contexts
CTX_CHAR_LIMIT = 1200   # truncate each context chunk
MAX_TOKENS = 120        # cap answer length (fast)
TIMEOUT = 600           # seconds

SYSTEM = (
    "You are a careful financial QA assistant. "
    "Answer ONLY using the provided context. "
    "If the context is insufficient, say 'Insufficient context'. "
    "Return a short, direct answer."
)

def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_prompt(question: str, contexts):
    contexts = contexts[:TOP_K]
    cleaned = []
    for c in contexts:
        c = (c or "").strip()
        if not c:
            continue
        cleaned.append(c[:CTX_CHAR_LIMIT])

    ctx = "\n\n---\n\n".join(cleaned)
    return (
        f"{SYSTEM}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:"
    )

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.0,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IN_PATH}. Run: python -m experiments.retrieve_financebench"
        )

    rows = load_rows(IN_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for row in tqdm(rows, desc="Generate answers (Ollama)"):
            q = row["question"]
            ctxs = row.get("retrieved_contexts", [])
            prompt = build_prompt(q, ctxs)

            ans = ollama_generate(prompt)

            row["generated_answer"] = ans
            row["generator"] = f"ollama:{MODEL}"
            row["gen_top_k"] = TOP_K
            row["gen_ctx_char_limit"] = CTX_CHAR_LIMIT
            row["gen_max_tokens"] = MAX_TOKENS

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
