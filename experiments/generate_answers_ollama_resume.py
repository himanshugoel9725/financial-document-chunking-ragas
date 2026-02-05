import json
from pathlib import Path
from typing import Dict, Any, Set, Tuple

import requests
from tqdm import tqdm

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

IN_PATH = Path("artifacts/retrieval_financebench.jsonl")
OUT_PATH = Path("artifacts/answers_financebench_ollama.jsonl")

MODEL = "llama3.2:3b"   # faster than 8b on CPU
TEMPERATURE = 0.0
MAX_TOKENS = 256        # keep small for speed/cost
TIMEOUT = 600           # seconds


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_id(row: Dict[str, Any]) -> str:
    # handle multiple possible key names
    for k in ["financebench_id", "id", "qid", "example_id"]:
        if k in row and row[k]:
            return str(row[k])
    # fallback: stable hash-ish from question
    q = row.get("question", "")
    return f"noid::{hash(q)}"


def get_chunker(row: Dict[str, Any]) -> str:
    # retrieval file usually contains chunker name
    for k in ["chunker", "chunker_name", "chunking", "method"]:
        if k in row and row[k]:
            return str(row[k])
    return "unknown"


def build_prompt(question: str, contexts) -> str:
    # contexts can be list[str] or list[dict] etc.
    ctx_texts = []
    if isinstance(contexts, list):
        for c in contexts:
            if isinstance(c, str):
                ctx_texts.append(c.strip())
            elif isinstance(c, dict):
                # common keys
                for kk in ["text", "chunk", "context", "content"]:
                    if kk in c and c[kk]:
                        ctx_texts.append(str(c[kk]).strip())
                        break
    elif isinstance(contexts, str):
        ctx_texts.append(contexts.strip())

    ctx = "\n\n---\n\n".join([t for t in ctx_texts if t])

    return (
        "You are answering a question using ONLY the provided context.\n"
        "If the answer is not in the context, say: NOT_ENOUGH_INFORMATION.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "ANSWER (short, direct):"
    )


def ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def load_done_keys(path: Path) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not path.exists():
        return done
    for row in load_jsonl(path):
        qid = get_id(row)
        ch = get_chunker(row)
        done.add((qid, ch))
    return done


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing input: {IN_PATH}")

    # make sure ollama server is up
    try:
        _ = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
    except Exception as e:
        raise SystemExit(
            "Ollama server not responding. Start it in another terminal with:\n\n"
            "  ollama serve\n"
        ) from e

    done = load_done_keys(OUT_PATH)

    rows = list(load_jsonl(IN_PATH))
    pbar = tqdm(rows, desc=f"Generate answers (Ollama: {MODEL})")

    wrote = 0
    skipped = 0

    for row in pbar:
        qid = get_id(row)
        ch = get_chunker(row)
        key = (qid, ch)

        if key in done:
            skipped += 1
            continue

        question = row.get("question", "")
        contexts = row.get("retrieved_contexts") or row.get("contexts") or row.get("retrieved") or []

        prompt = build_prompt(question, contexts)

        try:
            answer = ollama_generate(prompt)
        except Exception as e:
            # save the error so you can inspect later and still continue
            out = {
                "financebench_id": qid,
                "chunker": ch,
                "question": question,
                "answer": None,
                "error": str(e),
            }
            append_jsonl(OUT_PATH, out)
            done.add(key)
            wrote += 1
            continue

        out = {
            "financebench_id": qid,
            "chunker": ch,
            "question": question,
            "answer": answer,
            "model": MODEL,
        }
        append_jsonl(OUT_PATH, out)
        done.add(key)
        wrote += 1

    print(f"\nDone. wrote={wrote}, skipped(existing)={skipped}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
