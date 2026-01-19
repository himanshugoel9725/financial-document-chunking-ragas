import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
from pathlib import Path
from statistics import mean
from datasets import load_from_disk
from src.chunkers import chunk_fixed_chars, chunk_by_layout_breaks

def financebench_docs(limit=50):
    ds = load_from_disk("data/financebench")["train"]
    docs = []
    for i in range(min(limit, len(ds))):
        evidence = ds[i].get("evidence")
        parts = []
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict):
                    parts.append((item.get("evidence_text") or item.get("evidence_text_full_page") or "").strip())
        docs.append("\n\n".join([p for p in parts if p]))
    return [d for d in docs if d.strip()]

def tatqa_docs(limit=50):
    data = json.loads(Path("data/tatqa_raw/tatqa_dataset_train.json").read_text(encoding="utf-8"))
    docs = []
    for item in data[:limit]:
        rows = item.get("table", {}).get("table", [])
        table_text = "\n".join([" | ".join([str(c) for c in r]) for r in rows])

        paras = sorted(item.get("paragraphs", []), key=lambda x: x.get("order", 0))
        paras_text = "\n".join([p.get("text", "") for p in paras])

        docs.append((table_text + "\n\n" + paras_text).strip())
    return [d for d in docs if d.strip()]

def summarize(title, counts, lengths):
    print(f"\n=== {title} ===")
    print("Docs:", len(counts))
    print("Avg chunks/doc:", round(mean(counts), 2))
    print("Min/Max chunks/doc:", min(counts), "/", max(counts))
    print("Avg chunk length (chars):", int(mean(lengths)))
    print("Min/Max chunk length:", min(lengths), "/", max(lengths))

def run_stats(dataset_name, docs):
    fixed_counts, fixed_lens = [], []
    layout_counts, layout_lens = [], []

    for d in docs:
        fixed = chunk_fixed_chars(d, chunk_size=1000, overlap=200)
        layout = chunk_by_layout_breaks(d, max_chars=1200)

        fixed_counts.append(len(fixed))
        fixed_lens.extend([len(c.text) for c in fixed])

        layout_counts.append(len(layout))
        layout_lens.extend([len(c.text) for c in layout])

    summarize(f"{dataset_name} | fixed_chars(1000,200)", fixed_counts, fixed_lens)
    summarize(f"{dataset_name} | layout_breaks(max=1200)", layout_counts, layout_lens)

def main():
    fb = financebench_docs(limit=50)
    tq = tatqa_docs(limit=50)

    run_stats("FinanceBench", fb)
    run_stats("TAT-QA", tq)

if __name__ == "__main__":
    main()
