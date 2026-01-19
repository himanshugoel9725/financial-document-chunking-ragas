import json
from pathlib import Path
from datasets import load_from_disk

def normalize_financebench_evidence(evidence):
    """
    evidence is typically a list[dict] with keys like:
    - evidence_text
    - evidence_text_full_page
    """
    if not evidence:
        return ""
    if isinstance(evidence, str):
        return evidence

    chunks = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                txt = item.get("evidence_text") or item.get("evidence_text_full_page") or ""
                if txt:
                    chunks.append(txt.strip())
            else:
                chunks.append(str(item).strip())
        return "\n\n".join([c for c in chunks if c])

    # fallback
    return str(evidence)

def load_financebench_examples(limit=50):
    ds = load_from_disk("data/financebench")["train"]
    examples = []
    for i in range(min(limit, len(ds))):
        row = ds[i]
        doc_text = normalize_financebench_evidence(row.get("evidence"))
        examples.append({
            "dataset": "financebench",
            "id": row["financebench_id"],
            "question": row["question"],
            "gold_answer": row["answer"],
            "doc_text": doc_text,
        })
    return examples

def flatten_table(table_dict):
    rows = table_dict.get("table", [])
    lines = []
    for r in rows:
        lines.append(" | ".join([str(c) for c in r]))
    return "\n".join(lines)

def load_tatqa_examples(split_file="data/tatqa_raw/tatqa_dataset_train.json", limit=50):
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    examples = []
    count = 0
    for item in data:
        table_text = flatten_table(item.get("table", {}))
        paras = item.get("paragraphs", [])
        paras_sorted = sorted(paras, key=lambda x: x.get("order", 0))
        paras_text = "\n".join([p.get("text", "") for p in paras_sorted])

        doc_text = table_text + "\n\n" + paras_text

        for q in item.get("questions", []):
            examples.append({
                "dataset": "tatqa",
                "id": q.get("uid", f"q_{count}"),
                "question": q.get("question", ""),
                "gold_answer": str(q.get("answer", "")),
                "doc_text": doc_text,
            })
            count += 1
            if count >= limit:
                return examples
    return examples

def main():
    fb = load_financebench_examples(limit=3)
    tq = load_tatqa_examples(limit=3)

    print("FinanceBench sample:")
    for ex in fb:
        print(ex["dataset"], ex["id"])
        print("Q:", ex["question"][:120])
        print("DOC:", ex["doc_text"][:200])
        print("A:", ex["gold_answer"])
        print("---")

    print("\nTAT-QA sample:")
    for ex in tq:
        print(ex["dataset"], ex["id"])
        print("Q:", ex["question"][:120])
        print("DOC:", ex["doc_text"][:200])
        print("A:", ex["gold_answer"])
        print("---")

if __name__ == "__main__":
    main()
