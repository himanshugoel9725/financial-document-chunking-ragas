import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets import load_from_disk
from src.chunker_registry import CHUNKERS

def get_one_financebench_doc():
    ds = load_from_disk("data/financebench")["train"]
    row = ds[0]
    evidence = row.get("evidence", [])
    parts = []
    for item in evidence:
        if isinstance(item, dict):
            parts.append((item.get("evidence_text") or item.get("evidence_text_full_page") or "").strip())
    return "\n\n".join([p for p in parts if p])

def main():
    doc = get_one_financebench_doc()
    print("Doc length (chars):", len(doc))

    for name, chunker in CHUNKERS.items():
        chunks = chunker(doc)
        sizes = [len(c.text) for c in chunks]
        print(f"\n== {name} ==")
        print("chunks:", len(chunks))
        print("min/avg/max size:", min(sizes), "/", int(sum(sizes)/len(sizes)), "/", max(sizes))

if __name__ == "__main__":
    main()
