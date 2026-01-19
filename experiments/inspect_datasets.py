import json
from pathlib import Path
from datasets import load_from_disk

def inspect_financebench():
    print("\n=== FinanceBench (from data/financebench) ===")
    fb_path = Path("data/financebench")
    ds = load_from_disk(str(fb_path))
    # ds is usually a DatasetDict with splits
    print("Type:", type(ds))
    if hasattr(ds, "keys"):
        print("Splits:", list(ds.keys()))
        split = list(ds.keys())[0]
        d = ds[split]
    else:
        d = ds
    print("Rows:", len(d))
    print("Columns:", d.column_names)
    ex = d[0]
    print("\nSample keys:", list(ex.keys()))
    for k in list(ex.keys())[:10]:
        v = ex[k]
        s = str(v)
        print(f"- {k}: {type(v).__name__} | {s[:200]}{'...' if len(s)>200 else ''}")

def inspect_tatqa_raw():
    print("\n=== TAT-QA RAW (from data/tatqa_raw/*.json) ===")
    train_file = Path("data/tatqa_raw/tatqa_dataset_train.json")
    with train_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # TAT-QA is typically a list of items
    print("Top-level type:", type(data).__name__)
    if isinstance(data, list):
        print("Items:", len(data))
        ex = data[0]
    elif isinstance(data, dict):
        print("Keys:", list(data.keys())[:20])
        # pick first value if dict-of-lists
        first_key = next(iter(data.keys()))
        ex = data[first_key][0] if isinstance(data[first_key], list) else data[first_key]
    else:
        print("Unexpected top-level structure.")
        return

    print("\nSample item keys:", list(ex.keys()))
    for k in list(ex.keys())[:12]:
        v = ex[k]
        s = str(v)
        print(f"- {k}: {type(v).__name__} | {s[:200]}{'...' if len(s)>200 else ''}")

def main():
    inspect_financebench()
    inspect_tatqa_raw()

if __name__ == "__main__":
    main()
