from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "next-tat/TAT-QA"
OUT_DIR = Path("data/tatqa_raw")

FILES = [
    "tatqa_dataset_train.json",
    "tatqa_dataset_dev.json",
    "tatqa_dataset_test.json",
    "tatqa_dataset_test_gold.json",
    "README.md",
]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset")
        target = OUT_DIR / fname
        target.write_bytes(Path(local_path).read_bytes())
        print(f"Saved: {target}")
    print("TAT-QA raw files downloaded to data/tatqa_raw")

if __name__ == "__main__":
    main()
