from datasets import load_dataset

DATASET_ID = "next-tat/TAT-QA"
OUT_DIR = "data/tatqa"

def main():
    ds = load_dataset(DATASET_ID)
    ds.save_to_disk(OUT_DIR)
    print(f"{DATASET_ID} downloaded to {OUT_DIR}")

if __name__ == "__main__":
    main()
