from datasets import load_dataset

DATASET_ID = "ibm-research/finqa"
OUT_DIR = "data/finqa"

def main():
    ds = load_dataset(DATASET_ID)
    ds.save_to_disk(OUT_DIR)
    print(f"{DATASET_ID} downloaded to {OUT_DIR}")

if __name__ == "__main__":
    main()
