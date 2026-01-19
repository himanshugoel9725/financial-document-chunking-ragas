import sys
from pathlib import Path

def main():
    print("OK: smoke test running")
    print("Python:", sys.version.split()[0])
    print("Project root:", Path(__file__).resolve().parents[1])

if __name__ == "__main__":
    main()
