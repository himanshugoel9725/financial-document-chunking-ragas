from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ARTIFACTS = Path("artifacts")
PLOTS = Path("artifacts/plots")
PLOTS.mkdir(parents=True, exist_ok=True)

# Prefer the "fast openai" output if present, else fall back to ragas_results.csv
candidates = [
    ARTIFACTS / "ragas_results_openai_fast.csv",
    ARTIFACTS / "ragas_results.csv",
]
csv_path = None
for c in candidates:
    if c.exists():
        csv_path = c
        break

if csv_path is None:
    # fallback: pick any ragas csv in artifacts
    any_csv = sorted(ARTIFACTS.glob("ragas_results*.csv"))
    if any_csv:
        csv_path = any_csv[0]

if csv_path is None or not csv_path.exists():
    raise SystemExit(
        "❌ Could not find a RAGAS results CSV in artifacts/. "
        "Expected ragas_results_openai_fast.csv or ragas_results.csv"
    )

print(f"✅ Using results file: {csv_path}")

df = pd.read_csv(csv_path)

# Normalize column names (some scripts output different column names)
# Common possibilities:
# chunker, context_precision, context_recall, faithfulness, answer_relevancy, n_examples
if "chunker" not in df.columns:
    # try other common name
    for alt in ["strategy", "chunking_strategy", "method"]:
        if alt in df.columns:
            df["chunker"] = df[alt]
            break

if "chunker" not in df.columns:
    raise SystemExit("❌ Could not find 'chunker' column in CSV.")

# Keep only the chunker + numeric metric columns we care about
metric_cols = [c for c in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"] if c in df.columns]

if not metric_cols:
    raise SystemExit(f"❌ No known metric columns found. CSV columns: {list(df.columns)}")

# Convert metrics to numeric safely
for c in metric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Keep a stable order (your four chunkers)
preferred_order = ["fixed", "layout", "recursive_rule", "semantic_adjacent"]
order = [c for c in preferred_order if c in set(df["chunker"].astype(str))]
# add any extras
order += [c for c in df["chunker"].astype(str).unique() if c not in order]

df["chunker"] = df["chunker"].astype(str)
df = df.set_index("chunker").loc[order].reset_index()

# Helper: save as both png + pdf for paper-ready figures
def save_fig(name: str):
    png = PLOTS / f"{name}.png"
    pdf = PLOTS / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    print(f"✅ Saved: {png}")
    print(f"✅ Saved: {pdf}")
    plt.close()

# -----------------------------
# FIGURE 1: Precision & Recall
# -----------------------------
if "context_precision" in df.columns and "context_recall" in df.columns:
    x = range(len(df))
    width = 0.38

    plt.figure()
    plt.bar([i - width/2 for i in x], df["context_precision"], width=width, label="context_precision")
    plt.bar([i + width/2 for i in x], df["context_recall"], width=width, label="context_recall")
    plt.xticks(list(x), df["chunker"], rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("RAGAS: Context Precision & Recall by Chunking Strategy")
    plt.legend()
    save_fig("fig1_precision_recall_by_chunker")
else:
    print("⚠️ Skipping Fig1 (missing context_precision/context_recall columns).")

# -----------------------------
# FIGURE 2: Faithfulness
# -----------------------------
if "faithfulness" in df.columns:
    plt.figure()
    plt.bar(df["chunker"], df["faithfulness"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("RAGAS: Faithfulness by Chunking Strategy")
    plt.xticks(rotation=15, ha="right")
    save_fig("fig2_faithfulness_by_chunker")
else:
    print("⚠️ Skipping Fig2 (missing faithfulness column).")

# -----------------------------
# FIGURE 3: Summary table image (simple)
# -----------------------------
# Make a clean table figure you can drop into appendix/slides
plt.figure(figsize=(10, 2.2))
plt.axis("off")

table_cols = ["chunker"] + metric_cols
table_df = df[table_cols].copy()

# Round for readability
for c in metric_cols:
    table_df[c] = table_df[c].round(3)

tbl = plt.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.4)

plt.title("RAGAS Metrics Summary (means)")
save_fig("fig3_metrics_summary_table")

print("\n✅ Done. All figures are in:", PLOTS)
