# Financial Document Chunking + RAGAS Evaluation

This repository contains the experimental code and evaluation pipeline for the research paper:

**An Empirical Study of Chunking Strategies in Financial and Regulatory Documents for Semantic and Numerical Integrity**

The goal of this project is to empirically analyze how different document chunking strategies affect retrieval quality, information coverage, and answer faithfulness in Retrieval-Augmented Generation (RAG) systems applied to financial and regulatory documents.

---

## Motivation

Financial and regulatory documents (e.g., earnings reports, regulatory filings, compliance documents) are:

- long and structurally complex,
- densely populated with numerical values, tables, and cross-references,
- highly sensitive to context fragmentation.

Chunking—segmenting documents into retrievable units—directly determines whether semantic and numerical relationships are preserved during retrieval and downstream reasoning.  
Despite this, chunking is often treated as a secondary preprocessing step and rarely evaluated rigorously in financial RAG pipelines.

This repository treats chunking as a **meaning-preservation problem**, not merely a technical convenience.

---

## Chunking Strategies Evaluated

All strategies are evaluated under identical retrieval, context-window, and latency constraints.

- **Fixed-size chunking**  
  Splits documents into uniform token-length chunks.

- **Layout-based chunking**  
  Aligns chunks with document structure such as sections or headings.

- **Recursive rule-based chunking**  
  Applies hierarchical splitting rules with minimum-size and table guards.

- **Semantic-adjacent chunking**  
  Groups adjacent segments that are semantically related to preserve contextual continuity.

---

## Evaluation Framework

- **Dataset**: FinanceBench (real-world financial QA benchmark)
- **Pipeline**: Retrieval → Answer Generation → Evaluation
- **Metrics** (via RAGAS):
  - Context Precision
  - Context Recall
  - Faithfulness
- **Evaluation Mode**: Fast LLM-based judge to reflect realistic cost and latency constraints

The evaluation isolates **chunking strategy** as the primary independent variable.

---

## Repository Structure

```text
.
├── src/
│   ├── chunker_registry.py        # Central registry for chunking strategies
│   ├── chunkers_semantic.py       # Semantic-adjacent chunking implementation
│   └── __init__.py
│
├── experiments/
│   ├── retrieve_financebench.py
│   ├── generate_answers_openai.py
│   ├── generate_answers_ollama.py
│   ├── generate_answers_ollama_resume.py
│   ├── eval_ragas_financebench.py
│   ├── eval_ragas_financebench_openai_fast.py
│   ├── batch_chunk_stats.py
│   ├── make_eval_table.py
│   └── make_paper_figures.py
│
├── artifacts/                     # Generated outputs (CSV results, plots)
├── constraints.txt
├── README.md
└── .gitignore
