# Reproducibility Statement

This repository accompanies the paper:

**An Empirical Study of Chunking Strategies in Financial and Regulatory Documents for Semantic and Numerical Integrity**

## Experimental Scope

The experiments evaluate how different document chunking strategies affect retrieval quality and answer faithfulness in Retrieval-Augmented Generation (RAG) systems applied to financial documents.

The study focuses on *relative comparisons* between chunking strategies rather than absolute performance optimization.

---

## Data

- **Dataset**: FinanceBench (financial question-answering benchmark)
- Source documents include earnings reports and regulatory filings.
- Access to FinanceBench may be subject to licensing restrictions.

The repository does not redistribute dataset contents.

---

## Chunking Strategies

The following chunking strategies are implemented and evaluated:

- Fixed-size chunking
- Layout-based chunking
- Recursive rule-based chunking
- Semantic-adjacent chunking

All strategies are evaluated under identical retrieval and context-window constraints.

---

## Evaluation Metrics

Evaluation is conducted using RAGAS metrics:

- Context Precision
- Context Recall
- Faithfulness

A fast LLM-based evaluator is used to reflect realistic latency and cost constraints.

---

## Determinism and Variability

Some components of the pipeline are non-deterministic due to:

- stochastic language model inference,
- API-based evaluation,
- retrieval ranking variability.

To mitigate this:
- identical prompts are used across strategies,
- evaluations are run under fixed retrieval settings,
- results are compared using aggregated metrics.

Reported results should be interpreted as *comparative trends*, not exact point estimates.

---

## Reproducing Results

To reproduce the experiments:

1. Obtain access to the FinanceBench dataset.
2. Configure API credentials where required.
3. Run scripts in the following order:

```bash
python experiments/retrieve_financebench.py
python experiments/generate_answers_openai.py
python experiments/eval_ragas_financebench_openai_fast.py
python experiments/make_paper_figures.py
