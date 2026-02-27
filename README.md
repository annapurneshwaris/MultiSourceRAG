# HeteroRAG: Multi-Source Heterogeneous RAG for Technical Support

A multi-source heterogeneous Retrieval-Augmented Generation framework that learns to route queries across documentation, bug reports, and work items using a contextual bandit.

Built on the VS Code open-source ecosystem (vscode-docs + microsoft/vscode issues).

## Architecture

```
Query -> Learned Source Router (LinUCB) -> Per-Source Retrieval -> Diversity-Constrained Re-Ranker -> LLM Generation
                ^                                                                                        |
                |_________________________ Utility Signal Feedback _______________________________________|
```

**7-step pipeline:** Route -> Retrieve -> Re-rank -> Generate -> Compute utility -> Update router -> Persist

## Three Contributions

1. **C1:** Multi-source heterogeneous RAG with learned source router (LinUCB contextual bandit) + diversity-constrained retrieval
2. **C2:** Resolution-centric evaluation metrics (RA = RCI+AS+VM, CSAS, MSUR)
3. **C3:** Open-source three-source benchmark dataset (VS Code ecosystem)

## Setup

```bash
# Clone
git clone https://github.com/ANU-Sav/MultiSourceRAG.git
cd MultiSourceRAG

# Install
pip install -r requirements.txt

# Environment
cp .env.example .env
# Add your OPENAI_API_KEY and optionally ANTHROPIC_API_KEY, GITHUB_TOKEN
```

## Reproduction Steps

### 1. Data Collection (ingestion)

```bash
python collect_all.py
```

Collects docs, bugs, work items, and comments into `data/raw/`.

### 2. Chunking (processing)

```bash
python -m MultiSourceRAG process
```

**Output files:**
| File | Description |
|------|-------------|
| `data/processed/doc_chunks.json` | ~2,000-4,000 doc chunks |
| `data/processed/bug_chunks.json` | ~8,000-14,000 bug chunks |
| `data/processed/workitem_chunks.json` | ~3,000-5,000 work item chunks |
| `data/processed/all_chunks.json` | Combined chunks |
| `data/processed/doc_tree.json` | Hierarchical doc tree |
| `data/processed/processing_stats.json` | Processing statistics |

### 3. Indexing

```bash
python -m MultiSourceRAG index
```

**Output files:**
| Directory | Description |
|-----------|-------------|
| `data/indices/combined/` | FAISS + metadata for all chunks |
| `data/indices/bm25/` | BM25 keyword index |

### 4. Query (single)

```bash
python -m MultiSourceRAG query "How to configure terminal colors?" --config DBW --router adaptive
```

### 5. Ablation Experiments

```bash
# Full ablation: 9 configs x 250 queries
python -m MultiSourceRAG ablation

# Subset
python -m MultiSourceRAG ablation --configs D,DBW --queries 10

# Router sweep (heuristic + adaptive + llm_zeroshot on DBW)
python -m MultiSourceRAG ablation --router-sweep
```

**Output:** `data/evaluation/ablation_results.json`

### 6. LLM Judge Scoring

```bash
# GPT-4o judge (primary)
python -m MultiSourceRAG judge

# Claude cross-validation
python -m MultiSourceRAG judge --judge-model claude
```

**Output:** `data/evaluation/judge_scores.json`

### 7. Offline Router Training

```bash
python -m MultiSourceRAG train --epochs 3
```

**Output:** `data/models/adaptive_router/`

### 8. API + UI

```bash
python -m MultiSourceRAG serve --port 8000
```

- API docs: `http://localhost:8000/docs`
- Gradio UI: launched alongside API

## Project Structure

```
MultiSourceRAG/
+-- ingestion/           # Data collection (GitHub API + docs clone)
+-- processing/          # Chunking (doc, bug, work item)
+-- indexing/            # Embedding + FAISS/BM25 index building
+-- retrieval/           # Pipeline, router, retrievers, reranker
+-- generation/          # LLM providers, prompts, citations
+-- evaluation/          # Ablation runner, LLM judge, metrics, query bank
+-- analysis/            # Paper tables and figures
+-- api/                 # FastAPI backend
+-- ui/                  # Gradio frontend
+-- configs/             # Experiment configuration (YAML)
+-- tests/               # Unit tests
+-- data/                # Raw data, processed chunks, indices, results
```

## Experiment Configurations

| Config | Sources | Description |
|--------|---------|-------------|
| D | docs | Documentation only |
| B | bugs | Bug reports only |
| W | work items | Work items only |
| DB | docs + bugs | Two-source |
| DW | docs + work items | Two-source |
| BW | bugs + work items | Two-source |
| DBW | all three | Full multi-source (main) |
| BM25 | all three | BM25 keyword baseline |
| Naive | all three | No routing, no diversity |

## Metrics

- **RA** (Resolution Adequacy): (RCI + AS + VM) / 6, where RCI = Root Cause Identification (0-2), AS = Actionable Steps (0-2), VM = Version Match (0-2)
- **CSAS** (Citation Source Alignment Score): Precision of citations matching expected sources
- **MSUR** (Multi-Source Utilization Rate): Fraction of queries using 2+ source types

## License

MIT
