# Testing Guide - Advanced Agentic RAG

## Quick Start

**One-time setup (required):**
```bash
uv sync  # Installs project in editable mode + all dependencies
```

**Then run any test:**
```bash
uv run python tests/integration/test_<name>.py
```

**Why this works:**
- `uv sync` installs your project as an editable package (PEP 660 standard)
- The `src` package becomes automatically importable
- No PYTHONPATH needed - works across all shells and platforms (Bash, Zsh, PowerShell, CMD)
- IDE navigation and type checking work correctly
- One-time setup, then tests "just work"

**Alternative (if you activated venv):**
```bash
source .venv/bin/activate  # Unix/Mac
# or
.venv\Scripts\activate  # Windows

python tests/integration/test_<name>.py  # No "uv run" needed when venv active
```

**Technical details:** The `uv sync` command reads `[build-system]` from `pyproject.toml` and installs the `src/` directory as an editable package using `.pth` files. This is the modern Python packaging standard (replacing legacy `PYTHONPATH` and `sys.path` manipulation).

---

## Import Patterns for New Tests

**Package name:** `advanced_agentic_rag_langgraph` (NOT `src`)

**Correct imports in test files:**
```python
# Always use the package name after uv sync
from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph
from advanced_agentic_rag_langgraph.evaluation import GoldenDatasetManager
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector
```

**Common mistakes to avoid:**
```python
# WRONG - Don't import from 'src'
from src.core import setup_retriever  # ModuleNotFoundError: No module named 'src'
from src.orchestration import ...

# WRONG - Don't use PYTHONPATH
# PYTHONPATH=. uv run python ...  # Unnecessary, creates confusion
```

**Why 'src' doesn't work:**
- Package name is defined in `pyproject.toml` as `advanced_agentic_rag_langgraph`
- Source lives in `src/advanced_agentic_rag_langgraph/` (directory structure)
- After `uv sync`, Python knows: `advanced_agentic_rag_langgraph` → `src/advanced_agentic_rag_langgraph/`
- The `src` directory is NOT a package itself

**Creating new test files:**
1. Always run `uv sync` once after cloning/setup
2. Use `from advanced_agentic_rag_langgraph.<module>` imports
3. Run with `uv run python <file>` (no PYTHONPATH needed)
4. If errors: check package name spelling, verify uv sync completed

**LangSmith Warning Suppression:**
```python
import os
import warnings
import logging

# Suppress LangSmith warnings - MUST be set BEFORE importing LangChain modules
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*Failed to.*LangSmith.*")
warnings.filterwarnings("ignore", message=".*langsmith.*")

# Suppress LangSmith logger (key fix for 403 warnings)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.WARNING)

# NOW import LangChain modules
from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph
```

**Why this order matters:**
- Environment variables must be set before LangChain imports
- Logging configuration prevents 403 Forbidden warnings from appearing in stderr
- All integration tests use this pattern for clean output

---

## Test Selection Matrix

| When to Run | Test File | Runtime | Key Purpose |
|-------------|-----------|---------|-------------|
| Verify imports/basic pipeline | test_pdf_pipeline.py | 30-45s | End-to-end validation, strategy selection, conversational rewriting |
| After profiling changes | test_document_profiling.py | 40-60s | Document profiling, corpus stats, metadata-aware retrieval |
| After retrieval logic changes | test_adaptive_retrieval.py | 30-45s | Quality-issue-based retrieval, strategy switching, self-correction |
| Before deployment/releases | test_golden_dataset_evaluation.py | 10-15min | Comprehensive metrics, regression detection, baseline validation |
| Quick smoke test (reranking) | test_cross_encoder.py | 5-10s | CrossEncoder reranking verification |
| Quick smoke test (groundedness) | test_groundedness.py | 10-15s | Hallucination detection verification |
| After NLI detector changes | test_nli_hallucination_detector.py | 20-30s | NLI-based hallucination detection validation |
| RAGAS quick validation | test_ragas_simple.py | 10-20s | RAGAS metrics smoke test |
| RAGAS comprehensive | test_ragas_evaluation.py | 2-3min | RAGAS vs custom metrics comparison |
| Portfolio architecture showcase | test_architecture_comparison.py | 70-85min | 4-tier A/B test (basic/intermediate/advanced/multi-agent) with F1@K, Groundedness, Similarity |
| Multi-agent RAG evaluation | test_multi_agent_evaluation.py | 15-20min | Orchestrator-worker pattern with parallel retrieval, cross-agent fusion |

---

## Integration Tests

### 1. test_pdf_pipeline.py (~30-45s)
Complete RAG pipeline validation from PDF loading through generation.

**Tests:** PDF loading with profiling, strategy selection, conversational query rewriting, full LangGraph workflow
**Method used:** Graph workflow with multi-query RRF fusion (uses `retrieve_without_reranking()` internally before final reranking)
**Run after:** Changes to core pipeline, PDF processing, or before commits
**Command:** `uv run python tests/integration/test_pdf_pipeline.py`

---

### 2. test_document_profiling.py (~40-60s)
Validates LLM-based document profiling and corpus statistics.

**Tests:** Document profiling with LLM, corpus stats (technical density, domains, types), metadata-aware retrieval
**Method used:** Direct `retriever.retrieve()` call (tests complete pipeline: retrieval → two-stage reranking)
**Run after:** Changes to document profiling logic or corpus analysis
**Command:** `uv run python tests/integration/test_document_profiling.py`

---

### 3. test_adaptive_retrieval.py (~30-45s)
Tests self-correcting retrieval with quality-issue-based strategy switching.

**Tests:** Retrieval quality evaluation (8 issue types), strategy switching with reasoning, query rewriting feedback loops
**Run after:** Changes to adaptive retrieval, strategy switching, or quality gates
**Command:** `uv run python tests/integration/test_adaptive_retrieval.py`
**Note:** Displays retrieval_quality_issues, strategy_switch_reason, and refinement_history

---

### 4. test_golden_dataset_evaluation.py (~10-15min)
Comprehensive offline evaluation using 20-example golden dataset.

**Tests:** Dataset validation, baseline performance, regression detection, cross-document retrieval, difficulty correlation
**Run before:** Releases, after major changes, or when establishing baselines
**Command:** `uv run python tests/integration/test_golden_dataset_evaluation.py`
**Thresholds:** Recall@5 ≥70%, Precision@5 ≥60%, F1@5 ≥65%, Groundedness ≥85%, Hallucination ≤15%
**Output:** Saves `evaluation/baseline_metrics.json` and `evaluation/evaluation_report.md`

---

### 5. test_cross_encoder.py (~5-10s)
Quick verification that CrossEncoder reranking executes correctly.

**Tests:** CrossEncoderReRanker initialization, document reranking, score calculation
**Run after:** Changes to CrossEncoder reranker or verifying sentence-transformers
**Command:** `uv run python tests/integration/test_cross_encoder.py`

---

### 6. test_groundedness.py (~10-15s)
Quick verification that groundedness checking executes.

**Tests:** Groundedness check node, hallucination detection, quality metrics
**Run after:** Changes to groundedness logic
**Command:** `uv run python tests/integration/test_groundedness.py`

---

### 7. test_nli_hallucination_detector.py (~20-30s)
Validates NLI-based hallucination detector with research-backed implementation.

**Tests:** Claim decomposition, research-backed label mapping (entailment >0.7 → SUPPORTED), zero-shot baseline (F1: 0.65-0.70)
**Run after:** Changes to NLI detector or model updates
**Command:** `uv run python tests/integration/test_nli_hallucination_detector.py`
**Test cases:** Semantically similar claims, factually incorrect, completely hallucinated, detailed NLI scores

---

### 8. test_ragas_evaluation.py (~2-3min)
Comprehensive RAGAS metrics evaluation and comparison with custom metrics.

**Tests:** RAGASEvaluator initialization, single/subset evaluation, RAGAS vs custom metrics, correlation analysis
**Run after:** RAGAS updates, generating evaluation reports, benchmarking
**Command:** `uv run python tests/integration/test_ragas_evaluation.py`
**Metrics:** Faithfulness, Context Precision, Response Relevancy
**Output:** `evaluation/ragas_evaluation_results.json` and `evaluation/ragas_comparison_report.md`

---

### 9. test_ragas_simple.py (~10-20s)
Quick verification that RAGAS evaluation executes correctly.

**Tests:** RAGASEvaluator initialization, sample preparation, single sample evaluation
**Run after:** RAGAS library updates or quick validation
**Command:** `uv run python tests/integration/test_ragas_simple.py`

---

### 10. test_architecture_comparison.py (~70-85min)
4-tier A/B test comparing Basic (1), Intermediate (5), Advanced (17), Multi-Agent (20 features).

**Tests:** F1@K, Groundedness, Semantic Similarity, Factual Accuracy across tiers using BUDGET models
**Run after:** Architecture changes, before portfolio demos
**Command:** `uv run python tests/integration/test_architecture_comparison.py`
**Options:** `--dataset standard|hard`, `--quick` (2 examples, ~6-8min)
**Output:** `evaluation/architecture_comparison_report_{dataset}_latest.md`

---

### 11. test_multi_agent_evaluation.py (~15-20min)
Evaluates the Multi-Agent RAG variant using the Orchestrator-Worker pattern.

**Tests:** Query decomposition, parallel retrieval workers, cross-agent RRF fusion, multi-agent specific metrics
**Run after:** Changes to multi-agent RAG variant, before portfolio demos
**Command:** `uv run python tests/integration/test_multi_agent_evaluation.py`
**Options:** `--dataset standard|hard`, `--quick` (2 examples, ~2-3min)
**Metrics:** F1@K, Groundedness, Semantic Similarity + avg sub-queries, cross-agent doc overlap %
**Output:** `evaluation/multi_agent_evaluation_report_{dataset}_latest.md`

---

## Common Issues & Solutions

**"ModuleNotFoundError: No module named 'src'" or "No module named 'advanced_agentic_rag_langgraph'"**
- **Cause:** Project not installed in editable mode
- **Solution:** Run `uv sync` once to install the package
- **Why:** Tests import from package, which must be installed first

**"No PDFs found in docs/"**
- Ensure PDF files exist in `docs/` directory
- Check file extensions are `.pdf` (case-sensitive on Unix)

**"OpenAI API key not found"**
- Create `.env` file from `.env.example`
- Add `OPENAI_API_KEY=sk-...` to `.env`

**"403 Forbidden LangSmith warnings"**
- **Status:** Suppressed automatically (see "Import Patterns for New Tests" for implementation)
- **Note:** To enable for demos, set `LANGSMITH_TRACING=true` in `.env`

**Test fails: test_pdf_pipeline.py**
- Verify PDFs exist: `ls docs/*.pdf`
- Check API key valid
- Verify corpus loaded: `uv run python -c "from advanced_agentic_rag_langgraph.core import setup_retriever; r = setup_retriever(); print(f'Loaded {len(r.vectorstore.docstore._dict)} chunks')"`

**Test fails: test_golden_dataset_evaluation.py**
- Check dataset exists: `ls evaluation/golden_set.json`
- Verify dataset valid: `uv run python -c "from advanced_agentic_rag_langgraph.evaluation import GoldenDatasetManager; m=GoldenDatasetManager('evaluation/golden_set.json'); m.print_statistics()"`
- If chunk IDs don't match: Run `uv run python tests/utils/create_golden_dataset_helper.py`

---

## Performance Expectations

**Retrieval Quality:**
- Recall@5: 70-80% (70% minimum)
- Precision@5: 60-70% (60% minimum)
- F1@5: 65-75% (65% minimum)

**Generation Quality:**
- Groundedness: 90-100% (85% minimum)
- Hallucination rate: 0-10% (15% maximum)
- Confidence: 85-100% (80% minimum)

**Speed:**
- Reranking: < 1 second
- Full pipeline: < 5 seconds per query
- Golden dataset: ~30 seconds per example

---

## Running Test Suites

**Recommended suite (fast, < 2 min):**
```bash
uv run python tests/integration/test_pdf_pipeline.py
uv run python tests/integration/test_adaptive_retrieval.py
```

**Full validation (comprehensive, ~15 min):**
```bash
uv run python tests/integration/test_pdf_pipeline.py
uv run python tests/integration/test_document_profiling.py
uv run python tests/integration/test_adaptive_retrieval.py
uv run python tests/integration/test_golden_dataset_evaluation.py
```

**RAGAS evaluation:**
```bash
# Quick smoke test (~20s)
uv run python tests/integration/test_ragas_simple.py

# Comprehensive suite (~3 min)
uv run python tests/integration/test_ragas_evaluation.py

# Full dataset evaluation (~15 min)
uv run python -c "from tests.integration.test_ragas_evaluation import test_ragas_full_dataset_evaluation; test_ragas_full_dataset_evaluation()"
```

**Run all tests sequentially:**
```bash
# Takes 30-40 minutes total
for test in tests/integration/test_*.py; do
    echo "Running $test..."
    uv run python "$test"
done
```

---

## Quick Command Reference

```bash
# Core pipeline test (fast)
uv run python tests/integration/test_pdf_pipeline.py

# Document profiling test
uv run python tests/integration/test_document_profiling.py

# Adaptive retrieval test (newest feature)
uv run python tests/integration/test_adaptive_retrieval.py

# Comprehensive evaluation (slow, ~15 min)
uv run python tests/integration/test_golden_dataset_evaluation.py

# Quick smoke tests
uv run python tests/integration/test_cross_encoder.py
uv run python tests/integration/test_groundedness.py

# NLI hallucination detection test
uv run python tests/integration/test_nli_hallucination_detector.py

# RAGAS evaluation tests
uv run python tests/integration/test_ragas_simple.py
uv run python tests/integration/test_ragas_evaluation.py

# Multi-agent RAG evaluation
uv run python tests/integration/test_multi_agent_evaluation.py
uv run python tests/integration/test_multi_agent_evaluation.py --quick  # Fast mode

# Check dataset stats
uv run python -c "from advanced_agentic_rag_langgraph.evaluation import GoldenDatasetManager; m = GoldenDatasetManager('evaluation/golden_set.json'); m.print_statistics()"

# Verify corpus loaded
uv run python -c "from advanced_agentic_rag_langgraph.core import setup_retriever; r = setup_retriever(); print(f'Loaded {len(r.vectorstore.docstore._dict)} chunks')"
```

---

## Related Documentation

- **Main Project Docs:** `../CLAUDE.md`
- **Golden Dataset Guide:** `../evaluation/README.md`
- **Evaluation Best Practices:** `../references/Best Practices for Evaluating Retrieved RAG Documents.md`
- **RAGAS Integration:** `../references/RAGAS Integration with LangGraph for python RAG pipeline.md`

---

*Note: All tests use ASCII-only output per project guidelines (no emojis/unicode)*
*Last Updated: 2025-11-16*
