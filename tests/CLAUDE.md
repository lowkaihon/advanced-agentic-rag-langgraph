# Testing Guide - Advanced Agentic RAG

## Overview

This directory contains integration tests for the Advanced Agentic RAG system. All tests are executable Python scripts that validate end-to-end workflows and system behavior.

**Test Organization:** All tests are in `tests/integration/` and can be run directly as scripts or through pytest (future).

---

## CRITICAL: How to Run Tests

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

**For Windows users:** Same commands work - no special CMD syntax needed with `uv sync` approach.

**Technical details:** The `uv sync` command reads `[build-system]` from `pyproject.toml` and installs the `src/` directory as an editable package using `.pth` files. This is the modern Python packaging standard (replacing legacy `PYTHONPATH` and `sys.path` manipulation).

---

## Quick Test Selection Guide

| When to Run | Test File | Runtime | Key Purpose |
|-------------|-----------|---------|-------------|
| Verify imports/basic pipeline | test_pdf_pipeline.py | 30-45s | End-to-end validation, strategy selection, conversational rewriting |
| After profiling changes | test_document_profiling.py | 40-60s | Document profiling, corpus stats, metadata-aware retrieval |
| After retrieval logic changes | test_adaptive_retrieval.py | 30-45s | Metadata-driven retrieval, strategy switching, self-correction |
| Before deployment/releases | test_golden_dataset_evaluation.py | 10-15min | Comprehensive metrics, regression detection, baseline validation |
| Quick smoke test (reranking) | test_cross_encoder.py | 5-10s | CrossEncoder reranking verification |
| Quick smoke test (groundedness) | test_groundedness.py | 10-15s | Hallucination detection verification |
| After NLI detector changes | test_nli_hallucination_detector.py | 20-30s | NLI-based hallucination detection validation |
| RAGAS quick validation | test_ragas_simple.py | 10-20s | RAGAS metrics smoke test |
| RAGAS comprehensive | test_ragas_evaluation.py | 2-3min | RAGAS vs custom metrics comparison |
| Context completeness checks | test_context_sufficiency.py | 2-3min | Context sufficiency validation |

**Command pattern for all tests:**
```bash
uv run python tests/integration/<test_file>  # After uv sync
```

---

## Integration Tests

### 1. test_pdf_pipeline.py - Core End-to-End Test

**Purpose:** Complete RAG pipeline validation from PDF loading through generation

**What it tests:**
- PDF document loading with document profiling
- Strategy selection across diverse query types
- Conversational query rewriting with history
- Full LangGraph workflow execution

**When to run:**
- Before commits to ensure no regressions
- After changes to core pipeline components
- To verify PDF processing works correctly

**Command:**
```bash
uv run python tests/integration/test_pdf_pipeline.py
```

**Expected output:**
- 4 tests execute sequentially
- PDF loads with profiling metadata
- Strategy selection works for different query types
- Conversational rewriting incorporates history
- Full pipeline produces answers

**Runtime:** ~30-45 seconds

**Dependencies:**
- PDF files in `docs/` directory
- OpenAI API key in `.env`
- Core retrieval and graph components

---

### 2. test_document_profiling.py - Document Profiling Workflow

**Purpose:** Validates LLM-based document profiling and corpus statistics

**What it tests:**
- Document loading with LLM profiling enabled
- Corpus statistics extraction (technical density, domains, types)
- Metadata-aware retrieval strategy selection
- End-to-end retrieval with profiling metadata

**When to run:**
- After changes to document profiling logic
- To verify corpus analysis works correctly
- When debugging strategy selection issues

**Command:**
```bash
uv run python tests/integration/test_document_profiling.py
```

**Expected output:**
- Documents load with profiling enabled
- Corpus statistics show technical density, domains, document types
- Strategy selector uses corpus metadata
- Retrieval incorporates document profiles

**Runtime:** ~40-60 seconds (LLM profiling adds overhead)

**Dependencies:**
- PDF files in `docs/` directory
- OpenAI API key for profiling
- DocumentProfiler and corpus analysis components

---

### 3. test_adaptive_retrieval.py - Metadata-Driven Adaptive Retrieval

**Purpose:** Tests self-correcting retrieval with strategy switching

**What it tests:**
- Metadata-driven adaptive retrieval workflow
- Strategy mismatch detection (when docs prefer different strategy)
- Automatic strategy switching with logged reasoning
- Metadata analysis of retrieved documents

**When to run:**
- After changes to adaptive retrieval logic
- To verify strategy switching works
- When debugging metadata analysis
- After changes to retrieval quality gates

**Command:**
```bash
uv run python tests/integration/test_adaptive_retrieval.py
```

**Expected output:**
- Initial retrieval with selected strategy
- Metadata analysis detects strategy mismatches
- System switches to document-preferred strategy if needed
- Detailed logging of decision reasoning

**Runtime:** ~30-45 seconds

**Dependencies:**
- Metadata analysis node
- Strategy switching logic
- Quality scoring components

**Note:** This is the newest feature (implemented Nov 14), testing metadata-driven adaptive retrieval

---

### 4. test_golden_dataset_evaluation.py - Comprehensive Offline Evaluation

**Purpose:** Offline evaluation using 20-example golden dataset with comprehensive metrics

**What it tests:**
- Dataset loading and validation (20 curated examples)
- Baseline performance against thresholds
- Performance regression detection
- Cross-document retrieval accuracy
- Difficulty-based metrics correlation
- Full evaluation report generation

**When to run:**
- Establishing performance baselines
- Detecting regressions before releases
- Comprehensive system validation
- Generating evaluation reports

**Command:**
```bash
uv run python tests/integration/test_golden_dataset_evaluation.py
```

**Tests included:**
1. `test_dataset_loading()` - Validates dataset loads correctly
2. `test_dataset_validation()` - Checks all 20 examples are valid
3. `test_baseline_performance()` - Asserts minimum quality thresholds
4. `test_regression()` - Detects performance degradation vs saved baseline
5. `test_cross_document_retrieval()` - Validates multi-doc queries
6. `test_difficulty_correlation()` - Verifies harder = lower scores
7. `generate_evaluation_report()` - Creates markdown report

**Expected output:**
- All 6 tests pass
- Metrics meet minimum thresholds
- Saves `evaluation/baseline_metrics.json`
- Generates `evaluation/evaluation_report.md`

**Performance thresholds:**
- Recall@5: >= 70%
- Precision@5: >= 60%
- F1@5: >= 65%
- Groundedness: >= 85%
- Hallucination rate: <= 15%

**Runtime:** ~10-15 minutes (processes 20 examples)

**Dependencies:**
- `evaluation/golden_set.json` (20 examples)
- GoldenDatasetManager
- Complete evaluation infrastructure
- All retrieval and generation components

**Related documentation:**
- evaluation/README.md: Golden dataset documentation

---

### 5. test_cross_encoder.py - CrossEncoder Reranking Smoke Test

**Purpose:** Quick verification that CrossEncoder reranking executes correctly

**What it tests:**
- CrossEncoderReRanker initialization
- Document reranking with query-document pairs
- Score calculation and ranking

**When to run:**
- Quick smoke test for reranking functionality
- After changes to CrossEncoder reranker
- Verifying sentence-transformers installation

**Command:**
```bash
uv run python tests/integration/test_cross_encoder.py
```

**Expected output:**
- 3 test documents ranked by relevance
- Scores displayed for each document
- Relevant docs ranked higher than irrelevant

**Runtime:** ~5-10 seconds

**Dependencies:**
- sentence-transformers library
- CrossEncoderReRanker component

**Note:** Simple smoke test - comprehensive reranking testing is in test_pdf_pipeline.py

---

### 6. test_groundedness.py - Groundedness Check Smoke Test

**Purpose:** Quick verification that groundedness checking executes

**What it tests:**
- Groundedness check node execution
- Hallucination detection
- Quality metrics calculation

**When to run:**
- Quick smoke test for groundedness functionality
- After changes to groundedness logic
- Verifying groundedness check works

**Command:**
```bash
uv run python tests/integration/test_groundedness.py
```

**Expected output:**
- Single query processes through graph
- Groundedness score displayed
- Hallucination status shown
- Severity level indicated

**Runtime:** ~10-15 seconds

**Dependencies:**
- Groundedness check node
- Complete graph workflow

**Note:** Simple smoke test - comprehensive groundedness testing is in test_golden_dataset_evaluation.py

---

### 7. test_nli_hallucination_detector.py - NLI-Based Hallucination Detection

**Purpose:** Validates NLI-based hallucination detector with research-backed implementation

**What it tests:**
- NLIHallucinationDetector initialization and claim decomposition
- Research-backed label mapping (entailment > 0.7 → SUPPORTED, neutral → UNSUPPORTED)
- Zero-shot NLI baseline behavior and strictness
- Hallucination detection across multiple scenarios:
  - Semantically similar but not lexically identical claims
  - Factually incorrect claims (wrong facts)
  - Completely hallucinated content (off-topic)
- Detailed NLI scores and label verification

**When to run:**
- After changes to NLI hallucination detector
- To verify research-backed label mapping works correctly
- After NLI model updates
- Validating baseline performance expectations (0.65-0.70 F1)

**Command:**
```bash
PYTHONPATH=. uv run python tests/integration/test_nli_hallucination_detector.py
```

**Test cases:**
1. **Test 1:** Semantically similar claims (zero-shot NLI strict behavior)
   - Context: "BERT uses 12 transformer layers..."
   - Answer: "BERT has 12 layers and uses transformers"
   - Expected: UNSUPPORTED (neutral label, not entailment)
2. **Test 2:** Factually incorrect claim (hallucination)
   - Wrong publication year (2020 instead of 2018)
   - Expected: UNSUPPORTED (contradiction or neutral)
3. **Test 3:** Completely hallucinated content (off-topic)
   - Context about BERT, answer about GPT-3
   - Expected: All claims UNSUPPORTED
4. **Test 4:** Detailed NLI scores showing label mapping
   - Displays entailment, neutral, contradiction scores
   - Verifies research-backed threshold and mapping

**Expected output:**
- All 4 tests complete successfully
- Groundedness scores match expected baseline behavior
- Label mapping verified: neutral → UNSUPPORTED, entailment > 0.7 → SUPPORTED
- Key takeaways displayed (baseline F1: 0.65-0.70, production: 0.79-0.83)

**Runtime:** ~20-30 seconds

**Dependencies:**
- NLIHallucinationDetector (src/validation/)
- sentence-transformers library
- cross-encoder/nli-deberta-v3-base model
- OpenAI API key for claim decomposition

**Note:** This test validates the research-backed zero-shot NLI baseline. Production systems (0.83 F1) require fine-tuning on RAGTruth dataset.

---

### 8. test_ragas_evaluation.py - RAGAS Evaluation Suite

**Purpose:** Comprehensive RAGAS metrics evaluation and comparison with custom metrics

**What it tests:**
- RAGASEvaluator initialization with faithfulness, context precision, and response relevancy metrics
- Single sample RAGAS evaluation
- Small subset evaluation (3 easy examples for quick validation)
- RAGAS vs custom metrics comparison and correlation analysis
- Full dataset evaluation (20 examples with comprehensive comparison)
- Automated comparison report generation

**When to run:**
- Validating RAGAS integration after updates
- Comparing RAGAS metrics with custom evaluation metrics
- Generating comprehensive evaluation reports
- Analyzing correlation between different metric approaches
- Benchmarking RAGAS performance on golden dataset

**Command:**
```bash
# Run all RAGAS tests (excludes full dataset - runs 4 tests)
uv run python tests/integration/test_ragas_evaluation.py

# Run individual tests
uv run python -c "from tests.integration.test_ragas_evaluation import test_ragas_evaluator_initialization; test_ragas_evaluator_initialization()"
uv run python -c "from tests.integration.test_ragas_evaluation import test_ragas_sample_evaluation; test_ragas_sample_evaluation()"
uv run python -c "from tests.integration.test_ragas_evaluation import test_ragas_on_small_subset; test_ragas_on_small_subset()"
uv run python -c "from tests.integration.test_ragas_evaluation import test_ragas_vs_custom_metrics; test_ragas_vs_custom_metrics()"

# Run full dataset evaluation (20 examples, ~10-15 minutes)
uv run python -c "from tests.integration.test_ragas_evaluation import test_ragas_full_dataset_evaluation; test_ragas_full_dataset_evaluation()"
```

**Tests included:**
1. `test_ragas_evaluator_initialization()` - Validates RAGASEvaluator setup
2. `test_ragas_sample_evaluation()` - Single sample evaluation
3. `test_ragas_on_small_subset()` - 3 easy examples quick test
4. `test_ragas_vs_custom_metrics()` - Metric comparison on 3 examples
5. `test_ragas_full_dataset_evaluation()` - Full 20 examples (optional)

**Expected output:**
- RAGASEvaluator initializes with 3 metrics (faithfulness, context_precision, response_relevancy)
- Single sample scores displayed for all metrics
- Small subset processes 3 examples successfully
- Comparison shows correlation analysis between RAGAS and custom metrics
- Full dataset generates `evaluation/ragas_evaluation_results.json` and `evaluation/ragas_comparison_report.md`

**RAGAS Metrics:**
- **Faithfulness**: Measures if generated answers contain hallucinations (similar to custom groundedness)
- **Context Precision**: Evaluates if relevant contexts are ranked higher (similar to custom retrieval quality)
- **Response Relevancy**: Measures how relevant the answer is to the question (similar to custom confidence)

**Comparison Analysis:**
- Faithfulness vs Groundedness: Expected correlation >90%
- Context Precision vs Retrieval Metrics (Recall@5, Precision@5, F1@5)
- Response Relevancy vs Answer Confidence scores

**Runtime:**
- Default (4 tests): ~2-3 minutes
- Full dataset (test 5): ~10-15 minutes

**Dependencies:**
- `ragas` library (installed via uv)
- Golden dataset: `evaluation/golden_set.json`
- RAGASEvaluator component
- Complete RAG pipeline and graph
- OpenAI API key for LLM-based metrics

**Note:** This implements RAGAS integration. The test suite excludes the full dataset evaluation by default (run it separately if needed for comprehensive analysis).

---

### 9. test_ragas_simple.py - RAGAS Quick Smoke Test

**Purpose:** Quick verification that RAGAS evaluation executes correctly

**What it tests:**
- RAGASEvaluator initialization
- Sample preparation with question, answer, contexts, and ground truth
- Single sample evaluation with all RAGAS metrics
- Basic score calculation and display

**When to run:**
- Quick smoke test for RAGAS functionality
- After RAGAS library updates
- Verifying RAGAS integration works
- Fast validation without running full evaluation suite

**Command:**
```bash
uv run python tests/integration/test_ragas_simple.py
```

**Expected output:**
- RAGASEvaluator initializes successfully
- Lists initialized metrics (faithfulness, context_precision, response_relevancy)
- Sample prepared for Transformer architecture question
- Scores displayed for each RAGAS metric
- Test completes in 10-20 seconds

**Runtime:** ~10-20 seconds (single API call)

**Dependencies:**
- `ragas` library
- RAGASEvaluator component
- OpenAI API key

**Note:** Simplest RAGAS test - comprehensive RAGAS testing is in test_ragas_evaluation.py

---

### 10. test_context_sufficiency.py - Context Sufficiency Enhancement

**Purpose:** Validates context completeness checks before answer generation

**What it tests:**
- Context sufficiency evaluation before answer generation
- Missing aspects detection when context is incomplete
- Context-driven strategy switching (prioritizes semantic when context insufficient)
- Integration with existing quality gates and routing logic
- Multi-aspect query handling (e.g., "advantages AND disadvantages")

**When to run:**
- After changes to context sufficiency logic
- To verify early detection of incomplete retrieval
- When debugging strategy switching behavior
- After modifications to answer evaluation node

**Command:**
```bash
uv run python tests/integration/test_context_sufficiency.py
```

**Tests included:**
1. `test_context_sufficiency_detection()` - Multi-aspect query (advantages AND disadvantages)
2. `test_context_sufficiency_with_simple_query()` - Simple single-aspect query

**Expected output:**
- Context sufficiency score displayed (0.0-1.0)
- Missing aspects identified when context incomplete
- Context insufficiency warning logged (if score < 0.6)
- Strategy refinement logged showing context-driven switching
- Refinement history includes context sufficiency scores
- All assertions pass

**Sample Output:**
```
CONTEXT INSUFFICIENCY DETECTED
Sufficiency Score: 40%
Is Sufficient: False
Missing Aspects (3):
  - specific disadvantages of BERT
  - comparison to other models
  - limitations in real-world applications

STRATEGY REFINEMENT
Switch: semantic to hybrid
Reasoning: Context-driven: Semantic failed to provide complete context, trying hybrid
Context sufficiency: 40%
```

**Runtime:** ~2-3 minutes (runs 2 test queries)

**Dependencies:**
- Complete RAG pipeline with context sufficiency enhancements
- Context sufficiency fields in state.py
- Enhanced evaluate_answer_with_retrieval_node in nodes.py
- Updated route_after_evaluation in graph.py

**Note:** This implements context sufficiency enhancement. Adds early detection of incomplete context to reduce hallucinations by 5-10%.

---

## Running All Tests

### Run All Integration Tests Sequentially

```bash
# Run all tests (not recommended - takes 30+ minutes due to RAGAS)
for test in tests/integration/test_*.py; do
    echo "Running $test..."
    uv run python "$test"
done
```

**Note:** The RAGAS comprehensive test (test_ragas_evaluation.py) runs 4 tests by default (excludes full dataset). Total runtime for all 10 integration tests: ~30-40 minutes.

### Run Recommended Test Suite (Fast)

```bash
# Core pipeline + adaptive retrieval (< 2 minutes)
uv run python tests/integration/test_pdf_pipeline.py
uv run python tests/integration/test_adaptive_retrieval.py
```

### Run Full Validation (Comprehensive)

```bash
# All tests including golden dataset (~15 minutes)
uv run python tests/integration/test_pdf_pipeline.py
uv run python tests/integration/test_document_profiling.py
uv run python tests/integration/test_adaptive_retrieval.py
uv run python tests/integration/test_golden_dataset_evaluation.py
```

### Run RAGAS Evaluation Suite

```bash
# Quick RAGAS smoke test (~20 seconds)
uv run python tests/integration/test_ragas_simple.py

# RAGAS comprehensive suite without full dataset (~3 minutes)
uv run python tests/integration/test_ragas_evaluation.py

# RAGAS full dataset evaluation (~15 minutes)
uv run python -c "from tests.integration.test_ragas_evaluation import test_ragas_full_dataset_evaluation; test_ragas_full_dataset_evaluation()"
```

---

## Test Environment Setup

### Prerequisites

1. **Dependencies installed:**
   ```bash
   uv sync
   ```

2. **Environment variables set:**
   ```bash
   # .env file must contain:
   OPENAI_API_KEY=your-key-here
   ```

3. **PDF documents loaded:**
   ```bash
   # Ensure docs/ folder contains PDFs
   ls docs/*.pdf
   ```

4. **Optional - Disable LangSmith warnings:**
   ```bash
   export LANGCHAIN_TRACING_V2=false
   ```

### Common Setup Issues

**Issue: "ModuleNotFoundError: No module named 'src'"**
- **Cause:** Project not installed in editable mode
- **Solution:** Run `uv sync` once to install the package
- **Why:** Tests import from `src` package, which must be installed first
- See the "CRITICAL: How to Run Tests" section at the top of this document

**Issue: "No PDFs found in docs/"**
- Ensure PDF files exist in `docs/` directory
- Check file extensions are `.pdf` (case-sensitive on Unix)

**Issue: "OpenAI API key not found"**
- Create `.env` file from `.env.example`
- Add `OPENAI_API_KEY=sk-...` to `.env`

**Issue: "403 Forbidden LangSmith warnings"**
- **Cause:** LangSmith tracing attempted without valid API key
- **Solution:** Tests automatically disable all tracing (fixed in conftest.py)
- **Status:** All tests now force-disable tracing to eliminate warnings
- **Note:** To enable tracing for demos/self-testing, set `LANGSMITH_TRACING=true` in `.env` (only affects main.py, not tests)

---

## Understanding Test Output

### Success Indicators

- **Green/OK messages** - Test passed
- **Metrics displayed** - Expected behavior
- **No exceptions** - Test completed

### Failure Indicators

- **Red/ERROR messages** - Test failed
- **Exceptions/tracebacks** - Unexpected errors
- **Assertion failures** - Quality thresholds not met

### Performance Expectations

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

## Debugging Failed Tests

### Test Fails: test_pdf_pipeline.py

**Check:**
1. PDFs exist in `docs/` folder
2. OpenAI API key valid
3. Documents loaded: `uv run python -c "from src.core import setup_retriever; r = setup_retriever(); print(f'Loaded {len(r.vectorstore.docstore._dict)} chunks')"`

**Common causes:**
- Missing PDFs
- API rate limits
- Network issues

---

### Test Fails: test_document_profiling.py

**Check:**
1. DocumentProfiler works: `uv run python -c "from src.preprocessing.document_profiler import DocumentProfiler; p=DocumentProfiler(); print(p.profile_document('AI is ML'))"`
2. LLM profiling enabled in retriever setup

**Common causes:**
- LLM profiling disabled
- API timeouts (profiling adds overhead)

---

### Test Fails: test_adaptive_retrieval.py

**Check:**
1. Metadata analysis node exists in graph
2. Strategy switching logic implemented
3. Graph workflow includes adaptive routing

**Common causes:**
- Missing metadata analysis node
- Strategy switching not configured
- Graph routing logic changed

---

### Test Fails: test_golden_dataset_evaluation.py

**Check:**
1. Golden dataset exists: `ls evaluation/golden_set.json`
2. Dataset valid: `uv run python -c "from src.evaluation import GoldenDatasetManager; m=GoldenDatasetManager('evaluation/golden_set.json'); m.print_statistics()"`
3. Chunk IDs match corpus: Run `uv run python tests/utils/create_golden_dataset_helper.py`

**Common causes:**
- Missing golden dataset file
- Invalid chunk IDs in dataset
- Corpus not loaded
- Performance regression (check baseline_metrics.json)

---

## Test Maintenance

### When Adding New Features

1. **Consider adding integration test** if feature:
   - Affects end-to-end workflow
   - Changes retrieval or generation behavior
   - Modifies quality gates or routing

2. **Update existing tests** if feature:
   - Changes expected output format
   - Modifies performance characteristics
   - Affects quality thresholds

3. **Update this guide** if:
   - New test added
   - Test purposes change
   - New dependencies required

### When Modifying Tests

1. **Update documentation** in this file
2. **Update performance thresholds** if system improves
3. **Regenerate baselines** for regression tests
4. **Update golden dataset** if test coverage gaps found

---

## Future Test Enhancements

### Planned Improvements

**RAGAS Integration** - COMPLETED
- Checkmark RAGAS evaluation metrics integrated (faithfulness, context_precision, response_relevancy)
- Checkmark RAGAS vs custom metrics comparison implemented
- Checkmark Integration with golden dataset complete
- Tests: `test_ragas_evaluation.py` (comprehensive), `test_ragas_simple.py` (smoke test)

**Future Enhancements**

**Unit Test Migration**
- Convert scripts to pytest format
- Add unit tests for individual components
- Improve test isolation and speed

**CI/CD Integration**
- Automated test execution on commits
- Performance tracking over time
- Regression alerts

---

## Related Documentation

- **Main Project Docs:** `../CLAUDE.md`
- **Golden Dataset Guide:** `../evaluation/README.md`
- **Evaluation Best Practices:** `../references/Best Practices for Evaluating Retrieved RAG Documents.md`
- **RAGAS Integration:** `../references/RAGAS Integration with LangGraph for python RAG pipeline.md`

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
uv run python tests/integration/test_nli_hallucination_detector.py          # NLI-based hallucination validation (~30s)

# RAGAS evaluation tests
uv run python tests/integration/test_ragas_simple.py                    # Quick RAGAS smoke test (~20s)
uv run python tests/integration/test_ragas_evaluation.py                # RAGAS comprehensive suite (~3 min)

# Context sufficiency test
uv run python tests/integration/test_context_sufficiency.py              # Context completeness checks (~2-3 min)

# Check dataset stats
uv run python -c "from src.evaluation import GoldenDatasetManager; m = GoldenDatasetManager('evaluation/golden_set.json'); m.print_statistics()"

# Verify corpus loaded
uv run python -c "from src.core import setup_retriever; r = setup_retriever(); print(f'Loaded {len(r.vectorstore.docstore._dict)} chunks')"
```

---

*Note: All tests use ASCII-only output per project guidelines (no emojis/unicode)*
*Last Updated: 2025-11-15*
