# Testing Guide - Advanced Agentic RAG

## Overview

This directory contains integration tests for the Advanced Agentic RAG system. All tests are executable Python scripts that validate end-to-end workflows and system behavior.

**Test Organization:** All tests are in `tests/integration/` and can be run directly as scripts or through pytest (future).

**Quick Start:** Run any test with `uv run python tests/integration/test_<name>.py`

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

**Note:** This is the newest feature (implemented Nov 14), testing Phase 4 functionality from HANDOVER.md

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
- HANDOVER.md lines 151-252: Phase 4 implementation details
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

## Running All Tests

### Run All Integration Tests Sequentially

```bash
# Run all tests (not recommended - takes 20+ minutes)
for test in tests/integration/test_*.py; do
    echo "Running $test..."
    uv run python "$test"
done
```

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

**Issue: "No PDFs found in docs/"**
- Ensure PDF files exist in `docs/` directory
- Check file extensions are `.pdf` (case-sensitive on Unix)

**Issue: "OpenAI API key not found"**
- Create `.env` file from `.env.example`
- Add `OPENAI_API_KEY=sk-...` to `.env`

**Issue: "403 Forbidden LangSmith warnings"**
- Benign - LangSmith tracing not configured
- Tests automatically set `LANGCHAIN_TRACING_V2=false`
- Warnings don't affect test execution

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
3. Chunk IDs match corpus: Run `tests/utils/create_golden_dataset_helper.py`

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

**Phase 5: RAGAS Integration** (from HANDOVER.md)
- Add RAGAS evaluation metrics
- Compare RAGAS vs custom metrics
- Integration with golden dataset

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
- **Handover Document:** `../HANDOVER.md` (Phase 4: Golden Dataset, lines 151-252)
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

# Check dataset stats
uv run python -c "from src.evaluation import GoldenDatasetManager; m = GoldenDatasetManager('evaluation/golden_set.json'); m.print_statistics()"

# Verify corpus loaded
uv run python -c "from src.core import setup_retriever; r = setup_retriever(); print(f'Loaded {len(r.vectorstore.docstore._dict)} chunks')"
```

---

*Note: All tests use ASCII-only output per project guidelines (no emojis/unicode)*
*Last Updated: 2025-11-15*
