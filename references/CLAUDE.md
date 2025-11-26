# References Directory Guide

## Overview

This directory contains compiled research documents from Perplexity.ai and other sources that informed the design and implementation of this Advanced Agentic RAG system. These references provide deep-dive technical guidance on RAG evaluation, reranking strategies, dataset creation, and document processing techniques.

**Organization:** Documents are categorized into **Evaluation References** (measuring and improving RAG quality) and **Implementation References** (technical implementation patterns).

**Usage:** Reference these documents when implementing new features, debugging quality issues, or understanding the theoretical foundations of the system's design decisions.

---

## Foundations

### 0. Agentic RAG: Definition and Core Characteristics

**File:** `Agentic RAG_ Definition and Core Characteristics.md`

**Purpose:** Foundational definitions of agentic RAG characteristics and patterns

**Note:** Core concepts already covered in main `CLAUDE.md` (lines 3-18, "Agentic Architecture") and `README.md` (lines 25-89, "Why This Qualifies as Agentic RAG"). This reference provides supplementary academic context. For implementation guidance, see documents 1-11 below.

**Implementation Status:** System implements Dynamic Planning and Execution Agents pattern (distributed decision-making across 4 routing functions in graph structure).

---

## Evaluation References

### 1. Best Practices for Evaluating Retrieved RAG Documents

**File:** `Best Practices for Evaluating Retrieved RAG Documents.md`

**Purpose:** Comprehensive guide to RAG evaluation methodologies, metrics, and quality assessment frameworks

**When to Use:**
- Implementing retrieval quality scoring, evaluation pipelines, and metric selection
- Understanding LLM-as-judge patterns and automated quality gates

**Key Sections:**
- Core RAG evaluation metrics (relevance, precision, recall, NDCG, MRR)
- LLM-as-judge evaluation patterns
- Binary vs. graded relevance scoring
- Multi-dimensional quality assessment
- Automated evaluation frameworks
- Quality threshold tuning strategies

**Implementation Status:**
- [IMPLEMENTED] Retrieval quality scoring in `src/orchestration/nodes.py:grade_documents()`
- [IMPLEMENTED] Answer quality assessment in `src/orchestration/nodes.py:assess_answer()`
- [IMPLEMENTED] LLM-as-judge reranking in `src/retrieval/retrievers.py`
- [IMPLEMENTED] Quality gates and conditional routing in `src/orchestration/graph.py`

**Related Code:**
- `src/orchestration/nodes.py` - Quality scoring and grading functions
- `src/retrieval/retrievers.py` - Reranking with relevance scoring
- `src/core/state.py` - Quality score tracking in state

---

### 2. RAG Golden Dataset Creation for Technical Documents

**File:** `RAG Golden Dataset Creation for Technical Document.md`

**Purpose:** Methodology for creating high-quality evaluation datasets for RAG systems

**When to Use:**
- Building test datasets with ground-truth question-answer pairs and query variations
- Establishing baseline performance metrics and regression testing

**Key Sections:**
- Dataset creation strategies for technical content
- Question generation techniques
- Ground truth annotation methods
- Query diversity and coverage
- Dataset quality validation
- Sampling strategies for complex documents

**Implementation Status:**
- [PARTIAL] Manual test cases in `test_pdf_pipeline.py`
- [PLANNED] Automated golden dataset generation pipeline
- [PLANNED] Systematic evaluation with test datasets

**Related Code:**
- `test_pdf_pipeline.py` - Example queries for testing
- `src/evaluation/` - Evaluation framework (in development)

---

### 3. RAGAS Integration with LangGraph for Python RAG Pipeline

**File:** `RAGAS Integration with LangGraph for python RAG pipeline.md`

**Purpose:** Guide to integrating RAGAS (RAG Assessment) framework with LangGraph-based pipelines

**When to Use:**
- Implementing RAGAS metrics (faithfulness, answer relevance, context precision/recall) in LangGraph workflows
- Building evaluation dashboards and comparing retrieval strategies

**Key Sections:**
- RAGAS framework overview and metrics
- LangGraph integration patterns
- Evaluation dataset preparation
- Metric calculation and interpretation
- Automated evaluation workflows
- Performance benchmarking strategies

**Implementation Status:**
- [NOT IMPLEMENTED] RAGAS framework integration
- [ALTERNATIVE] Custom LLM-as-judge evaluation implemented
- [PLANNED] RAGAS metrics for comprehensive benchmarking

**Notes:** Current system uses custom evaluation logic. RAGAS provides standardized metrics that could complement existing evaluators.

---

### 4. Best Practices for LLM Prompts in RAG Pipelines

**File:** `Best Practices for LLM Prompts in RAG Pipelines.md`

**Purpose:** Comprehensive guide to prompt engineering strategies for RAG systems across retrieval, generation, and evaluation stages

**When to Use:**
- Designing system prompts for query rewriting, answer generation, and quality assessment
- Optimizing LLM-as-judge prompts for relevance scoring and hallucination detection
- Implementing structured output prompts with clear grounding instructions

**Key Sections:**
- Retrieval-stage prompting (query optimization, expansion decisions)
- Generation-stage prompting (context grounding, citation requirements)
- Evaluation-stage prompting (LLM-as-judge, quality scoring)
- Structured output techniques (TypedDict, Pydantic validation)
- Temperature and sampling strategies
- Prompt debugging and iteration patterns

**Implementation Status:**
- [IMPLEMENTED] Structured prompts throughout pipeline
- [IMPLEMENTED] LLM-as-judge prompting in `src/retrieval/retrievers.py`
- [IMPLEMENTED] Quality assessment prompts in `src/orchestration/nodes.py`
- [IN PROGRESS] Hallucination correction prompts

**Related Code:**
- `src/orchestration/nodes.py` - Query expansion, retrieval quality, answer assessment prompts
- `src/retrieval/retrievers.py` - Reranking and relevance scoring prompts
- `src/preprocessing/query_processing.py` - Conversational rewriting prompts

---

### 5. Research-Backed Best Practices for Self-Correction and Retry Mechanisms in RAG Systems

**File:** `Research-Backed Best Practices for Self-Correction and Retry Mechanisms in RAG Systems.md`

**Purpose:** Research-backed guide to implementing feedback loops and retry mechanisms in RAG systems, covering CRAG, Self-RAG, and multi-query strategies

**When to Use:**
- Implementing self-correction loops with quality gates and retry limits
- Designing confidence-based routing (CRAG three-tier thresholds)
- Optimizing retry limits and iteration counters (standard: 2-3 groundedness, 3-5 overall)
- Implementing NLI-based hallucination detection for retry triggering

**Key Sections:**
- Multi-query rewriting strategies (HyDE, DMQR-RAG, Step-back prompting)
- CRAG confidence thresholds (High/Ambiguous/Low tiers)
- Self-RAG reflection tokens (IsRelevant, isSupportive, IsUse)
- Retry limit standards and iteration control
- NLI hallucination detection patterns
- Prompt modifications for retry (stricter grounding, temperature reduction)
- Production framework implementations (LangChain, LlamaIndex, DSPy)

**Implementation Status:**
- [IMPLEMENTED] Three feedback loops (retrieval quality, answer quality, groundedness)
- [IMPLEMENTED] Retry limits: 2 query rewrites, 3 retrieval attempts, 1 groundedness retry
- [IMPLEMENTED] NLI-based hallucination detection in `src/validation/nli_hallucination_detector.py`
- [PARTIAL] Groundedness retry prompt strengthening (needs implementation)
- [NOT IMPLEMENTED] CRAG three-tier confidence system
- [NOT IMPLEMENTED] Self-RAG reflection tokens

**Related Code:**
- `src/orchestration/graph.py` - Conditional routing and feedback loops (lines 16-146)
- `src/orchestration/nodes.py` - Quality gates and retry logic
- `src/validation/nli_hallucination_detector.py` - NLI verification for retry triggering

**Notes:** Research validates current retry limits. Section 3b provides prompt-only hallucination correction techniques achieving 50-70% reduction with minimal overhead.

---

## Implementation References

### 6. Best Practices for Document Profiling and Metadata Extraction in RAG Pipelines

**File:** `Best Practices for Document Profiling and Metadata Extraction in RAG Pipelines.md`

**Purpose:** Comprehensive merged guide combining metadata architecture with research-backed LLM optimization strategies for document profiling

**When to Use:**
- Designing metadata schemas (4-layer strategy: document, content, structural, contextual)
- Implementing LLM-based metadata extraction with Pydantic validation
- Optimizing document profiling token budget and sampling strategies (4,000-8,000 tokens)
- Improving code/math detection accuracy in technical documents

**Key Sections:**
- Four-layer metadata architecture design
- Pydantic-based schema definition patterns
- LLM-based automated metadata extraction
- **Optimal input length and sampling strategies** (4,000-8,000 tokens, stratified positional sampling)
- **Cost-benefit analysis** (accuracy vs. token cost trade-offs)
- **Stratified positional sampling** (first 30% + last 20% + middle sections)
- **Document type-specific strategies** (research papers, legal docs, technical specs, business reports)
- **Code/math detection** (regex pre-detection + LLM verification)
- Quality assessment and deduplication
- Production metadata enrichment workflows

**Implementation Status:**
- [IMPLEMENTED] Document profiling with structured output (DocumentProfile TypedDict)
- [IMPLEMENTED] LLM-based metadata extraction in `src/preprocessing/document_profiler.py`
- [IMPLEMENTED] Stratified positional sampling (5,000 tokens) in `src/preprocessing/document_profiler.py:_stratified_sample()`
- [IMPLEMENTED] Regex signal pre-detection in `src/preprocessing/document_profiler.py:_detect_signals()`
- [IMPLEMENTED] Metadata attachment to chunks in `src/preprocessing/profiling_pipeline.py`
- [IMPROVED] Accuracy: Classification 72% → 87%, Code/Math 68% → 89%, Concepts 55% → 82%
- [COST] 5x token increase ($0.0001 → $0.0005 per doc, acceptable for ingestion)
- [PARTIAL] Structural metadata (postponed per portfolio scope decision)

**Related Code:**
- `src/preprocessing/document_profiler.py` - LLM-based document profiling with stratified sampling
- `src/preprocessing/document_profiler.py:154-183` - Enhanced profile_document()
- `src/preprocessing/document_profiler.py:71-94` - Regex signal detection
- `src/preprocessing/document_profiler.py:96-152` - Stratified sampling implementation
- `src/preprocessing/profiling_pipeline.py` - Metadata enrichment pipeline
- `src/core/config.py` - Corpus statistics and profile storage

**Notes:** Research findings directly informed document profiler improvements, achieving +15-27 percentage point accuracy gains. Merged document combines architectural best practices with optimization strategies.

---

### 7. Best Practices for Query Rewriting and Query Expansion in RAG Pipelines

**File:** `Best Practices for Query Rewriting and Query Expansion in RAG Pipelines.md`

**Purpose:** Strategic guide to query optimization through rewriting (precision) and expansion (recall)

**When to Use:**
- Implementing query rewriting for conversational context or clarification
- Deciding between expansion (recall) vs. rewriting (precision) strategies
- Optimizing multi-query fusion and variant generation

**Key Sections:**
- Query rewriting vs. expansion trade-offs (precision vs. recall)
- Classification-based routing (simple, multi-doc, complex queries)
- Core rewriting techniques (HyDE, Q2D, step-back prompting)
- LLM-based expansion decision logic
- Multi-query fusion strategies (RRF)
- Selective expansion with confidence thresholds

**Implementation Status:**
- [IMPLEMENTED] Conversational query rewriting in `src/preprocessing/query_processing.py:ConversationalRewriter`
- [IMPLEMENTED] LLM-based expansion decision in `src/orchestration/nodes.py:_should_skip_expansion_llm()`
- [IMPLEMENTED] Selective query expansion in `src/orchestration/nodes.py:query_expansion_node()`
- [IMPLEMENTED] RRF multi-query fusion in `src/orchestration/nodes.py:retrieve_with_expansion_node()`
- [DESIGN CHOICE] Prioritized precision (rewriting) over recall (expansion) for portfolio demonstration

**Related Code:**
- `src/preprocessing/query_processing.py` - Conversational rewriting
- `src/orchestration/nodes.py:74-156` - Expansion decision and query expansion
- `src/orchestration/nodes.py:204-247` - RRF-based multi-query retrieval
- `src/retrieval/query_optimization.py` - Query expansion utilities

**Notes:** Current implementation uses domain-agnostic LLM-based expansion decisions, avoiding heuristic over-expansion anti-patterns.

---

### 8. CrossEncoder Implementation

**File:** `CrossEncoder Implementation.md`

**Purpose:** Technical guide for implementing CrossEncoder-based reranking in RAG pipelines

**When to Use:**
- Implementing or optimizing reranking strategies (cross-encoder vs. bi-encoder architectures)
- Improving retrieval precision, troubleshooting performance, and comparing approaches

**Key Sections:**
- CrossEncoder architecture and theory
- Implementation patterns for reranking
- Model selection and fine-tuning
- Performance optimization techniques
- Integration with retrieval pipelines
- Trade-offs: accuracy vs. latency

**Implementation Status:**
- [IMPLEMENTED] LLM-as-judge reranking (alternative to CrossEncoder)
- [IMPLEMENTED] Relevance scoring in `src/retrieval/retrievers.py`
- [CONSIDERED] CrossEncoder models for future optimization

**Related Code:**
- `src/retrieval/retrievers.py` - Reranking implementation
- `src/retrieval/hybrid_reranker.py` - Hybrid reranking strategies
- `src/retrieval/cross_encoder_reranker.py` - CrossEncoder utilities (if implemented)

**Notes:** System currently uses LLM-based reranking. CrossEncoder models offer faster inference for production scale.

---

### 9. NLI-Based Hallucination Detection for RAG

**File:** `NLI-Based Hallucination Detection for RAG - Best Practices and Production Strategies.md`

**Purpose:** Research-backed guide to NLI-based hallucination detection achieving 0.79-0.83 F1 score in production RAG systems

**When to Use:**
- Implementing NLI-based hallucination detection (zero-shot baseline: 0.65-0.70 F1, production target: 0.79-0.83 F1)
- Selecting NLI models (LettuceDetect, Luna, HHEM-2.1) and implementing two-tier filtering (0.93 F1)

**Key Sections:**
- Zero-shot NLI limitations and label distribution mismatch
- Production NLI models comparison (LettuceDetect 79.22% F1, Luna 65.4% F1)
- Research-backed label mapping (neutral → UNSUPPORTED, entailment > 0.7)
- Fine-tuning on RAGTruth dataset (2-5k examples)
- Two-tier filtering architecture (verifiability + NLI)
- Alternative approaches (AlignScore, TRUE, G-Eval, LUMINA)
- Context window requirements (8k+ tokens for long-form RAG)

**Implementation Status:**
- [IMPLEMENTED] Zero-shot NLI baseline (0.65-0.70 F1) using cross-encoder/nli-deberta-v3-base
- [IMPLEMENTED] Research-backed label mapping and thresholds
- [IMPLEMENTED] Claim decomposition + NLI verification pipeline
- [PLANNED] Fine-tuning on RAGTruth for production (0.79-0.83 F1 target)

**Related Code:**
- `src/validation/nli_hallucination_detector.py` - NLI detector implementation
- `src/orchestration/nodes.py` - Groundedness check node integration
- `tests/integration/test_nli_hallucination_detector.py` - Validation tests

**Notes:** Critical reference for understanding why zero-shot NLI cannot achieve 0.83 F1 and documenting production upgrade path via fine-tuning.

---

### 10. RAG OCR Research Papers

**File:** `RAG_OCR_Research_papers.md`

**Purpose:** Recommendations and best practices for OCR tools tailored to parsing research papers with complex layouts

**When to Use:**
- Implementing OCR for scanned documents with complex layouts (tables, figures, equations)
- Improving text extraction quality and processing non-standard document formats

**Key Sections:**
- OCR tools comparison for academic papers
- Layout analysis techniques
- Table and figure extraction
- Equation recognition
- Multi-column text processing
- Quality validation strategies

**Implementation Status:**
- [NOT IMPLEMENTED] Advanced OCR processing
- [CURRENT] Using PyMuPDF for basic PDF text extraction
- [FUTURE] OCR enhancement for scanned/complex documents

**Related Code:**
- `src/preprocessing/loaders.py` - PDF loading with PyMuPDF
- Future: Enhanced OCR pipeline for complex layouts

**Notes:** Current implementation handles standard PDFs well. OCR enhancements needed for scanned documents or complex layouts.

---

### 11. Python `__pycache__` Best Practices

**File:** `Python '__pycache__' Best Practices.md`

**Purpose:** Comprehensive guide to Python bytecode cache management, covering best practices for development, testing, and production deployments

**When to Use:**
- Debugging stale bytecode issues, configuring CI/CD pipelines, and optimizing .gitignore
- Understanding cache invalidation mechanisms (timestamp vs. hash-based) and troubleshooting imports

**Key Sections:**
- .gitignore best practices for `__pycache__` directories
- Preventing stale bytecode cache issues
- Cache invalidation mechanisms (timestamp-based vs. hash-based Python 3.7+)
- Development approaches (PYTHONDONTWRITEBYTECODE, PYTHONPYCACHEPREFIX)
- Production deployment considerations
- Version compatibility and cross-platform issues
- Cleanup commands and automation strategies

**Implementation Status:**
- [IMPLEMENTED] .gitignore configuration for `__pycache__/` and `*.pyc` files
- [DOCUMENTED] Cache cleanup commands in `../CLAUDE.md` (lines 81-93)
- [RECOMMENDED] PYTHONPYCACHEPREFIX for centralized caching (Python 3.8+)
- [BEST PRACTICE] Standard `__pycache__` behavior with proper .gitignore

**Related Code:**
- `.gitignore` - Excludes `__pycache__/`, `*.pyc`, `*.pyo` files
- `../CLAUDE.md` - Python Cache Management section (lines 81-93)

**Notes:** Current project uses standard `__pycache__` behavior with proper .gitignore. Reference provides advanced strategies for centralized caching, hash-based invalidation, and troubleshooting stale bytecode issues.

---

### 12. Model Selection and Tier Optimization for Advanced Agentic RAG

**File:** `Model Selection and Tier Optimization for Advanced Agentic RAG.md`

**Purpose:** Comprehensive guide to model selection strategies at scale, comparing two production-viable approaches: uniform GPT-5 mini foundation (simplicity-optimized) vs. mixed task-specific routing (cost-optimized). Includes task-by-task allocation, cost-quality trade-offs, and decision framework.

**When to Use:**
- Deciding between GPT-5 mini, GPT-4o, GPT-4o-mini for 12-task RAG pipeline
- Optimizing cost-quality trade-offs at 100K+ queries/day scale
- Implementing conditional model routing and task-specific optimization
- Planning deployment strategy (rapid launch vs. production optimization)
- Evaluating reasoning_effort parameter tuning and prompt caching ROI

**Key Sections:**
- Model capabilities and tier overview (Budget/Balanced/Premium comparison)
- Strategy A: Uniform GPT-5 mini foundation ($2,330/day, 82% quality, 2-week implementation)
- Strategy B: Mixed task-specific routing ($508/day, 88% quality, 4-6 week implementation)
- Trade-off analysis and decision framework (when to choose each strategy)
- Prompting strategies (concise GPT-5 vs. detailed GPT-4o styles)
- Latency optimization and streaming patterns (TTFT < 1s with streaming)
- Production rollout plans (Month 1-4 for A, Week 1-8 for B)
- Hybrid phased approach (start with A, migrate to B incrementally)

**Implementation Status:**
- [DECISION PENDING] Choose between Strategy A (uniform) or Strategy B (mixed)
- [RECOMMENDED] Hybrid approach: Start Strategy A for rapid launch, profile usage, migrate to B
- Strategy A: Simpler (2 weeks), unified prompts, higher cost ($3,147 per quality point)
- Strategy B: Complex (4-6 weeks), conditional routing, superior ROI ($577 per quality point, 5.5x better)

**Cost-Efficiency Comparison:**
- Strategy A (Uniform): $2,330-2,700/day, 82% quality, $3,147 per quality point
- Strategy B (Mixed): $508/day, 88% quality, $577 per quality point (5.5x better ROI)

**Related Documents:**
- `Configurable Model Tier System - Model-Specific vs Model-Agnostic Prompts.md` (prompting deep-dive)
- `Latency-Focused Deep Dive - GPT-4o-mini vs GPT-5 mini for Budget Tier.md` (latency analysis)

**Related Code:**
- Future: `src/core/model_config.py` - Model tier configuration and routing logic
- Future: `src/core/task_router.py` - Conditional model selection by task complexity

**Notes:** This document consolidates two previously separate analyses with contradictory recommendations. Both strategies are production-viable; choice depends on priority (speed-to-market vs cost-optimization). Hybrid phased approach recommended for most teams: launch with Strategy A simplicity, migrate incrementally to Strategy B optimization based on profiling data.

---

### 13. Configurable Model Tier System - Model-Specific vs Model-Agnostic Prompts

**File:** `Configurable Model Tier System - Model-Specific vs Model-Agnostic Prompts.md`

**Purpose:** Deep-dive guide to prompt engineering strategies across model tiers (GPT-5, GPT-4o, GPT-4o-mini), covering model-specific vs model-agnostic approaches, reasoning_effort vs explicit scaffolding trade-offs, and structured output optimization with Pydantic.

**When to Use:**
- Designing system prompts for different model tiers (GPT-5 mini vs GPT-4o vs GPT-4o-mini)
- Optimizing reasoning_effort parameter vs explicit chain-of-thought scaffolding
- Implementing structured outputs (Pydantic/TypedDict) across tiers
- Deciding between model-agnostic base prompts vs model-specific variants
- Understanding GPT-5 instruction sensitivity and prompt precision requirements

**Key Sections:**
- GPT-5 vs GPT-4o prompt style differences (concise vs verbose scaffolding)
- Reasoning effort vs scaffolding trade-off (when to use each)
- Structured outputs with Pydantic (field naming, schema design, few-shot examples)
- GPT-5-nano vs GPT-5.1 prompting strategies (lightweight model considerations)
- Best practices by RAG task category (query rewriting, expansion, evaluation, generation)
- When model-specific prompts provide measurable gains (15-25% improvement scenarios)
- Three-phase implementation strategy (model-agnostic base → tier-aware parameters → selective variants)

**Implementation Status:**
- [RECOMMENDED] Phase 1: Model-agnostic base prompts with semantic clarity (favors GPT-5)
- [RECOMMENDED] Phase 2: Tier-specific reasoning_effort and verbosity parameters (no prompt changes)
- [OPTIONAL] Phase 3: Model-specific variants for hallucination detection and quality evaluation (if >10% gaps)

**Key Findings:**
- GPT-5 performs better with concise, unambiguous prompts; GPT-4o tolerates verbose scaffolding
- For GPT-5 with reasoning_effort="high", use minimal explicit CoT scaffolding (let model reason internally)
- Generic "think step by step" provides only 2.9% average improvement for reasoning models
- Field naming dramatically impacts structured output performance (4.5% → 95% accuracy with better names)
- GPT-5-nano needs simplified, binary-decision-point prompts (struggles with long-winded policies)

**Cost-Benefit Analysis:**
- Pure model-agnostic: 20-30 hours, baseline quality
- Agnostic + param tuning: 30-40 hours, +10-15% via reasoning_effort/verbosity (recommended)
- Selective model-specific: 40-60 hours, +15-25% on high-impact tasks
- Full model-specific: 60-100 hours, +20-30% overall (not recommended—complexity exceeds gains)

**Related Documents:**
- `Model Selection and Tier Optimization for Advanced Agentic RAG.md` (Document 12) - Main strategy document
- `Latency-Focused Deep Dive - GPT-4o-mini vs GPT-5 mini for Budget Tier.md` (Document 14) - Latency optimization

**Related Code:**
- Future: Model-specific prompt templates in `src/prompts/` directory
- `src/orchestration/nodes.py` - Task-specific system prompts
- `src/retrieval/retrievers.py` - Reranking and evaluation prompts

**Notes:** Complements Document 12 by providing implementation-level guidance on HOW to prompt each model tier, while Document 12 focuses on WHICH models to use. Three-phase approach balances implementation complexity with performance gains. Model-agnostic base + tier-aware parameters achieves ~15-20% uplift with minimal complexity.

---

### 14. Latency-Focused Deep Dive - GPT-4o-mini vs GPT-5 mini for Budget Tier

**File:** `Latency-Focused Deep Dive - GPT-4o-mini vs GPT-5 mini for Budget Tier.md`

**Purpose:** Production latency benchmark analysis for Budget tier model selection, comparing GPT-4o-mini vs GPT-5 mini with focus on first-token latency (TTFT), sequential task latency, streaming optimization, and real-world RAG latency benchmarks.

**When to Use:**
- Optimizing latency for Budget tier deployment (GPT-4o-mini vs GPT-5 mini decision)
- Implementing streaming to reduce perceived latency
- Understanding first-token latency (TTFT) vs total latency trade-offs
- Benchmarking production RAG system performance (acceptable latency thresholds)
- Evaluating reasoning_effort impact on latency vs quality

**Key Sections:**
- First-token latency comparison (GPT-4o-mini: 250ms, GPT-5 mini medium: 300ms, minimal: 200ms)
- Sequential latency for 5 critical tasks (GPT-4o-mini: 20.75s, GPT-5 mini: 27.5s)
- Streaming as primary UX lever (TTFT dominates perceived latency)
- Real-world RAG latency benchmarks (complex analysis: 3-10s acceptable)
- Prompt caching ROI for latency (2-3% savings for sequential tasks)
- Reasoning_effort impact on latency (15-30% reduction from medium → low)

**Critical Findings:**
- **Streaming makes perceived latency <1s** regardless of model (TTFT 250-300ms)
- GPT-5 mini 6.75s slower than GPT-4o-mini, but **worth +12% quality gain** for RAG
- Prompt caching provides minimal latency benefit for sequential tasks (only 2-3%)
- Acceptable latency for complex RAG analysis: 20-30s total (not real-time chat <2s)
- Minimal reasoning_effort improves GPT-5 mini TTFT to match/beat GPT-4o-mini

**Production Recommendations:**
1. Deploy GPT-5 mini for Budget tier (quality dominates latency penalty)
2. Implement streaming for all sequential tasks (TTFT <1s perception)
3. Use reasoning_effort: minimal (Task 2), low (Tasks 3-4, 9), medium (Tasks 5-8, 10-11)
4. Implement prompt caching for async tasks (+$418/day savings, not latency)
5. Monitor production latency; if >4s consistently, consider task parallelization

**Hybrid Approach Analysis:**
- GPT-4o-mini for sequential (faster) + GPT-5 mini for async (better quality)
- Saves 6.75s latency vs all-GPT-5 mini
- More complexity; only viable if latency >4s proven business blocker

**Related Documents:**
- `Model Selection and Tier Optimization for Advanced Agentic RAG.md` (Document 12) - Main strategy document
- `Configurable Model Tier System - Model-Specific vs Model-Agnostic Prompts.md` (Document 13) - Prompting optimization

**Related Code:**
- Future: Streaming implementation in `src/orchestration/streaming.py`
- `src/orchestration/graph.py` - Sequential task execution flow
- Future: `src/core/model_config.py` - Reasoning effort configuration

**Notes:** Complements Document 12 by providing detailed latency analysis for Budget tier decision (Strategy A focus). Key insight: Streaming makes latency differences imperceptible (<1s TTFT), so optimize for quality (GPT-5 mini) rather than chasing marginal latency gains. Document debunks common misconception that prompt caching significantly reduces user-facing latency (only helps cost, not sequential task latency).

---

## Quick Reference Guide

**Task** → **Recommended Document**

| Task | Primary Reference | Related Code |
|------|------------------|--------------|
| Design metadata schemas | Document Profiling and Metadata... | `src/preprocessing/document_profiler.py` |
| Implement document profiling | Document Profiling and Metadata... | `src/preprocessing/profiling_pipeline.py` |
| Optimize profiling token budget | LLM-Based Document Profiling... | `src/preprocessing/document_profiler.py:_stratified_sample()` |
| Improve code/math detection | LLM-Based Document Profiling... | `src/preprocessing/document_profiler.py:_detect_signals()` |
| Implement query rewriting | Query Rewriting and Expansion... | `src/preprocessing/query_processing.py` |
| Decide expansion vs. rewriting | Query Rewriting and Expansion... | `src/orchestration/nodes.py:_should_skip_expansion_llm()` |
| Implement retrieval quality scoring | Best Practices for Evaluating... | `src/orchestration/nodes.py:grade_documents()` |
| Set up answer quality gates | Best Practices for Evaluating... | `src/orchestration/nodes.py:assess_answer()` |
| Create evaluation datasets | RAG Golden Dataset Creation... | `test_pdf_pipeline.py`, `src/evaluation/` |
| Integrate RAGAS metrics | RAGAS Integration... | Future: `src/evaluation/` |
| Optimize reranking | CrossEncoder Implementation | `src/retrieval/retrievers.py` |
| Implement NLI hallucination detection | NLI-Based Hallucination Detection... | `src/validation/nli_hallucination_detector.py` |
| Plan production fine-tuning | NLI-Based Hallucination Detection... | Future: RAGTruth fine-tuning pipeline |
| Improve PDF processing | RAG OCR Research Papers | `src/preprocessing/loaders.py` |
| Understand LLM-as-judge | Best Practices for Evaluating... | `src/retrieval/retrievers.py:rerank_documents()` |
| Build test datasets | RAG Golden Dataset Creation... | `test_datasets/` |
| Benchmark performance | RAGAS Integration... | Future evaluation pipeline |
| Handle complex PDFs | RAG OCR Research Papers | Future OCR enhancement |
| Manage Python bytecode cache | Python `__pycache__` Best Practices | `.gitignore`, `../CLAUDE.md` |
| Design RAG system prompts | Best Practices for LLM Prompts... | `src/orchestration/nodes.py`, `src/retrieval/retrievers.py` |
| Implement feedback loops | Research-Backed Best Practices for Self-Correction... | `src/orchestration/graph.py:16-146` |
| Set retry limits | Research-Backed Best Practices for Self-Correction... | `src/orchestration/graph.py` |
| Fix hallucination on retry | Research-Backed Best Practices for Self-Correction... (Section 3b) | `src/orchestration/nodes.py:574-612` |
| Add citation requirements | Research-Backed Best Practices for Self-Correction... (Section 3b) | `src/orchestration/nodes.py:answer_generation_node` |
| Strengthen retry prompts | Research-Backed Best Practices for Self-Correction... (Section 3b) | `src/orchestration/nodes.py:answer_generation_node` |
| Optimize profiling sampling | Best Practices for Document Profiling... (merged) | `src/preprocessing/document_profiler.py:_stratified_sample()` |
| Select models for RAG tasks | Model Selection and Tier Optimization... | Future: `src/core/model_config.py` |
| Optimize cost-quality trade-offs | Model Selection and Tier Optimization... (Section 4) | Future: `src/core/task_router.py` |
| Tune reasoning_effort parameter | Model Selection and Tier Optimization... (Sections 2.4, 3.3) | Task-specific configuration |
| Implement task-specific routing | Model Selection and Tier Optimization... (Section 3.5) | Future: `src/core/task_router.py` |
| Plan deployment strategy | Model Selection and Tier Optimization... (Sections 7, 8) | Production rollout configuration |
| Optimize prompts for model tier | Configurable Model Tier System... (Document 13) | Model-specific prompt variants |
| Design model-specific prompts | Configurable Model Tier System... (Document 13, Section 5.1) | `src/prompts/`, `src/orchestration/nodes.py` |
| Optimize reasoning_effort vs scaffolding | Configurable Model Tier System... (Document 13, Section 5.2) | Task-specific prompt configuration |
| Implement structured outputs (Pydantic) | Configurable Model Tier System... (Document 13, Section 5.3) | Schema design with semantic field names |
| Reduce perceived latency with streaming | Latency-Focused Deep Dive... (Document 14, Section 6.5) | Future: `src/orchestration/streaming.py` |
| Benchmark production latency | Latency-Focused Deep Dive... (Document 14, Section 6.4) | Latency monitoring framework |
| Optimize first-token latency (TTFT) | Latency-Focused Deep Dive... (Document 14, Sections 6.1, 6.3) | Streaming + reasoning_effort tuning |

---

## Development Guidelines

**Development Workflow:**
- Adding features: Consult references, check status, update guide, link code
- Debugging quality: Review evaluation best practices, validate metrics/thresholds, test datasets, optimize reranking
- Optimizing performance: Review CrossEncoder guide, profile reranking, evaluate trade-offs, test with datasets

---

## Additional Resources

**Main Project Documentation:**
- `../CLAUDE.md` - Project instructions and LangChain/LangGraph resources
- `../README.md` - Project overview and setup
- `../HANDOVER.md` - Implementation handover documentation

**LangChain/LangGraph Documentation:**
See `../CLAUDE.md` for comprehensive links to official documentation

**Evaluation Tools:**
- LangSmith: https://docs.langchain.com/langsmith/evaluate-rag-tutorial
- RAGAS: Referenced in `RAGAS Integration...` document

---

*Last Updated: 2025-11-18 (Consolidated: 4 new model selection documents → 14 total documents)*
*Note: This guide uses ASCII-only characters per project guidelines (no emojis/unicode)*
