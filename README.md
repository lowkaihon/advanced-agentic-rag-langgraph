# Advanced Agentic RAG using LangGraph

This Advanced Agentic RAG uses LangGraph to implement features including multi-strategy retrieval (semantic + keyword), LLM-based reranking, intelligent query expansion and rewriting, automatic strategy switching, and self-correcting agent loops with quality evaluation.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Example Usage](#example-usage)
- [How It Works: Example Flow](#how-it-works-example-flow)
- [Self-Correction Example](#self-correction-example)
- [Complete Flow](#complete-flow)
- [Future Improvements](#future-improvements)
  - [Immediate Customization Options](#immediate-customization-options)
  - [Planned Enhancements](#planned-enhancements)
- [Technology Stack](#technology-stack)

## Features

### 1. Document & Corpus Profiling
- LLM-based profiling of documents before chunking to analyze technical density, document types, and domain characteristics

### 2. Conversational Query Rewriting
- Transforms follow-up queries into self-contained questions using conversation history

### 3. Query Optimization
- Generates query variations and rewrites unclear queries to improve retrieval coverage
- RRF (Reciprocal Rank Fusion) merges results across query variants BEFORE reranking using ranking scores instead of naive deduplication

### 4. Intelligent Strategy Selection
- Pure LLM-based system to select optimal retrieval strategy (semantic/keyword/hybrid) per query

### 5. Multi-Strategy Retrieval
- Three retrieval approaches (FAISS semantic, BM25 keyword, or hybrid) with dynamic selection
- RRF fusion applied BEFORE reranking: Documents appearing in multiple query results accumulate higher scores (formula: sum(1/(rank + 60)) across variants)

### 6. Quality Gates
- Conditional routing at retrieval and answer generation stages with adaptive thresholds

### 7. Three-Tier Self-Correction Loops
- **Query Rewriting Loop**: Poor retrieval (score <0.6) → issue-specific rewriting guidance (8 issue types) → retry (max 2 rewrites)
  - Example: `missing_key_info` → "Add specific keywords, technical terms, or entities that might appear in relevant documents"
- **Hallucination Correction Loop** (three-tier retrieval-aware routing):
  - MODERATE (0.6-0.8): Likely NLI false positive → log warning, proceed without retry
  - SEVERE + good retrieval (≥0.6): LLM hallucination → NLI verification → list unsupported claims → regenerate with strict grounding → retry (max 2)
  - SEVERE + poor retrieval (<0.6): Retrieval-caused hallucination → flag for re-retrieval with strategy change (46% hallucination reduction vs regeneration)
- **Strategy Switching Loop**: Insufficient answer → content-driven mapping (missing_key_info → semantic, off_topic → keyword) → regenerate query expansions for new strategy → retry (max 3 attempts)
  - Regenerates query expansions when strategy changes (not reused from previous strategy)

### 8. Multi-turn Conversations
- Preserves conversation context across queries with state persistence and thread management

### 9. Real-time Streaming
- Streams execution progress in real-time
- Shows node transitions and quality scores
- Verbose mode for detailed debugging

### 10. Dual-Tier Content-Driven Strategy Switching
- **Early detection** (route_after_retrieval): Detects obvious strategy mismatches (off_topic, wrong_domain) and switches immediately before wasting retrieval attempts (saves 30-50% tokens)
- **Late detection** (route_after_evaluation): Maps retrieval quality issues to optimal strategies (missing_key_info → semantic, off_topic/wrong_domain → keyword, partial_coverage → intelligent fallback) after answer proves insufficient
- Both tiers regenerate query expansions optimized for new strategy
- Tracks refinement history with reasoning and detected issues

### 11. Two-Stage Reranking
- Applied AFTER RRF multi-query fusion to the fused candidate pool
- Stage 1: CrossEncoder (ms-marco-MiniLM-L-6-v2) filters to top-15
- Stage 2: LLM-as-judge scores each document 0-100 for relevance, selects top-4
- Temperature 0 for consistency, metadata-aware scoring
- 3-5x faster than pure LLM reranking
- 5-10x cheaper while maintaining quality

### 12. NLI-Based Hallucination Detection
- Claim decomposition: LLM extracts individual claims from answers
- NLI verification: cross-encoder/nli-deberta-v3-base validates each claim
- Research-backed label mapping: entailment (>0.7) → SUPPORTED
- Zero-shot baseline: ~0.65-0.70 F1 score
- Three-tier routing: MODERATE (0.6-0.8) protects against false positives, SEVERE (<0.6) triggers retrieval-aware correction (distinguishes LLM vs retrieval-caused hallucination)
- Sets retrieval_caused_hallucination flag when poor retrieval (<0.6) causes hallucinations, triggering re-retrieval with strategy change

### 13. Comprehensive Evaluation Framework
- Retrieval metrics: Recall@K, Precision@K, F1@K, nDCG, MRR, Hit Rate
- Generation metrics: Groundedness, hallucination rate, confidence, answer quality
- Golden dataset: 20 validated examples with graded relevance
- RAGAS integration: 4 industry-standard metrics
- Answer evaluation: vRAG-Eval framework (Relevance, Completeness, Accuracy) with 8 issue types and adaptive thresholds (65%/50% based on retrieval quality)

### 14. RAGAS Integration
- Faithfulness: Measures hallucinations (LLM extracts + verifies claims)
- Context Recall: Evaluates retrieval completeness vs ground truth
- Context Precision: Checks if relevant contexts ranked higher
- Answer Relevancy: Embedding similarity between question and answer

### 15. Context Sufficiency Enhancement
- Pre-generation validation: Checks if retrieved context is complete
- Missing aspects detection: Identifies gaps before answer generation
- Context-driven routing: Switches to semantic when context insufficient
- Reduces hallucinations by 5-10% through early validation

## Architecture Overview

### System Components

**1. Document Profiling** (`preprocessing/document_profiler.py`)
- LLM-based profiling with 29 document types across academic, educational, technical, business, legal, and general domains
- Analyzes corpus characteristics: technical density (0.0-1.0), document type, domain tags
- Profiles full documents BEFORE chunking, then attaches metadata to all chunks
- Informs retrieval strategy selection based on content patterns

**2. Query Analysis & Optimization** (`preprocessing/query_processing.py`, `retrieval/query_optimization.py`)
- **Conversational Rewriting**: Makes queries self-contained using conversation history
- **Query Expansion**: Generates 3 variations (technical, simple, different aspect)
- **RRF Fusion**: Reciprocal Rank Fusion aggregates rankings across query variants BEFORE reranking (3-5% MRR improvement vs naive deduplication)
- **Intent Classification**: factual, conceptual, comparative, procedural
- **Complexity Assessment**: simple, moderate, complex

**3. Intelligent Strategy Selection** (`retrieval/strategy_selection.py`)
- Pure LLM classification - domain-agnostic, handles all edge cases
- Analyzes query characteristics + corpus statistics
- Selects semantic/keyword/hybrid with confidence score + reasoning

**4. Multi-Strategy Retrieval** (`retrieval/retrievers.py`, `retrieval/two_stage_reranker.py`)
- **Semantic**: FAISS vector search for meaning-based retrieval
- **Keyword**: BM25 lexical search for exact term matching
- **Hybrid**: Combines both approaches with RRF-based fusion (replaces naive set deduplication)
- **RRF Multi-Query Fusion**: Aggregates rankings across query variants BEFORE reranking using formula: score(doc) = sum(1/(rank + 60))
- **Two-Stage Reranking** (applied AFTER RRF fusion):
  - Stage 1: CrossEncoder (`cross_encoder_reranker.py`) filters to top-15 (200-300ms)
  - Stage 2: LLM-as-judge (`llm_metadata_reranker.py`) selects top-4 with metadata awareness

**5. LangGraph Orchestration** (`orchestration/graph.py`, `orchestration/nodes.py`)
- 9 nodes with conditional routing based on quality scores
- Integrated metadata analysis within retrieval evaluation (not separate node)
- Quality gates at retrieval and answer generation stages
- Three feedback loops with issue-specific guidance:
  1. Query rewriting: 8 issue types → actionable rewriting instructions
  2. Hallucination correction: Lists unsupported claims → strict grounding prompt
  3. Strategy switching: Maps issues to strategies + regenerates query expansions
- Streams execution progress in real-time

**6. Evaluation & Validation** (`evaluation/`, `validation/`)
- **Retrieval Metrics** (`retrieval_metrics.py`): Recall@K, Precision@K, F1@K, nDCG, MRR, Hit Rate
- **Golden Dataset** (`golden_dataset.py`): 20 validated examples, graded relevance, evaluation pipeline
- **RAGAS Integration** (`ragas_evaluator.py`): 4 industry-standard metrics (see Features #14)
- **NLI Hallucination Detection** (`validation/nli_hallucination_detector.py`): Claim decomposition + NLI verification
- **Answer Assessment**: Semantic similarity, factual accuracy, completeness scoring

**7. State Management** (`core/state.py`)
- TypedDict schema (AdvancedRAGState) for performance
- MemorySaver checkpointer for conversation persistence
- Tracks: queries, documents, quality scores, attempts, conversation history

## Quick Start

**Prerequisites:** Python 3.11 or higher

```bash
# 1. Install package + dependencies (uses uv, not pip)
uv sync  # Installs project in editable mode + all dependencies

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 3. Run tests (10 integration tests available - no PYTHONPATH needed)
# Quick smoke tests (~30s each)
uv run python tests/integration/test_cross_encoder.py      # Reranking validation
uv run python tests/integration/test_groundedness.py       # Groundedness validation
uv run python tests/integration/test_ragas_simple.py       # RAGAS smoke test

# Core pipeline tests (~1-2 min)
uv run python tests/integration/test_pdf_pipeline.py       # Complete pipeline
uv run python tests/integration/test_adaptive_retrieval.py # Adaptive retrieval

# Comprehensive evaluation (10-15 min each)
uv run python tests/integration/test_golden_dataset_evaluation.py   # Golden dataset
uv run python tests/integration/test_ragas_evaluation.py            # RAGAS comprehensive

# See tests/CLAUDE.md for complete guide with all 10 tests

# 4. Run interactive demo
uv run python main.py
```

### Example Usage

```python
from main import run_advanced_rag

# Single query
result = run_advanced_rag(
    question="How does multi-head attention work?",
    thread_id="demo-1",
    verbose=True
)

# Multi-turn conversation (context is preserved)
run_advanced_rag("What is a transformer?", thread_id="conv-1")
run_advanced_rag("How does it work?", thread_id="conv-1")
# Automatically rewrites to: "How does the transformer work?"

run_advanced_rag("What are the key components?", thread_id="conv-1")
# Rewrites to: "What are the key components of the transformer?"
```

### How It Works: Example Flow

**User Query**: "What is self-attention?"

**1. Query Analysis**
```
- Question type: "what" (factual)
- Intent: conceptual
- Technical score: 0.17
- Complexity: simple
```

**2. Strategy Selection**
```
LLM analyzes query characteristics:
- Intent: conceptual question about core concept
- Corpus: technical research papers
- Best match: semantic search for conceptual understanding

Selected: SEMANTIC (confidence: 0.85)
Reasoning: "Conceptual query best matched by semantic similarity"
```

**3. Query Expansion**
```
Original: "What is self-attention?"
- Technical: "Explain the self-attention mechanism in transformers"
- Simple: "What does self-attention mean?"
- Different: "How is self-attention different from regular attention?"
```

**4. Retrieval**
```
- Search with all 3 query variations
- RRF fusion: Documents appearing in multiple results get higher scores → 8 unique documents
- Two-stage reranking: CrossEncoder filters to top-15, then LLM-as-judge selects top-4
```

**5. Quality Check**
```
LLM evaluates: "Do these docs answer the query?"
Score: 85/100 → PASS (threshold: 60)
```

**6. Answer Generation**
```
- Generate answer with quality-aware instructions
- High retrieval quality → confident answer
```

**7. Answer Evaluation**
```
Checks: relevant? complete? accurate?
Confidence: 88% → SUFFICIENT → return answer
```

**Result**:
- Strategy: SEMANTIC
- Retrieval attempts: 1
- Final confidence: 88%

### Self-Correction Example

**Scenario**: Query with poor initial retrieval

**User Query**: "MultiHeadAttention function"

**Attempt 1**:
```
Strategy: KEYWORD (exact lookup)
Retrieval: 3 documents found
Quality score: 42/100 → FAIL (< 60)
```

**Self-Correction: Query Rewriting**
```
Rewritten query: "MultiHeadAttention function parameters and usage"
Retrieval: 5 documents found
Quality score: 68/100 → PASS
Generate answer...
```

**Answer Evaluation**:
```
Confidence: 65%
Assessment: Not complete → INSUFFICIENT
```

**Self-Correction: Strategy Switching**
```
Switch: KEYWORD → SEMANTIC
Retrieval: 6 documents found
Quality score: 75/100
Generate answer...
```

**Answer Evaluation**:
```
Confidence: 78%
Assessment: SUFFICIENT → return answer
```

**Result**:
- Total attempts: 3 (1 rewrite + 1 strategy switch)
- Final strategy: SEMANTIC
- Final confidence: 78%

## Complete Flow

The system uses a 9-node LangGraph workflow with conditional routing and self-correction loops:

```
┌─────────────────────────────────────────────────────────────────┐
│ START: User Question                                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 1: Conversational Rewriting                                │
│ • Checks conversation history                                   │
│ • Makes query self-contained (resolves pronouns, references)    │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 2: Query Expansion                                         │
│ • Generates 3 variations (technical, simple, different aspect)  │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 3: Strategy Selection                                      │
│ • Analyzes query features + corpus characteristics             │
│ • Pure LLM classification (domain-agnostic)                    │
│ • Selects: SEMANTIC, KEYWORD, or HYBRID                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 4: Retrieval with Expansion                               │
│ • Retrieves using selected strategy with all query variations  │
│ • Semantic: FAISS vector search                                │
│ • Keyword: BM25 lexical search                                 │
│ • Hybrid: combines both                                        │
│ • RRF fusion FIRST: Ranks docs by cross-query consensus (3-5% MRR gain) │
│ • Two-stage reranking AFTER RRF: CrossEncoder filters to top-15, LLM selects top-4 │
│ • Integrated metadata analysis: evaluates retrieval quality (0-100 score) and detects issues │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌────────┴────────┐
                    │ Quality Gate #1  │
                    │ Quality ≥ 60%?   │
                    │ OR attempts ≥ 3? │
                    └────┬────────┬────┘
                   YES   │        │   NO
                         ↓        ↓
              ┌──────────┘        └──────────────┐
              │                                   │
              │            ┌──────────────────────┴──────────────────┐
              │            │ Node 5: Rewrite and Refine              │
              │            │ • Maps 8 issue types to specific rewriting guidance │
              │            │ • Example: missing_key_info → add keywords │
              │            │ • Increments attempt counter            │
              │            └──────────┬──────────────────────────────┘
              │                       │
              │                       └─────────┐ (retry retrieval)
              │                                 ↓
              │                          (back to Node 4)
              │
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 6: Answer Generation with Quality Context                 │
│ • Structured RAG prompting with XML markup                     │
│ • Quality-aware instructions (>0.8, >0.6, ≤0.6 thresholds)     │
│ • Prepends hallucination feedback when retry_needed=True       │
│ • Lists specific unsupported claims for targeted regeneration  │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 7: Groundedness Check (retrieval-aware routing)           │
│ • NLI-based hallucination detection with claim verification    │
│ • Three-tier thresholds: NONE (≥0.8), MODERATE (0.6-0.8), SEVERE (<0.6) │
│ • MODERATE: NLI false positive protection → proceed            │
│ • SEVERE + good retrieval: LLM hallucination → retry generation│
│ • SEVERE + poor retrieval: Sets retrieval_caused flag → re-retrieval │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 8: Evaluate Answer (vRAG-Eval framework)                  │
│ • Evaluates: Relevance, Completeness, Accuracy                 │
│ • Detects 8 issue types for content-driven routing             │
│ • Adaptive threshold: 65% (good retrieval) or 50% (poor)       │
│ • Computes confidence score                                    │
│ • Missing aspects detection for incomplete context            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌────────┴────────┐
                    │ Quality Gate #2  │
                    │ Answer sufficient?│
                    │ OR attempts ≥ 3? │
                    └────┬────────┬────┘
                   YES   │        │   NO
                         │        │
                         │        └────────────────────────────┐
                         │                                     │
                         ↓                          ┌──────────┴──────────────────┐
              ┌──────────────────────┐              │ Content-Driven Switching:   │
              │ END: Return Answer   │              │ Maps issues to strategies:  │
              │ • Final answer       │              │ missing_key_info → semantic │
              │ • Confidence score   │              │ off_topic → keyword         │
              │ • Strategy used      │              │ Regenerates query expansions│
              │ • Attempts made      │              └──────────┬──────────────────┘
              └──────────────────────┘                         │
                                                               └─────────┐
                                                                         ↓
                                                               (back to Node 2: regenerate expansions)

Self-Correction Loops:
• Loop 1 (Query Rewriting): Quality < 0.6 AND attempts < 2 → issue-specific feedback (8 types) → rewrite query → retry
• Loop 2 (Hallucination Correction - retrieval-aware):
  - MODERATE (0.6-0.8): Log warning, proceed (NLI false positive protection)
  - SEVERE + good retrieval (≥0.6): LLM hallucination → NLI verification → list unsupported claims → regenerate with strict grounding (max 2 retries)
  - SEVERE + poor retrieval (<0.6): Retrieval-caused → flag for re-retrieval with strategy change
• Loop 3 (Dual-Tier Strategy Switching):
  - Early tier (after retrieval): Detects off_topic/wrong_domain → immediate switch → saves 30-50% tokens
  - Late tier (after evaluation): Answer insufficient AND attempts < 3 → content-driven mapping (missing_key_info → semantic, off_topic → keyword)
  - Both tiers regenerate query expansions when strategy changes (routes to Node 2)
```

**Key Points**:
- Not a linear pipeline - uses conditional routing based on quality scores and content analysis
- Integrated metadata analysis within retrieval evaluation (not separate node)
- Three self-correction loops with issue-specific feedback:
  1. Query rewriting: 8 issue types → actionable guidance
  2. Hallucination correction: NLI verification → unsupported claims list → strict regeneration
  3. Strategy switching: Content-driven mapping + query expansion regeneration
- Strategy switching uses retrieval_quality_issues for intelligent adaptation (missing_key_info → semantic, off_topic → keyword)
- Groundedness check uses NLI-based claim verification (zero-shot F1: 0.65-0.70)
- Query expansions regenerated when strategy changes (not reused)
- 9 nodes total

## Future Improvements

### Immediate Customization Options

These features can be implemented by extending existing components:

- **Add custom retrieval strategies** - Implement retrievers in `retrieval/retrievers.py` and update `StrategySelector` LLM prompt
- **Adjust quality thresholds** - Customize retrieval/answer quality thresholds in `orchestration/graph.py` routing functions (default: 0.6)
- **Extend document profiling** - Add custom analysis features in `preprocessing/document_profiler.py` (now using stratified sampling + regex pre-detection)
- **Integrate external reranking** - Replace internal reranking with Cohere or Pinecone using `ContextualCompressionRetriever`

### Planned Enhancements

**Data & Model Optimization:**
- Fine-tune NLI hallucination detector on RAGTruth dataset to improve F1 score
- Expand golden dataset from 20 to 100 examples using RAGAS TestsetGenerator with human validation
- Benchmark domain-specific embeddings (Specter, SciBERT) against OpenAI for 10-20% retrieval improvement
- Systematically optimize chunk size (256-2048 tokens) and overlap (0-200) for better recall and context sufficiency

**Production & Monitoring:**
- Integrate LangSmith tracing, user feedback collection, and real-time quality dashboards for continuous evaluation
- Historical performance tracking: learn which strategies work best for query types over time
- User feedback loop: learn from user ratings to improve strategy selection

**Infrastructure & Scalability:**
- Query result caching: cache common queries to reduce latency and costs
- Document clustering: group similar documents for faster retrieval
- Hybrid vector stores: support for multiple vector databases (Pinecone, Weaviate, Qdrant)

## Technology Stack

- **LLM Framework**: LangChain 1.0 (production-ready)
- **Orchestration**: LangGraph 1.0 (state-based workflows)
- **Vector Store**: FAISS (semantic search)
- **Lexical Search**: BM25 (keyword matching)
- **LLM**: OpenAI GPT-4o-mini (strategy selection, reranking, generation)
- **PDF Processing**: PyMuPDF
- **Package Manager**: uv (faster than pip)
- **Reranking**: sentence-transformers (CrossEncoder models)
- **Evaluation**: RAGAS (industry-standard RAG metrics), datasets (RAGAS dependency)
- **Hallucination Detection**: cross-encoder/nli-deberta-v3-base (NLI model)
- **Python**: 3.11+