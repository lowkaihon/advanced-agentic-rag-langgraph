# Advanced Agentic RAG using LangGraph

An advanced Agentic RAG system that autonomously adapts its retrieval strategy and reasoning process through dynamic decision-making, iterative self-correction, and intelligent tool selection. Built with LangGraph's StateGraph pattern, the system embeds autonomous reasoning into a 7-node workflow where routing functions and conditional edges provide distributed intelligence—no central "agent" orchestrator needed.

**What makes it "Agentic"**: The system continuously evaluates retrieval quality and answer sufficiency, autonomously deciding whether to proceed, rewrite queries, switch strategies, or retry generation based on intermediate results. This follows the "Dynamic Planning and Execution Agents" pattern where the graph structure itself encodes planning logic and decision-making flows.

**What makes it "Advanced"**: Implements research-backed enhancements (CRAG, PreQRAG, RAG-Fusion, vRAG-Eval) including early strategy switching, NLI-based hallucination detection, strategy-specific query optimization (13-14% MRR improvement), and issue-specific feedback loops across 16 quality dimensions.

## Table of Contents

- [Why This Qualifies as Agentic RAG](#why-this-qualifies-as-agentic-rag)
- [Advanced Features Beyond Basic RAG](#advanced-features-beyond-basic-rag)
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

## Why This Qualifies as Agentic RAG

This system demonstrates all four core characteristics that define Agentic RAG (as opposed to traditional static RAG pipelines):

### 1. Autonomous Decision-Making and Planning

**Traditional RAG**: Fixed pipeline (query → retrieve → generate)
**This System**: Dynamic routing based on quality evaluation

The system autonomously plans next steps based on intermediate results:
- `route_after_retrieval`: Decides to proceed, rewrite query, or switch strategy based on quality scores and detected issues
- `route_after_evaluation`: Evaluates answer quality, decides to retry generation or return result

**Not a fixed sequence** - uses conditional routing at 2 decision points throughout the workflow.

### 2. Iterative Refinement and Self-Correction

**Traditional RAG**: Single-pass retrieval and generation
**This System**: Two self-correction loops with quality gates

**Loop 1 - Query Rewriting**: Poor retrieval quality (<0.6) -> 8 issue types detected -> issue-specific feedback -> rewrite -> retry (max 3)

**Loop 2 - Generation Retry**: Consolidated evaluation (refusal detection + NLI hallucination + quality assessment) -> unified feedback -> regenerate with adaptive temperature -> retry (max 3)

**Early Strategy Switching**: off_topic/wrong_domain detected at retrieval -> immediate strategy switch -> saves 30-50% tokens

**Key Principle**: Fix generation problems with generation strategies, not by retrieving more documents (no re-retrieval after generation).

### 3. Context Management and Multi-Turn Reasoning

**Traditional RAG**: Stateless, no conversation memory
**This System**: Persistent state across conversation turns

- `conversational_rewrite_node`: Transforms follow-up queries into self-contained questions using conversation history
- `messages` field with add_messages reducer: Tracks conversation history (LangGraph best practice)
- MemorySaver checkpointer (graph.py:441): Persists state across multi-turn conversations with thread management

### 4. Intelligent Tool Use and Source Selection

**Traditional RAG**: Single retrieval method for all queries
**This System**: Three retrieval strategies with intelligent selection

**"Tools" in Broader Context**: In agentic RAG literature, "tools" refers to retrieval methods, processing techniques, and verification mechanisms—not just LLM function-calling tools.

**This system's "tool" capabilities**:
- **Retrieval strategies** (the "tools"): Semantic (FAISS vector), Keyword (BM25 lexical), Hybrid (RRF fusion)
- **Decision-making** (the "intelligence"): `decide_retrieval_strategy_node` analyzes corpus stats + query characteristics → selects optimal strategy
- **Dynamic adaptation** (the "selection"): Switches strategies mid-execution based on content analysis (off_topic → keyword, missing_key_info → semantic)

The system assesses query intent and selects specialized retrieval methods based on what the query requires—exactly matching the definition of intelligent tool selection in agentic systems.

### Architecture Pattern: Dynamic Planning and Execution Agents

**No "Main Agent" Needed**: The LangGraph StateGraph itself IS the agent. Decision-making is distributed across routing functions and conditional edges rather than centralized in a single LLM orchestrator. This is a more sophisticated pattern than traditional ReAct agents because:
- Decision logic is specialized per routing point (2 routing functions, each with distinct responsibilities)
- More controllable and debuggable than single-agent decision-making
- Maintains full autonomy through quality-driven conditional routing

This follows the "Dynamic Planning and Execution Agents" pattern where graph structure encodes planning logic and routing functions provide reasoning.

## Advanced Features Beyond Basic RAG

What elevates this from "agentic RAG" to "advanced agentic RAG":

### Research-Backed Enhancements

**CRAG (Corrective RAG)**: Confidence-based action triggering with issue-specific feedback
- Early detection at retrieval stage (off_topic/wrong_domain triggers immediate strategy switch)
- Saves 30-50% tokens by avoiding wasted retrieval attempts
- Key principle: Fix generation problems with generation strategies, not re-retrieval

**PreQRAG**: Strategy-specific query optimization (13-14% MRR improvement)
- Keyword strategy → adds specific terms, identifiers, proper nouns
- Semantic strategy → broadens to conceptual phrasing, semantic relationships
- Hybrid strategy → balances specificity with conceptual framing

**RAG-Fusion**: Multi-query retrieval with RRF ranking fusion (3-5% MRR improvement)
- Strategy-agnostic expansions → select best strategy → apply to ALL variants
- RRF aggregates rankings across query variants BEFORE reranking
- Two-stage reranking: CrossEncoder (top-10) → LLM-as-judge (top-4)

**vRAG-Eval**: Answer quality evaluation with adaptive thresholds
- 8 issue types for content-driven routing (incomplete_synthesis, lacks_specificity, missing_details, unsupported_claims, partial_answer, wrong_focus, retrieval_limited, contextual_gaps)
- Adaptive thresholds: 65% (good retrieval), 50% (poor retrieval)
- Maps issues to strategies for targeted improvement

### NLI-Based Hallucination Detection

Integrated into the consolidated evaluate_answer node:
- Claim decomposition: LLM extracts individual claims from answers
- NLI verification: cross-encoder/nli-deberta-v3-base validates each claim against retrieved context
- Groundedness scoring: Percentage of claims supported (threshold: 0.8)
- Unified feedback: Unsupported claims listed in retry_feedback for targeted regeneration

### Early Strategy Switching

**Detection** (route_after_retrieval): Catches obvious mismatches immediately
- Detects: off_topic, wrong_domain in initial retrieval results
- Action: Immediate strategy switch before wasting retrieval attempts
- Benefit: Saves 30-50% tokens vs naive retry-all approach
- Optimizes query for new strategy and regenerates expansions

### Issue-Specific Feedback Loops (16 Quality Dimensions)

Traditional systems give generic "try again" feedback. This system provides actionable guidance:

**8 Retrieval Quality Issues**: partial_coverage, missing_key_info, incomplete_context, domain_misalignment, low_confidence, mixed_relevance, off_topic, wrong_domain

**8 Answer Quality Issues**: incomplete_synthesis, lacks_specificity, missing_details, unsupported_claims, partial_answer, wrong_focus, retrieval_limited, contextual_gaps

Each issue maps to specific rewriting instructions or strategy changes, ensuring every retry has targeted improvement.

## Features

### 1. Document & Corpus Profiling
- LLM-based profiling of documents before chunking to analyze technical density, document types, and domain characteristics

### 2. Conversational Query Rewriting
- Transforms follow-up queries into self-contained questions using conversation history

### 3. Query Optimization
- Generates query variations and rewrites unclear queries to improve retrieval coverage
- RAG-Fusion architecture: Strategy-agnostic expansions → select optimal strategy → apply to all variants (differs from PreQRAG parallel multi-strategy retrieval)
- RRF (Reciprocal Rank Fusion) merges results across query variants BEFORE reranking using ranking scores instead of naive deduplication

### 4. Intelligent Strategy Selection
- Pure LLM-based system to select optimal retrieval strategy (semantic/keyword/hybrid) per query

### 5. Multi-Strategy Retrieval
- Three retrieval approaches (FAISS semantic, BM25 keyword, or hybrid) with dynamic selection
- RRF fusion applied BEFORE reranking: Documents appearing in multiple query results accumulate higher scores (formula: sum(1/(rank + 60)) across variants)

### 6. Quality Gates
- Conditional routing at retrieval and answer generation stages with adaptive thresholds

### 7. Two Self-Correction Loops
- **Query Rewriting Loop**: Poor retrieval (score <0.6) -> issue-specific rewriting guidance (8 issue types) -> retry (max 3)
  - Example: `missing_key_info` -> "Add specific keywords, technical terms, or entities that might appear in relevant documents"
- **Generation Retry Loop**: Consolidated evaluation (refusal + NLI hallucination + quality) -> unified feedback -> regenerate with adaptive temperature (0.3/0.7/0.5) -> retry (max 3)
  - Key principle: Fix generation problems with generation strategies, not by retrieving more documents

### 8. Multi-turn Conversations
- Preserves conversation context across queries with state persistence and thread management

### 9. Real-time Streaming
- Streams execution progress in real-time
- Shows node transitions and quality scores
- Verbose mode for detailed debugging

### 10. Early Strategy Switching
- **Detection** (route_after_retrieval): Detects obvious strategy mismatches (off_topic, wrong_domain) and switches immediately before wasting retrieval attempts (saves 30-50% tokens)
- Optimizes queries using strategy-specific guidance (keyword=specific terms/identifiers, semantic=conceptual phrasing, hybrid=balanced approach) and regenerates query expansions (13-14% MRR improvement from CRAG/PreQRAG research)
- Tracks refinement history with reasoning and detected issues

### 11. Two-Stage Reranking
- Applied AFTER RRF multi-query fusion to the fused candidate pool
- Stage 1: CrossEncoder (ms-marco-MiniLM-L-6-v2) filters to top-10
- Stage 2: LLM-as-judge scores each document 0-100 for relevance, selects top-4
- Temperature 0 for consistency, metadata-aware scoring
- 3-5x faster than pure LLM reranking
- 5-10x cheaper while maintaining quality

### 12. NLI-Based Hallucination Detection
- Claim decomposition: LLM extracts individual claims from answers
- NLI verification: cross-encoder/nli-deberta-v3-base validates each claim against retrieved context
- Research-backed label mapping: entailment (>0.7) -> SUPPORTED
- Zero-shot baseline: ~0.65-0.70 F1 score
- Groundedness threshold: 0.8 (unsupported claims trigger regeneration with unified feedback)

### 13. Comprehensive Evaluation Framework
- Retrieval metrics: Recall@K, Precision@K, F1@K, nDCG, MRR, Hit Rate
- Generation metrics: Groundedness, hallucination rate, confidence, answer quality
- **Two-Tier Evaluation Strategy**:
  - **Standard Dataset** (`golden_set_standard.json`): 20 questions (2-3 avg chunks, 10% cross-doc) - Target: F1@5 65-75%, Groundedness 95-97%, k_final=4
  - **Hard Dataset** (`golden_set_hard.json`): 10 questions (3.3 avg chunks, 50% cross-doc) - Target: F1@5 32-40%, Groundedness 92-95%, k_final=6
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

This system follows the **Dynamic Planning and Execution Agents** pattern, where autonomous decision-making is distributed across specialized routing functions rather than centralized in a single LLM orchestrator. The 7-node LangGraph StateGraph workflow uses conditional edges to create an adaptive, quality-driven pipeline that autonomously decides next steps based on intermediate results.

**Key Architectural Principles**:
- **Distributed Intelligence**: 2 routing functions with specialized decision logic (retrieval quality assessment, answer evaluation)
- **Quality-Driven Flow**: Conditional routing at each stage based on quality scores, not predetermined sequences
- **Autonomous Adaptation**: System decides whether to proceed, rewrite, switch strategies, or retry without human intervention
- **State Persistence**: TypedDict schema with MemorySaver checkpointer enables multi-turn conversations

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
  - Stage 1: CrossEncoder (`cross_encoder_reranker.py`) filters to top-10 (200-300ms)
  - Stage 2: LLM-as-judge (`llm_metadata_reranker.py`) selects top-4 with metadata awareness

**5. LangGraph Orchestration** (`orchestration/graph.py`, `orchestration/nodes.py`)
- 7 nodes with conditional routing based on quality scores
- Integrated metadata analysis within retrieval evaluation (not separate node)
- Quality gates at retrieval and answer generation stages
- Two self-correction loops:
  1. Query rewriting: 8 issue types -> actionable rewriting instructions
  2. Generation retry: Unified feedback (hallucination + quality) -> regenerate with adaptive temperature
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
- Tracks: queries, documents, quality scores, retrieval_attempts, generation_attempts, retry_feedback, conversation history
- **State Management Patterns:**
  - `add_messages`: Message history (idempotent, deduplicates by ID) - LangGraph best practice
  - `operator.add`: Documents and refinement_history (accumulate across iterations)
  - Direct replacement: query_expansions (regenerated per iteration for expansion-query alignment)

## Quick Start

**Prerequisites:** Python 3.11 or higher

```bash
# 1. Install package + dependencies (uses uv, not pip)
uv sync  # Installs project in editable mode + all dependencies

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env
# Set MODEL_TIER (budget|balanced|premium) - defaults to budget

# Example .env:
OPENAI_API_KEY=sk-your-key-here
MODEL_TIER=budget  # Options: budget, balanced, premium

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

## Model Tier Configuration

Control cost-quality tradeoffs via `MODEL_TIER` environment variable:

| Tier | Models | Use Case |
|------|--------|----------|
| **budget** | All GPT-4o-mini | Development, demos, architecture showcase |
| **balanced** | GPT-4o-mini + GPT-5-mini | Production (cost-conscious) |
| **premium** | GPT-5.1 + GPT-5-mini + GPT-5-nano | Production (quality-critical) |

**Portfolio Strategy:** "Architecture baseline + model upgrades"
1. **Budget tier** showcases RAG architecture value using only GPT-4o-mini
2. **Balanced tier** adds selective GPT-5-mini upgrades for improved quality
3. **Premium tier** uses best models for all tasks for maximum quality

**Tier System Exceptions:**

The system uses **13 LLM instances total**: 11 are tier-aware (runtime operations like query rewriting, answer generation, strategy selection) and 2 are hardcoded for quality-critical operations:

- **Golden dataset evaluation**: Hardcoded to `gpt-5-mini` regardless of tier. Rationale: Consistent evaluation quality across all tiers (85-90% agreement with human judges), enables accurate tier comparison metrics.
- **Document profiling**: Hardcoded to `gpt-4o-mini` regardless of tier. Rationale: Tier-independence (ingestion happens once before runtime), already optimal for the task, minimal cost impact.

**Design principle:** Evaluation and ingestion prioritize quality/appropriateness; runtime components prioritize tier-based cost-quality flexibility.

**Configuration:**
```bash
# Set in .env
MODEL_TIER=budget    # Default - best cost-efficiency
MODEL_TIER=balanced  # Best cost-quality tradeoff
MODEL_TIER=premium   # Maximum quality
```

**Validation:** Run tier comparison test to measure performance on your dataset:
```bash
uv run python tests/integration/test_tier_comparison.py
```

See `evaluation/tier_comparison_report.md` for detailed results.

## Architecture Tier Comparison

Showcase the value of advanced RAG architecture through 4-tier A/B testing. All tiers use the same **BUDGET model tier** (gpt-4o-mini) to isolate architectural improvements from model quality differences.

| Tier | Features | Graph Structure | Description |
|------|----------|-----------------|-------------|
| **Basic** | 1 feature | Simplest (2 nodes, no routing) | Semantic vector search, top-k chunks, no reranking |
| **Intermediate** | 5 features (+4) | Linear (4 nodes, no routing) | + Query expansion, hybrid retrieval, CrossEncoder reranking, RRF fusion |
| **Advanced** | 17 features (+12) | Full agentic (7 nodes, 2 routers) | + Strategy selection, two-stage reranking, NLI hallucination detection, quality gates, adaptive loops |
| **Multi-Agent** | 20 features (+3) | Orchestrator-worker (7 nodes, parallel) | + Query decomposition, parallel retrieval workers, cross-agent RRF fusion |

### Key Differentiators

**Basic -> Intermediate (+4 features):**
- Query expansion (3 variants with RRF fusion)
- Hybrid retrieval (semantic + BM25 keyword)
- CrossEncoder reranking (top-k)
- RRF fusion across query variants

**Intermediate -> Advanced (+12 features):**
- Conversational query rewriting
- LLM-based strategy selection (semantic/keyword/hybrid)
- Two-stage reranking (CrossEncoder → LLM-as-judge)
- Retrieval quality gates (8 issue types)
- Answer quality evaluation (8 issue types)
- Adaptive thresholds (65% good retrieval, 50% poor)
- Query rewriting loop (issue-specific feedback, max 3)
- Early strategy switching (off_topic/wrong_domain detection)
- Generation retry loop (adaptive temperature 0.3/0.7/0.5)
- NLI-based hallucination detection
- Refusal detection
- Conversation context preservation (multi-turn)

**Advanced -> Multi-Agent (+3 features):**
- Complexity classification (simple vs complex routing)
- Query decomposition (2-4 sub-queries)
- Parallel worker retrieval with cross-agent RRF merge + LLM coverage selection

### Run Comparison Test

```bash
# Architecture comparison test (~70-85 min, portfolio showcase)
uv run python tests/integration/test_architecture_comparison.py
```

**Results:** See `evaluation/architecture_comparison_report.md` for:
- **F1@5** (Retrieval Quality): Harmonic mean of Precision@5 and Recall@5
- **Groundedness** (Anti-Hallucination): % claims supported by context (NLI-based)
- **Confidence** (Answer Quality): LLM confidence score
- **Delta Analysis**: Shows incremental improvements (Basic -> Intermediate -> Advanced)
- **Feature Justification**: Which features drove each improvement
- **Portfolio Narrative**: Architecture value independent of model quality

**Expected Progression:**
- Basic -> Intermediate: +10-15% improvement (hybrid search, query expansion, reranking)
- Intermediate -> Advanced: +30-50% improvement (NLI, strategy switching, adaptive loops, quality gates)
- Advanced -> Multi-Agent: +5-15% improvement (query decomposition, parallel retrieval, cross-agent fusion)
- Basic -> Multi-Agent: +50-90% overall improvement

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
- Two-stage reranking: CrossEncoder filters to top-10, then LLM-as-judge selects top-4
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

The system uses a 7-node LangGraph workflow with **autonomous decision-making** at every stage. Quality gates and routing functions provide distributed intelligence—each decision point evaluates intermediate results and autonomously determines the next action.

![Graph Architecture](mermaid%20chart.png)

### 7 Nodes
1. **conversational_rewrite**: Makes query self-contained using conversation history
2. **decide_strategy**: Selects optimal retrieval strategy (semantic/keyword/hybrid)
3. **query_expansion**: Generates query variations, optimizes for strategy
4. **retrieve_with_expansion**: RRF fusion + two-stage reranking + quality evaluation
5. **rewrite_and_refine**: Issue-specific query rewriting (8 issue types)
6. **answer_generation**: Structured RAG prompting with adaptive temperature
7. **evaluate_answer**: Consolidated refusal + NLI hallucination + quality assessment

### 2 Decision Points
1. **route_after_retrieval**: Quality >= 0.6? Proceed to generation, else rewrite or switch strategy
2. **route_after_evaluation**: Answer sufficient? Return result, else retry generation (max 3)

### 2 Self-Correction Loops
- **Query Rewriting Loop**: Poor retrieval -> issue-specific feedback -> rewrite -> retry (max 3)
- **Generation Retry Loop**: Quality issues -> unified feedback (hallucination + quality) -> regenerate with adaptive temperature -> retry (max 3)

**Key Principle**: Fix generation problems with generation strategies, not by retrieving more documents. No re-retrieval after generation begins.

**What Makes This Agentic**: The system continuously evaluates its own performance and autonomously decides whether to proceed, retry with modifications, or switch approaches. Every path through the graph is determined by quality metrics at runtime.

## Future Improvements

**Advanced Query Optimization:**
- HyDE (Hypothetical Document Embeddings): Generate hypothetical answer first, then embed and retrieve using answer-document similarity
- Step-back prompting: Generate higher-level conceptual questions alongside specific queries for better multi-hop reasoning
- Adaptive multi-query rewriting: Learn which rewriting strategies (sparse/dense/semantic) work best per query type

**Data & Model Optimization:**
- Fine-tune NLI hallucination detector on RAGTruth dataset
- CRAG three-tier confidence system: Implement full framework with lightweight T5 evaluator and web search fallback (+7-36% accuracy)
- Two-tier hallucination detection: Add verifiability classifier before NLI verification
- Extend document profiling: Add custom analysis features beyond current stratified sampling + regex pre-detection (+15-27 pt accuracy gains)
- Expand golden dataset from 20 to 100 examples using RAGAS TestsetGenerator with human validation
- Benchmark domain-specific embeddings (Specter, SciBERT) against OpenAI for 10-20% retrieval improvement
- Systematically optimize chunk size (256-2048 tokens) and overlap (0-200) for better recall and context sufficiency

**Generation Quality:**
- Chain-of-thought answer generation: Structured reasoning steps (identify facts → cite sources → synthesize → verify)
- Mandatory inline citations: Require citation (Doc #, lines X-Y) for each factual claim (43% hallucination reduction)

**Production & Monitoring:**
- Integrate LangSmith tracing, user feedback collection, and real-time quality dashboards for continuous evaluation
- Context compression: Reduce prompt tokens by 75% while maintaining accuracy (4x faster inference)
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
- **LLMs**: OpenAI GPT-4o-mini/GPT-5-mini/GPT-5.1/GPT-5-nano (configurable via MODEL_TIER)
- **PDF Processing**: PyMuPDF
- **Package Manager**: uv (faster than pip)
- **Reranking**: sentence-transformers (CrossEncoder models)
- **Evaluation**: RAGAS (industry-standard RAG metrics), datasets (RAGAS dependency)
- **Hallucination Detection**: cross-encoder/nli-deberta-v3-base (NLI model)
- **Python**: 3.11+