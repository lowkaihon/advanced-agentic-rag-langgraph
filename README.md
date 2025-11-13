# Advanced Agentic RAG using LangGraph

This Advanced Agentic RAG uses LangGraph to implement features including multi-strategy retrieval (semantic + keyword), LLM-based reranking, intelligent query expansion and rewriting, automatic strategy switching, and self-correcting agent loops with quality evaluation.

## Features

### 1. Document & Corpus Profiling
- Analyzes technical density (0.0-1.0), document types, domain tags
- Computes aggregate corpus statistics
- Informs intelligent strategy selection

### 2. Conversational Query Rewriting
- Makes queries self-contained using conversation history
- Resolves pronouns and context references
- Enables natural multi-turn conversations

### 3. Query Optimization
- **Expansion**: Generates 3 query variations (technical, simple, different aspect)
- **Rewriting**: Rewrites unclear queries when retrieval quality < 60%
- Multi-query retrieval with deduplication

### 4. Intelligent Strategy Selection
- 10 heuristic rules analyze query features + corpus characteristics
- LLM fallback for ambiguous cases (confidence < 70%)
- Selects semantic, keyword, or hybrid with reasoning

### 5. Multi-Strategy Retrieval
- **Semantic**: FAISS vector search (meaning-based)
- **Keyword**: BM25 lexical search (exact term matching)
- **Hybrid**: Combines both with deduplication
- Dynamic strategy selection per query

### 6. LLM-as-Judge Reranking
- Scores each document 0-100 for relevance
- Returns top 4 most relevant documents
- Temperature 0 for consistency

### 7. Quality Gates
- **Retrieval Quality**: LLM scores if docs answer query (threshold: 60%)
- **Answer Evaluation**: Checks relevance, completeness, accuracy
- **Adaptive Thresholds**: Lowers standards when retrieval is poor

### 8. Self-Correction Loops
- **Query Rewriting Loop**: Poor retrieval → rewrite query → retry (max 2 rewrites)
- **Strategy Switching Loop**: Insufficient answer → switch strategy → retry (max 3 attempts)
- Progressive strategy order: hybrid → semantic → keyword

### 9. Multi-turn Conversations
- State persistence with MemorySaver checkpointer
- Tracks conversation history across queries
- Thread-based conversation management

### 10. Real-time Streaming
- Streams execution progress in real-time
- Shows node transitions and quality scores
- Verbose mode for detailed debugging

## Architecture Overview

### System Components

**1. Document Profiling** (`src/preprocessing/document_profiling.py`)
- Analyzes corpus characteristics: technical density (0.0-1.0), document type, domain tags
- Profiles entire corpus to compute aggregate statistics
- Informs retrieval strategy selection based on content patterns

**2. Query Analysis & Optimization** (`src/preprocessing/query_processing.py`, `src/retrieval/query_optimization.py`)
- **Conversational Rewriting**: Makes queries self-contained using conversation history
- **Query Expansion**: Generates 3 variations (technical, simple, different aspect)
- **Intent Classification**: factual, conceptual, comparative, procedural
- **Complexity Assessment**: simple, moderate, complex

**3. Intelligent Strategy Selection** (`src/retrieval/strategy_selection.py`)
- 10 heuristic rules analyze query features + corpus characteristics
- LLM fallback for ambiguous cases (when confidence < 0.7)
- Selects semantic/keyword/hybrid with confidence score + reasoning

**4. Multi-Strategy Retrieval** (`src/retrieval/retrievers.py`, `src/retrieval/reranking.py`)
- **Semantic**: FAISS vector search for meaning-based retrieval
- **Keyword**: BM25 lexical search for exact term matching
- **Hybrid**: Combines both approaches with deduplication
- **Reranking**: LLM-as-Judge scores each document 0-100 for relevance

**5. LangGraph Orchestration** (`src/orchestration/graph.py`, `src/orchestration/nodes.py`)
- 7 nodes with conditional routing based on quality scores
- Quality gates at retrieval and answer generation stages
- Self-correction loops for query rewriting and strategy switching
- Streams execution progress in real-time

**6. State Management** (`src/core/state.py`)
- TypedDict schema (AdvancedRAGState) for performance
- MemorySaver checkpointer for conversation persistence
- Tracks: queries, documents, quality scores, attempts, conversation history

### Quick Start

```bash
# 1. Install dependencies (uses uv, not pip)
uv sync

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 3. Run comprehensive tests
uv run python test_pdf_pipeline.py

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
Heuristic rules fire:
- Conceptual questions → semantic (+0.35)
- Short queries (<5 words) → semantic (+0.2)

Selected: SEMANTIC (confidence: 0.85)
Reasoning: "Conceptual query with simple phrasing"
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
- Deduplicate results → 8 documents
- LLM reranks by relevance → top 4 documents
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

### Extending the System

**Add a New Retrieval Strategy**:
1. Implement retriever in `src/retrieval/retrievers.py`
2. Add strategy case in `HybridRetriever.retrieve()` method
3. Update `StrategySelector` heuristics in `src/retrieval/strategy_selection.py`

**Customize Quality Thresholds**:
```python
# In src/orchestration/graph.py
def route_after_retrieval(state):
    quality = state["retrieval_quality_score"]
    threshold = 0.6  # Adjust this (default: 0.6)

    if quality > threshold or state["retrieval_attempts"] >= 2:
        return "answer_generation_with_quality"
    return "rewrite_and_refine"
```

**Add New Document Profiling Features**:
```python
# In src/preprocessing/document_profiling.py
class DocumentProfiler:
    def profile_document(self, doc_text: str):
        # Add your custom analysis here
        custom_feature = self._analyze_custom_feature(doc_text)

        profile["custom_feature"] = custom_feature
        return profile
```

**Integrate External Reranking** (Cohere, Pinecone):
```python
# Replace LLMReranker in src/retrieval/reranking.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank(cohere_api_key="...", top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

### Project Status

**Production-Ready Features**:
- ✅ LangChain 1.0 & LangGraph 1.0 (stable APIs with stability commitment)
- ✅ 10 research papers in demo corpus (transformers, diffusion, RAG, vision)
- ✅ Comprehensive test suite (4 tests covering all components)
- ✅ Multi-turn conversation support with state persistence
- ✅ Real-time streaming execution with progress tracking
- ✅ Quality-driven self-correction loops

**Potential Enhancements**:
- [ ] **Historical Performance Tracking**: Learn which strategies work best for query types over time
- [ ] **External Reranking Models**: Integrate Cohere Rerank or Pinecone Rerank for production scale
- [ ] **Query Result Caching**: Cache common queries to reduce latency and costs
- [ ] **Multi-Document Fusion**: Cross-encoder reranking for better relevance
- [ ] **Automated RAG Evaluation**: Metrics for retrieval accuracy, answer correctness, groundedness
- [ ] **User Feedback Loop**: Learn from user ratings to improve strategy selection
- [ ] **Document Clustering**: Group similar documents for faster retrieval
- [ ] **Hybrid Vector Stores**: Support for multiple vector databases (Pinecone, Weaviate, Qdrant)

### Technology Stack

- **LLM Framework**: LangChain 1.0 (production-ready)
- **Orchestration**: LangGraph 1.0 (state-based workflows)
- **Vector Store**: FAISS (semantic search)
- **Lexical Search**: BM25 (keyword matching)
- **LLM**: OpenAI GPT-4o-mini (strategy selection, reranking, generation)
- **PDF Processing**: PyMuPDF
- **Package Manager**: uv (faster than pip)
- **Python**: 3.10+

## Complete Flow

The system uses a 7-node LangGraph workflow with conditional routing and self-correction loops:

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
│ • 10 heuristic rules + LLM fallback                            │
│ • Selects: SEMANTIC, KEYWORD, or HYBRID                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 4: Retrieval with Expansion                               │
│ • Retrieves using selected strategy with all query variations  │
│ • Semantic: FAISS vector search                                │
│ • Keyword: BM25 lexical search                                 │
│ • Hybrid: combines both                                        │
│ • Deduplicates results                                         │
│ • LLM reranks by relevance (0-100 scores)                      │
│ • LLM evaluates retrieval quality (0-100 score)                │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌────────┴────────┐
                    │ Quality Gate #1  │
                    │ Quality ≥ 60%?   │
                    │ OR attempts ≥ 2? │
                    └────┬────────┬────┘
                   YES   │        │   NO
                         ↓        ↓
              ┌──────────┘        └──────────────┐
              │                                   │
              │            ┌──────────────────────┴──────────────────┐
              │            │ Node 5: Rewrite and Refine              │
              │            │ • Rewrites query for clarity            │
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
│ • Adjusts system prompt based on retrieval quality             │
│ • High quality: confident answer                               │
│ • Low quality: notes gaps and uncertainty                      │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node 7: Evaluate Answer                                         │
│ • Checks: relevance, completeness, accuracy                    │
│ • Adaptive threshold (lower if retrieval was poor)             │
│ • Computes confidence score                                    │
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
                         ↓                          ┌──────────┴──────────────┐
              ┌──────────────────────┐              │ Switch Strategy:        │
              │ END: Return Answer   │              │ hybrid → semantic       │
              │ • Final answer       │              │ semantic → keyword      │
              │ • Confidence score   │              │ keyword → give up       │
              │ • Strategy used      │              └──────────┬──────────────┘
              │ • Attempts made      │                         │
              └──────────────────────┘                         └─────────┐
                                                                         ↓
                                                               (back to Node 4)

Self-Correction Loops:
• Loop 1 (Query Rewriting): Quality < 60% AND attempts < 2 → rewrite query → retry
• Loop 2 (Strategy Switching): Answer insufficient AND attempts < 3 → switch strategy → retry
```

**Key Points**:
- Not a linear pipeline - uses conditional routing based on quality scores
- Two self-correction loops ensure high-quality results
- Maximum 2 query rewrites, maximum 3 total retrieval attempts
- Quality thresholds: 60% for retrieval, adaptive for answers
- Strategy switching follows progression: hybrid → semantic → keyword