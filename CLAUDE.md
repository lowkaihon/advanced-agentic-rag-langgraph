# Advanced Agentic RAG using LangGraph

A portfolio project showcasing advanced RAG and LangGraph capabilities through an intelligent, adaptive retrieval pipeline. The system analyzes both corpus characteristics (document types, technical density, content patterns) and query context (intent, conversational history, complexity) to dynamically select optimal retrieval strategies. Built-in quality gates, self-correction loops, and automatic strategy switching ensure retrieved documents meet relevance thresholds. While demonstrated using research papers, the architecture generalizes to diverse document types and use cases, making it a foundation for production-grade RAG systems.

**Framework Status**: LangChain 1.0 & LangGraph 1.0 (production-ready, stability commitment)
**Requirements**: Python 3.11+
**Migration Guide**: https://docs.langchain.com/oss/python/migrate/langchain-v1

## Key Design Patterns

This system demonstrates advanced RAG patterns that remain stable across implementation changes:

**Quality-Driven Architecture**
- Retrieval quality scoring → conditional routing → retry or proceed
- Answer evaluation → strategy switching → improved results
- Adaptive thresholds based on retrieval performance

**LangGraph Workflow Pattern**
- 9 nodes with conditional edges (not linear pipeline)
- Metadata analysis node examines retrieved documents for strategy alignment
- State accumulation using TypedDict with `Annotated[list, operator.add]`
- Quality gates determine routing: retrieval quality → answer generation, answer quality → retry/end

**Self-Correction Loops**
- Query rewriting loop: poor retrieval quality → rewrite query → retry (max 2 rewrites)
- Strategy switching loop: insufficient answer → switch strategy → retry (max 3 attempts)
- Metadata-driven switching: uses document preferences when detected, fallback: hybrid → semantic → keyword

**Multi-Strategy Retrieval**
- Three approaches: semantic (vector), keyword (BM25), hybrid (combined)
- Strategy selection based on corpus characteristics + query analysis
- LLM-as-Judge reranking for relevance scoring

**Evaluation & Quality Assurance**
- Two-stage reranking: CrossEncoder (stage 1, top-10) → LLM-as-judge (stage 2, top-4)
- NLI-based hallucination detection: Claim decomposition → cross-encoder/nli-deberta-v3-base verification
- Comprehensive metrics: Recall@K, Precision@K, F1@K, nDCG, MRR, Hit Rate
- RAGAS integration: Faithfulness, Context Recall, Context Precision, Answer Relevancy
- Golden dataset: 20 validated examples with graded relevance (0-3 scale)
- Context sufficiency: Pre-generation completeness validation
- Answer quality: Semantic similarity, factual accuracy, completeness scoring

**Intelligent Adaptation**
- Document profiling: analyzes technical density, type, domain
- Query analysis: LLM-based intent classification and expansion decisions
- Strategy selector: pure LLM classification (domain-agnostic, handles all edge cases)
- Conversational rewriting: injects context from conversation history

**Metadata-Driven Adaptation**
- Post-retrieval metadata analysis: examines retrieved document characteristics
- Strategy mismatch detection: identifies when docs prefer different strategy (>60% threshold)
- Intelligent refinement: switches to document-preferred strategy with logged reasoning
- Quality issue tracking: detects low confidence, complexity mismatches, domain misalignment

**State Management**
- Uses TypedDict (best performance) not Pydantic
- MemorySaver checkpointer for multi-turn conversations
- Tracks: queries, documents, quality scores, attempts, conversation history

## Development Commands

### Important: No Emojis or Unicode Characters
Never use emojis or Unicode special characters (like ✓, ✅, ✗, ❌, →, etc.) in code, print statements, or comments. Windows console doesn't support these characters and will cause Unicode encoding errors. Use ASCII-only characters.

### Important: Bash Command Syntax
This project runs in a Unix bash environment (even on Windows). **Always use Unix commands:**
- Use: `mv` (NOT `move`)
- Use: `rm -rf` (NOT `del`)
- Use: `cp` (NOT `copy`)
- Use: `ls` (NOT `dir`)

### Package Manager
This project uses **uv** for dependency management. Do not use pip or conda.

### Setup
```bash
uv sync                              # Install/sync all dependencies
cp .env.example .env                 # Create environment file (add your OPENAI_API_KEY)
```

### Python Cache Management

Centralize bytecode cache to avoid `__pycache__` clutter (optional):
```bash
export PYTHONPYCACHEPREFIX="$HOME/.cache/cpython/"  # Unix/Mac/Git Bash
$env:PYTHONPYCACHEPREFIX = "$env:USERPROFILE\.cache\cpython"  # Windows PowerShell
```

Clear cache if stale code issues after refactoring:
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null  # Unix/Mac/Git Bash
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force  # Windows PS
```

### Testing
```bash
# Fast tests (~1-2 min)
uv run python tests/integration/test_pdf_pipeline.py
uv run python tests/integration/test_adaptive_retrieval.py

# Comprehensive evaluation (~10-15 min)
uv run python tests/integration/test_golden_dataset_evaluation.py
```
See `tests/CLAUDE.md` for all 10 tests, selection matrix, and detailed documentation.

### Development
```bash
uv run jupyter notebook              # Start Jupyter for notebook examples
uv run python -m pytest              # Run test suite (when tests are added)
```

### Dependencies
```bash
uv add <package>                     # Add a new dependency
uv remove <package>                  # Remove a dependency
uv sync                              # Sync after updating pyproject.toml
```

### Common Tasks
```bash
# Load PDFs for retrieval
uv run python -c "from advanced_agentic_rag_langgraph.core import setup_retriever; setup_retriever()"  # All PDFs
uv run python -c "from advanced_agentic_rag_langgraph.core import setup_retriever; setup_retriever(pdfs='filename.pdf')"  # Specific PDF
```

### Test File Organization

**Permanent tests:** `tests/integration/test_<name>.py`
- Multiple test cases, meant to run repeatedly
- Examples: test_nli_hallucination_detector.py, test_ragas_evaluation.py, test_context_sufficiency.py

**Temporary debugging:** Root directory with `debug_*.py` prefix (delete after use)
- One-off exploration, no formal assertions
- Example: debug_nli.py (deleted after understanding model output)

**Rule:** Permanent → tests/integration/, Temporary → root with debug_ prefix

### Import Best Practices

**Package name**: `advanced_agentic_rag_langgraph` (after `uv sync`)

**Correct imports**:
```python
from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph
```

**Wrong**: `from src.core import ...` (ModuleNotFoundError)
**Wrong**: Using PYTHONPATH (unnecessary with editable install)

**Run with**: `uv run python <file>` or activate venv first
See `tests/CLAUDE.md` for detailed explanation and common issues.

## Quick Reference by Task

**Building Retrieval Pipelines**
- Start: [Retrieval Guide](#rag--retrieval)
- Advanced: [Build a Custom RAG Agent](#rag--retrieval)

**Implementing Quality Checks & Evaluation**
- Retrieval quality: [RAG Evaluation Guide](#rag--retrieval), [Document Grading Pattern](#reranking--relevance-scoring)
- Answer assessment: [Application-Specific Evaluation](#rag--retrieval)

**Query Optimization & Rewriting**
- Query rewriting: [RePhraseQuery](#query-optimization--enhancement), [Custom RAG Agent](#rag--retrieval)

**Adding Reranking**
- Implementations: [Cohere Reranker](#reranking--relevance-scoring), [Pinecone Rerank](#reranking--relevance-scoring)
- Custom scoring: [Document Relevance Grading Pattern](#reranking--relevance-scoring)

**Hybrid Search (Semantic + Keyword)**
- Keyword component: [BM25 Retriever](#hybrid--multi-strategy-retrieval)
- Semantic component: [Vector Stores](#tools--integrations)

**Discovering Retrieval Tools**
- Catalog: [Retrievers Overview](#tools--integrations), [Vector Stores](#tools--integrations)

**Building Self-Correction Loops**
- Patterns: [Custom RAG Agent](#rag--retrieval) (quality gates, strategy switching)
- Graph mechanics: [Graph API Usage](#langgraph-core), [Persistence](#langgraph-core)

**Agent Construction**
- LangGraph: [Graph API](#langgraph-core), [Graph API Usage](#langgraph-core)
- Optimization: [Context Engineering](#agents)

**Deployment & Monitoring**
- Testing: [LangSmith Studio](#langsmith--deployment)
- Production: [Streaming API](#langsmith--deployment)

---

## Resources

### RAG & Retrieval

**RAG Evaluation Guide** [Tutorial] - LangSmith-based evaluation with metrics and test sets
https://docs.langchain.com/langsmith/evaluate-rag-tutorial
Covers: Retrieval relevance scoring, answer correctness, groundedness evaluation
Use for: Offline/batch evaluation pipelines with comprehensive metrics and test datasets

**Retrieval Guide** [Reference] - RAG architecture patterns and component overview
https://docs.langchain.com/oss/python/langchain/retrieval
Covers: 2-step RAG, agentic RAG, hybrid RAG architectures, document loaders, text splitting
Use for: Understanding different RAG architectures and choosing the right design pattern

**Build a Custom RAG Agent** [Tutorial, Advanced] - Multi-step RAG with LangGraph
https://docs.langchain.com/oss/python/langgraph/agentic-rag
Covers: Query rewriting, retrieval quality grading, self-correction loops, conditional routing
Includes: Document relevance scoring, answer quality gates, strategy switching
Use for: Building advanced RAG with self-correction and quality checks
Code Pattern:
```python
# Grade documents for relevance (see detailed pattern in Reranking section)
class RelevanceGrade(TypedDict):
    explanation: str
    relevant: bool

grader = model.with_structured_output(RelevanceGrade)
grade = grader.invoke([
    {"role": "system", "content": "Grade if docs contain keywords/meaning related to question"},
    {"role": "user", "content": f"FACTS: {docs}\nQUESTION: {question}"}
])
# Route based on relevance
return "generate" if grade["relevant"] else "rewrite"
```

**Application-Specific Evaluation Approaches** [Guide] - Custom evaluation with LLM-as-judge
https://docs.langchain.com/langsmith/evaluation-approaches
Covers: Quality scoring, confidence metrics, answer assessment, custom evaluators
Use for: Runtime quality gates and evaluation within agent self-correction loops

---

### Query Optimization & Enhancement

**RePhraseQuery Retriever** [Integration] - LLM-based query preprocessing and rewriting
https://docs.langchain.com/oss/python/integrations/retrievers/re_phrase
Covers: Query reformulation, irrelevant info filtering, custom transformation logic
Use for: Improving unclear queries before retrieval
Code Pattern:
```python
# Rewrite query to improve retrieval quality
rewritePrompt = "Reason about semantic intent and formulate improved question"
response = rewritePrompt.pipe(model).invoke({"question": original_query})
```

---

### Reranking & Relevance Scoring

**Cohere Reranker** [Integration] - Production-ready reranking with contextual compression
https://docs.langchain.com/oss/python/integrations/retrievers/cohere-reranker
Covers: ContextualCompressionRetriever setup, Cohere Rerank API, complete RAG pipeline
Use for: Improving relevance of initial retrieval results with external reranking service (best for production scale)
Implementation: Use ContextualCompressionRetriever with CohereRerank compressor

**Pinecone Rerank** [Integration] - Relevance scoring with custom field targeting
https://docs.langchain.com/oss/python/integrations/retrievers/pinecone_rerank
Covers: PineconeRerank class, top_n filtering, rank_fields customization, score normalization
Use for: Reranking with Pinecone infrastructure or custom relevance criteria

**Document Relevance Grading** [Pattern] - LLM-as-judge for retrieval quality assessment
Source: [Custom RAG Agent](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
Use for: Custom relevance scoring without external services (best for prototyping or cost optimization)
Code Pattern:
```python
# Grade retrieved documents for relevance (binary yes/no)
class RelevanceGrade(TypedDict):
    explanation: str  # Reasoning for the score
    relevant: bool    # True if relevant, False if not

# Create grader with structured output
grader = model.with_structured_output(RelevanceGrade)
grade = grader.invoke([
    {"role": "system", "content": "Grade if docs contain keywords/meaning related to question"},
    {"role": "user", "content": f"FACTS: {docs}\nQUESTION: {question}"}
])
# Use result to route workflow
return grade["relevant"]
```

---

### Hybrid & Multi-Strategy Retrieval

**BM25 Retriever** [Integration] - Keyword-based sparse search using BM25 algorithm
https://docs.langchain.com/oss/python/integrations/retrievers/bm25
Covers: Keyword ranking, preprocessing functions, sparse search alternative to vector search
Use for: The keyword/sparse component of hybrid retrieval systems (exact term matching, proper nouns, technical terms)

---

### LangGraph Core

**Graph API Guide** [Reference] - StateGraph and MessageGraph API documentation
https://docs.langchain.com/oss/python/langgraph/graph-api
Covers: Node/edge definitions, state schemas, graph compilation
Use for: API reference and core concepts when building graphs

**Graph API Usage Guide** [Tutorial - Patterns] - Practical patterns for graph construction
https://docs.langchain.com/oss/python/langgraph/use-graph-api
Covers: Conditional routing, branching, Send API for dynamic fan-out
Includes: Multi-path graphs, decision nodes, routing functions
Use for: Implementing conditional logic and complex routing in agent workflows
Code Pattern:
```python
# Conditional routing based on function output
builder.add_conditional_edges(
    source_node,
    router_function,  # Function that returns key from mapping
    {"option1": "node1", "option2": "node2"}  # Map routing keys to target nodes
)
```

**Subgraphs Guide** [Tutorial] - Composing graphs and sharing state
https://docs.langchain.com/oss/python/langgraph/use-subgraphs
Covers: Invoking child graphs, adding subgraphs as nodes, state sharing patterns
Use for: Building modular, reusable graph components

**Streaming Guide** [Tutorial] - Stream graph execution in real-time
https://docs.langchain.com/oss/python/langgraph/streaming
Covers: Stream modes (values, updates, messages), real-time token streaming
Use for: Building responsive user interfaces with live updates

**Persistence & Checkpointing** [Tutorial] - Save and restore graph state
https://docs.langchain.com/oss/python/langgraph/persistence
Covers: Checkpointers (memory, SQLite, Postgres), thread management, conversation memory
Use for: Multi-turn conversations, resumable workflows
Code Pattern:
```python
# State schema with message history accumulation
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Accumulates messages across turns
```

**State Management Best Practices** [Best Practice]
Performance hierarchy for state definitions:
- **TypedDict** (best performance) - Use for most cases
- **dataclass** (for defaults) - Use when you need default values
- **Pydantic BaseModel** (validation only) - Use only when recursive validation needed
See [Graph API Guide](#langgraph-core) for state schema patterns

**Durable Execution** [Concept] - Fault tolerance and execution guarantees
https://docs.langchain.com/oss/python/langgraph/durable-execution
Covers: Execution modes, task persistence, replay mechanisms, failure recovery
Use for: Understanding execution guarantees and building fault-tolerant production systems (complements Persistence for state storage)

**Interrupts & Human-in-the-Loop** [Tutorial] - Pause execution for human input
https://docs.langchain.com/oss/python/langgraph/interrupts
Covers: Breakpoints, approval flows, dynamic interrupts, state updates during pauses
Use for: Human approval workflows, interactive debugging

---

### Agents

**Context Engineering in Agents** [Guide] - Optimize agent context and prompts
https://docs.langchain.com/oss/python/langchain/context-engineering
Covers: Context provision strategies, prompt engineering, tool selection optimization
Use for: Improving agent decision-making and reducing token usage

---

### Tools & Integrations

**Tool Calling Guide** [Tutorial] - Bind tools to LLMs for agent capabilities
https://docs.langchain.com/oss/python/langchain/tools
Covers: Tool definition with @tool decorator, tool binding, structured outputs
Use for: Creating custom tools for agents

**Retrievers Overview** [Index] - Comprehensive catalog of retriever integrations
https://docs.langchain.com/oss/python/integrations/retrievers/index
Includes: Query optimizers (RePhraseQuery, LLMLingua), rerankers (Cohere, Pinecone), hybrid search, BM25, specialized retrievers
Use for: Discovering available retrieval tools and patterns
**NOTE**: Some retriever utilities moved to `langchain-classic` package (install separately if needed)

**Vector Stores Overview** [Index] - Comprehensive listing of vector store integrations
https://docs.langchain.com/oss/python/integrations/vectorstores/index
Includes: FAISS, Chroma, Pinecone, Weaviate, Qdrant, and many more
Use for: The semantic/dense component of hybrid retrieval systems (semantic similarity, conceptual matching)

---

### LangSmith & Deployment

**LangSmith Studio Overview** [Product] - Agent IDE for development and testing
https://docs.langchain.com/langsmith/studio
Covers: Interactive testing, debugging, trace inspection
Use for: Development workflow and agent iteration

**How to Use Studio** [Tutorial] - Studio features and workflows
https://docs.langchain.com/langsmith/use-studio
Covers: Creating projects, running agents, viewing traces, evaluation
Use for: Getting started with LangSmith Studio

**Streaming API** [Reference] - Real-time streaming for production deployments
https://docs.langchain.com/langsmith/streaming
Covers: Server-sent events, streaming modes, production patterns
Use for: Deploying agents with real-time response streaming

---

## Troubleshooting

**Retrieval returns irrelevant documents**
- Try [Query Rewriting](#query-optimization--enhancement) to improve query clarity
- Add [Reranking](#reranking--relevance-scoring) to filter results by relevance
- Adjust similarity threshold or top_k parameter

**Agent loops infinitely**
- Implement max iteration limits in graph
- Add quality gates to exit when answer is sufficient
- Check [Graph API Usage](#langgraph-core) for conditional routing patterns

**Slow retrieval performance**
- Use [Hybrid Search](#hybrid--multi-strategy-retrieval) for better precision
- Implement result caching for common queries
- Optimize vector store configuration

**Inconsistent answer quality**
- Add [Evaluation](#rag--retrieval) to measure quality metrics
- Implement [Self-Correction Loops](#rag--retrieval) with quality thresholds
- Review prompt engineering and context limits

**State management issues**
- Review [Persistence & Checkpointing](#langgraph-core) patterns
- Verify state schema definitions match usage
- Check thread management in multi-turn conversations
