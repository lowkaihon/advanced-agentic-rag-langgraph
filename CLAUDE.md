# Advanced Agentic RAG using LangGraph

A portfolio project showcasing advanced RAG and LangGraph capabilities through an intelligent, adaptive retrieval pipeline. The system analyzes both corpus characteristics (document types, technical density, content patterns) and query context (intent, conversational history, complexity) to dynamically select optimal retrieval strategies. Built-in quality gates, self-correction loops, and automatic strategy switching ensure retrieved documents meet relevance thresholds. While demonstrated using research papers, the architecture generalizes to diverse document types and use cases, making it a foundation for production-grade RAG systems.

**Architecture**: 7-node StateGraph with distributed intelligence, not linear pipeline
**Pattern**: Dynamic Planning and Execution Agents (graph structure encodes planning logic)
**Framework**: LangChain 1.0 & LangGraph 1.0 (production-ready)
**Requirements**: Python 3.11+

## Agentic Architecture

The agentic RAG system implements the "Dynamic Planning and Execution Agents" pattern. The LangGraph StateGraph itself IS the agent — decision-making is distributed across 2 specialized routing functions rather than centralized in a single LLM orchestrator. Autonomous adaptation through quality-driven conditional routing at every stage.

**No Central Agent Orchestrator**: The StateGraph itself provides autonomous decision-making through:
- **2 routing functions** with specialized logic: retrieval quality assessment (route_after_retrieval), answer evaluation (route_after_evaluation)
- **Conditional edges** that change behavior based on state (quality scores, detected issues, attempt counts)
- **Quality-driven flow**: Each routing point evaluates intermediate results and autonomously decides next action (proceed/rewrite/switch/retry)

**"Tools" in Broader Context**: In agentic RAG, "tools" = retrieval strategies (semantic/keyword/hybrid), processing techniques (reranking, HHEM), not just LLM function-calling. The "intelligence" is deciding which to use when based on content analysis.

## Key Design Patterns

This system demonstrates advanced RAG patterns that remain stable across implementation changes:

**Quality-Driven Architecture**
- Retrieval quality scoring → conditional routing → retry or proceed
- Answer evaluation → strategy switching → improved results
- Adaptive thresholds based on retrieval performance

**LangGraph Workflow Pattern** (Dynamic Planning and Execution Agents)
- 7 nodes with conditional edges (not linear pipeline)
- Integrated metadata analysis within retrieval evaluation
- **State management pattern**:
  - `add_messages` for messages (conversation history), `operator.add` for: retrieved_docs
  - Direct replacement (no operator.add) for: query_expansions (regenerated fresh per iteration to ensure expansion-query alignment)
- Quality gates determine routing: retrieval quality -> answer generation, answer quality -> retry/end
- Single-use nodes: conversational_rewrite and decide_strategy used only on initial flow; retry paths skip them
- Key principle: Fix generation problems with generation strategies, not by retrieving more documents

**Self-Correction Loops**
- Retrieval correction loop: poor retrieval quality (score <0.6) -> single correction cycle (max 2 attempts, research-backed CRAG/Self-RAG principle showing diminishing returns after first cycle)
  - Path A (off_topic/wrong_domain): Strategy switch (precision correction, pure strategy change)
  - Path B (other issues): Keyword injection for query enrichment (coverage correction)
- Generation retry loop: Consolidated evaluation in single node (refusal detection + HHEM hallucination + quality assessment) -> unified feedback -> regenerate with low temperature (0.3) -> retry (max 3 attempts)
- No re-retrieval after generation: Generation problems fixed with generation strategies, not by retrieving more documents (CRAG research principle)

**Multi-Strategy Retrieval**
- Three approaches: semantic (vector), keyword (BM25), hybrid (combined)
- Strategy selection based on corpus characteristics + query analysis
- RAG-Fusion pattern: Strategy-agnostic expansions → select best strategy → use for ALL variants (differs from PreQRAG parallel multi-strategy approach)
- RRF-based multi-query fusion: Reciprocal Rank Fusion aggregates rankings across query variants BEFORE reranking (3-5% MRR improvement)
- Two-stage reranking applied to RRF-fused results for final relevance scoring

**Evaluation & Quality Assurance**
- Two-stage reranking (applied after RRF fusion): CrossEncoder (stage 1, top-10) → LLM-as-judge (stage 2, top-4)
- HHEM-based hallucination detection: Claim decomposition → vectara/hallucination_evaluation_model (HHEM-2.1-Open) verification → hallucination feedback lists specific unsupported claims for targeted regeneration
- Comprehensive metrics: F1@K, Precision@K, Recall@K, MRR, nDCG
- Golden datasets: 30 validated examples across 2 datasets (Standard: 20, Hard: 10) with graded relevance (0-3 scale)
- Retrieval quality evaluation: Issue-specific detection (missing_key_info, partial_coverage, incomplete_context, wrong_domain, off_topic)
- Answer quality evaluation (vRAG-Eval framework): Relevance, Completeness, Accuracy scoring with 5 issue types (incomplete_synthesis, lacks_specificity, missing_details, partial_answer, wrong_focus) and adaptive thresholds (65% for good retrieval, 50% for poor)

**Intelligent Adaptation**
- Document profiling: Stratified sampling (5K tokens), regex signal pre-detection, +15-27 pt accuracy gains
- Query analysis: LLM-based intent classification and expansion decisions
- Strategy selector: pure LLM classification (domain-agnostic, handles all edge cases)
- Conversational rewriting: injects context from conversation history
- Early strategy switching (route_after_retrieval): Detects obvious strategy mismatches (off_topic, wrong_domain) and switches immediately before wasting retrieval attempts, regenerates query expansions for new strategy
- Tiered model architecture: Configurable quality/cost trade-offs through three model tiers (budget: GPT-4o-mini only, balanced: GPT-4o-mini + GPT-5-mini, premium: GPT-5.1 + GPT-5-mini + GPT-5-nano). Demonstrates architectural value independent of model quality—budget tier showcases graph intelligence (adaptive retrieval, self-correction, multi-stage processing) while balanced/premium tiers show incremental gains from model upgrades. Architecture provides baseline, models add polish.
- LLM allocation strategy: 13 total instances (11 tier-aware for runtime operations, 2 hardcoded for quality-critical operations). Golden dataset evaluation uses gpt-5-mini (portfolio quality showcase, 85-90% agreement with human judges). Document profiling uses gpt-4o-mini (tier-independence principle, already optimal, one-time ingestion cost). Design principle: Evaluation and ingestion use best/appropriate models regardless of tier; runtime components scale with tier for cost-quality flexibility.
- Hallucination-aware answer generation: Structured RAG prompting with XML markup, quality-aware instructions, unified retry_feedback (hallucination + quality issues) prepended on regeneration
- Query expansion regeneration: Expansions regenerated when strategy changes OR query rewritten, ensuring retrieval pool always matches current query context

**State Management**
- Uses TypedDict (best performance) not Pydantic
- MemorySaver checkpointer for multi-turn conversations
- Tracks: queries, documents, quality scores, retrieval_attempts, generation_attempts, retry_feedback, conversation history

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
cp .env.example .env                 # Create environment file (add OPENAI_API_KEY + MODEL_TIER)
```

### Python Cache Management

**For comprehensive guidance, see:** `references/Python '__pycache__' Best Practices.md`

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
# Architecture comparison test (portfolio showcase)
PYTHONIOENCODING=utf-8:replace uv run python tests/integration/test_architecture_comparison.py

# Options:
#   --dataset standard|hard   Dataset selection
#   --tiers basic intermediate advanced multi_agent   Tier selection (default: all)
#   --output-dir PATH             Results directory (default: evaluation/)
```

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

**Main test:** `tests/integration/test_architecture_comparison.py`
- Comprehensive 4-tier architecture comparison (Basic/Intermediate/Advanced/Multi-Agent)
- Evaluates on golden dataset with graded relevance

**Temporary debugging:** Root directory with `debug_*.py` prefix (delete after use)
- One-off exploration, no formal assertions

**Rule:** Main test in tests/integration/, temporary scripts in root with debug_ prefix

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

**File paths**: Use same package-based resolution for project files:
```python
import advanced_agentic_rag_langgraph
PROJECT_ROOT = Path(advanced_agentic_rag_langgraph.__file__).parent.parent.parent
```

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

### LangGraph Best Practices

**State Reducers** - add_messages vs operator.add for state management
Use `add_messages` for message history (idempotent, deduplicates by message ID), `operator.add` for generic list accumulation. This system uses selective accumulation: `add_messages` for messages, `operator.add` for docs/history, direct replacement for query_expansions (ensures expansion-query alignment).
https://docs.langchain.com/oss/python/langgraph/use-graph-api#messagesstate
Code Pattern:
```python
# state.py - Selective accumulation
class AdvancedRAGState(TypedDict):
    # Always set (no Optional)
    user_question: str
    baseline_query: str

    # Conditionally set (Optional)
    active_query: Optional[str]
    query_expansions: Optional[list[str]]  # Regenerated per iteration (not accumulated)

    # Accumulated with reducers
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history (LangGraph best practice)
    retrieved_docs: Annotated[list[str], operator.add]
```

**Router Function Purity** - Deterministic conditional edges without side effects
Router functions should be pure (deterministic output based only on state, no side effects). Enables reliable checkpointing, testability, and observability. Avoid: random(), datetime.now(), DB queries, global state mutations.
https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching
Code Pattern:
```python
# graph.py - Pure router with three-way branching
def route_after_retrieval(state: AdvancedRAGState) -> Literal["answer_generation", "rewrite_and_refine", "query_expansion"]:
    quality = state.get("retrieval_quality_score", 0)
    attempts = state.get("retrieval_attempts", 0)
    issues = state.get("retrieval_quality_issues", [])

    if quality >= 0.6:
        return "answer_generation"
    if attempts >= 2:
        return "answer_generation"  # Max attempts
    if ("off_topic" in issues or "wrong_domain" in issues) and attempts == 1:
        return "query_expansion"  # Early strategy switch
    return "rewrite_and_refine"  # Semantic rewrite
```

**Error Handling** - Quality gates and retry policies over exceptions
This system uses quality gates (quality scores + retry limits) over try/catch. LangGraph retries transient errors (network, rate limits, 5xx) automatically but NOT programming errors (ValueError, TypeError). Store LLM-recoverable errors in state for feedback loops.
https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph#handle-errors-appropriately
Code Pattern:
```python
# Quality-driven routing (not exception handling)
if state["retrieval_quality_score"] < 0.6 and state["retrieval_attempts"] < 3:
    return "rewrite"  # Poor retrieval -> retry
return "generate"
```

**Async & Parallel Execution** - Supersteps and concurrent branches
Use parallel edges from START or Send API for fan-out. LangGraph executes nodes in supersteps (transactional boundaries): all parallel branches succeed or all fail. Successful results ARE checkpointed even if superstep fails, so retries don't repeat work. Current system: synchronous (invoke() not ainvoke()).
https://docs.langchain.com/oss/python/langgraph/workflows-agents#parallelization
Code Pattern:
```python
# Parallel branches from START
builder.add_edge(START, "node_a")
builder.add_edge(START, "node_b")  # Concurrent with node_a
```

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
