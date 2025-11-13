# Advanced Agentic RAG using LangGraph

This Advanced Agentic RAG uses LangGraph to implement features including multi-strategy retrieval (semantic + keyword), LLM-based reranking, intelligent query expansion and rewriting, automatic strategy switching, and self-correcting agent loops with quality evaluation.

## Quick Reference by Task

**Building Retrieval Pipelines**
- Start: [Retrieval Guide](#rag--retrieval), [Semantic Search Tutorial](#rag--retrieval)
- With tools: [Build a RAG Agent](#rag--retrieval)

**Implementing Quality Checks & Evaluation**
- Retrieval quality: [RAG Evaluation Guide](#rag--retrieval), [Document Grading Pattern](#reranking--relevance-scoring)
- Answer assessment: [Application-Specific Evaluation](#rag--retrieval)

**Query Optimization & Rewriting**
- Query rewriting: [RePhraseQuery](#query-optimization--enhancement), [Custom RAG Agent](#rag--retrieval)
- Discovery: [Retrievers Overview](#tools--integrations)

**Adding Reranking**
- Implementations: [Cohere Reranker](#reranking--relevance-scoring), [Pinecone Rerank](#reranking--relevance-scoring)
- Custom scoring: [Document Relevance Grading Pattern](#reranking--relevance-scoring)

**Hybrid Search (Semantic + Keyword)**
- Keyword component: [BM25 Retriever](#hybrid--multi-strategy-retrieval)
- Semantic component: [Vector Stores](#tools--integrations)
- Discovery: [Retrievers Overview](#tools--integrations)

**Building Self-Correction Loops**
- Patterns: [Custom RAG Agent](#rag--retrieval) (quality gates, strategy switching)
- Graph mechanics: [Graph API Usage](#langgraph-core), [Persistence](#langgraph-core)

**Agent Construction**
- LangGraph: [LangGraph Overview](#langgraph-core), [Graph API](#langgraph-core)
- Agents: [Agents Guide](#agents), [Context Engineering](#agents)

**Deployment & Monitoring**
- Testing: [LangSmith Studio](#langsmith--deployment)
- Production: [Streaming API](#langsmith--deployment)

---

## Resources

### RAG & Retrieval

**RAG Evaluation Guide** [Tutorial] - LangSmith-based evaluation with metrics and test sets
https://docs.langchain.com/langsmith/evaluate-rag-tutorial
Covers: Retrieval relevance scoring, answer correctness, groundedness evaluation
Use for: Production evaluation pipelines

**Retrieval Guide** [Reference] - RAG architecture patterns and component overview
https://docs.langchain.com/oss/python/langchain/retrieval
Covers: 2-step RAG, agentic RAG, hybrid RAG architectures, document loaders, text splitting
Use for: Understanding RAG system design patterns

**Build a RAG Agent with LangChain** [Tutorial] - Create RAG system with tool-based retrieval
https://docs.langchain.com/oss/python/langchain/rag
Covers: Retrieval as agent tool, document context management, basic RAG flow
Use for: Implementing basic RAG with LangChain agents
Code Pattern:
```python
# Define retrieval as an agent tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    return serialized_docs, retrieved_docs
```

**Build a Custom RAG Agent** [Tutorial, Advanced] - Multi-step RAG with LangGraph
https://docs.langchain.com/oss/python/langgraph/agentic-rag
Covers: Query rewriting, retrieval quality grading, self-correction loops, conditional routing
Includes: Document relevance scoring, answer quality gates, strategy switching
Use for: Building advanced RAG with self-correction and quality checks
Code Pattern:
```python
# Grade documents and route: "generate" if relevant, "rewrite" if not
score = model.with_structured_output({"binaryScore": "yes|no"}).invoke({
    "question": q,
    "context": docs
})
# Route based on relevance score
return "generate" if score["binaryScore"] == "yes" else "rewrite"
```

**Semantic Search Engine Tutorial** [Tutorial] - Build vector search with embeddings
https://docs.langchain.com/oss/python/langchain/knowledge-base
Covers: FAISS/Chroma setup, similarity search, metadata filtering, basic retrieval
Use for: Setting up foundational semantic search

**Application-Specific Evaluation Approaches** [Guide] - Custom evaluation with LLM-as-judge
https://docs.langchain.com/langsmith/evaluation-approaches
Covers: Quality scoring, confidence metrics, answer assessment, custom evaluators
Use for: Real-time quality gates in agent loops

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
Use for: Improving relevance of initial retrieval results with external reranking service
Implementation: Use ContextualCompressionRetriever with CohereRerank compressor

**Pinecone Rerank** [Integration] - Relevance scoring with custom field targeting
https://docs.langchain.com/oss/python/integrations/retrievers/pinecone_rerank
Covers: PineconeRerank class, top_n filtering, rank_fields customization, score normalization
Use for: Reranking with Pinecone infrastructure or custom relevance criteria

**Document Relevance Grading** [Pattern] - LLM-as-judge for retrieval quality assessment
Source: [Custom RAG Agent](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
Use for: Custom relevance scoring without external services
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

**LangGraph Overview** [Concept] - Stateful agent orchestration framework introduction
https://docs.langchain.com/oss/python/langgraph/overview
Covers: Graph-based agent architecture, state management, cycles, streaming
Use for: Understanding LangGraph fundamentals and use cases

**Graph API Guide** [Reference] - StateGraph and MessageGraph API documentation
https://docs.langchain.com/oss/python/langgraph/graph-api
Covers: Node/edge definitions, state schemas, graph compilation
Use for: API reference when building graphs

**Graph API Usage Guide** [Tutorial] - Practical patterns for graph construction
https://docs.langchain.com/oss/python/langgraph/use-graph-api
Covers: Conditional routing, branching, Send API for dynamic fan-out
Includes: Multi-path graphs, decision nodes, routing functions
Use for: Implementing conditional logic in agent workflows
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

**Durable Execution** [Concept] - Fault tolerance and execution guarantees
https://docs.langchain.com/oss/python/langgraph/durable-execution
Covers: Execution modes, task persistence, replay mechanisms, failure recovery
Use for: Production systems requiring reliability and resumability

**Interrupts & Human-in-the-Loop** [Tutorial] - Pause execution for human input
https://docs.langchain.com/oss/python/langgraph/interrupts
Covers: Breakpoints, approval flows, dynamic interrupts, state updates during pauses
Use for: Human approval workflows, interactive debugging

---

### Agents

**Agents Guide** [Tutorial] - Agent architectures and tool binding
https://docs.langchain.com/oss/python/langchain/agents
Covers: ReAct prompting, agent executors, tool selection, reasoning loops
Use for: Building tool-using agents with LangChain

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

**Agent Chat UI** [Tutorial] - Build chat interfaces for agents
https://docs.langchain.com/oss/python/langchain/ui
Covers: UI components, streaming responses, chat history
Use for: Creating user-facing agent interfaces

**Agent Server Changelog** [Reference] - Latest updates to LangSmith agent server
https://docs.langchain.com/langsmith/agent-server-changelog
Use for: Staying updated on new features and changes

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
- Check [Conditional Routing](#langgraph-core) logic

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
