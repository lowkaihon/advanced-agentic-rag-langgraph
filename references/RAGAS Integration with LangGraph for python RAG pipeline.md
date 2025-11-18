## RAGAS Integration with LangGraph for Python RAG Pipeline

**RAGAS** (Retrieval-Augmented Generation Assessment) is a specialized evaluation framework designed to measure RAG pipeline performance through reference-free metrics, making it ideal for production systems. **LangGraph** is a state-based orchestration framework that structures AI workflows as directed graphs. Integrating these two creates a powerful system for building and evaluating complex RAG pipelines systematically.

### Core Architecture and Integration Strategy

LangGraph manages your RAG workflow through **state management, nodes (computation units), and edges (routing logic)**. The framework maintains a centralized `GraphState` object that flows through each node, enabling precise tracking of queries, retrieved documents, and generated answers. RAGAS evaluates the outputs of each pipeline by converting LangGraph's message sequences into its evaluation format and calculating quality metrics.[^1_1][^1_2][^1_3][^1_4]

The integration follows this pattern: your LangGraph nodes handle retrieval and generation, while RAGAS metrics assess the outputs asynchronously without requiring ground truth data for online evaluation. This enables continuous evaluation of production traces.[^1_5]

### Key RAGAS Metrics for RAG Assessment

RAGAS provides comprehensive metrics split into **retriever-focused** and **generator-focused** categories:[^1_6][^1_7]

**Retriever Metrics:**

- **Context Precision**: Measures whether retrieved contexts are ranked correctly (higher relevance first)
- **Context Recall**: Determines if retrieved contexts contain all information needed to answer the question (requires ground truth)
- **Context Entities Recall**: Evaluates entity-level retrieval accuracy
- **Noise Sensitivity**: Assesses the signal-to-noise ratio in retrieved documents

**Generator Metrics:**

- **Faithfulness**: Checks if generated answers contain hallucinations or unsupported claims relative to retrieved context
- **Response Relevancy**: Measures how relevant and on-topic the answer is to the question
- **Answer Correctness**: Compares generated answers against reference answers

**Agent/Tool Metrics:**

- **Tool Call Accuracy**: Evaluates whether the LLM correctly identifies and invokes required tools
- **Agent Goal Accuracy**: Measures whether the LLM achieved the user's stated objective[^1_4]


### Implementation Pattern: Converting LangGraph State to RAGAS Format

RAGAS requires data in specific formats: **SingleTurnSample** for single-turn interactions or **MultiTurnSample** for multi-turn conversations. LangGraph message sequences must be converted to this format using RAGAS's integration utilities.[^1_8][^1_9][^1_4]

```python
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample
from ragas.metrics import Faithfulness, ContextRecall

# After LangGraph node execution, convert messages
ragas_trace = convert_to_ragas_messages(
    messages=langraph_result["messages"]
)

# Create evaluation sample
sample = SingleTurnSample(
    user_input=query,
    retrieved_contexts=contexts,
    response=answer,
    reference=ground_truth  # Optional for reference-free metrics
)

# Score asynchronously
scorer = Faithfulness()
score = await scorer.single_turn_ascore(sample)
```


### LangGraph Node Integration Pattern

LangGraph nodes should return state updates that RAGAS can consume. A typical RAG pipeline with evaluation looks like:[^1_10][^1_11]

```python
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    evaluation_scores: dict

def retrieval_node(state: GraphState):
    """Retrieve relevant documents"""
    documents = retriever.invoke(state["question"])
    return {"documents": documents}

def generation_node(state: GraphState):
    """Generate answer from retrieved documents"""
    context = "\n\n".join([d.page_content for d in state["documents"]])
    generation = rag_chain.invoke({
        "context": context,
        "question": state["question"]
    })
    return {"generation": generation}

def evaluation_node(state: GraphState):
    """Evaluate using RAGAS metrics"""
    ragas_trace = convert_to_ragas_messages(state.get("messages", []))
    
    sample = SingleTurnSample(
        user_input=state["question"],
        retrieved_contexts=[d.page_content for d in state["documents"]],
        response=state["generation"]
    )
    
    # Run evaluation metrics
    scores = {}
    for metric in metrics:
        scores[metric.name] = await metric.single_turn_ascore(sample)
    
    return {"evaluation_scores": scores}

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)
workflow.add_node("evaluate", evaluation_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_edge("evaluate", END)

graph = workflow.compile()
```


### Initialization of RAGAS Metrics

RAGAS metrics require LLM and embedding model initialization. Use **LangchainLLMWrapper** and **LangchainEmbeddingsWrapper** to integrate with LangChain-based models:[^1_12][^1_5]

```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness, 
    ContextRecall, 
    ContextPrecision, 
    ResponseRelevancy
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.run_config import RunConfig

# Initialize LLMs and embeddings
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
evaluator_embeddings = OpenAIEmbeddings()

# Wrap for RAGAS compatibility
llm_wrapper = LangchainLLMWrapper(evaluator_llm)
embeddings_wrapper = LangchainEmbeddingsWrapper(evaluator_embeddings)

# Initialize metrics
metrics = [
    Faithfulness(),
    ContextRecall(),
    ContextPrecision(),
    ResponseRelevancy()
]

# Configure metrics
for metric in metrics:
    if hasattr(metric, 'llm'):
        metric.llm = llm_wrapper
    if hasattr(metric, 'embeddings'):
        metric.embeddings = embeddings_wrapper
    
    run_config = RunConfig()
    metric.init(run_config)
```


### Evaluation Strategies: Offline vs. Online

**Offline Evaluation** involves batch evaluation on static datasets before deployment. This approach:[^1_13]

- Tests system changes before production impact
- Runs comprehensive, computationally expensive metrics
- Uses a "golden dataset" with known good answers and contexts
- Allows regression testing against benchmarks
- Cannot capture dynamic data drift or user behavior changes

**Online Evaluation** assesses live production traffic in real-time. Benefits include:[^1_13]

- Measures actual user experience
- Detects live performance degradation
- Captures data drift and evolving query patterns
- Enables A/B testing of system versions
- Provides direct business impact metrics

For production RAG systems, integrate both: use offline evaluation to validate changes before deployment, then deploy with online monitoring to capture real-world performance.[^1_13]

### Asynchronous Batch Evaluation

For production efficiency, RAGAS supports async evaluation to score multiple samples in parallel:[^1_14]

```python
import asyncio
from ragas.dataset_schema import EvaluationDataset

# Create evaluation dataset
dataset = EvaluationDataset(samples=[sample1, sample2, sample3])

# Run async batch evaluation
async def evaluate_batch():
    from ragas import evaluate
    
    results = await evaluate(
        dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embeddings_wrapper
    )
    return results

# Execute
results = asyncio.run(evaluate_batch())
```


### LangGraph Persistence for Evaluation State

LangGraph supports **checkpointing** to persist graph state across executions. This enables:[^1_15][^1_16]

- **Memory**: Access conversation history and prior states
- **Human-in-the-loop**: Pause execution for validation before continuing
- **Fault tolerance**: Resume from checkpoints after failures
- **Time travel**: Inspect and replay past states

```python
from langgraph.checkpoint.memory import InMemorySaver

# For production, use PostgresSaver or other durable checkpointers
checkpointer = InMemorySaver()

graph = workflow.compile(checkpointer=checkpointer)

# Invoke with thread_id for persistence
result = graph.invoke(
    {"question": "What is RAG?"},
    config={"configurable": {"thread_id": "user_123"}}
)

# Access checkpoint history
history = graph.get_state_history(
    config={"configurable": {"thread_id": "user_123"}}
)
```


### Integration with LangSmith/Langfuse for Tracing

For comprehensive observability, integrate evaluation with tracing platforms:[^1_17][^1_5]

```python
from langfuse.langchain import CallbackHandler

# Initialize tracing
langfuse_handler = CallbackHandler()

# Invoke graph with tracing
result = graph.invoke(
    {"question": "What is RAG?"},
    config={"callbacks": [langfuse_handler]}
)

# RAGAS scores are automatically traced
for metric_name, score in result["evaluation_scores"].items():
    langfuse.create_score(
        name=metric_name,
        value=score,
        trace_id=trace_id
    )
```


### Best Practices for Production Deployment

**Use LangGraph for adaptive RAG architectures**: Combine conditional routing with RAGAS evaluation to implement adaptive retrieval that adjusts depth and strategy based on query complexity. Route low-confidence retrievals to retry with modified queries, then evaluate results.[^1_18]

**Implement hierarchical checkpointing**: Use **PostgresSaver** or production-grade checkpointers for persistent state management, enabling pause/resume functionality and process recovery.[^1_16]

**Minimize token costs**: Run reference-free RAGAS metrics online for continuous monitoring, reserve expensive ground-truth metrics for periodic offline batches.[^1_5]

**Establish evaluation data pipelines**: Create feedback loops where insights from online evaluation (problematic queries, user feedback) augment offline golden datasets, ensuring evolving relevance.[^1_13]

**Monitor evaluation latency**: RAGAS metrics may exceed timeout thresholds with local LLMs or network delays. Configure appropriate timeout windows for async scoring.[^1_19]

### Common Integration Patterns

**Self-correcting RAG**: Use document grading nodes to evaluate retrieval quality, then conditionally route to query rewriting if scores fall below threshold. Evaluate final output with RAGAS metrics.[^1_20]

**Multi-turn conversation evaluation**: For chatbot applications, use **MultiTurnSample** and **AgentGoalAccuracyWithReference** to evaluate whether the agent achieves multi-step user objectives.[^1_4]

**Batch scoring for reporting**: Periodically sample production traces, format as EvaluationDataset, and run offline batch evaluation to generate performance reports.[^1_5]

This integrated approach provides systematic, measurable assessment of RAG pipeline quality while maintaining production performance through efficient asynchronous evaluation patterns.
<span style="display:none">[^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49]</span>

<div align="center">⁂</div>

[^1_1]: https://docs.langchain.com/oss/python/langgraph/agentic-rag

[^1_2]: https://aws.amazon.com/blogs/machine-learning/build-multi-agent-systems-with-langgraph-and-amazon-bedrock/

[^1_3]: https://milvus.io/blog/langchain-vs-langgraph.md

[^1_4]: https://docs.ragas.io/en/stable/howtos/integrations/_langgraph_agent_evaluation/

[^1_5]: https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas

[^1_6]: https://agenta.ai/blog/how-to-evaluate-rag-metrics-evals-and-best-practices

[^1_7]: https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more

[^1_8]: https://docs.ragas.io/en/v0.2.0/concepts/components/eval_dataset/

[^1_9]: https://docs.ragas.io/en/stable/concepts/components/eval_dataset/

[^1_10]: https://newsletter.theaiedge.io/p/how-to-build-ridiculously-complex

[^1_11]: https://www.leanware.co/insights/langgraph-rag-agentic

[^1_12]: https://docs.ragas.io/en/v0.2.10/getstarted/rag_testset_generation/

[^1_13]: https://apxml.com/courses/optimizing-rag-for-production/chapter-6-advanced-rag-evaluation-monitoring/rag-offline-online-evaluation

[^1_14]: https://docs.ragas.io/en/latest/references/metrics/

[^1_15]: https://docs.langchain.com/oss/python/langgraph/persistence

[^1_16]: https://sparkco.ai/blog/mastering-langgraph-checkpointing-best-practices-for-2025

[^1_17]: https://langfuse.com/guides/cookbook/example_langgraph_agents

[^1_18]: https://www.chitika.com/adaptive-rag-systems-langchain-langgraph/

[^1_19]: https://github.com/explodinggradients/ragas/issues/1100

[^1_20]: https://www.datacamp.com/tutorial/corrective-rag-crag

[^1_21]: https://www.ibm.com/think/tutorials/evaluate-rag-pipeline-using-ragas-in-python-with-watsonx

[^1_22]: https://github.com/explodinggradients/ragas

[^1_23]: https://blog.langchain.com/evaluating-rag-pipelines-with-ragas-langsmith/

[^1_24]: https://docs.ragas.io/en/stable/

[^1_25]: https://pathway.com/blog/evaluating-rag

[^1_26]: https://medium.aiplanet.com/evaluating-naive-rag-and-advanced-rag-pipeline-using-langchain-v-0-1-0-and-ragas-17d24e74e5cf

[^1_27]: https://www.zenml.io/blog/best-llm-evaluation-tools

[^1_28]: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

[^1_29]: https://sparkco.ai/blog/langchain-vs-langgraph-a-deep-dive-comparison

[^1_30]: https://realpython.com/langgraph-python/

[^1_31]: https://www.reddit.com/r/Rag/comments/1mxs81z/finally_figured_out_the_langchain_vs_langgraph_vs/

[^1_32]: https://auth0.com/blog/building-a-secure-python-rag-agent-using-auth0-fga-and-langgraph/

[^1_33]: https://www.linkedin.com/posts/jjmachan_if-youre-building-with-langgraph-do-checkout-activity-7262849412207419393-iNnc

[^1_34]: https://medium.aiplanet.com/evaluate-and-monitor-your-hybrid-search-rag-langgraph-qdrant-minicoil-opik-and-deepseek-r1-a7ac70981ac3

[^1_35]: https://docs.ragas.io/en/latest/howtos/applications/compare_llms/

[^1_36]: https://docs.ragas.io/en/v0.2.13/howtos/integrations/langchain/

[^1_37]: https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/list-of-eval-metrics

[^1_38]: https://redis.io/blog/get-better-rag-responses-with-ragas/

[^1_39]: https://docs.ragas.io/en/v0.2.0/concepts/metrics/available_metrics/

[^1_40]: https://qdrant.tech/blog/rag-evaluation-guide/

[^1_41]: https://dev.to/syamaner/beyond-basic-rag-measuring-embedding-and-generation-performance-with-ragas-ddk

[^1_42]: https://github.com/langchain-ai/langgraph/discussions/924

[^1_43]: https://github.com/chitralputhran/Advanced-RAG-LangGraph

[^1_44]: https://v2docs.galileo.ai/cookbooks/use-cases/multi-agent-langgraph/multi-agent-langgraph

[^1_45]: https://docs.ragas.io/en/latest/references/evaluation_schema/

[^1_46]: https://milvus.io/ai-quick-reference/what-is-the-difference-between-online-and-offline-evaluation-of-recommender-systems

[^1_47]: https://developer.couchbase.com/tutorial-langgraph-persistence-checkpoint/

[^1_48]: https://docs.ragas.io/en/stable/references/prompt/

[^1_49]: https://www.reddit.com/r/LangChain/comments/1bijg75/why_is_everyone_using_ragas_for_rag_evaluation/

