"""Quick test to verify groundedness check executes"""
import os
import warnings
import logging

# Suppress LangSmith warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*Failed to.*LangSmith.*")
warnings.filterwarnings("ignore", message=".*langsmith.*")

# Suppress LangSmith logging
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.WARNING)

from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph

# Simple query
result = advanced_rag_graph.invoke({
    "question": "What is attention?",
    "original_query": "What is attention?",
    "retrieval_attempts": 0,
    "query_expansions": [],
    "messages": [],
    "retrieved_docs": [],
}, config={"configurable": {"thread_id": "test-groundedness"}})

print("\n" + "="*60)
print("GROUNDEDNESS TEST RESULTS")
print("="*60)
print(f"Groundedness Score: {result.get('groundedness_score', 'N/A')}")
print(f"Has Hallucination: {result.get('has_hallucination', 'N/A')}")
print(f"Severity: {result.get('groundedness_severity', 'N/A')}")
print(f"Unsupported Claims: {result.get('unsupported_claims', [])}")
print("="*60)
