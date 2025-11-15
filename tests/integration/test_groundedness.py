"""Quick test to verify groundedness check executes"""
from src.orchestration.graph import advanced_rag_graph

# Simple query
result = advanced_rag_graph.invoke({
    "question": "What is attention?",
    "original_query": "What is attention?",
    "conversation_history": [],
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
