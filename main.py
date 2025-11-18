from advanced_agentic_rag_langgraph.orchestration import advanced_rag_graph
from advanced_agentic_rag_langgraph.core import setup_retriever
import advanced_agentic_rag_langgraph.orchestration.nodes as nodes
import uuid

def run_advanced_rag(question: str, thread_id: str = None, verbose: bool = True):
    """Run the complete advanced RAG system with all techniques"""
    
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    # Initialize retriever in global
    if nodes.adaptive_retriever is None:
        nodes.adaptive_retriever = setup_retriever()
    
    # Initial state
    initial_state = {
        "baseline_query": question,
        "query_expansions": [],
        "active_query": question,
        "retrieval_strategy": "hybrid",
        "messages": [],
        "retrieved_docs": [],
        "retrieval_quality_score": 0.0,
        "is_answer_sufficient": False,
        "retrieval_attempts": 0,
        "final_answer": "",
        "confidence_score": 0.0,
    }
    
    # Config with thread
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n{'='*70}")
    print(f"ADVANCED AGENTIC RAG SYSTEM")
    print(f"{'='*70}")
    print(f"Question: {question}\n")
    
    # Track execution
    step_count = 0
    
    # Stream the graph execution
    for step in advanced_rag_graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_state in step.items():
            if node_name != "__root__":
                step_count += 1
                print(f"\n[Step {step_count}] {node_name.upper()}")
                print("-" * 50)
                
                # Print node-specific info
                if "query_expansions" in node_state and node_state["query_expansions"]:
                    print(f"Query variations: {len(node_state['query_expansions'])}")
                    for i, q in enumerate(node_state["query_expansions"][1:], 1):
                        print(f"  {i}. {q[:60]}...")
                
                if "retrieval_strategy" in node_state:
                    print(f"Strategy: {node_state['retrieval_strategy'].upper()}")
                
                if "retrieval_quality_score" in node_state and node_state["retrieval_quality_score"] > 0:
                    quality = node_state["retrieval_quality_score"]
                    bar = "#" * int(quality * 10) + "-" * (10 - int(quality * 10))
                    print(f"Retrieval Quality: [{bar}] {quality:.0%}")
                
                if "active_query" in node_state and node_state["active_query"]:
                    current_query = node_state["active_query"]
                    baseline_query = node_state.get("baseline_query", "")
                    if current_query != baseline_query:
                        print(f"Rewritten Query: {current_query}")
                
                if "final_answer" in node_state and node_state["final_answer"]:
                    answer = node_state["final_answer"]
                    preview = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"Answer: {preview}")
                
                if "confidence_score" in node_state and node_state["confidence_score"] > 0:
                    conf = node_state["confidence_score"]
                    print(f"Confidence: {conf:.0%}")
                
                if "retrieval_attempts" in node_state:
                    print(f"Attempts: {node_state['retrieval_attempts']}")
    
    # Get final state
    final_state = advanced_rag_graph.get_state(config)
    final_values = final_state.values
    
    print(f"\n{'='*70}")
    print("FINAL RESULT")
    print(f"{'='*70}\n")
    print(final_values.get("final_answer", "No answer generated"))
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Confidence: {final_values.get('confidence_score', 0):.0%}")
    print(f"Retrieval Attempts: {final_values.get('retrieval_attempts', 0)}")
    print(f"Retrieval Quality: {final_values.get('retrieval_quality_score', 0):.0%}")
    print(f"Strategy Used: {final_values.get('retrieval_strategy', 'hybrid').upper()}")
    print(f"Query Rewritten: {'Yes' if final_values.get('active_query') != final_values.get('baseline_query') else 'No'}")
    print(f"Query Variations: {len(final_values.get('query_expansions', []))}")
    
    return final_values

# Demo
if __name__ == "__main__":
    test_questions = [
        "What is the relationship between machine learning and deep learning?",
        "How does LangGraph enable building stateful agents?",
        "Explain retrieval-augmented generation and its benefits",
    ]
    
    for question in test_questions:
        run_advanced_rag(question, verbose=True)
        print("\n" + "="*70 + "\n")