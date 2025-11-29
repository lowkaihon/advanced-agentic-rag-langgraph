from advanced_agentic_rag_langgraph.orchestration import advanced_rag_graph
from advanced_agentic_rag_langgraph.core import setup_retriever
import advanced_agentic_rag_langgraph.orchestration.nodes as nodes
import uuid
import io
from contextlib import redirect_stdout

def run_advanced_rag(question: str, thread_id: str = None, verbose: bool = True):
    """Run the complete advanced RAG system with all techniques"""
    
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    # Initialize retriever in global
    if nodes.adaptive_retriever is None:
        nodes.adaptive_retriever = setup_retriever()
    
    # Initial state
    initial_state = {
        "user_question": question,
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
    
    # Stream the graph execution (nodes print their own detailed output)
    if verbose:
        for step in advanced_rag_graph.stream(initial_state, config=config, stream_mode="updates"):
            pass  # Nodes handle their own output via print statements
    else:
        # Suppress node output when not verbose
        with redirect_stdout(io.StringIO()):
            for step in advanced_rag_graph.stream(initial_state, config=config, stream_mode="updates"):
                pass
    
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
        "What does CLIP stand for?",
#        "How does the attention mechanism work in Transformers?",
#        "How do consistency models differ from traditional diffusion models?",
    ]
    
    for question in test_questions:
        run_advanced_rag(question, verbose=True)
        print("\n" + "="*70 + "\n")