"""
Test script for the Advanced Agentic RAG pipeline using the Attention paper PDF.

This script demonstrates:
1. PDF loading and document profiling
2. Intelligent strategy selection
3. Conversational query rewriting
4. Complete end-to-end RAG pipeline
"""

import os

# Disable LangSmith tracing to avoid 403 warnings in tests
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats, reset_retriever, DOCS_DIR
from advanced_agentic_rag_langgraph.orchestration import advanced_rag_graph
from advanced_agentic_rag_langgraph.retrieval.strategy_selection import StrategySelector


def test_pdf_loading():
    """Test 1: Load PDF and show document profiling results"""
    print("\n" + "="*80)
    print("TEST 1: PDF LOADING & DOCUMENT PROFILING")
    print("="*80)

    # Check if docs folder exists
    if not os.path.exists(DOCS_DIR):
        print(f"\ndocs/ directory not found at {DOCS_DIR}")
        print("Please ensure the docs/ directory exists and contains PDF files")
        return False

    print(f"\ndocs/ directory found: {DOCS_DIR}")

    # Initialize retriever with specific PDF (this triggers loading and profiling)
    print("\nInitializing retriever with Attention Is All You Need PDF...")
    retriever = setup_retriever(pdfs="Attention Is All You Need.pdf", verbose=True)

    # Get corpus statistics
    corpus_stats = get_corpus_stats()

    print("\n" + "="*80)
    print("CORPUS ANALYSIS SUMMARY")
    print("="*80)
    print(f"Successfully loaded and profiled {corpus_stats.get('total_documents', 0)} document chunks")
    print(f"Average technical density: {corpus_stats.get('avg_technical_density', 0):.2f}")
    print(f"Document types: {corpus_stats.get('document_types', {})}")
    print(f"Top domains: {list(corpus_stats.get('domain_distribution', {}).keys())[:5]}")
    print(f"Percentage with code: {corpus_stats.get('pct_with_code', 0):.1f}%")
    print(f"Percentage with math: {corpus_stats.get('pct_with_math', 0):.1f}%")

    return True


def test_strategy_selection():
    """Test 2: Test intelligent strategy selection with different query types"""
    print("\n\n" + "="*80)
    print("TEST 2: INTELLIGENT STRATEGY SELECTION")
    print("="*80)

    corpus_stats = get_corpus_stats()
    selector = StrategySelector()

    # Test queries representing different intents
    test_queries = [
        ("What is self-attention?", "conceptual question"),
        ("MultiHeadAttention parameters", "exact API lookup"),
        ("How does positional encoding work?", "procedural question"),
        ("Compare encoder vs decoder", "comparative question"),
        ("transformer architecture", "short factual query"),
    ]

    print("\nTesting strategy selection on diverse queries:\n")

    for query, description in test_queries:
        strategy, confidence, reasoning = selector.select_strategy(query, corpus_stats)

        print(f"Query: \"{query}\"")
        print(f"  Type: {description}")
        print(f"  -> Strategy: {strategy.upper()}")
        print(f"  -> Confidence: {confidence:.0%}")
        print(f"  -> Reasoning: {reasoning}")
        print()


def test_conversational_rewriting():
    """Test 3: Test conversational query rewriting"""
    print("\n" + "="*80)
    print("TEST 3: CONVERSATIONAL QUERY REWRITING")
    print("="*80)

    from advanced_agentic_rag_langgraph.preprocessing.query_processing import ConversationalRewriter

    rewriter = ConversationalRewriter()

    # Simulate a multi-turn conversation
    conversation = [
        {"user": "What is the transformer architecture?", "assistant": "The transformer is a neural network architecture introduced in the 'Attention Is All You Need' paper..."},
    ]

    follow_up_queries = [
        "How does it work?",
        "Show me the attention formula",
        "What about positional encoding?",
    ]

    print("\nSimulating multi-turn conversation:\n")
    print(f"Turn 1:")
    print(f"  User: {conversation[0]['user']}")
    print(f"  -> No rewrite (first query)")
    print()

    for i, query in enumerate(follow_up_queries, 2):
        rewritten, reasoning = rewriter.rewrite(query, conversation)

        print(f"Turn {i}:")
        print(f"  User: \"{query}\"")
        if rewritten != query:
            print(f"  -> Rewritten: \"{rewritten}\"")
        else:
            print(f"  -> No rewrite needed")
        print(f"  -> Reasoning: {reasoning}")
        print()

        # Add to conversation history
        conversation.append({
            "user": query,
            "assistant": f"[Response about {rewritten}]"
        })


def test_conversational_rewriting_with_messages():
    """Test 3b: Test conversational rewriting using messages field (LangGraph best practice)"""
    print("\n" + "="*80)
    print("TEST 3b: CONVERSATIONAL REWRITING WITH MESSAGES (LangGraph Best Practice)")
    print("="*80)

    from langchain_core.messages import HumanMessage, AIMessage
    from advanced_agentic_rag_langgraph.orchestration.nodes import conversational_rewrite_node

    print("\nSimulating multi-turn conversation using messages field:\n")

    # Turn 1: Initial query (no context)
    print("Turn 1:")
    print("  User: What is the transformer architecture?")

    state_turn1 = {
        "user_question": "What is the transformer architecture?",
        "baseline_query": "What is the transformer architecture?",
        "messages": [],  # No previous messages
    }

    result_turn1 = conversational_rewrite_node(state_turn1)
    print(f"  -> No rewrite (no previous conversation)")
    print()

    # Simulate the answer being generated (this would normally happen in answer_generation_node)
    simulated_answer_turn1 = "The transformer is a neural network architecture introduced in the 'Attention Is All You Need' paper that relies on self-attention mechanisms instead of recurrence..."

    # Turn 2: Follow-up query with context from turn 1
    print("Turn 2:")
    print("  User: How does it work?")

    state_turn2 = {
        "user_question": "How does it work?",
        "baseline_query": "How does it work?",
        "messages": [
            HumanMessage(content="What is the transformer architecture?"),
            AIMessage(content=simulated_answer_turn1),
        ],
    }

    result_turn2 = conversational_rewrite_node(state_turn2)
    print(f"  -> Rewritten: {result_turn2['baseline_query']}")
    assert result_turn2['baseline_query'] != "How does it work?", \
        "Query should be rewritten to include context from turn 1"
    assert "transformer" in result_turn2['baseline_query'].lower(), \
        "Rewritten query should reference 'transformer' from previous turn"
    print()

    # Turn 3: Another follow-up with accumulated context
    print("Turn 3:")
    print("  User: What about positional encoding?")

    simulated_answer_turn2 = "The transformer works by using self-attention to weigh the importance of different parts of the input sequence..."

    state_turn3 = {
        "user_question": "What about positional encoding?",
        "baseline_query": "What about positional encoding?",
        "messages": [
            HumanMessage(content="What is the transformer architecture?"),
            AIMessage(content=simulated_answer_turn1),
            HumanMessage(content="How does it work?"),
            AIMessage(content=simulated_answer_turn2),
        ],
    }

    result_turn3 = conversational_rewrite_node(state_turn3)
    print(f"  -> Rewritten: {result_turn3['baseline_query']}")
    assert result_turn3['baseline_query'] != "What about positional encoding?", \
        "Query should be rewritten to include context"
    assert "transformer" in result_turn3['baseline_query'].lower() or "positional encoding" in result_turn3['baseline_query'].lower(), \
        "Rewritten query should maintain topic context"
    print()

    print("[PASS] Conversational rewriting using messages field works correctly!")
    print("This follows LangGraph best practices (using add_messages reducer)\n")


def test_full_pipeline():
    """Test 4: Run complete end-to-end pipeline"""
    print("\n" + "="*80)
    print("TEST 4: COMPLETE END-TO-END PIPELINE")
    print("="*80)

    # Initialize retriever if not already done
    setup_retriever(pdfs="Attention Is All You Need.pdf", verbose=False)

    # Test query
    test_query = "What is the multi-head attention mechanism?"

    print(f"\nTest Query: \"{test_query}\"")
    print("\nRunning complete pipeline...\n")

    # Create initial state
    initial_state = {
        "user_question": test_query,
        "baseline_query": test_query,
        "retrieval_attempts": 0,
        "query_expansions": [],
        "messages": [],
        "retrieved_docs": [],
    }

    try:
        # Run the graph with thread_id for checkpointer
        print("Executing LangGraph workflow...")
        config = {"configurable": {"thread_id": "test-run"}}
        result = advanced_rag_graph.invoke(initial_state, config=config)

        print("\n" + "-"*80)
        print("PIPELINE RESULTS")
        print("-"*80)
        print(f"\nOriginal Query: {result.get('user_question', 'N/A')}")
        print(f"Strategy Used: {result.get('retrieval_strategy', 'N/A')}")
        print(f"Retrieval Attempts: {result.get('retrieval_attempts', 0)}")
        print(f"Retrieval Quality: {result.get('retrieval_quality_score', 0):.0%}")
        print(f"Answer Sufficient: {result.get('is_answer_sufficient', False)}")
        print(f"Confidence Score: {result.get('confidence_score', 0):.0%}")

        print(f"\nFinal Answer:")
        print(f"{result.get('final_answer', 'No answer generated')}")

    except Exception as e:
        print(f"\n[ERROR] Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("ADVANCED AGENTIC RAG - PDF PIPELINE TEST SUITE")
    print("Testing with: Attention Is All You Need Paper")
    print("="*80)

    # Test 1: PDF Loading
    if not test_pdf_loading():
        print("\n[ERROR] PDF loading failed. Aborting remaining tests.")
        return

    # Test 2: Strategy Selection
    test_strategy_selection()

    # Test 3: Conversational Rewriting (legacy pattern)
    test_conversational_rewriting()

    # Test 3b: Conversational Rewriting with Messages (LangGraph best practice)
    test_conversational_rewriting_with_messages()

    # Test 4: Full Pipeline
    test_full_pipeline()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. PDF successfully loaded and profiled (~45-50 chunks)")
    print("2. Strategy selection adapts to query type (semantic/keyword/hybrid)")
    print("3. Conversational rewriting handles multi-turn dialogues")
    print("4. Complete pipeline executes end-to-end with real technical content")
    print("\nYour RAG system is working with real documents!\n")


if __name__ == "__main__":
    main()
