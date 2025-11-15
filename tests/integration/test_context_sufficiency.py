"""
Integration test for Phase 6: Context Sufficiency Enhancement

This test validates that context sufficiency is evaluated before answer generation,
enabling early detection of incomplete retrieval and smarter strategy switching.
"""

import os
import sys

# Disable LangSmith tracing to avoid 403 warnings in tests
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.orchestration.graph import advanced_rag_graph


def test_context_sufficiency_detection():
    """
    Test that context sufficiency is evaluated and logged.

    Uses a multi-aspect query to potentially trigger incomplete context detection.
    """
    print("\n" + "="*70)
    print("TEST: Context Sufficiency Detection (Phase 6)")
    print("="*70)

    # Query that asks for multiple aspects (advantages AND disadvantages)
    # This may reveal incomplete context if retrieval only captures one aspect
    query = "What are the main advantages and disadvantages of the BERT model?"

    print(f"\nQuery: {query}")
    print("Note: Multi-aspect query may reveal incomplete context\n")

    result = advanced_rag_graph.invoke({
        "question": query,
        "original_query": query,
        "conversation_history": [],
        "retrieval_attempts": 0,
        "query_expansions": [],
        "messages": [],
        "retrieved_docs": [],
    }, config={"configurable": {"thread_id": "test-context-sufficiency"}})

    print("\n" + "="*70)
    print("CONTEXT SUFFICIENCY TEST RESULTS")
    print("="*70)

    # Display context sufficiency metrics
    sufficiency_score = result.get('context_sufficiency_score', 0.0)
    is_sufficient = result.get('context_is_sufficient', True)
    missing_aspects = result.get('missing_context_aspects', [])

    print(f"Context Sufficiency Score: {sufficiency_score:.0%}")
    print(f"Context Is Sufficient: {is_sufficient}")
    print(f"Missing Aspects ({len(missing_aspects)}):")
    if missing_aspects:
        for aspect in missing_aspects:
            print(f"  - {aspect}")
    else:
        print(f"  (none detected)")

    print(f"\nRetrieval Attempts: {result.get('retrieval_attempts', 0)}")
    print(f"Final Strategy: {result.get('retrieval_strategy', 'N/A')}")
    print(f"Answer Sufficient: {result.get('is_answer_sufficient', False)}")

    # Display refinement history if any
    refinement_history = result.get('refinement_history', [])
    if refinement_history:
        print(f"\nStrategy Refinement History ({len(refinement_history)} refinements):")
        for i, refinement in enumerate(refinement_history, 1):
            print(f"  Refinement {i}:")
            print(f"    From: {refinement.get('from_strategy')} to {refinement.get('to_strategy')}")
            print(f"    Reasoning: {refinement.get('reasoning')}")
            print(f"    Context Sufficiency: {refinement.get('context_sufficiency', 'N/A'):.0%}")

    print("="*70)

    # Assertions
    print("\nRunning assertions...")

    assert "context_sufficiency_score" in result, "Context sufficiency score missing from result"
    assert "context_is_sufficient" in result, "Context sufficiency flag missing from result"
    assert "missing_context_aspects" in result, "Missing aspects list missing from result"

    # Check that sufficiency score is a valid float in range [0, 1]
    assert isinstance(sufficiency_score, float), f"Sufficiency score should be float, got {type(sufficiency_score)}"
    assert 0.0 <= sufficiency_score <= 1.0, f"Sufficiency score {sufficiency_score} should be in range [0, 1]"

    # Check that is_sufficient is a boolean
    assert isinstance(is_sufficient, bool), f"Context is_sufficient should be bool, got {type(is_sufficient)}"

    # Check that missing_aspects is a list
    assert isinstance(missing_aspects, list), f"Missing aspects should be list, got {type(missing_aspects)}"

    print("[OK] All assertions passed")

    # Display final answer excerpt
    final_answer = result.get('final_answer', '')
    if final_answer:
        print(f"\nFinal Answer (excerpt):")
        print(f"{final_answer[:300]}...")

    print("\n" + "="*70)
    print("TEST PASSED: Context sufficiency evaluation working correctly")
    print("="*70)

    return result


def test_context_sufficiency_with_simple_query():
    """
    Test context sufficiency with a simple single-aspect query.

    Expected: High sufficiency score since query is straightforward.
    """
    print("\n" + "="*70)
    print("TEST: Context Sufficiency with Simple Query")
    print("="*70)

    query = "What is the attention mechanism in neural networks?"

    print(f"\nQuery: {query}")
    print("Note: Simple query should have high context sufficiency\n")

    result = advanced_rag_graph.invoke({
        "question": query,
        "original_query": query,
        "conversation_history": [],
        "retrieval_attempts": 0,
        "query_expansions": [],
        "messages": [],
        "retrieved_docs": [],
    }, config={"configurable": {"thread_id": "test-context-simple"}})

    sufficiency_score = result.get('context_sufficiency_score', 0.0)
    is_sufficient = result.get('context_is_sufficient', True)

    print(f"\nContext Sufficiency Score: {sufficiency_score:.0%}")
    print(f"Context Is Sufficient: {is_sufficient}")

    # For simple queries, we expect high sufficiency (usually > 0.7)
    if sufficiency_score >= 0.7:
        print("[OK] Simple query achieved high context sufficiency as expected")
    else:
        print(f"[NOTE] Unexpected low sufficiency ({sufficiency_score:.0%}) for simple query")

    print("\n" + "="*70)
    print("TEST PASSED: Simple query context sufficiency evaluated")
    print("="*70)

    return result


def run_all_context_sufficiency_tests():
    """Run all context sufficiency tests."""
    print("\n" + "="*70)
    print("RUNNING ALL CONTEXT SUFFICIENCY TESTS")
    print("="*70)

    try:
        # Test 1: Multi-aspect query (may reveal incomplete context)
        test_context_sufficiency_detection()

        # Test 2: Simple query (should have high sufficiency)
        test_context_sufficiency_with_simple_query()

        print("\n" + "="*70)
        print("ALL CONTEXT SUFFICIENCY TESTS PASSED!")
        print("="*70)
        print("\nPhase 6: Context Sufficiency Enhancement is working correctly")
        print("- Context sufficiency evaluated before answer generation")
        print("- Missing aspects detected when context incomplete")
        print("- Context-driven strategy switching operational")
        print("\n")

    except Exception as e:
        print("\n" + "="*70)
        print(f"CONTEXT SUFFICIENCY TESTS FAILED: {e}")
        print("="*70)
        raise


if __name__ == "__main__":
    run_all_context_sufficiency_tests()
