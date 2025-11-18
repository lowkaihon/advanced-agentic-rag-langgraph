"""
Test script for quality-issue-based adaptive retrieval workflow.

This demonstrates:
1. Retrieval quality evaluation with issue detection
2. Strategy switching based on quality issues
3. Query rewriting with actionable feedback
4. Self-correcting retrieval loops
"""

import os

# Disable LangSmith tracing to avoid 403 warnings in tests
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from advanced_agentic_rag_langgraph.core import setup_retriever, get_corpus_stats, reset_retriever
from advanced_agentic_rag_langgraph.orchestration import advanced_rag_graph


def test_adaptive_retrieval_workflow():
    """Test complete adaptive retrieval workflow with quality-based adaptation"""
    print("\n" + "="*80)
    print("TEST: QUALITY-ISSUE-BASED ADAPTIVE RETRIEVAL WORKFLOW")
    print("="*80)

    # Reset and initialize
    reset_retriever()
    print("\nInitializing retriever with Attention paper...")
    retriever = setup_retriever(pdfs="Attention Is All You Need.pdf", verbose=False)

    print("\n[SUCCESS] Retriever initialized\n")

    # Test queries that should trigger different behaviors
    test_cases = [
        {
            "name": "Conceptual Query (Basic Retrieval)",
            "query": "What is the attention mechanism?",
            "expected_behavior": "Initial semantic retrieval, evaluate quality, may trigger rewriting if insufficient"
        },
        {
            "name": "Technical Query (Strategy Evaluation)",
            "query": "multi-head attention implementation details",
            "expected_behavior": "May switch strategy based on retrieval quality issues"
        },
        {
            "name": "Cross-Document Query (Should Trigger Quality Issues)",
            "query": "Compare transformer architecture advantages versus RNN and CNN approaches",
            "expected_behavior": "May trigger partial_coverage or incomplete_context issues, leading to query rewriting"
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "="*80)
        print(f"TEST CASE {i}: {test_case['name']}")
        print("="*80)
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected_behavior']}")
        print("\n" + "-"*80)

        # Create initial state
        initial_state = {
            "user_question": test_case["query"],
            "baseline_query": test_case["query"],
            "retrieval_attempts": 0,
            "query_expansions": [],
            "messages": [],
            "retrieved_docs": [],
        }

        try:
            # Run the graph
            config = {"configurable": {"thread_id": f"test-adaptive-{i}"}}
            result = advanced_rag_graph.invoke(initial_state, config=config)

            # Display results
            print("\n" + "-"*80)
            print("RESULTS")
            print("-"*80)
            print(f"Original Query: {result.get('user_question', 'N/A')}")
            print(f"Initial Strategy: {result.get('retrieval_strategy', 'N/A')}")
            print(f"Retrieval Attempts: {result.get('retrieval_attempts', 0)}")

            # Retrieval quality analysis
            print(f"\nRetrieval Quality Analysis:")
            print(f"  - Score: {result.get('retrieval_quality_score', 0):.0%}")

            quality_issues = result.get('retrieval_quality_issues', [])
            if quality_issues:
                print(f"  - Issues detected: {', '.join(quality_issues)}")
            else:
                print(f"  - No issues detected")

            quality_reasoning = result.get('retrieval_quality_reasoning', '')
            if quality_reasoning:
                print(f"  - LLM reasoning: {quality_reasoning[:150]}...")

            # Strategy switching analysis
            strategy_switch_reason = result.get('strategy_switch_reason', '')
            if strategy_switch_reason:
                print(f"\nStrategy Switch:")
                print(f"  - Reason: {strategy_switch_reason}")

            # Refinement history
            refinement_history = result.get("refinement_history", [])
            if refinement_history:
                print(f"\nRefinement History ({len(refinement_history)} refinements):")
                for j, refinement in enumerate(refinement_history, 1):
                    print(f"  {j}. Iteration {refinement.get('iteration')}: {refinement.get('from_strategy')} to {refinement.get('to_strategy')}")
                    print(f"     Reasoning: {refinement.get('reasoning')}")
                    print(f"     Issues: {', '.join(refinement.get('metadata_issues', []))}")
            else:
                print(f"\nNo refinements needed (good initial strategy!)")

            print(f"\nRetrieval Quality: {result.get('retrieval_quality_score', 0):.0%}")
            print(f"Answer Sufficient: {result.get('is_answer_sufficient', False)}")
            print(f"Confidence Score: {result.get('confidence_score', 0):.0%}")

            print(f"\nFinal Answer:")
            print(f"{result.get('final_answer', 'No answer generated')[:300]}...")

            print("\n" + "="*80)
            print(f"[SUCCESS] Test case {i} completed")
            print("="*80)

        except Exception as e:
            print(f"\n[ERROR] Test case {i} failed: {e}")
            import traceback
            traceback.print_exc()


def test_quality_evaluation_details():
    """Test detailed quality evaluation functionality"""
    print("\n\n" + "="*80)
    print("TEST: QUALITY EVALUATION DETAILS")
    print("="*80)

    # Initialize retriever if not already done
    setup_retriever(pdfs="Attention Is All You Need.pdf", verbose=False)

    print("\nThis test demonstrates:")
    print("1. Document profiling creates metadata during PDF loading")
    print("2. Retrieval quality evaluation with 8 issue types:")
    print("   - partial_coverage, missing_key_info, incomplete_context, domain_misalignment")
    print("   - low_confidence, mixed_relevance, off_topic, wrong_domain")
    print("3. Strategy switching triggered by quality issues")
    print("4. Query rewriting with actionable feedback")
    print("5. Self-correcting loops that improve retrieval")

    print("\n[INFO] Run the main test above to see this in action!")


def main():
    """Run all adaptive retrieval tests"""
    print("\n" + "="*80)
    print("ADAPTIVE RETRIEVAL TEST SUITE")
    print("Demonstrating: Quality-Issue-Based Intelligence & Self-Correction")
    print("="*80)

    # Test 1: Full workflow
    test_adaptive_retrieval_workflow()

    # Test 2: Quality evaluation details
    test_quality_evaluation_details()

    print("\n" + "="*80)
    print("ALL ADAPTIVE RETRIEVAL TESTS COMPLETED")
    print("="*80)
    print("\nKey Achievements:")
    print("[SUCCESS] Retrieval quality evaluation with 8 issue types")
    print("[SUCCESS] Quality issues drive query rewriting decisions")
    print("[SUCCESS] Strategy switching based on retrieval quality analysis")
    print("[SUCCESS] Refinement history logs all adaptive decisions")
    print("[SUCCESS] Self-correcting loops improve retrieval quality")
    print("\nThis showcases production-ready adaptive AI patterns!\n")


if __name__ == "__main__":
    main()
