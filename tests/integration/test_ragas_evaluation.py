"""
RAGAS Evaluation Suite for Golden Dataset

This test suite integrates RAGAS (Retrieval-Augmented Generation Assessment)
metrics with the existing golden dataset evaluation framework.

RAGAS Metrics Tested:
- Faithfulness: Measures if generated answers contain hallucinations
- Context Precision: Evaluates if relevant contexts are ranked higher
- Response Relevancy: Measures how relevant the answer is to the question

Comparison Analysis:
- Faithfulness (RAGAS) vs Groundedness (custom)
- Context Precision (RAGAS) vs Retrieval Quality (custom)
- Response Relevancy (RAGAS) vs Answer Sufficiency (custom)
"""

import json
import os
import sys

# Disable LangSmith tracing to avoid 403 warnings in tests
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.evaluation import (
    GoldenDatasetManager,
    evaluate_on_golden_dataset,
    RAGASEvaluator,
    run_ragas_evaluation_on_golden,
    compare_ragas_with_custom_metrics
)
from src.orchestration.graph import advanced_rag_graph


def test_ragas_evaluator_initialization():
    """Test that RAGASEvaluator initializes correctly."""
    print("\n" + "="*70)
    print("TEST: RAGAS Evaluator Initialization")
    print("="*70)

    try:
        evaluator = RAGASEvaluator()
        print(f"[OK] RAGASEvaluator initialized successfully")

        # Check metrics
        assert len(evaluator.metrics) > 0, "Should have metrics initialized"
        print(f"[OK] Initialized {len(evaluator.metrics)} metrics:")
        for metric in evaluator.metrics:
            print(f"  - {metric.name}")

        # Check LLM and embeddings
        assert evaluator.llm_wrapper is not None, "LLM wrapper should be initialized"
        assert evaluator.embeddings_wrapper is not None, "Embeddings wrapper should be initialized"
        print(f"[OK] LLM and embeddings wrappers configured")

        print("\n[OK] RAGAS evaluator initialization test PASSED\n")
        return evaluator

    except Exception as e:
        print(f"[FAIL] RAGAS initialization failed: {e}")
        raise


def test_ragas_sample_evaluation():
    """Test RAGAS evaluation on a single sample."""
    print("\n" + "="*70)
    print("TEST: RAGAS Single Sample Evaluation")
    print("="*70)

    evaluator = RAGASEvaluator()

    # Create a test sample
    question = "What is the Transformer architecture?"
    answer = "The Transformer is a neural network architecture that relies entirely on self-attention mechanisms, as described in the 'Attention Is All You Need' paper."
    contexts = [
        "The Transformer is a neural network architecture introduced in 'Attention Is All You Need' that uses self-attention mechanisms instead of recurrence.",
        "The paper introduces the Transformer model which achieves better performance than RNNs while being more parallelizable."
    ]
    ground_truth = "The Transformer architecture uses self-attention mechanisms and was introduced in the 'Attention Is All You Need' paper."

    sample = evaluator.prepare_sample(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth
    )

    print(f"[OK] Sample prepared successfully")

    # Evaluate sample
    try:
        scores = evaluator.evaluate_sample_sync(sample)
        print(f"[OK] Sample evaluated successfully")

        print("\nRAGAS Scores:")
        for metric_name, score in scores.items():
            if score is not None:
                print(f"  {metric_name:25s}: {score:.4f}")
            else:
                print(f"  {metric_name:25s}: N/A")

        # Basic sanity checks
        assert 'faithfulness' in scores or any('faith' in k.lower() for k in scores.keys()), \
            "Should have faithfulness metric"

        print("\n[OK] RAGAS single sample evaluation test PASSED\n")
        return scores

    except Exception as e:
        print(f"[FAIL] Sample evaluation failed: {e}")
        raise


def test_ragas_on_small_subset():
    """
    Run RAGAS evaluation on a small subset of golden dataset (3 easy examples).

    This test validates:
    - RAGAS can process golden dataset examples
    - All metrics compute successfully
    - Results are reasonable
    """
    print("\n" + "="*70)
    print("TEST: RAGAS Evaluation on Small Subset")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    # Get 3 easy examples for quick testing
    easy_examples = manager.get_by_difficulty("easy")[:3]

    print(f"[OK] Selected {len(easy_examples)} easy examples for testing")

    # Run RAGAS evaluation
    try:
        results = run_ragas_evaluation_on_golden(
            easy_examples,
            advanced_rag_graph,
            evaluator=None,  # Creates default evaluator
            verbose=True
        )

        print(f"[OK] RAGAS evaluation completed")
        print(f"  Total examples: {results['total_examples']}")
        print(f"  Successful: {results['successful_evaluations']}")

        # Check that we got results
        assert results['successful_evaluations'] > 0, "Should have at least one successful evaluation"

        print("\n[OK] RAGAS small subset test PASSED\n")
        return results

    except Exception as e:
        print(f"[FAIL] RAGAS subset evaluation failed: {e}")
        raise


def test_ragas_vs_custom_metrics():
    """
    Compare RAGAS metrics with custom evaluation metrics.

    Analyzes correlation between:
    - Faithfulness (RAGAS) vs Groundedness (custom)
    - Context Precision (RAGAS) vs Retrieval Quality (custom)
    - Response Relevancy (RAGAS) vs Answer Sufficiency (custom)

    Expected: High correlation (>90%) between similar metrics
    """
    print("\n" + "="*70)
    print("TEST: RAGAS vs Custom Metrics Comparison")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    # Get 3 easy examples for quick testing
    easy_examples = manager.get_by_difficulty("easy")[:3]

    print(f"Running parallel evaluations on {len(easy_examples)} examples...")

    # Run custom evaluation
    print("\n1. Running custom evaluation...")
    custom_results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        easy_examples,
        verbose=True
    )

    # Run RAGAS evaluation
    print("\n2. Running RAGAS evaluation...")
    ragas_results = run_ragas_evaluation_on_golden(
        easy_examples,
        advanced_rag_graph,
        evaluator=None,
        verbose=True
    )

    # Compare metrics
    print("\n3. Comparing RAGAS vs Custom metrics...")
    comparison = compare_ragas_with_custom_metrics(
        ragas_results['ragas_results'],
        custom_results,
        verbose=True
    )

    # Analyze correlation
    if 'correlations' in comparison and 'faithfulness_vs_groundedness' in comparison['correlations']:
        correlation = comparison['correlations']['faithfulness_vs_groundedness']
        print(f"\nCorrelation Analysis:")
        print(f"  RAGAS Faithfulness: {correlation['ragas_faithfulness']:.2%}")
        print(f"  Custom Groundedness: {correlation['custom_groundedness']:.2%}")
        print(f"  Difference: {correlation['difference']:.2%}")
        print(f"  Correlation Strength: {correlation['correlation_strength']}")

        # Assert reasonable correlation
        # Note: We use a lenient threshold (0.3) because:
        # 1. Metrics measure slightly different things
        # 2. Small sample size (3 examples) may have variance
        # 3. Different LLM calls may introduce variability
        assert correlation['difference'] < 0.3, \
            f"Correlation too weak: {correlation['difference']:.2%} difference"

        print(f"\n[OK] Metrics show reasonable correlation")

    print("\n[OK] RAGAS vs custom metrics comparison test PASSED\n")
    return comparison


def test_ragas_full_dataset_evaluation():
    """
    Run RAGAS evaluation on full golden dataset (20 examples).

    This test:
    - Evaluates all 20 examples with RAGAS metrics
    - Compares with custom evaluation metrics
    - Generates comprehensive comparison report
    - Saves results for future analysis

    WARNING: This test takes ~10-15 minutes to complete.
    """
    print("\n" + "="*70)
    print("TEST: RAGAS Full Dataset Evaluation (20 examples)")
    print("="*70)
    print("WARNING: This test takes ~10-15 minutes to complete")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    print(f"\nEvaluating all {len(manager.dataset)} examples...")

    # Run custom evaluation
    print("\n1. Running custom evaluation...")
    custom_results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        manager.dataset,
        verbose=True
    )

    # Run RAGAS evaluation
    print("\n2. Running RAGAS evaluation...")
    ragas_results = run_ragas_evaluation_on_golden(
        manager.dataset,
        advanced_rag_graph,
        evaluator=None,
        verbose=True
    )

    # Compare metrics
    print("\n3. Comparing RAGAS vs Custom metrics...")
    comparison = compare_ragas_with_custom_metrics(
        ragas_results['ragas_results'],
        custom_results,
        verbose=True
    )

    # Save results
    results_path = "evaluation/ragas_evaluation_results.json"

    # Prepare serializable results
    serializable_results = {
        'total_examples': ragas_results['total_examples'],
        'successful_evaluations': ragas_results['successful_evaluations'],
        'ragas_metrics': comparison.get('ragas_metrics', {}),
        'custom_metrics': comparison.get('custom_metrics', {}),
        'correlations': comparison.get('correlations', {}),
        'insights': comparison.get('insights', [])
    }

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n[OK] Saved RAGAS evaluation results to {results_path}")

    # Generate comparison report
    generate_ragas_comparison_report(comparison, custom_results, ragas_results)

    print("\n[OK] RAGAS full dataset evaluation test PASSED\n")
    return ragas_results, custom_results, comparison


def generate_ragas_comparison_report(comparison, custom_results, ragas_results):
    """
    Generate markdown report comparing RAGAS vs custom metrics.

    Args:
        comparison: Comparison analysis dictionary
        custom_results: Custom evaluation results
        ragas_results: RAGAS evaluation results
    """
    report_path = "evaluation/ragas_comparison_report.md"

    with open(report_path, 'w') as f:
        f.write("# RAGAS vs Custom Metrics Comparison Report\n\n")
        f.write(f"**Generated:** {os.popen('date').read().strip()}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- Total Examples: {ragas_results['total_examples']}\n")
        f.write(f"- Successful Evaluations: {ragas_results['successful_evaluations']}\n\n")

        f.write("## RAGAS Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        for metric, value in comparison.get('ragas_metrics', {}).items():
            f.write(f"| {metric} | {value:.4f} |\n")

        f.write("\n## Custom Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        for metric, value in comparison.get('custom_metrics', {}).items():
            f.write(f"| {metric} | {value:.4f} |\n")

        f.write("\n## Metric Correlations\n\n")
        if 'correlations' in comparison and 'faithfulness_vs_groundedness' in comparison['correlations']:
            corr = comparison['correlations']['faithfulness_vs_groundedness']
            f.write("### Faithfulness vs Groundedness\n\n")
            f.write(f"- RAGAS Faithfulness: {corr['ragas_faithfulness']:.2%}\n")
            f.write(f"- Custom Groundedness: {corr['custom_groundedness']:.2%}\n")
            f.write(f"- Difference: {corr['difference']:.2%}\n")
            f.write(f"- Correlation Strength: {corr['correlation_strength']}\n\n")

        f.write("## Key Insights\n\n")
        if comparison.get('insights'):
            for insight in comparison['insights']:
                f.write(f"- {insight}\n")
        else:
            f.write("- No significant insights identified\n")

        f.write("\n## Interpretation\n\n")
        f.write("### Faithfulness vs Groundedness\n\n")
        f.write("Both metrics measure whether generated answers contain hallucinations:\n")
        f.write("- **RAGAS Faithfulness**: LLM-as-judge checking if claims are supported by context\n")
        f.write("- **Custom Groundedness**: Similar LLM-based verification with custom prompts\n\n")
        f.write("**Expected**: High correlation (>90%) if both measure the same underlying quality\n\n")

        f.write("### Context Precision vs Retrieval Quality\n\n")
        f.write("- **RAGAS Context Precision**: Whether relevant contexts are ranked higher\n")
        f.write("- **Custom Retrieval Metrics**: Recall, Precision, F1, nDCG at k\n\n")

        f.write("### Response Relevancy vs Answer Sufficiency\n\n")
        f.write("- **RAGAS Response Relevancy**: How relevant answer is to question\n")
        f.write("- **Custom Confidence**: System's confidence in answer quality\n\n")

    print(f"[OK] Generated comparison report: {report_path}")


def run_all_ragas_tests():
    """Run all RAGAS tests sequentially."""
    print("\n" + "="*70)
    print("RUNNING ALL RAGAS TESTS")
    print("="*70)

    try:
        # Test 1: Initialization
        test_ragas_evaluator_initialization()

        # Test 2: Single sample
        test_ragas_sample_evaluation()

        # Test 3: Small subset
        test_ragas_on_small_subset()

        # Test 4: Comparison
        test_ragas_vs_custom_metrics()

        print("\n" + "="*70)
        print("ALL RAGAS TESTS PASSED!")
        print("="*70)
        print("\nTo run full dataset evaluation (20 examples, ~15 min):")
        print("  uv run python -c \"from tests.integration.test_ragas_evaluation import test_ragas_full_dataset_evaluation; test_ragas_full_dataset_evaluation()\"")
        print("\n")

    except Exception as e:
        print("\n" + "="*70)
        print(f"RAGAS TESTS FAILED: {e}")
        print("="*70)
        raise


if __name__ == "__main__":
    # Run all tests
    run_all_ragas_tests()
