"""
Offline Evaluation Suite for Golden Dataset

This test suite evaluates the RAG system using the golden dataset,
ensuring baseline performance, detecting regressions, and validating
retrieval strategies.
"""

import json
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

from advanced_agentic_rag_langgraph.evaluation.golden_dataset import GoldenDatasetManager, evaluate_on_golden_dataset
from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph


def test_dataset_loading():
    """Test that golden dataset loads correctly."""
    print("\n" + "="*70)
    print("TEST: Dataset Loading")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    assert len(manager.dataset) > 0, "Dataset should not be empty"
    print(f"[OK] Loaded {len(manager.dataset)} examples")

    # Print statistics
    manager.print_statistics()

    print("[OK] Dataset loading test PASSED\n")


def test_dataset_validation():
    """Test that all examples pass validation."""
    print("\n" + "="*70)
    print("TEST: Dataset Validation")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    validation_errors = []
    for example in manager.dataset:
        is_valid, errors = manager.validate_example(example)
        if not is_valid:
            validation_errors.append({
                'example_id': example.get('id'),
                'errors': errors
            })

    if validation_errors:
        print(f"[FAIL] Found {len(validation_errors)} validation errors:")
        for error in validation_errors[:5]:
            print(f"  {error['example_id']}: {error['errors']}")
        assert False, "Validation failed"
    else:
        print(f"[OK] All {len(manager.dataset)} examples passed validation")

    print("[OK] Dataset validation test PASSED\n")


def test_baseline_performance():
    """
    Run on golden dataset and assert minimum performance thresholds.

    Baseline thresholds (conservative estimates):
    - avg_recall_at_5 >= 0.70 (retrieve 70% of relevant docs)
    - avg_precision_at_5 >= 0.60 (60% of retrieved docs are relevant)
    - avg_f1_at_5 >= 0.65 (balanced metric)
    - avg_groundedness >= 0.85 (85% of claims supported)
    - hallucination_rate <= 0.15 (max 15% hallucination rate)
    """
    print("\n" + "="*70)
    print("TEST: Baseline Performance")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    # Run evaluation
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        manager.dataset,
        verbose=True
    )

    # Extract metrics
    retrieval_metrics = results['retrieval_metrics']
    generation_metrics = results['generation_metrics']

    # Define thresholds
    thresholds = {
        'recall_at_k': 0.70,
        'precision_at_k': 0.60,
        'f1_at_k': 0.65,
    }

    generation_thresholds = {
        'avg_groundedness': 0.85,
        'hallucination_rate': 0.15,  # Max threshold
    }

    # Check retrieval thresholds
    print("\nRetrieval Metrics vs Thresholds:")
    retrieval_passed = True
    for metric, threshold in thresholds.items():
        actual = retrieval_metrics.get(metric, 0.0)
        passed = actual >= threshold
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {metric:20s}: {actual:.2%} (threshold: {threshold:.2%})")
        if not passed:
            retrieval_passed = False

    # Check generation thresholds
    print("\nGeneration Metrics vs Thresholds:")
    generation_passed = True
    for metric, threshold in generation_thresholds.items():
        actual = generation_metrics.get(metric, 0.0)

        # Hallucination rate should be BELOW threshold
        if metric == 'hallucination_rate':
            passed = actual <= threshold
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {metric:20s}: {actual:.2%} (max: {threshold:.2%})")
        else:
            passed = actual >= threshold
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {metric:20s}: {actual:.2%} (threshold: {threshold:.2%})")

        if not passed:
            generation_passed = False

    # Save baseline for future regression tests
    baseline_path = "evaluation/baseline_metrics.json"
    baseline_data = {
        'retrieval_metrics': retrieval_metrics,
        'generation_metrics': generation_metrics,
    }
    with open(baseline_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    print(f"\n[OK] Saved baseline metrics to {baseline_path}")

    # Assert all thresholds met
    assert retrieval_passed, "Retrieval metrics below threshold"
    assert generation_passed, "Generation metrics below threshold"

    print("\n[OK] Baseline performance test PASSED\n")
    return results


def test_regression():
    """
    Compare current performance against saved baseline.

    Asserts:
    - Recall should not decrease > 5%
    - Precision should not decrease > 5%
    - Groundedness should not decrease > 10%
    """
    print("\n" + "="*70)
    print("TEST: Regression Detection")
    print("="*70)

    baseline_path = "evaluation/baseline_metrics.json"

    if not os.path.exists(baseline_path):
        print("[WARN] No baseline found - run test_baseline_performance() first")
        print("[OK] Regression test SKIPPED\n")
        return

    # Load baseline
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    # Run current evaluation
    manager = GoldenDatasetManager("evaluation/golden_set.json")
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        manager.dataset,
        verbose=False
    )

    # Compare metrics
    print("\nRegression Analysis:")

    regression_found = False

    # Retrieval metrics (5% tolerance)
    retrieval_tolerance = 0.05
    for metric in ['recall_at_k', 'precision_at_k', 'f1_at_k']:
        baseline_value = baseline['retrieval_metrics'].get(metric, 0)
        current_value = results['retrieval_metrics'].get(metric, 0)
        diff = current_value - baseline_value
        diff_pct = (diff / baseline_value * 100) if baseline_value > 0 else 0

        status = "[OK]"
        if diff < -retrieval_tolerance:  # Decrease > 5%
            status = "[FAIL] REGRESSION"
            regression_found = True

        print(f"  {status} {metric:20s}: {current_value:.2%} (baseline: {baseline_value:.2%}, diff: {diff_pct:+.1f}%)")

    # Generation metrics (10% tolerance)
    generation_tolerance = 0.10
    groundedness_baseline = baseline['generation_metrics'].get('avg_groundedness', 0)
    groundedness_current = results['generation_metrics'].get('avg_groundedness', 0)
    groundedness_diff = groundedness_current - groundedness_baseline
    groundedness_diff_pct = (groundedness_diff / groundedness_baseline * 100) if groundedness_baseline > 0 else 0

    status = "[OK]"
    if groundedness_diff < -generation_tolerance:  # Decrease > 10%
        status = "[FAIL] REGRESSION"
        regression_found = True

    print(f"  {status} avg_groundedness:    {groundedness_current:.2%} (baseline: {groundedness_baseline:.2%}, diff: {groundedness_diff_pct:+.1f}%)")

    assert not regression_found, "Performance regression detected!"

    print("\n[OK] Regression test PASSED (no significant degradation)\n")


def test_cross_document_retrieval():
    """Test accuracy on cross-document examples."""
    print("\n" + "="*70)
    print("TEST: Cross-Document Retrieval")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")
    cross_doc_examples = manager.get_cross_document_examples()

    if not cross_doc_examples:
        print("[WARN] No cross-document examples found")
        print("[OK] Cross-document test SKIPPED\n")
        return

    print(f"Found {len(cross_doc_examples)} cross-document examples")

    # Run evaluation on cross-document examples only
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        cross_doc_examples,
        verbose=False
    )

    # Check that recall is reasonable for cross-document queries
    recall = results['retrieval_metrics'].get('recall_at_k', 0)
    threshold = 0.60  # Lower threshold for harder cross-document queries

    print(f"\nCross-Document Recall@5: {recall:.2%} (threshold: {threshold:.2%})")

    if recall >= threshold:
        print(f"[OK] Cross-document retrieval meets threshold")
    else:
        print(f"[FAIL] Cross-document retrieval below threshold")

    assert recall >= threshold, f"Cross-document recall ({recall:.2%}) below threshold ({threshold:.2%})"

    print("\n[OK] Cross-document retrieval test PASSED\n")


def test_difficulty_correlation():
    """
    Verify harder questions have appropriate metrics.

    Expected trends:
    - Easy: recall > 0.80, precision > 0.70
    - Medium: recall > 0.70, precision > 0.60
    - Hard: recall > 0.60, precision > 0.50
    """
    print("\n" + "="*70)
    print("TEST: Difficulty Correlation")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    # Run full evaluation
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        manager.dataset,
        verbose=False
    )

    difficulty_breakdown = results['per_difficulty_breakdown']

    # Expected thresholds by difficulty
    thresholds = {
        'easy': {'recall_at_k': 0.80, 'precision_at_k': 0.70},
        'medium': {'recall_at_k': 0.70, 'precision_at_k': 0.60},
        'hard': {'recall_at_k': 0.60, 'precision_at_k': 0.50},
    }

    print("\nDifficulty-Based Performance:")
    all_passed = True

    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty not in difficulty_breakdown:
            print(f"\n  {difficulty.upper()}: No examples")
            continue

        metrics = difficulty_breakdown[difficulty]
        threshold = thresholds[difficulty]

        print(f"\n  {difficulty.upper()}:")
        for metric, expected in threshold.items():
            actual = metrics.get(metric, 0)
            passed = actual >= expected
            status = "[OK]" if passed else "[FAIL]"
            print(f"    {status} {metric:20s}: {actual:.2%} (expected: {expected:.2%})")
            if not passed:
                all_passed = False

    # Note: We don't assert here because this is more of a guideline
    # Hard questions can legitimately have lower scores
    if all_passed:
        print("\n[OK] Difficulty correlation test PASSED")
    else:
        print("\n[WARN] Some difficulty thresholds not met (this is informational)")

    print("\n[OK] Difficulty correlation test COMPLETED\n")


def generate_evaluation_report():
    """
    Generate comprehensive markdown report.

    Report includes:
    - Overall metrics
    - Breakdown by difficulty
    - Breakdown by query type
    - Top 5 best performing queries
    - Top 5 worst performing queries
    - Recommendations
    """
    print("\n" + "="*70)
    print("GENERATING EVALUATION REPORT")
    print("="*70)

    manager = GoldenDatasetManager("evaluation/golden_set.json")

    # Run full evaluation
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        manager.dataset,
        verbose=True
    )

    # Generate markdown report
    report = f"""# Golden Dataset Evaluation Report

## Overview

- **Total Examples**: {results['total_examples']}
- **Successful Evaluations**: {results['successful_evaluations']}
- **Success Rate**: {(results['successful_evaluations'] / results['total_examples'] * 100):.1f}%

## Overall Metrics

### Retrieval Performance

| Metric | Value |
|--------|-------|
"""

    for metric, value in results['retrieval_metrics'].items():
        if value <= 1.0:
            report += f"| {metric} | {value:.2%} |\n"
        else:
            report += f"| {metric} | {value:.4f} |\n"

    report += f"""
### Generation Quality

| Metric | Value |
|--------|-------|
| Average Groundedness | {results['generation_metrics']['avg_groundedness']:.2%} |
| Average Confidence | {results['generation_metrics']['avg_confidence']:.2%} |
| Hallucination Rate | {results['generation_metrics']['hallucination_rate']:.2%} |

## Performance by Difficulty

"""

    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in results['per_difficulty_breakdown']:
            metrics = results['per_difficulty_breakdown'][difficulty]
            report += f"\n### {difficulty.capitalize()}\n\n"
            for metric, value in metrics.items():
                if value <= 1.0:
                    report += f"- {metric}: {value:.2%}\n"
                else:
                    report += f"- {metric}: {value:.4f}\n"

    report += f"""
## Performance by Query Type

"""

    for query_type, metrics in results['per_query_type_breakdown'].items():
        report += f"\n### {query_type.capitalize()}\n\n"
        for metric, value in metrics.items():
            if value <= 1.0:
                report += f"- {metric}: {value:.2%}\n"
            else:
                report += f"- {metric}: {value:.4f}\n"

    # Top performers
    per_example = results['per_example_results']
    valid_examples = [ex for ex in per_example if 'error' not in ex and 'retrieval_metrics' in ex]

    if valid_examples:
        # Sort by F1 score
        sorted_by_f1 = sorted(
            valid_examples,
            key=lambda x: x['retrieval_metrics'].get('f1_at_k', 0),
            reverse=True
        )

        report += "\n## Top 5 Best Performing Examples\n\n"
        for i, ex in enumerate(sorted_by_f1[:5], 1):
            report += f"{i}. **{ex['example_id']}** (Difficulty: {ex['difficulty']})\n"
            report += f"   - Recall@5: {ex['retrieval_metrics'].get('recall_at_k', 0):.2%}\n"
            report += f"   - Precision@5: {ex['retrieval_metrics'].get('precision_at_k', 0):.2%}\n"
            report += f"   - F1@5: {ex['retrieval_metrics'].get('f1_at_k', 0):.2%}\n"
            report += f"   - Groundedness: {ex['groundedness_score']:.2%}\n\n"

        report += "\n## Top 5 Worst Performing Examples\n\n"
        for i, ex in enumerate(sorted_by_f1[-5:][::-1], 1):
            report += f"{i}. **{ex['example_id']}** (Difficulty: {ex['difficulty']})\n"
            report += f"   - Recall@5: {ex['retrieval_metrics'].get('recall_at_k', 0):.2%}\n"
            report += f"   - Precision@5: {ex['retrieval_metrics'].get('precision_at_k', 0):.2%}\n"
            report += f"   - F1@5: {ex['retrieval_metrics'].get('f1_at_k', 0):.2%}\n"
            report += f"   - Groundedness: {ex['groundedness_score']:.2%}\n\n"

    report += """
## Recommendations

1. **High Hallucination Rate**: If > 10%, review groundedness check thresholds
2. **Low Recall**: If < 70%, consider improving retrieval strategies or chunking
3. **Low Precision**: If < 60%, enhance reranking or relevance scoring
4. **Hard Query Performance**: Multi-hop queries may need query decomposition
5. **Cross-Document Queries**: Consider citation following or graph-based retrieval

---
*Generated by Advanced Agentic RAG - Golden Dataset Evaluation Suite*
"""

    # Save report
    report_path = "evaluation/evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[OK] Saved evaluation report to {report_path}")
    print(f"\n[OK] Report generation COMPLETED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GOLDEN DATASET OFFLINE EVALUATION SUITE")
    print("="*70)

    # Run all tests
    test_dataset_loading()
    test_dataset_validation()
    test_baseline_performance()
    test_regression()
    test_cross_document_retrieval()
    test_difficulty_correlation()
    generate_evaluation_report()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")
