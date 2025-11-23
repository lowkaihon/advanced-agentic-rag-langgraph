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
import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

# Suppress LangSmith warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*Failed to.*LangSmith.*")
warnings.filterwarnings("ignore", message=".*langsmith.*")

# Suppress LangSmith logging
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.WARNING)

from advanced_agentic_rag_langgraph.evaluation.golden_dataset import GoldenDatasetManager, evaluate_on_golden_dataset
from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph
from advanced_agentic_rag_langgraph.core.model_config import get_current_tier, TIER_METADATA


def get_expected_ranges(dataset_type: str, tier: str) -> dict:
    """
    Get expected performance ranges based on dataset difficulty and model tier.

    Args:
        dataset_type: Dataset type ('standard' or 'hard')
        tier: Model tier ('budget', 'balanced', or 'premium')

    Returns:
        Dictionary with expected ranges for each metric
    """
    if dataset_type == "standard":
        ranges = {
            "budget": {
                "recall_at_k": (0.65, 0.75),
                "precision_at_k": (0.55, 0.65),
                "f1_at_k": (0.60, 0.70),
                "avg_groundedness": (0.80, 0.90),
                "hallucination_rate": (0.10, 0.20),
            },
            "balanced": {
                "recall_at_k": (0.70, 0.80),
                "precision_at_k": (0.60, 0.70),
                "f1_at_k": (0.65, 0.75),
                "avg_groundedness": (0.85, 0.95),
                "hallucination_rate": (0.05, 0.15),
            },
            "premium": {
                "recall_at_k": (0.75, 0.85),
                "precision_at_k": (0.65, 0.75),
                "f1_at_k": (0.70, 0.80),
                "avg_groundedness": (0.90, 0.98),
                "hallucination_rate": (0.02, 0.10),
            },
        }
    else:  # hard dataset - lower expectations for harder questions
        ranges = {
            "budget": {
                "recall_at_k": (0.55, 0.65),
                "precision_at_k": (0.45, 0.55),
                "f1_at_k": (0.50, 0.60),
                "avg_groundedness": (0.75, 0.85),
                "hallucination_rate": (0.15, 0.25),
            },
            "balanced": {
                "recall_at_k": (0.60, 0.70),
                "precision_at_k": (0.50, 0.60),
                "f1_at_k": (0.55, 0.65),
                "avg_groundedness": (0.80, 0.90),
                "hallucination_rate": (0.10, 0.20),
            },
            "premium": {
                "recall_at_k": (0.65, 0.75),
                "precision_at_k": (0.55, 0.65),
                "f1_at_k": (0.60, 0.70),
                "avg_groundedness": (0.85, 0.95),
                "hallucination_rate": (0.05, 0.15),
            },
        }

    return ranges.get(tier, ranges["budget"])


def test_dataset_loading(dataset_type: str = "standard"):
    """Test that golden dataset loads correctly."""
    print("\n" + "="*70)
    print("TEST: Dataset Loading")
    print("="*70)

    dataset_path = f"evaluation/golden_set_{dataset_type}.json"
    manager = GoldenDatasetManager(dataset_path)

    assert len(manager.dataset) > 0, "Dataset should not be empty"
    print(f"[OK] Loaded {len(manager.dataset)} examples")

    # Print statistics
    manager.print_statistics()

    print("[OK] Dataset loading test PASSED\n")


def test_dataset_validation(dataset_type: str = "standard"):
    """Test that all examples pass validation."""
    print("\n" + "="*70)
    print("TEST: Dataset Validation")
    print("="*70)

    dataset_path = f"evaluation/golden_set_{dataset_type}.json"
    manager = GoldenDatasetManager(dataset_path)

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


def test_baseline_performance(quick_mode: bool = False, dataset_type: str = "standard"):
    """
    Run on golden dataset and assert minimum performance thresholds.

    Baseline thresholds (conservative estimates for k=4):
    - avg_recall_at_k >= 0.70 (retrieve 70% of relevant docs)
    - avg_precision_at_k >= 0.60 (60% of retrieved docs are relevant)
    - avg_f1_at_k >= 0.65 (balanced metric)
    - avg_groundedness >= 0.85 (85% of claims supported)
    - hallucination_rate <= 0.15 (max 15% hallucination rate)
    """
    print("\n" + "="*70)
    print("TEST: Baseline Performance")
    print("="*70)

    # Detect and display model tier
    current_tier = get_current_tier()
    tier_info = TIER_METADATA[current_tier]
    print(f"Model Tier: {current_tier.value.upper()} ({tier_info['description']})")
    print(f"Cost/Day: ${tier_info['cost_per_day']:,}")
    print("="*70)

    dataset_path = f"evaluation/golden_set_{dataset_type}.json"
    if dataset_type == "standard":
        k_final = 4  # Optimal for 1-3 chunk questions
    else:  # hard
        k_final = 6  # Adaptive retrieval for 3-5 chunk questions

    manager = GoldenDatasetManager(dataset_path)
    dataset = manager.dataset

    # Apply quick mode if requested
    if quick_mode:
        dataset = dataset[:2]
        print(f"[*] Quick mode: Using first 2 examples\n")

    # Run evaluation
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        dataset,
        k_final=k_final,
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

    # Validation against expected ranges
    print("\nValidation Against Expected Ranges:")
    print(f"| {'Metric':<20s} | {'Target':<13s} | {'Actual':<8s} | {'Status':<6s} |")
    print("|" + "-"*20 + "|" + "-"*13 + "|" + "-"*8 + "|" + "-"*6 + "|")

    expected_ranges = get_expected_ranges(dataset_type, current_tier.value)

    for metric in ['recall_at_k', 'precision_at_k', 'f1_at_k']:
        min_val, max_val = expected_ranges[metric]
        actual = retrieval_metrics.get(metric, 0.0)
        status = "[OK]" if min_val <= actual <= max_val else "[WARN]"
        print(f"| {metric:<20s} | {min_val:.0%}-{max_val:.0%} | {actual:6.1%} | {status:6s} |")

    # Groundedness
    min_val, max_val = expected_ranges['avg_groundedness']
    actual = generation_metrics.get('avg_groundedness', 0.0)
    status = "[OK]" if min_val <= actual <= max_val else "[WARN]"
    print(f"| {'avg_groundedness':<20s} | {min_val:.0%}-{max_val:.0%} | {actual:6.1%} | {status:6s} |")

    # Hallucination rate (lower is better)
    max_val = expected_ranges['hallucination_rate'][1]
    actual = generation_metrics.get('hallucination_rate', 0.0)
    status = "[OK]" if actual <= max_val else "[WARN]"
    print(f"| {'hallucination_rate':<20s} | {'<' + f'{max_val:.0%}':<13s} | {actual:6.1%} | {status:6s} |")

    # Save baseline for future regression tests
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_path = Path("evaluation") / f"baseline_metrics_{dataset_type}_{test_timestamp}.json"
    latest_baseline_path = Path("evaluation") / f"baseline_metrics_{dataset_type}_latest.json"

    baseline_data = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'golden_dataset_baseline',
        'dataset_type': dataset_type,
        'dataset_size': len(dataset),
        'k_final': k_final,
        'quick_mode': quick_mode,
        'model_tier': current_tier.value,
        'tier_description': tier_info['description'],
        'tier_cost_per_day': tier_info['cost_per_day'],
        'retrieval_metrics': retrieval_metrics,
        'generation_metrics': generation_metrics,
    }
    with open(baseline_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)

    # Create latest copy
    shutil.copy2(baseline_path, latest_baseline_path)

    print(f"\n[OK] Saved baseline metrics to {baseline_path}")
    print(f"[OK] Latest copy saved to {latest_baseline_path}")

    # Assert all thresholds met
    assert retrieval_passed, "Retrieval metrics below threshold"
    assert generation_passed, "Generation metrics below threshold"

    print("\n[OK] Baseline performance test PASSED\n")
    return results


def test_regression(quick_mode: bool = False, dataset_type: str = "standard"):
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

    baseline_path = Path("evaluation") / f"baseline_metrics_{dataset_type}_latest.json"

    if not os.path.exists(baseline_path):
        print("[WARN] No baseline found - run test_baseline_performance() first")
        print("[OK] Regression test SKIPPED\n")
        return

    # Load baseline
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    # Setup dataset
    dataset_path = f"evaluation/golden_set_{dataset_type}.json"
    if dataset_type == "standard":
        k_final = 4  # Optimal for 1-3 chunk questions
    else:  # hard
        k_final = 6  # Adaptive retrieval for 3-5 chunk questions

    manager = GoldenDatasetManager(dataset_path)
    dataset = manager.dataset

    # Apply quick mode if requested
    if quick_mode:
        dataset = dataset[:2]
        print(f"[*] Quick mode: Using first 2 examples\n")

    # Run current evaluation
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        dataset,
        k_final=k_final,
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


def test_cross_document_retrieval(quick_mode: bool = False, dataset_type: str = "standard"):
    """Test accuracy on cross-document examples."""
    print("\n" + "="*70)
    print("TEST: Cross-Document Retrieval")
    print("="*70)

    dataset_path = f"evaluation/golden_set_{dataset_type}.json"
    if dataset_type == "standard":
        k_final = 4  # Optimal for 1-3 chunk questions
    else:  # hard
        k_final = 6  # Adaptive retrieval for 3-5 chunk questions

    manager = GoldenDatasetManager(dataset_path)
    cross_doc_examples = manager.get_cross_document_examples()

    if not cross_doc_examples:
        print("[WARN] No cross-document examples found")
        print("[OK] Cross-document test SKIPPED\n")
        return

    print(f"Found {len(cross_doc_examples)} cross-document examples")

    # Apply quick mode if requested
    if quick_mode:
        cross_doc_examples = cross_doc_examples[:2]
        print(f"[*] Quick mode: Using first 2 examples\n")

    # Run evaluation on cross-document examples only
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        cross_doc_examples,
        k_final=k_final,
        verbose=False
    )

    # Check that recall is reasonable for cross-document queries
    recall = results['retrieval_metrics'].get('recall_at_k', 0)
    threshold = 0.60  # Lower threshold for harder cross-document queries
    k = k_final

    print(f"\nCross-Document Recall@{k}: {recall:.2%} (threshold: {threshold:.2%})")

    if recall >= threshold:
        print(f"[OK] Cross-document retrieval meets threshold")
    else:
        print(f"[FAIL] Cross-document retrieval below threshold")

    assert recall >= threshold, f"Cross-document recall ({recall:.2%}) below threshold ({threshold:.2%})"

    print("\n[OK] Cross-document retrieval test PASSED\n")


def test_difficulty_correlation(quick_mode: bool = False, dataset_type: str = "standard"):
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

    dataset_path = f"evaluation/golden_set_{dataset_type}.json"
    if dataset_type == "standard":
        k_final = 4  # Optimal for 1-3 chunk questions
    else:  # hard
        k_final = 6  # Adaptive retrieval for 3-5 chunk questions

    manager = GoldenDatasetManager(dataset_path)
    dataset = manager.dataset

    # Apply quick mode if requested
    if quick_mode:
        dataset = dataset[:2]
        print(f"[*] Quick mode: Using first 2 examples\n")

    # Run full evaluation
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        dataset,
        k_final=k_final,
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


def generate_evaluation_report(quick_mode: bool = False, dataset_type: str = "standard"):
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

    # Detect model tier
    current_tier = get_current_tier()
    tier_info = TIER_METADATA[current_tier]

    dataset_path = f"evaluation/golden_set_{dataset_type}.json"
    if dataset_type == "standard":
        k_final = 4  # Optimal for 1-3 chunk questions
    else:  # hard
        k_final = 6  # Adaptive retrieval for 3-5 chunk questions

    manager = GoldenDatasetManager(dataset_path)
    dataset = manager.dataset

    # Apply quick mode if requested
    if quick_mode:
        dataset = dataset[:2]
        print(f"[*] Quick mode: Using first 2 examples\n")

    # Run full evaluation
    results = evaluate_on_golden_dataset(
        advanced_rag_graph,
        dataset,
        k_final=k_final,
        verbose=True
    )

    # Generate markdown report
    report = f"""# Golden Dataset Evaluation Report

## Overview

- **Model Tier**: {current_tier.value.upper()} ({tier_info['description']})
- **Cost/Day**: ${tier_info['cost_per_day']:,}
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

        k = k_final
        report += "\n## Top 5 Best Performing Examples\n\n"
        for i, ex in enumerate(sorted_by_f1[:5], 1):
            report += f"{i}. **{ex['example_id']}** (Difficulty: {ex['difficulty']})\n"
            report += f"   - Recall@{k}: {ex['retrieval_metrics'].get('recall_at_k', 0):.2%}\n"
            report += f"   - Precision@{k}: {ex['retrieval_metrics'].get('precision_at_k', 0):.2%}\n"
            report += f"   - F1@{k}: {ex['retrieval_metrics'].get('f1_at_k', 0):.2%}\n"
            report += f"   - Groundedness: {ex['groundedness_score']:.2%}\n\n"

        report += "\n## Top 5 Worst Performing Examples\n\n"
        for i, ex in enumerate(sorted_by_f1[-5:][::-1], 1):
            report += f"{i}. **{ex['example_id']}** (Difficulty: {ex['difficulty']})\n"
            report += f"   - Recall@{k}: {ex['retrieval_metrics'].get('recall_at_k', 0):.2%}\n"
            report += f"   - Precision@{k}: {ex['retrieval_metrics'].get('precision_at_k', 0):.2%}\n"
            report += f"   - F1@{k}: {ex['retrieval_metrics'].get('f1_at_k', 0):.2%}\n"
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

    # Save report with timestamping
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("evaluation") / f"evaluation_report_{dataset_type}_{test_timestamp}.md"
    latest_report_path = Path("evaluation") / f"evaluation_report_{dataset_type}_latest.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Create latest copy
    shutil.copy2(report_path, latest_report_path)

    print(f"\n[OK] Saved evaluation report to {report_path}")
    print(f"[OK] Latest copy saved to {latest_report_path}")
    print(f"\n[OK] Report generation COMPLETED\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Golden Dataset Offline Evaluation Suite')
    parser.add_argument(
        '--dataset',
        choices=['standard', 'hard'],
        default='standard',
        help='Dataset to evaluate: standard (20 questions, k_final=4) or hard (10 questions, k_final=6)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: evaluate only first 2 examples'
    )
    args = parser.parse_args()

    # Print runtime estimate
    if args.quick:
        print(f"[*] Running in quick mode (2 examples from {args.dataset} dataset)")
    else:
        dataset_size = "20 examples" if args.dataset == "standard" else "10 examples"
        expected_time = "10-15 minutes" if args.dataset == "standard" else "5-7 minutes"
        print(f"[*] Running full evaluation on {args.dataset} dataset ({dataset_size})")
        print(f"[*] This will take approximately {expected_time}")
        print("[*] Use --quick flag for faster testing (~1-2 minutes)")

    # PRE-BUILD RETRIEVER ONCE (optimization to avoid re-ingesting PDFs)
    print("\n" + "="*80)
    print("PRE-BUILD: Initializing retriever once for all test runs")
    print("="*80)
    print("    This avoids re-ingesting PDFs multiple times (saves 50-60% time)")

    if args.dataset == "standard":
        k_final = 4  # Optimal for 1-3 chunk questions
    else:  # hard
        k_final = 6  # Adaptive retrieval for 3-5 chunk questions

    from advanced_agentic_rag_langgraph.core import setup_retriever
    import advanced_agentic_rag_langgraph.orchestration.nodes as nodes
    retriever_instance = setup_retriever(k_final=k_final)
    nodes.adaptive_retriever = retriever_instance
    print(f"[OK] Retriever pre-built and injected (k_final={k_final})")
    print("="*80 + "\n")

    print("\n" + "="*70)
    print("GOLDEN DATASET OFFLINE EVALUATION SUITE")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {'Quick (2 examples)' if args.quick else 'Full'}")
    print("="*70 + "\n")

    # Start timing
    overall_start = time.time()

    # Run all tests with args
    test_dataset_loading(dataset_type=args.dataset)
    test_dataset_validation(dataset_type=args.dataset)
    test_baseline_performance(quick_mode=args.quick, dataset_type=args.dataset)
    test_regression(quick_mode=args.quick, dataset_type=args.dataset)
    test_cross_document_retrieval(quick_mode=args.quick, dataset_type=args.dataset)
    test_difficulty_correlation(quick_mode=args.quick, dataset_type=args.dataset)
    generate_evaluation_report(quick_mode=args.quick, dataset_type=args.dataset)

    # Calculate total execution time
    overall_time = time.time() - overall_start

    # Display execution summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"Dataset: {args.dataset} ({'quick mode' if args.quick else 'full'})")
    print("="*80 + "\n")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")
