"""
Multi-Agent RAG Golden Dataset Evaluation.

Evaluates the multi-agent RAG variant on golden dataset with:
- Standard retrieval metrics (F1@K, Precision@K, Recall@K)
- Multi-agent specific metrics (sub-query count, cross-agent fusion stats)
- Generation metrics (groundedness, semantic similarity, factual accuracy, completeness)

Key Differences from Advanced RAG:
- Query decomposition into 2-4 sub-queries
- Parallel retrieval workers (one per sub-query)
- Cross-agent RRF fusion with multi-agent boost
- top-6 final documents (vs top-4 in advanced)

Usage:
    uv run python tests/integration/test_multi_agent_evaluation.py
    uv run python tests/integration/test_multi_agent_evaluation.py --quick
    uv run python tests/integration/test_multi_agent_evaluation.py --dataset hard

Outputs:
    - evaluation/multi_agent_evaluation_results_{dataset}_{timestamp}.json
    - evaluation/multi_agent_evaluation_report_{dataset}_{timestamp}.md
    - evaluation/multi_agent_evaluation_results_{dataset}_latest.json
    - evaluation/multi_agent_evaluation_report_{dataset}_latest.md
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
from typing import Dict, List, Any
from collections import defaultdict

# Suppress LangSmith warnings - MUST be set BEFORE importing LangChain modules
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*Failed to.*LangSmith.*")
warnings.filterwarnings("ignore", message=".*langsmith.*")

# Suppress LangSmith logging
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.WARNING)

from advanced_agentic_rag_langgraph.variants import multi_agent_rag_graph
import advanced_agentic_rag_langgraph.variants.multi_agent_rag_graph as multi_agent_module
from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.evaluation.golden_dataset import GoldenDatasetManager, compare_answers
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics
from advanced_agentic_rag_langgraph.validation import HHEMHallucinationDetector
from advanced_agentic_rag_langgraph.core.model_config import get_current_tier, TIER_METADATA


# ========== EVALUATION FUNCTIONS ==========

def run_multi_agent_on_golden_dataset(
    dataset: List[Dict],
    k_final: int = 6,
    verbose: bool = True
) -> List[Dict]:
    """
    Run multi-agent RAG on golden dataset.

    Args:
        dataset: List of golden examples
        k_final: Number of documents for metrics calculation
        verbose: Print progress

    Returns:
        List of result dictionaries with metrics per example
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running MULTI-AGENT RAG on Golden Dataset")
        print(f"{'='*70}")
        print(f"Examples: {len(dataset)}")
        print(f"k_final: {k_final}")
        print(f"{'='*70}\n")

    results = []
    hhem_detector = HHEMHallucinationDetector()

    for i, example in enumerate(dataset, 1):
        query = example["question"]
        ground_truth_docs = example["relevant_doc_ids"]

        if verbose:
            print(f"\n[{i}/{len(dataset)}] Processing: {example.get('id', query[:50])}")
            print(f"  Query: {query[:80]}...")

        try:
            # Initialize state for multi-agent graph
            initial_state = {
                "user_question": query,
                "baseline_query": query,
                "messages": [],
                "sub_agent_results": [],  # Required for operator.add reducer
                "ground_truth_doc_ids": ground_truth_docs,
            }

            # Run graph
            result = multi_agent_rag_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"multi_agent_{i}"}}
            )

            # Extract results
            answer = result.get("final_answer", "")
            retrieved_docs = result.get("unique_docs_list", [])
            multi_agent_metrics = result.get("multi_agent_metrics", {})
            sub_queries = result.get("sub_queries", [])

            # Calculate retrieval metrics
            metrics = calculate_retrieval_metrics(retrieved_docs, ground_truth_docs, k_final)
            f1_at_k = metrics["f1_at_k"]
            precision = metrics["precision_at_k"]
            recall = metrics["recall_at_k"]

            # Calculate groundedness using HHEM (per-chunk verification)
            if retrieved_docs and answer:
                chunks = [doc.page_content for doc in retrieved_docs[:k_final]]
                groundedness_result = hhem_detector.verify_groundedness(answer, chunks)
                groundedness_score = groundedness_result["groundedness_score"]
            else:
                groundedness_score = 0.0

            # Calculate answer quality vs ground truth
            if answer:
                answer_comparison = compare_answers(
                    generated=answer,
                    ground_truth=example.get("ground_truth_answer", "")
                )
                semantic_similarity = answer_comparison.get("semantic_similarity", 0.0)
                factual_accuracy = answer_comparison.get("factual_accuracy", 0.0)
                completeness = answer_comparison.get("completeness", 0.0)
            else:
                semantic_similarity = 0.0
                factual_accuracy = 0.0
                completeness = 0.0

            if verbose:
                num_workers = multi_agent_metrics.get("workers", 0)
                multi_docs = multi_agent_metrics.get("multi_agent_docs", 0)
                print(f"  Sub-queries: {len(sub_queries)} | Workers: {num_workers} | Multi-agent docs: {multi_docs}")
                print(f"  F1@{k_final}: {f1_at_k:.0%} | Ground: {groundedness_score:.0%} | Sim: {semantic_similarity:.0%} | Fact: {factual_accuracy:.0%}")

            # Extract doc IDs for storage
            retrieved_doc_ids = [
                doc.metadata.get('id', f'doc_{j}')
                for j, doc in enumerate(retrieved_docs[:k_final])
            ]

            results.append({
                "example_id": example.get("id", query[:30]),
                "query": query,
                "answer": answer,
                "ground_truth_docs": ground_truth_docs,
                "retrieved_docs": retrieved_doc_ids,
                "f1_at_k": f1_at_k,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "groundedness_score": groundedness_score,
                "semantic_similarity": semantic_similarity,
                "factual_accuracy": factual_accuracy,
                "completeness": completeness,
                # Multi-agent specific
                "sub_queries": sub_queries,
                "num_sub_queries": len(sub_queries),
                "num_workers": multi_agent_metrics.get("workers", 0),
                "total_unique_docs": multi_agent_metrics.get("total_unique_docs", 0),
                "multi_agent_docs": multi_agent_metrics.get("multi_agent_docs", 0),
                "avg_worker_quality": multi_agent_metrics.get("avg_quality", 0.0),
            })

        except Exception as e:
            print(f"  [ERROR] Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "example_id": example.get("id", query[:30]),
                "query": query,
                "error": str(e),
                "f1_at_k": 0.0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "groundedness_score": 0.0,
                "semantic_similarity": 0.0,
                "factual_accuracy": 0.0,
                "completeness": 0.0,
                "sub_queries": [],
                "num_sub_queries": 0,
                "num_workers": 0,
                "total_unique_docs": 0,
                "multi_agent_docs": 0,
                "avg_worker_quality": 0.0,
            })

    return results


def calculate_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate aggregate metrics from per-example results.

    Args:
        results: List of per-example results

    Returns:
        Dictionary of aggregate metrics
    """
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {
            "avg_f1_at_k": 0.0,
            "avg_precision_at_k": 0.0,
            "avg_recall_at_k": 0.0,
            "avg_groundedness": 0.0,
            "avg_semantic_similarity": 0.0,
            "avg_factual_accuracy": 0.0,
            "avg_completeness": 0.0,
            "avg_sub_queries": 0.0,
            "avg_workers": 0.0,
            "avg_total_unique_docs": 0.0,
            "avg_multi_agent_docs": 0.0,
            "multi_agent_doc_ratio": 0.0,
            "avg_worker_quality": 0.0,
            "total_examples": len(results),
            "successful_examples": 0,
            "error_rate": 1.0,
        }

    n = len(valid_results)

    # Standard metrics
    metrics = {
        "avg_f1_at_k": sum(r["f1_at_k"] for r in valid_results) / n,
        "avg_precision_at_k": sum(r["precision_at_k"] for r in valid_results) / n,
        "avg_recall_at_k": sum(r["recall_at_k"] for r in valid_results) / n,
        "avg_groundedness": sum(r["groundedness_score"] for r in valid_results) / n,
        "avg_semantic_similarity": sum(r["semantic_similarity"] for r in valid_results) / n,
        "avg_factual_accuracy": sum(r["factual_accuracy"] for r in valid_results) / n,
        "avg_completeness": sum(r["completeness"] for r in valid_results) / n,
        "total_examples": len(results),
        "successful_examples": n,
        "error_rate": (len(results) - n) / len(results) if results else 0.0,
    }

    # Multi-agent specific metrics
    metrics["avg_sub_queries"] = sum(r["num_sub_queries"] for r in valid_results) / n
    metrics["avg_workers"] = sum(r["num_workers"] for r in valid_results) / n
    metrics["avg_total_unique_docs"] = sum(r["total_unique_docs"] for r in valid_results) / n
    metrics["avg_multi_agent_docs"] = sum(r["multi_agent_docs"] for r in valid_results) / n
    metrics["avg_worker_quality"] = sum(r["avg_worker_quality"] for r in valid_results) / n

    # Calculate multi-agent doc ratio (docs appearing in multiple workers / total unique docs)
    total_unique = sum(r["total_unique_docs"] for r in valid_results)
    total_multi = sum(r["multi_agent_docs"] for r in valid_results)
    metrics["multi_agent_doc_ratio"] = total_multi / total_unique if total_unique > 0 else 0.0

    return metrics


def generate_evaluation_report(
    metrics: Dict[str, float],
    results: List[Dict],
    k_final: int,
    dataset_type: str,
    test_timestamp: str,
    current_tier,
    tier_info: Dict,
) -> str:
    """
    Generate markdown evaluation report.

    Args:
        metrics: Aggregate metrics
        results: Per-example results
        k_final: k value used for evaluation
        dataset_type: 'standard' or 'hard'
        test_timestamp: Timestamp for report
        current_tier: Model tier enum
        tier_info: Tier metadata

    Returns:
        Markdown formatted report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_label = "Standard" if dataset_type == "standard" else "Hard"

    report = f"""# Multi-Agent RAG Evaluation Report

**Generated:** {timestamp}
**Model Tier:** {current_tier.value.upper()} ({tier_info['description']})
**Dataset:** {dataset_label} ({metrics['total_examples']} examples)
**k_final:** {k_final}

---

## Executive Summary

This report evaluates the Multi-Agent RAG architecture using the Orchestrator-Worker pattern.
Complex queries are decomposed into sub-queries, processed in parallel by retrieval workers,
then merged using cross-agent RRF fusion.

### Key Results

| Metric | Value |
|--------|-------|
| **F1@{k_final}** | {metrics['avg_f1_at_k']:.1%} |
| **Groundedness** | {metrics['avg_groundedness']:.1%} |
| **Semantic Similarity** | {metrics['avg_semantic_similarity']:.1%} |
| **Factual Accuracy** | {metrics['avg_factual_accuracy']:.1%} |
| **Completeness** | {metrics['avg_completeness']:.1%} |

### Multi-Agent Architecture Metrics

| Metric | Value |
|--------|-------|
| Avg Sub-Queries | {metrics['avg_sub_queries']:.1f} |
| Avg Workers | {metrics['avg_workers']:.1f} |
| Avg Unique Docs Retrieved | {metrics['avg_total_unique_docs']:.1f} |
| Avg Multi-Agent Docs | {metrics['avg_multi_agent_docs']:.1f} |
| Multi-Agent Doc Ratio | {metrics['multi_agent_doc_ratio']:.1%} |
| Avg Worker Quality | {metrics['avg_worker_quality']:.1%} |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| F1@{k_final} | {metrics['avg_f1_at_k']:.1%} |
| Precision@{k_final} | {metrics['avg_precision_at_k']:.1%} |
| Recall@{k_final} | {metrics['avg_recall_at_k']:.1%} |

---

## Generation Metrics

| Metric | Value |
|--------|-------|
| Groundedness | {metrics['avg_groundedness']:.1%} |
| Semantic Similarity | {metrics['avg_semantic_similarity']:.1%} |
| Factual Accuracy | {metrics['avg_factual_accuracy']:.1%} |
| Completeness | {metrics['avg_completeness']:.1%} |

---

## Success Rate

- **Successful:** {metrics['successful_examples']}/{metrics['total_examples']} ({(metrics['successful_examples']/metrics['total_examples']*100):.0f}%)
- **Error Rate:** {metrics['error_rate']:.1%}

---

## Per-Example Results

### Top Performing Examples

{_format_top_examples(results, n=5, k=k_final)}

### Examples with Most Sub-Queries

{_format_most_decomposed(results, n=5)}

---

## Methodology

**Architecture:** Multi-Agent RAG with Orchestrator-Worker pattern
**Pattern:** Query decomposition -> Parallel retrieval workers -> Cross-agent RRF fusion
**Workers:** Each worker runs full retrieval pipeline (strategy selection, query expansion, retrieval, retry loop)
**Fusion:** RRF with sqrt(N) boost for docs appearing in multiple workers
**Final K:** {k_final} documents selected for generation

**Evaluation:**
- **F1@{k_final}:** Harmonic mean of Precision and Recall
- **Groundedness:** HHEM-based verification of answer claims
- **Similarity/Factual/Completeness:** LLM-as-judge comparison vs ground truth

---

*Report generated by `test_multi_agent_evaluation.py`*
"""

    return report


def _format_top_examples(results: List[Dict], n: int = 5, k: int = 6) -> str:
    """Format top N performing examples."""
    valid_results = [r for r in results if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x["f1_at_k"], reverse=True)[:n]

    lines = []
    for i, r in enumerate(sorted_results, 1):
        lines.append(
            f"{i}. **{r['example_id']}**: F1@{k}={r['f1_at_k']:.0%}, "
            f"Sim={r['semantic_similarity']:.0%}, Sub-Q={r['num_sub_queries']}"
        )

    return "\n".join(lines) if lines else "*No successful examples*"


def _format_most_decomposed(results: List[Dict], n: int = 5) -> str:
    """Format examples with most sub-queries."""
    valid_results = [r for r in results if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x["num_sub_queries"], reverse=True)[:n]

    lines = []
    for i, r in enumerate(sorted_results, 1):
        sub_q_preview = ", ".join(r.get("sub_queries", [])[:2])
        if len(r.get("sub_queries", [])) > 2:
            sub_q_preview += "..."
        lines.append(
            f"{i}. **{r['example_id']}**: {r['num_sub_queries']} sub-queries "
            f"(F1={r['f1_at_k']:.0%})"
        )

    return "\n".join(lines) if lines else "*No decomposed examples*"


# ========== MAIN TEST ==========

def test_multi_agent_evaluation(quick_mode: bool = False, dataset_type: str = "standard"):
    """
    Main evaluation test - run multi-agent RAG on golden dataset.

    Args:
        quick_mode: If True, evaluate only first 2 examples
        dataset_type: Dataset to evaluate ('standard' or 'hard')
    """
    print("\n" + "="*80)
    print("MULTI-AGENT RAG GOLDEN DATASET EVALUATION")
    print("="*80)
    current_tier = get_current_tier()
    tier_info = TIER_METADATA[current_tier]
    print(f"Model Tier: {current_tier.value.upper()} ({tier_info['description']})")
    print(f"Dataset: {dataset_type}")
    print(f"Mode: {'Quick (2 examples)' if quick_mode else 'Full'}")
    print("="*80 + "\n")

    # Load golden dataset with adaptive k_final
    print(f"[*] Loading {dataset_type} golden dataset...")
    if dataset_type == "standard":
        dataset_path = "evaluation/golden_set_standard.json"
        k_final = 6  # Multi-agent uses top-6 (vs top-4 in advanced)
    else:  # hard
        dataset_path = "evaluation/golden_set_hard.json"
        k_final = 6  # Same for hard dataset

    dataset_manager = GoldenDatasetManager(dataset_path)
    dataset = dataset_manager.dataset
    print(f"[OK] Loaded {len(dataset)} examples (k_final={k_final})")

    if not dataset:
        print("[ERROR] No examples in golden dataset")
        return

    # Apply quick mode if requested
    if quick_mode:
        dataset = dataset[:2]
        print(f"[*] Quick mode: Using first 2 examples\n")

    # PRE-BUILD RETRIEVER ONCE
    print(f"\n{'='*80}")
    print(f"PRE-BUILD: Initializing retriever once (k_final={k_final})")
    print(f"{'='*80}")
    print("    This avoids re-ingesting PDFs for each example (saves ~50% time)")

    # Use k_final=4 for retriever (workers use this), merge_results takes top-6
    shared_retriever = setup_retriever(k_final=4)

    # Inject into multi_agent module
    multi_agent_module.adaptive_retriever = shared_retriever
    print(f"[OK] Retriever pre-built and injected into multi_agent module")
    print(f"{'='*80}\n")

    # Start timing
    start_time = time.time()

    # Run evaluation
    print(f"\n{'='*80}")
    print("Running Multi-Agent RAG Evaluation...")
    print(f"{'='*80}")
    results = run_multi_agent_on_golden_dataset(dataset, k_final=k_final, verbose=True)

    # Calculate aggregate metrics
    metrics = calculate_aggregate_metrics(results)

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'F1@' + str(k_final):<20} {metrics['avg_f1_at_k']:<10.1%}")
    print(f"{'Precision@' + str(k_final):<20} {metrics['avg_precision_at_k']:<10.1%}")
    print(f"{'Recall@' + str(k_final):<20} {metrics['avg_recall_at_k']:<10.1%}")
    print(f"{'Groundedness':<20} {metrics['avg_groundedness']:<10.1%}")
    print(f"{'Semantic Similarity':<20} {metrics['avg_semantic_similarity']:<10.1%}")
    print(f"{'Factual Accuracy':<20} {metrics['avg_factual_accuracy']:<10.1%}")
    print(f"{'Completeness':<20} {metrics['avg_completeness']:<10.1%}")
    print("-" * 30)
    print(f"{'Avg Sub-Queries':<20} {metrics['avg_sub_queries']:<10.1f}")
    print(f"{'Avg Workers':<20} {metrics['avg_workers']:<10.1f}")
    print(f"{'Multi-Agent Doc %':<20} {metrics['multi_agent_doc_ratio']:<10.1%}")
    print(f"{'Avg Worker Quality':<20} {metrics['avg_worker_quality']:<10.1%}")
    print("=" * 120)
    print(f"Elapsed Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    print("=" * 120 + "\n")

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "multi_agent_evaluation",
        "dataset_type": dataset_type,
        "model_tier": f"{current_tier.value.upper()} ({tier_info['description']})",
        "dataset_size": len(dataset),
        "k_final": k_final,
        "quick_mode": quick_mode,
        "elapsed_time_seconds": elapsed_time,
        "metrics": metrics,
        "results": results,
    }

    # Save with timestamp
    results_path = Path("evaluation") / f"multi_agent_evaluation_results_{dataset_type}_{test_timestamp}.json"
    latest_results_path = Path("evaluation") / f"multi_agent_evaluation_results_{dataset_type}_latest.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print(f"[OK] Saved raw results to {results_path}")

    # Create latest copy
    shutil.copy2(results_path, latest_results_path)
    print(f"[OK] Latest copy saved to {latest_results_path}")

    # Generate and save report
    report = generate_evaluation_report(
        metrics, results, k_final, dataset_type,
        test_timestamp, current_tier, tier_info
    )

    report_path = Path("evaluation") / f"multi_agent_evaluation_report_{dataset_type}_{test_timestamp}.md"
    latest_report_path = Path("evaluation") / f"multi_agent_evaluation_report_{dataset_type}_latest.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] Saved evaluation report to {report_path}")

    # Create latest copy
    shutil.copy2(report_path, latest_report_path)
    print(f"[OK] Latest copy saved to {latest_report_path}")

    print("\n" + "="*80)
    print("MULTI-AGENT EVALUATION COMPLETE")
    print("="*80)
    print(f"Results: {results_path}")
    print(f"Report: {report_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Agent RAG Golden Dataset Evaluation')
    parser.add_argument(
        '--dataset',
        choices=['standard', 'hard'],
        default='standard',
        help='Dataset to evaluate: standard (20 questions) or hard (10 questions)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: evaluate only first 2 examples'
    )
    args = parser.parse_args()

    if args.quick:
        print(f"[*] Running in quick mode (2 examples from {args.dataset} dataset)")
    else:
        dataset_size = "20 examples" if args.dataset == "standard" else "10 examples"
        expected_time = "15-20 minutes" if args.dataset == "standard" else "8-12 minutes"
        print(f"[*] Running full evaluation on {args.dataset} dataset ({dataset_size})")
        print(f"[*] This will take approximately {expected_time}")
        print("[*] Use --quick flag for faster testing (~2-3 minutes)")

    test_multi_agent_evaluation(quick_mode=args.quick, dataset_type=args.dataset)
