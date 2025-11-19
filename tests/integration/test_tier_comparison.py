"""
Tier Comparison Test for Model Selection Strategy

This test evaluates the Advanced Agentic RAG system across three model tiers:
- Budget: All GPT-4o-mini ($1,200/day, 70-75% quality target)
- Balanced: Hybrid GPT-4o-mini + GPT-5-mini ($1,800/day, 78-80% quality target)
- Premium: GPT-5.1 + GPT-5-mini + GPT-5-nano ($12,060/day, 88-92% quality target)

Validates the portfolio narrative: "Architecture adds X%, model upgrades add Y%"

Runtime: ~8-12 minutes (quick mode, 5 examples)
         ~30-45 minutes (full test, 20 examples)
"""

import json
import os
import sys
import warnings
import logging
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Suppress LangSmith warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*Failed to.*LangSmith.*")
warnings.filterwarnings("ignore", message=".*langsmith.*")

# Suppress LangSmith logging
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.WARNING)

from advanced_agentic_rag_langgraph.evaluation.golden_dataset import GoldenDatasetManager, evaluate_on_golden_dataset
import advanced_agentic_rag_langgraph.orchestration.graph as graph_module


def load_tier_config(tier: str) -> Dict[str, Any]:
    """Load tier configuration metadata with metric-specific targets."""
    configs = {
        "budget": {
            "name": "Budget",
            "daily_cost": 1200,
            "quality_narrative": "70-75%",  # Aspirational overall quality description
            "models": "All GPT-4o-mini",
            # Metric-specific targets (empirically realistic for cross-domain RAG)
            "targets": {
                "f1_at_k": (0.20, 0.30),        # 20-30% F1@5 (retrieval quality)
                "groundedness": (0.85, 0.95),   # 85-95% hallucination prevention
                "confidence": (0.65, 0.80),     # 65-80% answer confidence
            }
        },
        "balanced": {
            "name": "Balanced",
            "daily_cost": 1800,
            "quality_narrative": "78-80%",
            "models": "Hybrid GPT-4o-mini + GPT-5-mini",
            "targets": {
                "f1_at_k": (0.28, 0.38),        # +8 pts over budget
                "groundedness": (0.88, 0.98),   # +3 pts
                "confidence": (0.72, 0.87),     # +7 pts
            }
        },
        "premium": {
            "name": "Premium",
            "daily_cost": 12060,
            "quality_narrative": "88-92%",
            "models": "GPT-5.1 + GPT-5-mini + GPT-5-nano",
            "targets": {
                "f1_at_k": (0.35, 0.50),        # +15 pts over budget
                "groundedness": (0.95, 1.00),   # +10 pts (near-perfect)
                "confidence": (0.85, 0.95),     # +20 pts
            }
        },
    }
    return configs.get(tier, {})


def reload_graph_with_tier(tier: str):
    """
    Reload the graph module with a new tier configuration.

    This is necessary because the graph is built once at module load time,
    so we need to reload the module to rebuild the graph with the new tier.
    """
    os.environ["MODEL_TIER"] = tier
    importlib.reload(graph_module)
    return graph_module.advanced_rag_graph


def run_tier_evaluation(tier: str, dataset: List[Dict], quick_mode: bool = False) -> Dict[str, Any]:
    """
    Run evaluation for a specific tier.

    Args:
        tier: Model tier ("budget", "balanced", "premium")
        dataset: Golden dataset examples
        quick_mode: If True, use only first 5 examples

    Returns:
        Dictionary with metrics and execution time
    """
    print(f"\n{'='*70}")
    print(f"TIER: {tier.upper()}")
    print(f"{'='*70}")

    config = load_tier_config(tier)
    print(f"Configuration: {config['name']}")
    print(f"  Models: {config['models']}")
    print(f"  Daily Cost: ${config['daily_cost']:,}")
    print(f"  Quality Narrative: {config['quality_narrative']}")
    print()

    # Reload graph with tier
    print(f"[*] Loading graph with {tier} tier...")
    graph = reload_graph_with_tier(tier)
    print(f"[OK] Graph loaded successfully")

    # Prepare dataset
    eval_dataset = dataset[:5] if quick_mode else dataset
    print(f"[*] Evaluating {len(eval_dataset)} examples...")

    # Run evaluation with timing
    start_time = time.time()
    results = evaluate_on_golden_dataset(graph, eval_dataset, verbose=True)
    execution_time = time.time() - start_time

    # Add tier metadata
    results["tier"] = tier
    results["tier_config"] = config
    results["execution_time_seconds"] = execution_time
    results["examples_evaluated"] = len(eval_dataset)
    results["avg_time_per_example"] = execution_time / len(eval_dataset)

    print(f"\n[OK] {tier.upper()} tier evaluation completed")
    print(f"  Execution time: {execution_time:.1f}s ({execution_time/60:.1f} min)")
    print(f"  Avg per example: {execution_time/len(eval_dataset):.1f}s")

    return results


def calculate_improvement(budget_value: float, tier_value: float) -> float:
    """Calculate percentage improvement over budget tier."""
    if budget_value == 0:
        return 0.0
    return ((tier_value - budget_value) / budget_value) * 100


def generate_comparison_report(tier_results: Dict[str, Dict], output_dir: str = "evaluation"):
    """
    Generate markdown comparison report.

    Args:
        tier_results: Dictionary mapping tier names to results
        output_dir: Directory to save report
    """
    output_path = Path(output_dir) / "tier_comparison_report.md"

    # Extract key metrics
    budget = tier_results["budget"]
    balanced = tier_results["balanced"]
    premium = tier_results["premium"]

    # Build report
    lines = [
        "# Model Tier Comparison Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Examples Evaluated:** {budget['examples_evaluated']}",
        "",
        "## Executive Summary",
        "",
        "This report compares three model tier configurations to validate the portfolio narrative:",
        '**"Architecture adds X%, model upgrades add Y%"**',
        "",
    ]

    # Calculate key improvements using multi-dimensional metrics
    # Primary metric: F1@5 (balanced retrieval quality)
    budget_f1 = budget["retrieval_metrics"]["f1_at_k"]
    balanced_f1 = balanced["retrieval_metrics"]["f1_at_k"]
    premium_f1 = premium["retrieval_metrics"]["f1_at_k"]

    # Secondary metric: Groundedness (generation quality)
    budget_ground = budget["generation_metrics"]["avg_groundedness"]
    balanced_ground = balanced["generation_metrics"]["avg_groundedness"]
    premium_ground = premium["generation_metrics"]["avg_groundedness"]

    # Calculate improvements for F1 (retrieval)
    balanced_f1_improvement = balanced_f1 - budget_f1
    premium_f1_improvement = premium_f1 - budget_f1
    premium_vs_balanced_f1 = premium_f1 - balanced_f1

    # Calculate improvements for groundedness (generation)
    balanced_ground_improvement = balanced_ground - budget_ground
    premium_ground_improvement = premium_ground - budget_ground
    premium_vs_balanced_ground = premium_ground - balanced_ground

    lines.extend([
        f"- **Budget Baseline:** F1@5={budget_f1:.1%}, Groundedness={budget_ground:.1%}",
        f"- **Balanced Improvement:** Retrieval +{balanced_f1_improvement:.1f} pts, Generation +{balanced_ground_improvement:.1f} pts",
        f"- **Premium Improvement:** Retrieval +{premium_f1_improvement:.1f} pts, Generation +{premium_ground_improvement:.1f} pts",
        "",
        "---",
        "",
        "## Tier Configurations",
        "",
        "| Tier | Models | Daily Cost | Quality Narrative | Retrieval F1@5 | Groundedness |",
        "|------|--------|------------|-------------------|----------------|--------------|",
    ])

    for tier in ["budget", "balanced", "premium"]:
        result = tier_results[tier]
        config = result["tier_config"]
        f1 = result["retrieval_metrics"]["f1_at_k"]
        groundedness = result["generation_metrics"]["avg_groundedness"]
        lines.append(
            f"| {config['name']} | {config['models']} | ${config['daily_cost']:,} | "
            f"{config['quality_narrative']} | {f1:.1%} | {groundedness:.1%} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Performance Comparison",
        "",
        "### Generation Quality Metrics",
        "",
        "| Metric | Budget | Balanced | Premium | Balanced vs Budget | Premium vs Budget |",
        "|--------|--------|----------|---------|-------------------|------------------|",
    ])

    # Generation metrics comparison
    gen_metrics_to_compare = [
        ("avg_groundedness", "Groundedness", "%"),
        ("hallucination_rate", "Hallucination Rate", "%"),
        ("avg_confidence", "Confidence", "%"),
        ("avg_semantic_similarity", "Answer Quality", "%"),
    ]

    for key, label, unit in gen_metrics_to_compare:
        budget_val = budget["generation_metrics"].get(key, 0) * 100  # Convert to percentage
        balanced_val = balanced["generation_metrics"].get(key, 0) * 100
        premium_val = premium["generation_metrics"].get(key, 0) * 100

        balanced_delta = balanced_val - budget_val
        premium_delta = premium_val - budget_val

        # For hallucination rate, lower is better (show as negative improvement)
        if key == "hallucination_rate":
            balanced_delta = -balanced_delta
            premium_delta = -premium_delta

        lines.append(
            f"| {label} | {budget_val:.1f}% | {balanced_val:.1f}% | {premium_val:.1f}% | "
            f"{balanced_delta:+.1f} pts | {premium_delta:+.1f} pts |"
        )

    lines.extend([
        "",
        "### Retrieval Quality Metrics",
        "",
        "| Metric | Budget | Balanced | Premium | Balanced vs Budget | Premium vs Budget |",
        "|--------|--------|----------|---------|-------------------|------------------|",
    ])

    retrieval_metrics = [
        ("recall_at_k", "Recall@5", "%"),
        ("precision_at_k", "Precision@5", "%"),
        ("f1_at_k", "F1@5", "%"),
        ("ndcg", "nDCG", ""),
        ("mrr", "MRR", ""),
    ]

    for key, label, unit in retrieval_metrics:
        budget_val = budget["retrieval_metrics"].get(key, 0) * (100 if unit == "%" else 1)
        balanced_val = balanced["retrieval_metrics"].get(key, 0) * (100 if unit == "%" else 1)
        premium_val = premium["retrieval_metrics"].get(key, 0) * (100 if unit == "%" else 1)

        balanced_delta = balanced_val - budget_val
        premium_delta = premium_val - budget_val

        lines.append(
            f"| {label} | {budget_val:.1f}{unit} | {balanced_val:.1f}{unit} | {premium_val:.1f}{unit} | "
            f"{balanced_delta:+.1f} pts | {premium_delta:+.1f} pts |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Cost-Quality Analysis",
        "",
        "### Cost Per Quality Point",
        "",
    ])

    # Calculate cost per quality point (using F1@5 as primary metric)
    for tier in ["budget", "balanced", "premium"]:
        result = tier_results[tier]
        config = result["tier_config"]
        f1_quality = result["retrieval_metrics"]["f1_at_k"]
        cost_per_point = config["daily_cost"] / f1_quality if f1_quality > 0 else 0

        lines.append(f"**{config['name']}:** ${cost_per_point:.2f} per F1 point")

    lines.extend([
        "",
        "### Incremental Cost-Benefit (Retrieval F1@5)",
        "",
        f"- **Budget → Balanced:** +{balanced_f1_improvement:.1f} pts for ${balanced['tier_config']['daily_cost'] - budget['tier_config']['daily_cost']:,}/day "
        f"(${(balanced['tier_config']['daily_cost'] - budget['tier_config']['daily_cost']) / balanced_f1_improvement if balanced_f1_improvement > 0 else 0:.2f} per point)",
        f"- **Balanced → Premium:** +{premium_vs_balanced_f1:.1f} pts for ${premium['tier_config']['daily_cost'] - balanced['tier_config']['daily_cost']:,}/day "
        f"(${(premium['tier_config']['daily_cost'] - balanced['tier_config']['daily_cost']) / premium_vs_balanced_f1 if premium_vs_balanced_f1 > 0 else 0:.2f} per point)",
        "",
        "---",
        "",
        "## Execution Performance",
        "",
        "| Tier | Total Time | Avg Per Example | Examples |",
        "|------|-----------|----------------|----------|",
    ])

    for tier in ["budget", "balanced", "premium"]:
        result = tier_results[tier]
        total_time = result["execution_time_seconds"]
        avg_time = result["avg_time_per_example"]
        examples = result["examples_evaluated"]

        lines.append(
            f"| {result['tier_config']['name']} | {total_time:.1f}s ({total_time/60:.1f} min) | "
            f"{avg_time:.1f}s | {examples} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Portfolio Narrative",
        "",
        "### Architecture Value (Budget Tier)",
        "",
        f"The Budget tier achieves **F1@5={budget_f1:.1%}, Groundedness={budget_ground:.1%}** using only GPT-4o-mini models across all components. ",
        "This demonstrates that the Advanced Agentic RAG architecture itself provides substantial value through:",
        "",
        "- Multi-stage query processing and expansion",
        "- Adaptive retrieval strategy selection",
        "- Two-stage reranking (CrossEncoder + LLM)",
        "- Self-correction loops with quality gates",
        "- NLI-based hallucination detection",
        "",
        f"**Architecture Contribution:** F1@5={budget_f1:.1%}, Groundedness={budget_ground:.1%}",
        "",
        "### Model Upgrade Impact",
        "",
        f"**Balanced Tier:** Selective GPT-5-mini upgrades for critical reasoning tasks add **Retrieval +{balanced_f1_improvement:.1f} pts, Generation +{balanced_ground_improvement:.1f} pts** ",
        f"(F1@5={balanced_f1:.1%}, Groundedness={balanced_ground:.1%}) for an additional ${balanced['tier_config']['daily_cost'] - budget['tier_config']['daily_cost']:,}/day.",
        "",
        f"**Premium Tier:** Full GPT-5.1 deployment adds **Retrieval +{premium_f1_improvement:.1f} pts, Generation +{premium_ground_improvement:.1f} pts** ",
        f"(F1@5={premium_f1:.1%}, Groundedness={premium_ground:.1%}) for an additional ${premium['tier_config']['daily_cost'] - budget['tier_config']['daily_cost']:,}/day.",
        "",
        "### Key Findings",
        "",
        f"1. **Architecture provides the foundation:** F1@5={budget_f1:.1%}, Groundedness={budget_ground:.1%} with budget models",
        f"2. **Balanced tier offers best ROI:** Retrieval +{balanced_f1_improvement:.1f} pts, Generation +{balanced_ground_improvement:.1f} pts for 50% cost increase",
        f"3. **Premium tier for critical applications:** Retrieval +{premium_f1_improvement:.1f} pts, Generation +{premium_ground_improvement:.1f} pts justifies 10x cost when quality is paramount",
        "",
        "---",
        "",
        "## Validation Against Targets (Multi-Dimensional)",
        "",
        "| Tier | Metric | Target | Actual | Status |",
        "|------|--------|--------|--------|--------|",
    ])

    # Multi-dimensional validation for each tier
    for tier in ["budget", "balanced", "premium"]:
        result = tier_results[tier]
        config = result["tier_config"]
        targets = config["targets"]

        # Validate F1@5 (retrieval quality)
        f1_min, f1_max = targets["f1_at_k"]
        f1_actual = result["retrieval_metrics"]["f1_at_k"]
        f1_status = "[OK]" if f1_min <= f1_actual <= f1_max else "[WARN]"

        lines.append(
            f"| {config['name']} | F1@5 (Retrieval) | {f1_min:.0%}-{f1_max:.0%} | {f1_actual:.1%} | {f1_status} |"
        )

        # Validate Groundedness (anti-hallucination)
        ground_min, ground_max = targets["groundedness"]
        ground_actual = result["generation_metrics"]["avg_groundedness"]
        ground_status = "[OK]" if ground_min <= ground_actual <= ground_max else "[WARN]"

        lines.append(
            f"| {config['name']} | Groundedness (Anti-Hallucination) | {ground_min:.0%}-{ground_max:.0%} | {ground_actual:.1%} | {ground_status} |"
        )

        # Validate Confidence (answer quality)
        conf_min, conf_max = targets["confidence"]
        conf_actual = result["generation_metrics"]["avg_confidence"]
        conf_status = "[OK]" if conf_min <= conf_actual <= conf_max else "[WARN]"

        lines.append(
            f"| {config['name']} | Confidence (Answer Quality) | {conf_min:.0%}-{conf_max:.0%} | {conf_actual:.1%} | {conf_status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        f"*Report generated by test_tier_comparison.py at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Comparison report saved to {output_path}")


def save_results(tier_results: Dict[str, Dict], output_dir: str = "evaluation"):
    """
    Save tier comparison results to JSON.

    Args:
        tier_results: Dictionary mapping tier names to results
        output_dir: Directory to save results
    """
    output_path = Path(output_dir) / "tier_comparison_results.json"

    # Add metadata
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "tier_comparison",
        "tiers_evaluated": list(tier_results.keys()),
        "results": tier_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"[OK] Results saved to {output_path}")


def test_tier_comparison(quick_mode: bool = False):
    """
    Main test function for tier comparison.

    Args:
        quick_mode: If True, evaluate only 5 examples instead of full dataset
    """
    print("\n" + "="*70)
    print("TEST: Model Tier Comparison")
    print("="*70)
    print(f"Mode: {'Quick (5 examples)' if quick_mode else 'Full (20 examples)'}")
    print()

    # Load golden dataset
    print("[*] Loading golden dataset...")
    manager = GoldenDatasetManager("evaluation/golden_set.json")
    dataset = manager.dataset
    print(f"[OK] Loaded {len(dataset)} examples")

    # Pre-build retriever once (tier-independent optimization)
    print("\n[*] Pre-building retriever (tier-independent components)...")
    print("    This avoids re-ingesting 10 PDFs for each tier (saves 50-60% time)")
    from advanced_agentic_rag_langgraph.core import setup_retriever
    retriever_instance = setup_retriever()
    print("[OK] Retriever built and cached\n")

    # Run evaluations for each tier
    tiers = ["budget", "balanced", "premium"]
    tier_results = {}

    overall_start = time.time()

    for tier in tiers:
        # Inject pre-built retriever before running evaluation
        # This avoids rebuilding FAISS/BM25 indexes for each tier
        import advanced_agentic_rag_langgraph.orchestration.nodes as nodes
        nodes.adaptive_retriever = retriever_instance

        tier_results[tier] = run_tier_evaluation(tier, dataset, quick_mode)

    overall_time = time.time() - overall_start

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Total execution time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print()

    print("Quality Scores (Multi-Dimensional):")
    for tier in tiers:
        result = tier_results[tier]
        f1 = result["retrieval_metrics"]["f1_at_k"]
        groundedness = result["generation_metrics"]["avg_groundedness"]
        config = result["tier_config"]
        print(f"  {config['name']:10s}: F1@5={f1:5.1f}%, Groundedness={groundedness:5.1f}% (narrative: {config['quality_narrative']})")

    print()
    print("Improvements vs Budget:")
    budget_f1 = tier_results["budget"]["retrieval_metrics"]["f1_at_k"]
    budget_ground = tier_results["budget"]["generation_metrics"]["avg_groundedness"]
    for tier in ["balanced", "premium"]:
        result = tier_results[tier]
        f1 = result["retrieval_metrics"]["f1_at_k"]
        groundedness = result["generation_metrics"]["avg_groundedness"]
        f1_improvement = f1 - budget_f1
        ground_improvement = groundedness - budget_ground
        config = result["tier_config"]
        print(f"  {config['name']:10s}: Retrieval +{f1_improvement:4.1f} pts, Generation +{ground_improvement:4.1f} pts")

    # Save results
    print()
    save_results(tier_results)
    generate_comparison_report(tier_results)

    print("\n" + "="*70)
    print("[OK] Tier comparison test COMPLETED")
    print("="*70)

    # Return results for further analysis if needed
    return tier_results


if __name__ == "__main__":
    # Check for --quick flag
    quick_mode = "--quick" in sys.argv

    if quick_mode:
        print("[*] Running in quick mode (5 examples)")
    else:
        print("[*] Running full evaluation (20 examples)")
        print("[*] This will take approximately 30-45 minutes")
        print("[*] Use --quick flag for faster testing (8-12 minutes)")

    test_tier_comparison(quick_mode=quick_mode)
