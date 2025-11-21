"""
4-Tier Architecture A/B Test for Portfolio Showcase.

Compares four RAG architecture implementations to demonstrate
the incremental value of advanced features:
- Pure Semantic: Simplest RAG (4 features, pure vector search)
- Basic: Simple RAG (8 features, linear flow)
- Intermediate: Enhanced RAG (18 features, conditional routing)
- Advanced: Full Agentic RAG (31 features, adaptive loops)

All tiers use BUDGET model tier (gpt-4o-mini) to isolate architectural
improvements from model quality differences.

Key Metrics:
- F1@5: Retrieval quality (harmonic mean of precision and recall)
- Groundedness: Anti-hallucination (% claims supported by context)
- Confidence: Answer quality (LLM confidence score)
- Avg Retrieval Attempts: Efficiency metric

Expected Progression:
- Pure Semantic → Basic: +10-15% (hybrid search, query expansion, reranking)
- Basic → Intermediate: +15-25% (quality gates, two-stage reranking)
- Intermediate → Advanced: +20-35% (NLI, strategy switching, adaptive loops)
- Pure Semantic → Advanced: +45-75% overall

Usage:
    uv run python tests/integration/test_architecture_comparison.py

Outputs:
    - evaluation/architecture_comparison_results.json (raw data)
    - evaluation/architecture_comparison_report.md (formatted report)
"""

import json
import os
import warnings
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Suppress LangSmith warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*Failed to.*LangSmith.*")
warnings.filterwarnings("ignore", message=".*langsmith.*")

# Suppress LangSmith logging
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.WARNING)

from advanced_agentic_rag_langgraph.variants import (
    pure_semantic_rag_graph,
    basic_rag_graph,
    intermediate_rag_graph,
    advanced_rag_graph,
)
# Import modules to access global adaptive_retriever variables
import advanced_agentic_rag_langgraph.variants.pure_semantic_rag_graph as pure_semantic_module
import advanced_agentic_rag_langgraph.variants.basic_rag_graph as basic_module
import advanced_agentic_rag_langgraph.variants.intermediate_rag_graph as intermediate_module
import advanced_agentic_rag_langgraph.orchestration.nodes as advanced_module
from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.evaluation.golden_dataset import GoldenDatasetManager
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import (
    calculate_retrieval_metrics,
)
from advanced_agentic_rag_langgraph.validation import NLIHallucinationDetector


# ========== TIER CONFIGURATION ==========

TIER_CONFIGS = {
    "pure_semantic": {
        "name": "Pure Semantic RAG",
        "features": 4,
        "graph": pure_semantic_rag_graph,
        "description": "Simplest RAG with pure vector search, top-4 chunks, no reranking",
    },
    "basic": {
        "name": "Basic RAG",
        "features": 8,
        "graph": basic_rag_graph,
        "description": "Linear flow with hybrid retrieval and CrossEncoder reranking",
    },
    "intermediate": {
        "name": "Intermediate RAG",
        "features": 18,
        "graph": intermediate_rag_graph,
        "description": "Conditional routing with quality gates and limited retry",
    },
    "advanced": {
        "name": "Advanced RAG",
        "features": 31,
        "graph": advanced_rag_graph,
        "description": "Full agentic with NLI, strategy switching, and adaptive loops",
    },
}


# ========== HELPER FUNCTIONS ==========

def run_tier_on_golden_dataset(
    tier_name: str,
    graph: Any,
    dataset: List[Dict],
    verbose: bool = True
) -> List[Dict]:
    """
    Run a specific architecture tier on golden dataset.

    Args:
        tier_name: Name of the tier (basic, intermediate, advanced)
        graph: Compiled LangGraph graph
        dataset: List of golden examples
        verbose: Print progress

    Returns:
        List of result dictionaries with metrics per example
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running {tier_name.upper()} Tier on Golden Dataset")
        print(f"{'='*70}")
        print(f"Examples: {len(dataset)}")
        print(f"Graph: {TIER_CONFIGS[tier_name]['name']}")
        print(f"Features: {TIER_CONFIGS[tier_name]['features']}")
        print(f"{'='*70}\n")

    results = []
    nli_detector = NLIHallucinationDetector()

    for i, example in enumerate(dataset, 1):
        query = example["question"]  # Fixed: golden dataset uses "question" not "query"
        ground_truth_docs = example["relevant_doc_ids"]

        if verbose:
            print(f"\n[{i}/{len(dataset)}] Processing: {example.get('id', query[:50])}")

        try:
            # Different graphs have different state schemas
            # Initialize with required/accumulated fields per reference code pattern
            if tier_name == "pure_semantic":
                initial_state = {
                    "user_question": query,
                    "retrieved_docs": [],  # Will be set by retrieve node
                }
            elif tier_name == "basic":
                initial_state = {
                    "user_question": query,
                    "query_expansions": [],  # Will be set by first node
                    "retrieved_docs": [],  # Accumulated field
                }
            elif tier_name == "intermediate":
                initial_state = {
                    "user_question": query,
                    "baseline_query": query,
                    "messages": [],
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                }
            else:  # advanced
                initial_state = {
                    "user_question": query,
                    "baseline_query": query,  # Required field for advanced tier
                    "messages": [],
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                    "query_expansions": [],
                }

            # Run graph
            result = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"{tier_name}_{i}"}}
            )

            # Extract results (different state schemas)
            answer = result.get("final_answer", "")
            confidence = result.get("confidence_score", 0.7)
            retrieved_docs = result.get("unique_docs_list", [])
            retrieval_attempts = result.get("retrieval_attempts", 1)

            # Calculate retrieval metrics
            retrieved_doc_ids = []
            for doc in retrieved_docs:
                # Extract chunk ID from metadata - fixed: use 'id' not 'chunk_id'
                if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                    retrieved_doc_ids.append(doc.metadata['id'])

            # Calculate F1@5
            relevant_retrieved = set(ground_truth_docs[:5]) & set(retrieved_doc_ids[:5])
            precision = len(relevant_retrieved) / 5 if len(retrieved_doc_ids) >= 5 else 0.0
            recall = len(relevant_retrieved) / min(len(ground_truth_docs), 5)
            f1_at_5 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Calculate groundedness using NLI
            if retrieved_docs and answer:
                context = "\n\n".join([
                    doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    for doc in retrieved_docs[:4]
                ])
                groundedness_result = nli_detector.verify_groundedness(answer, context)
                groundedness_score = groundedness_result["groundedness_score"]
            else:
                groundedness_score = 0.0

            if verbose:
                print(f"  F1@5: {f1_at_5:.0%} | Groundedness: {groundedness_score:.0%} | Confidence: {confidence:.0%} | Attempts: {retrieval_attempts}")

            results.append({
                "example_id": example.get("id", query[:30]),
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "ground_truth_docs": ground_truth_docs,
                "retrieved_docs": retrieved_doc_ids,
                "f1_at_5": f1_at_5,
                "precision_at_5": precision,
                "recall_at_5": recall,
                "groundedness_score": groundedness_score,
                "retrieval_attempts": retrieval_attempts,
            })

        except Exception as e:
            print(f"  [ERROR] Failed: {str(e)}")
            results.append({
                "example_id": example.get("id", query[:30]),
                "query": query,
                "error": str(e),
                "f1_at_5": 0.0,
                "precision_at_5": 0.0,
                "recall_at_5": 0.0,
                "groundedness_score": 0.0,
                "retrieval_attempts": 0,
            })

    return results


def calculate_tier_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate aggregate metrics for a tier.

    Args:
        results: List of per-example results

    Returns:
        Dictionary of aggregate metrics
    """
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {
            "avg_f1_at_5": 0.0,
            "avg_precision_at_5": 0.0,
            "avg_recall_at_5": 0.0,
            "avg_groundedness": 0.0,
            "avg_confidence": 0.0,
            "avg_retrieval_attempts": 0.0,
            "total_examples": len(results),
            "successful_examples": 0,
            "error_rate": 1.0,
        }

    metrics = {
        "avg_f1_at_5": sum(r["f1_at_5"] for r in valid_results) / len(valid_results),
        "avg_precision_at_5": sum(r["precision_at_5"] for r in valid_results) / len(valid_results),
        "avg_recall_at_5": sum(r["recall_at_5"] for r in valid_results) / len(valid_results),
        "avg_groundedness": sum(r["groundedness_score"] for r in valid_results) / len(valid_results),
        "avg_confidence": sum(r["confidence"] for r in valid_results) / len(valid_results),
        "avg_retrieval_attempts": sum(r["retrieval_attempts"] for r in valid_results) / len(valid_results),
        "total_examples": len(results),
        "successful_examples": len(valid_results),
        "error_rate": (len(results) - len(valid_results)) / len(results) if results else 0.0,
    }

    return metrics


def generate_comparison_report(
    pure_semantic_metrics: Dict[str, float],
    basic_metrics: Dict[str, float],
    intermediate_metrics: Dict[str, float],
    advanced_metrics: Dict[str, float],
    pure_semantic_results: List[Dict],
    basic_results: List[Dict],
    intermediate_results: List[Dict],
    advanced_results: List[Dict],
    dataset_type: str = "standard",
    test_timestamp: str = None,
) -> str:
    """
    Generate markdown comparison report with delta analysis.

    Args:
        *_metrics: Aggregate metrics for each tier
        *_results: Per-example results for each tier
        dataset_type: Dataset type ('standard' or 'hard')
        test_timestamp: Timestamp for report generation (YYYYMMDD_HHMMSS format)

    Returns:
        Markdown formatted report string
    """
    if test_timestamp is None:
        test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate deltas (all consecutive pairs + pure semantic to advanced)
    pure_to_basic_f1 = ((basic_metrics["avg_f1_at_5"] - pure_semantic_metrics["avg_f1_at_5"]) / pure_semantic_metrics["avg_f1_at_5"] * 100) if pure_semantic_metrics["avg_f1_at_5"] > 0 else 0
    basic_to_inter_f1 = ((intermediate_metrics["avg_f1_at_5"] - basic_metrics["avg_f1_at_5"]) / basic_metrics["avg_f1_at_5"] * 100) if basic_metrics["avg_f1_at_5"] > 0 else 0
    inter_to_adv_f1 = ((advanced_metrics["avg_f1_at_5"] - intermediate_metrics["avg_f1_at_5"]) / intermediate_metrics["avg_f1_at_5"] * 100) if intermediate_metrics["avg_f1_at_5"] > 0 else 0
    pure_to_adv_f1 = ((advanced_metrics["avg_f1_at_5"] - pure_semantic_metrics["avg_f1_at_5"]) / pure_semantic_metrics["avg_f1_at_5"] * 100) if pure_semantic_metrics["avg_f1_at_5"] > 0 else 0

    pure_to_basic_ground = ((basic_metrics["avg_groundedness"] - pure_semantic_metrics["avg_groundedness"]) / pure_semantic_metrics["avg_groundedness"] * 100) if pure_semantic_metrics["avg_groundedness"] > 0 else 0
    basic_to_inter_ground = ((intermediate_metrics["avg_groundedness"] - basic_metrics["avg_groundedness"]) / basic_metrics["avg_groundedness"] * 100) if basic_metrics["avg_groundedness"] > 0 else 0
    inter_to_adv_ground = ((advanced_metrics["avg_groundedness"] - intermediate_metrics["avg_groundedness"]) / intermediate_metrics["avg_groundedness"] * 100) if intermediate_metrics["avg_groundedness"] > 0 else 0
    pure_to_adv_ground = ((advanced_metrics["avg_groundedness"] - pure_semantic_metrics["avg_groundedness"]) / pure_semantic_metrics["avg_groundedness"] * 100) if pure_semantic_metrics["avg_groundedness"] > 0 else 0

    dataset_label = "Standard" if dataset_type == "standard" else "Hard"
    report = f"""# 4-Tier RAG Architecture Comparison Report

**Generated:** {timestamp}
**Model Tier:** BUDGET (gpt-4o-mini for all tiers)
**Dataset:** {dataset_label} ({pure_semantic_metrics['total_examples']} examples)

---

## Executive Summary

This report demonstrates the incremental value of advanced RAG architecture patterns
by comparing four implementation tiers using identical models (BUDGET tier) to isolate
architectural improvements.

### Key Findings

**Winner by Metric:**
- **F1@5 (Retrieval Quality):** {_get_winner(pure_semantic_metrics['avg_f1_at_5'], basic_metrics['avg_f1_at_5'], intermediate_metrics['avg_f1_at_5'], advanced_metrics['avg_f1_at_5'])}
- **Groundedness (Anti-Hallucination):** {_get_winner(pure_semantic_metrics['avg_groundedness'], basic_metrics['avg_groundedness'], intermediate_metrics['avg_groundedness'], advanced_metrics['avg_groundedness'])}
- **Confidence (Answer Quality):** {_get_winner(pure_semantic_metrics['avg_confidence'], basic_metrics['avg_confidence'], intermediate_metrics['avg_confidence'], advanced_metrics['avg_confidence'])}

**Overall Improvement (Pure Semantic → Advanced):**
- F1@5: **{pure_to_adv_f1:+.1f}%**
- Groundedness: **{pure_to_adv_ground:+.1f}%**

---

## Metrics Comparison

| Tier | Features | F1@5 | Groundedness | Confidence | Avg Attempts |
|------|----------|------|--------------|------------|--------------|
| **Pure Semantic** | 4 | {pure_semantic_metrics['avg_f1_at_5']:.1%} | {pure_semantic_metrics['avg_groundedness']:.1%} | {pure_semantic_metrics['avg_confidence']:.1%} | {pure_semantic_metrics['avg_retrieval_attempts']:.1f} |
| **Basic** | 8 | {basic_metrics['avg_f1_at_5']:.1%} | {basic_metrics['avg_groundedness']:.1%} | {basic_metrics['avg_confidence']:.1%} | {basic_metrics['avg_retrieval_attempts']:.1f} |
| **Intermediate** | 18 | {intermediate_metrics['avg_f1_at_5']:.1%} | {intermediate_metrics['avg_groundedness']:.1%} | {intermediate_metrics['avg_confidence']:.1%} | {intermediate_metrics['avg_retrieval_attempts']:.1f} |
| **Advanced** | 31 | {advanced_metrics['avg_f1_at_5']:.1%} | {advanced_metrics['avg_groundedness']:.1%} | {advanced_metrics['avg_confidence']:.1%} | {advanced_metrics['avg_retrieval_attempts']:.1f} |

---

## Delta Analysis

### Pure Semantic → Basic (+4 features)

| Metric | Delta |
|--------|-------|
| F1@5 | {pure_to_basic_f1:+.1f}% |
| Groundedness | {pure_to_basic_ground:+.1f}% |

**Key Features Added:**
1. Query expansion (3 variants with RRF fusion)
2. Hybrid retrieval (semantic + BM25 keyword)
3. CrossEncoder reranking (top-5)
4. Enhanced answer generation prompting

### Basic → Intermediate (+10 features)

| Metric | Delta |
|--------|-------|
| F1@5 | {basic_to_inter_f1:+.1f}% |
| Groundedness | {basic_to_inter_ground:+.1f}% |

**Key Features Added:**
1. Conversational query rewriting
2. LLM-based strategy selection
3. Two-stage reranking (CrossEncoder + LLM-as-judge)
4. Binary retrieval quality scoring
5. Query rewriting loop (max 1 rewrite)
6. Answer quality check
7. Conditional routing (2 router functions)
8. Limited retry logic
9. LLM-based expansion decision
10. Message accumulation for multi-turn

### Intermediate → Advanced (+13 features)

| Metric | Delta |
|--------|-------|
| F1@5 | {inter_to_adv_f1:+.1f}% |
| Groundedness | {inter_to_adv_ground:+.1f}% |

**Key Features Added:**
1. NLI-based hallucination detection
2. Three-tier groundedness routing
3. Root cause detection (LLM vs retrieval hallucination)
4. Dual-tier strategy switching (early + late)
5. Query optimization for new strategy
6. Expansion regeneration on strategy change
7. Issue-specific feedback (8 retrieval types)
8. Adaptive thresholds (65% good, 50% poor)
9. Answer quality framework (8 issue types)
10. Content-driven issue → strategy mapping
11. Document profiling metadata
12. Advanced retry logic (2 rewrites, 3 attempts, 2 groundedness)
13. Complete metrics suite

### Pure Semantic → Advanced (Overall: +27 features)

| Metric | Delta |
|--------|-------|
| F1@5 | {pure_to_adv_f1:+.1f}% |
| Groundedness | {pure_to_adv_ground:+.1f}% |

---

## Feature Justification

### Why Basic Outperforms Pure Semantic

1. **Query Expansion:** Multiple query variants with RRF fusion capture different phrasings
2. **Hybrid Retrieval:** Combines semantic similarity (concepts) with BM25 keyword matching (exact terms)
3. **CrossEncoder Reranking:** Re-scores top candidates for better relevance
4. **Enhanced Prompting:** Better structured answer generation

### Why Intermediate Outperforms Basic

1. **Two-Stage Reranking:** CrossEncoder + LLM-as-judge provides better relevance filtering than CrossEncoder alone
2. **Strategy Selection:** LLM chooses optimal retrieval strategy (semantic/keyword/hybrid) per query
3. **Quality Gates:** Binary retrieval assessment prevents wasting generation on poor context
4. **Limited Retry:** One query rewrite opportunity improves results for borderline cases
5. **Expansion Decision:** LLM decides when expansion helps vs adds noise

### Why Advanced Outperforms Intermediate

1. **NLI Hallucination Detection:** Catches and corrects unsupported claims that pass simple quality checks
2. **Dual-Tier Strategy Switching:** Early detection saves tokens, late detection recovers from mistakes
3. **Root Cause Analysis:** Distinguishes LLM hallucination (regenerate) from retrieval gaps (re-retrieve)
4. **Issue-Specific Feedback:** 8 retrieval issue types enable targeted query improvements
5. **Adaptive Thresholds:** Quality expectations adjust based on retrieval performance (65% vs 50%)

---

## Portfolio Narrative

This comparison demonstrates that **advanced RAG architecture provides measurable value
independent of model quality**:

1. **Graph architecture provides baseline intelligence:** Even basic conditional routing
   (Basic → Intermediate) shows {basic_to_inter_f1:.0f}% improvement in F1@5

2. **Quality gates and limited retry add measurable value:** Simple binary quality checks
   and 1-2 retries (Intermediate) improve results without complex adaptive loops

3. **Full self-correction and adaptation maximize performance:** Advanced features
   (NLI detection, dual-tier switching, root cause analysis) provide additional
   {inter_to_adv_f1:.0f}% F1@5 improvement

4. **The value is in the architecture, not just the model:** All tiers use identical
   BUDGET models (gpt-4o-mini), yet Advanced tier shows {pure_to_adv_f1:.0f}% improvement
   over Pure Semantic through architecture alone

---

## Validation Against Expected Progression

### Expected Improvement Ranges

Based on architectural features added at each tier, we expect the following improvements:

| Transition | Metric | Expected Range | Actual | Status |
|------------|--------|----------------|--------|--------|
| Pure→Basic | F1@5 | +10-15% | {pure_to_basic_f1:+.1f}% | {'[OK]' if 10 <= pure_to_basic_f1 <= 15 else '[WARN]'} |
| Pure→Basic | Groundedness | +5-10% | {pure_to_basic_ground:+.1f}% | {'[OK]' if 5 <= pure_to_basic_ground <= 10 else '[WARN]'} |
| Basic→Intermediate | F1@5 | +15-25% | {basic_to_inter_f1:+.1f}% | {'[OK]' if 15 <= basic_to_inter_f1 <= 25 else '[WARN]'} |
| Basic→Intermediate | Groundedness | +8-15% | {basic_to_inter_ground:+.1f}% | {'[OK]' if 8 <= basic_to_inter_ground <= 15 else '[WARN]'} |
| Intermediate→Advanced | F1@5 | +20-35% | {inter_to_adv_f1:+.1f}% | {'[OK]' if 20 <= inter_to_adv_f1 <= 35 else '[WARN]'} |
| Intermediate→Advanced | Groundedness | +10-20% | {inter_to_adv_ground:+.1f}% | {'[OK]' if 10 <= inter_to_adv_ground <= 20 else '[WARN]'} |
| Pure→Advanced (Overall) | F1@5 | +45-75% | {pure_to_adv_f1:+.1f}% | {'[OK]' if 45 <= pure_to_adv_f1 <= 75 else '[WARN]'} |
| Pure→Advanced (Overall) | Groundedness | +25-40% | {pure_to_adv_ground:+.1f}% | {'[OK]' if 25 <= pure_to_adv_ground <= 40 else '[WARN]'} |

### Interpretation

- **[OK]**: Actual improvement falls within expected range (architectural features performing as designed)
- **[WARN]**: Actual improvement outside expected range (may indicate dataset-specific behavior or need for tuning)

**Note:** Expected ranges are based on incremental architectural complexity:
- Pure→Basic adds 4 features (hybrid search, expansion, basic reranking)
- Basic→Intermediate adds 10 features (quality gates, two-stage reranking, limited retry)
- Intermediate→Advanced adds 13 features (NLI detection, dual-tier switching, adaptive thresholds)

---

## Per-Example Analysis

### Success Rate by Tier

- **Pure Semantic:** {pure_semantic_metrics['successful_examples']}/{pure_semantic_metrics['total_examples']} ({(pure_semantic_metrics['successful_examples']/pure_semantic_metrics['total_examples']*100):.0f}%)
- **Basic:** {basic_metrics['successful_examples']}/{basic_metrics['total_examples']} ({(basic_metrics['successful_examples']/basic_metrics['total_examples']*100):.0f}%)
- **Intermediate:** {intermediate_metrics['successful_examples']}/{intermediate_metrics['total_examples']} ({(intermediate_metrics['successful_examples']/intermediate_metrics['total_examples']*100):.0f}%)
- **Advanced:** {advanced_metrics['successful_examples']}/{advanced_metrics['total_examples']} ({(advanced_metrics['successful_examples']/advanced_metrics['total_examples']*100):.0f}%)

### Top Performing Examples (Advanced Tier)

{_format_top_examples(advanced_results, n=5)}

### Most Improved Examples (Basic → Advanced)

{_format_most_improved(basic_results, advanced_results, n=5)}

---

## Methodology

**Dataset:** {basic_metrics['total_examples']} validated examples from golden set
**Model Tier:** BUDGET (gpt-4o-mini) for all tiers
**Metrics:**
- **F1@5:** Harmonic mean of Precision@5 and Recall@5 (retrieval quality)
- **Groundedness:** NLI-based verification (% claims supported by context)
- **Confidence:** LLM confidence score (answer quality)
- **Avg Attempts:** Average retrieval attempts per query (efficiency)

**Evaluation:** Offline evaluation using ground truth relevance labels

---

*Report generated by `test_architecture_comparison.py`*
"""

    return report


def _get_winner(pure_semantic: float, basic: float, intermediate: float, advanced: float) -> str:
    """Determine which tier won for a metric."""
    scores = {
        "Pure Semantic": pure_semantic,
        "Basic": basic,
        "Intermediate": intermediate,
        "Advanced": advanced
    }
    winner = max(scores, key=scores.get)
    return f"{winner} ({scores[winner]:.1%})"


def _format_top_examples(results: List[Dict], n: int = 5) -> str:
    """Format top N performing examples."""
    valid_results = [r for r in results if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x["f1_at_5"], reverse=True)[:n]

    lines = []
    for i, r in enumerate(sorted_results, 1):
        lines.append(f"{i}. **{r['example_id']}**: F1@5={r['f1_at_5']:.0%}, Ground={r['groundedness_score']:.0%}")

    return "\n".join(lines) if lines else "*No successful examples*"


def _format_most_improved(basic_results: List[Dict], advanced_results: List[Dict], n: int = 5) -> str:
    """Format examples with biggest improvement from basic to advanced."""
    improvements = []

    for basic_r, adv_r in zip(basic_results, advanced_results):
        if "error" in basic_r or "error" in adv_r:
            continue

        delta_f1 = adv_r["f1_at_5"] - basic_r["f1_at_5"]
        improvements.append({
            "example_id": adv_r["example_id"],
            "delta_f1": delta_f1,
            "basic_f1": basic_r["f1_at_5"],
            "adv_f1": adv_r["f1_at_5"],
        })

    sorted_improvements = sorted(improvements, key=lambda x: x["delta_f1"], reverse=True)[:n]

    lines = []
    for i, imp in enumerate(sorted_improvements, 1):
        lines.append(
            f"{i}. **{imp['example_id']}**: {imp['basic_f1']:.0%} → {imp['adv_f1']:.0%} "
            f"(**{imp['delta_f1']:+.0%}**)"
        )

    return "\n".join(lines) if lines else "*No improvements found*"


# ========== MAIN TEST ==========

def test_architecture_comparison(quick_mode: bool = False, dataset_type: str = "standard"):
    """
    Main comparison test - run all 4 tiers on golden dataset.

    Args:
        quick_mode: If True, evaluate only first 2 examples
        dataset_type: Dataset to evaluate ('standard' or 'hard')

    Generates:
    - evaluation/architecture_comparison_results_{dataset_type}_{timestamp}.json (raw data)
    - evaluation/architecture_comparison_report_{dataset_type}_{timestamp}.md (formatted report)
    - evaluation/architecture_comparison_results_{dataset_type}_latest.json (convenience copy)
    - evaluation/architecture_comparison_report_{dataset_type}_latest.md (convenience copy)
    """
    print("\n" + "="*80)
    print("4-TIER ARCHITECTURE COMPARISON TEST")
    print("="*80)
    print("Model Tier: BUDGET (gpt-4o-mini for all tiers)")
    print("Tiers: Pure Semantic (4), Basic (8), Intermediate (18), Advanced (31 features)")
    print(f"Dataset: {dataset_type}")
    print(f"Mode: {'Quick (2 examples)' if quick_mode else 'Full'}")
    print("="*80 + "\n")

    # Load golden dataset with adaptive k_final
    print(f"[*] Loading {dataset_type} golden dataset...")
    if dataset_type == "standard":
        dataset_path = "evaluation/golden_set_standard.json"
        k_final = 4  # Optimal for 1-3 chunk questions
    else:  # hard
        dataset_path = "evaluation/golden_set_hard.json"
        k_final = 6  # Adaptive retrieval for 3-5 chunk questions

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

    # PRE-BUILD RETRIEVER ONCE (optimization to avoid redundant PDF re-ingestion)
    print(f"\n{'='*80}")
    print(f"PRE-BUILD: Initializing retriever once for all tiers (k_final={k_final})")
    print(f"{'='*80}")
    print("    This avoids re-ingesting PDFs for each tier (saves 50-60% time)")
    shared_retriever = setup_retriever(k_final=k_final)

    # Inject into all four variant modules
    pure_semantic_module.adaptive_retriever = shared_retriever
    basic_module.adaptive_retriever = shared_retriever
    intermediate_module.adaptive_retriever = shared_retriever
    advanced_module.adaptive_retriever = shared_retriever
    print(f"[OK] Retriever pre-built and injected into all tiers (k_final={k_final})")
    print(f"{'='*80}\n")

    # Run Pure Semantic Tier
    print(f"\n{'='*80}")
    print("[1/4] Running PURE SEMANTIC tier (4 features)...")
    print(f"{'='*80}")
    pure_semantic_results = run_tier_on_golden_dataset("pure_semantic", pure_semantic_rag_graph, dataset)
    pure_semantic_metrics = calculate_tier_metrics(pure_semantic_results)

    # Run Basic Tier
    print(f"\n{'='*80}")
    print("[2/4] Running BASIC tier (8 features)...")
    print(f"{'='*80}")
    basic_results = run_tier_on_golden_dataset("basic", basic_rag_graph, dataset)
    basic_metrics = calculate_tier_metrics(basic_results)

    # Run Intermediate Tier
    print(f"\n{'='*80}")
    print("[3/4] Running INTERMEDIATE tier (18 features)...")
    print(f"{'='*80}")
    intermediate_results = run_tier_on_golden_dataset("intermediate", intermediate_rag_graph, dataset)
    intermediate_metrics = calculate_tier_metrics(intermediate_results)

    # Run Advanced Tier
    print(f"\n{'='*80}")
    print("[4/4] Running ADVANCED tier (31 features)...")
    print(f"{'='*80}")
    advanced_results = run_tier_on_golden_dataset("advanced", advanced_rag_graph, dataset)
    advanced_metrics = calculate_tier_metrics(advanced_results)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Tier':<15} {'F1@5':<10} {'Groundedness':<15} {'Confidence':<12} {'Attempts':<10}")
    print("-" * 80)
    print(f"{'Pure Semantic':<15} {pure_semantic_metrics['avg_f1_at_5']:<10.1%} {pure_semantic_metrics['avg_groundedness']:<15.1%} {pure_semantic_metrics['avg_confidence']:<12.1%} {pure_semantic_metrics['avg_retrieval_attempts']:<10.1f}")
    print(f"{'Basic':<15} {basic_metrics['avg_f1_at_5']:<10.1%} {basic_metrics['avg_groundedness']:<15.1%} {basic_metrics['avg_confidence']:<12.1%} {basic_metrics['avg_retrieval_attempts']:<10.1f}")
    print(f"{'Intermediate':<15} {intermediate_metrics['avg_f1_at_5']:<10.1%} {intermediate_metrics['avg_groundedness']:<15.1%} {intermediate_metrics['avg_confidence']:<12.1%} {intermediate_metrics['avg_retrieval_attempts']:<10.1f}")
    print(f"{'Advanced':<15} {advanced_metrics['avg_f1_at_5']:<10.1%} {advanced_metrics['avg_groundedness']:<15.1%} {advanced_metrics['avg_confidence']:<12.1%} {advanced_metrics['avg_retrieval_attempts']:<10.1f}")
    print("=" * 80 + "\n")

    # Save raw results with timestamp
    os.makedirs("evaluation", exist_ok=True)

    # Generate consistent timestamp for both files
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "architecture_comparison",
        "dataset_type": dataset_type,
        "model_tier": "BUDGET (gpt-4o-mini)",
        "dataset_size": len(dataset),
        "k_final": k_final,
        "quick_mode": quick_mode,
        "tiers": {
            "pure_semantic": {
                "metrics": pure_semantic_metrics,
                "results": pure_semantic_results,
            },
            "basic": {
                "metrics": basic_metrics,
                "results": basic_results,
            },
            "intermediate": {
                "metrics": intermediate_metrics,
                "results": intermediate_results,
            },
            "advanced": {
                "metrics": advanced_metrics,
                "results": advanced_results,
            },
        },
    }

    # Save with timestamp
    results_path = Path("evaluation") / f"architecture_comparison_results_{dataset_type}_{test_timestamp}.json"
    latest_results_path = Path("evaluation") / f"architecture_comparison_results_{dataset_type}_latest.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print(f"[OK] Saved raw results to {results_path}")

    # Create latest copy
    shutil.copy2(results_path, latest_results_path)
    print(f"[OK] Latest copy saved to {latest_results_path}")

    # Generate and save report
    report = generate_comparison_report(
        pure_semantic_metrics, basic_metrics, intermediate_metrics, advanced_metrics,
        pure_semantic_results, basic_results, intermediate_results, advanced_results,
        dataset_type, test_timestamp
    )

    report_path = Path("evaluation") / f"architecture_comparison_report_{dataset_type}_{test_timestamp}.md"
    latest_report_path = Path("evaluation") / f"architecture_comparison_report_{dataset_type}_latest.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] Saved comparison report to {report_path}")

    # Create latest copy
    shutil.copy2(report_path, latest_report_path)
    print(f"[OK] Latest copy saved to {latest_report_path}")

    print("\n" + "="*80)
    print("COMPARISON TEST COMPLETE")
    print("="*80)
    print(f"Results: {results_path}")
    print(f"Report: {report_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='4-Tier Architecture Comparison Evaluation')
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

    if args.quick:
        print(f"[*] Running in quick mode (2 examples from {args.dataset} dataset)")
    else:
        dataset_size = "20 examples" if args.dataset == "standard" else "10 examples"
        expected_time = "75-90 minutes" if args.dataset == "standard" else "35-45 minutes"
        print(f"[*] Running full evaluation on {args.dataset} dataset ({dataset_size})")
        print(f"[*] This will take approximately {expected_time}")
        print("[*] Use --quick flag for faster testing (~6-7 minutes)")

    test_architecture_comparison(quick_mode=args.quick, dataset_type=args.dataset)
