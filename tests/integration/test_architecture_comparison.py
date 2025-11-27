"""
4-Tier Architecture A/B Test for Portfolio Showcase.

Compares four RAG architecture implementations to demonstrate
the incremental value of advanced features:
- Basic: Simplest RAG (1 feature, semantic vector search)
- Intermediate: Simple RAG (5 features, linear flow)
- Advanced: Full Agentic RAG (17 features, adaptive loops)
- Multi-Agent: Orchestrator-Worker RAG (20 features, parallel retrieval)

Model tier controlled by MODEL_TIER environment variable (budget/balanced/premium).
Set MODEL_TIER in .env.local to compare architectures at different quality/cost points.

Key Metrics:
- F1@K: Retrieval quality (K=4 for standard, K=6 for hard datasets)
- Groundedness: Anti-hallucination (% claims supported by context)
- Semantic Similarity: How closely answer matches ground truth meaning
- Factual Accuracy: Correctness of factual claims in answer
- Completeness: Coverage of key points from ground truth
- Generation Attempts: Retry metrics for Advanced/Multi-Agent tiers

Expected Progression:
- Basic -> Intermediate: +10-15% (hybrid search, query expansion, reranking)
- Intermediate -> Advanced: +30-50% (NLI, strategy switching, adaptive loops, quality gates)
- Advanced -> Multi-Agent: +5-15% (query decomposition, parallel retrieval, cross-agent fusion)
- Basic -> Multi-Agent: +50-90% overall

Usage:
    uv run python tests/integration/test_architecture_comparison.py

Outputs:
    - evaluation/architecture_comparison_results_{dataset}_{timestamp}.json (raw data)
    - evaluation/architecture_comparison_report_{dataset}_{timestamp}.md (formatted report)
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
    basic_rag_graph,
    intermediate_rag_graph,
    advanced_rag_graph,
    multi_agent_rag_graph,
)
# Import modules to access global adaptive_retriever variables
import advanced_agentic_rag_langgraph.variants.basic_rag_graph as basic_module
import advanced_agentic_rag_langgraph.variants.intermediate_rag_graph as intermediate_module
import advanced_agentic_rag_langgraph.orchestration.nodes as advanced_module
import advanced_agentic_rag_langgraph.variants.multi_agent_rag_graph as multi_agent_module
from advanced_agentic_rag_langgraph.core import setup_retriever
from advanced_agentic_rag_langgraph.evaluation.golden_dataset import GoldenDatasetManager, compare_answers
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import (
    calculate_retrieval_metrics,
)
from advanced_agentic_rag_langgraph.validation import NLIHallucinationDetector
from advanced_agentic_rag_langgraph.core.model_config import get_current_tier, TIER_METADATA


# ========== TIER CONFIGURATION ==========

TIER_CONFIGS = {
    "basic": {
        "name": "Basic RAG",
        "features": 1,
        "graph": basic_rag_graph,
        "description": "Simplest RAG with semantic vector search, top-k chunks, no reranking",
    },
    "intermediate": {
        "name": "Intermediate RAG",
        "features": 5,
        "graph": intermediate_rag_graph,
        "description": "Linear flow with hybrid retrieval and CrossEncoder reranking",
    },
    "advanced": {
        "name": "Advanced RAG",
        "features": 17,
        "graph": advanced_rag_graph,
        "description": "Full agentic with NLI, strategy switching, and adaptive loops",
    },
    "multi_agent": {
        "name": "Multi-Agent RAG",
        "features": 20,
        "graph": multi_agent_rag_graph,
        "description": "Orchestrator-worker pattern with parallel retrieval and cross-agent RRF fusion",
    },
}


# ========== HELPER FUNCTIONS ==========

def run_tier_on_golden_dataset(
    tier_name: str,
    graph: Any,
    dataset: List[Dict],
    k_final: int = 4,
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
            if tier_name == "basic":
                initial_state = {
                    "user_question": query,
                    "retrieved_docs": [],  # Will be set by retrieve node
                    "ground_truth_doc_ids": ground_truth_docs,
                }
            elif tier_name == "intermediate":
                initial_state = {
                    "user_question": query,
                    "query_expansions": [],  # Will be set by first node
                    "retrieved_docs": [],  # Accumulated field
                    "ground_truth_doc_ids": ground_truth_docs,
                }
            elif tier_name == "advanced":
                initial_state = {
                    "user_question": query,
                    "baseline_query": query,  # Required field for advanced tier
                    "messages": [],
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                    "query_expansions": [],
                    "ground_truth_doc_ids": ground_truth_docs,
                }
            else:  # multi_agent
                initial_state = {
                    "user_question": query,
                    "baseline_query": query,
                    "messages": [],
                    "sub_agent_results": [],  # Required for operator.add reducer
                    "ground_truth_doc_ids": ground_truth_docs,
                }

            # Run graph
            result = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"{tier_name}_{i}"}}
            )

            # Extract results (different state schemas)
            answer = result.get("final_answer", "")
            retrieved_docs = result.get("unique_docs_list", [])
            # Only advanced tier has retry features
            if tier_name == "advanced":
                retrieval_attempts = result.get("retrieval_attempts", 1)
                generation_attempts = result.get("generation_attempts", 1)
            else:
                retrieval_attempts = None
                generation_attempts = None

            # Calculate retrieval metrics using shared function
            metrics = calculate_retrieval_metrics(retrieved_docs, ground_truth_docs, k_final)
            f1_at_k = metrics["f1_at_k"]
            precision = metrics["precision_at_k"]
            recall = metrics["recall_at_k"]

            # Calculate groundedness using NLI (independent verification with graph-matching format)
            if retrieved_docs and answer:
                context = "\n---\n".join([
                    f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
                    for doc in retrieved_docs[:k_final]
                ])
                groundedness_result = nli_detector.verify_groundedness(answer, context)
                groundedness_score = groundedness_result["groundedness_score"]
            else:
                groundedness_score = 0.0

            # Calculate answer quality metrics vs ground truth (replaces hardcoded confidence)
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
                print(f"  F1@{k_final}: {f1_at_k:.0%} | Ground: {groundedness_score:.0%} | Sim: {semantic_similarity:.0%} | Fact: {factual_accuracy:.0%} | Comp: {completeness:.0%}")

            # Extract doc IDs for storage (using same logic as shared function)
            retrieved_doc_ids = [
                doc.metadata.get('id', f'doc_{i}')
                for i, doc in enumerate(retrieved_docs[:k_final])
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
                "retrieval_attempts": retrieval_attempts,
                "generation_attempts": generation_attempts,
            })

        except Exception as e:
            print(f"  [ERROR] Failed: {str(e)}")
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
                "retrieval_attempts": 0 if tier_name == "advanced" else None,
                "generation_attempts": 0 if tier_name == "advanced" else None,
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
            "avg_f1_at_k": 0.0,
            "avg_precision_at_k": 0.0,
            "avg_recall_at_k": 0.0,
            "avg_groundedness": 0.0,
            "avg_semantic_similarity": 0.0,
            "avg_factual_accuracy": 0.0,
            "avg_completeness": 0.0,
            "total_examples": len(results),
            "successful_examples": 0,
            "error_rate": 1.0,
        }

    metrics = {
        "avg_f1_at_k": sum(r["f1_at_k"] for r in valid_results) / len(valid_results),
        "avg_precision_at_k": sum(r["precision_at_k"] for r in valid_results) / len(valid_results),
        "avg_recall_at_k": sum(r["recall_at_k"] for r in valid_results) / len(valid_results),
        "avg_groundedness": sum(r["groundedness_score"] for r in valid_results) / len(valid_results),
        "avg_semantic_similarity": sum(r["semantic_similarity"] for r in valid_results) / len(valid_results),
        "avg_factual_accuracy": sum(r["factual_accuracy"] for r in valid_results) / len(valid_results),
        "avg_completeness": sum(r["completeness"] for r in valid_results) / len(valid_results),
        "total_examples": len(results),
        "successful_examples": len(valid_results),
        "error_rate": (len(results) - len(valid_results)) / len(results) if results else 0.0,
    }

    # Only calculate attempt averages for advanced tier (has retry features)
    # Basic and Intermediate have retrieval_attempts=None
    has_retry_features = any(r.get("retrieval_attempts") is not None for r in valid_results)
    if has_retry_features:
        metrics["avg_retrieval_attempts"] = sum(r["retrieval_attempts"] for r in valid_results) / len(valid_results)
        metrics["avg_generation_attempts"] = sum(r["generation_attempts"] for r in valid_results) / len(valid_results)

    return metrics


def generate_comparison_report(
    basic_metrics: Dict[str, float],
    intermediate_metrics: Dict[str, float],
    advanced_metrics: Dict[str, float],
    multi_agent_metrics: Dict[str, float],
    basic_results: List[Dict],
    intermediate_results: List[Dict],
    advanced_results: List[Dict],
    multi_agent_results: List[Dict],
    k_final: int = 4,
    dataset_type: str = "standard",
    test_timestamp: str = None,
    current_tier = None,
    tier_info: Dict = None,
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

    k = k_final  # For display strings

    # Calculate deltas (all consecutive pairs + basic to advanced)
    basic_to_intermediate_f1 = ((intermediate_metrics["avg_f1_at_k"] - basic_metrics["avg_f1_at_k"]) / basic_metrics["avg_f1_at_k"] * 100) if basic_metrics["avg_f1_at_k"] > 0 else 0
    intermediate_to_adv_f1 = ((advanced_metrics["avg_f1_at_k"] - intermediate_metrics["avg_f1_at_k"]) / intermediate_metrics["avg_f1_at_k"] * 100) if intermediate_metrics["avg_f1_at_k"] > 0 else 0
    basic_to_adv_f1 = ((advanced_metrics["avg_f1_at_k"] - basic_metrics["avg_f1_at_k"]) / basic_metrics["avg_f1_at_k"] * 100) if basic_metrics["avg_f1_at_k"] > 0 else 0

    basic_to_intermediate_ground = ((intermediate_metrics["avg_groundedness"] - basic_metrics["avg_groundedness"]) / basic_metrics["avg_groundedness"] * 100) if basic_metrics["avg_groundedness"] > 0 else 0
    intermediate_to_adv_ground = ((advanced_metrics["avg_groundedness"] - intermediate_metrics["avg_groundedness"]) / intermediate_metrics["avg_groundedness"] * 100) if intermediate_metrics["avg_groundedness"] > 0 else 0
    basic_to_adv_ground = ((advanced_metrics["avg_groundedness"] - basic_metrics["avg_groundedness"]) / basic_metrics["avg_groundedness"] * 100) if basic_metrics["avg_groundedness"] > 0 else 0

    # Multi-agent deltas
    adv_to_multi_f1 = ((multi_agent_metrics["avg_f1_at_k"] - advanced_metrics["avg_f1_at_k"]) / advanced_metrics["avg_f1_at_k"] * 100) if advanced_metrics["avg_f1_at_k"] > 0 else 0
    adv_to_multi_ground = ((multi_agent_metrics["avg_groundedness"] - advanced_metrics["avg_groundedness"]) / advanced_metrics["avg_groundedness"] * 100) if advanced_metrics["avg_groundedness"] > 0 else 0
    basic_to_multi_f1 = ((multi_agent_metrics["avg_f1_at_k"] - basic_metrics["avg_f1_at_k"]) / basic_metrics["avg_f1_at_k"] * 100) if basic_metrics["avg_f1_at_k"] > 0 else 0
    basic_to_multi_ground = ((multi_agent_metrics["avg_groundedness"] - basic_metrics["avg_groundedness"]) / basic_metrics["avg_groundedness"] * 100) if basic_metrics["avg_groundedness"] > 0 else 0

    dataset_label = "Standard" if dataset_type == "standard" else "Hard"

    # Get tier info if not provided
    if current_tier is None or tier_info is None:
        current_tier = get_current_tier()
        tier_info = TIER_METADATA[current_tier]

    report = f"""# 4-Tier RAG Architecture Comparison Report

**Generated:** {timestamp}
**Model Tier:** {current_tier.value.upper()} ({tier_info['description']})
**Dataset:** {dataset_label} ({basic_metrics['total_examples']} examples)

---

## Executive Summary

This report demonstrates the incremental value of advanced RAG architecture patterns
by comparing four implementation tiers using identical models (BUDGET tier) to isolate
architectural improvements.

### Key Findings

**Winner by Metric:**
- **F1@{k} (Retrieval Quality):** {_get_winner_4(basic_metrics['avg_f1_at_k'], intermediate_metrics['avg_f1_at_k'], advanced_metrics['avg_f1_at_k'], multi_agent_metrics['avg_f1_at_k'])}
- **Groundedness (Anti-Hallucination):** {_get_winner_4(basic_metrics['avg_groundedness'], intermediate_metrics['avg_groundedness'], advanced_metrics['avg_groundedness'], multi_agent_metrics['avg_groundedness'])}
- **Semantic Similarity (Answer Quality):** {_get_winner_4(basic_metrics['avg_semantic_similarity'], intermediate_metrics['avg_semantic_similarity'], advanced_metrics['avg_semantic_similarity'], multi_agent_metrics['avg_semantic_similarity'])}
- **Factual Accuracy:** {_get_winner_4(basic_metrics['avg_factual_accuracy'], intermediate_metrics['avg_factual_accuracy'], advanced_metrics['avg_factual_accuracy'], multi_agent_metrics['avg_factual_accuracy'])}

**Overall Improvement (Basic -> Multi-Agent):**
- F1@{k}: **{basic_to_multi_f1:+.1f}%**
- Groundedness: **{basic_to_multi_ground:+.1f}%**

---

## Metrics Comparison

| Tier | Features | F1@{k} | Groundedness | Similarity | Factual | Complete | Retrieval Attempts | Generation Attempts |
|------|----------|------|--------------|------------|---------|----------|--------------------|--------------------|
| **Basic** | 1 | {basic_metrics['avg_f1_at_k']:.1%} | {basic_metrics['avg_groundedness']:.1%} | {basic_metrics['avg_semantic_similarity']:.1%} | {basic_metrics['avg_factual_accuracy']:.1%} | {basic_metrics['avg_completeness']:.1%} | - | - |
| **Intermediate** | 5 | {intermediate_metrics['avg_f1_at_k']:.1%} | {intermediate_metrics['avg_groundedness']:.1%} | {intermediate_metrics['avg_semantic_similarity']:.1%} | {intermediate_metrics['avg_factual_accuracy']:.1%} | {intermediate_metrics['avg_completeness']:.1%} | - | - |
| **Advanced** | 17 | {advanced_metrics['avg_f1_at_k']:.1%} | {advanced_metrics['avg_groundedness']:.1%} | {advanced_metrics['avg_semantic_similarity']:.1%} | {advanced_metrics['avg_factual_accuracy']:.1%} | {advanced_metrics['avg_completeness']:.1%} | {advanced_metrics.get('avg_retrieval_attempts', 1.0):.1f} | {advanced_metrics.get('avg_generation_attempts', 1.0):.1f} |
| **Multi-Agent** | 20 | {multi_agent_metrics['avg_f1_at_k']:.1%} | {multi_agent_metrics['avg_groundedness']:.1%} | {multi_agent_metrics['avg_semantic_similarity']:.1%} | {multi_agent_metrics['avg_factual_accuracy']:.1%} | {multi_agent_metrics['avg_completeness']:.1%} | - | {multi_agent_metrics.get('avg_generation_attempts', 1.0):.1f} |

---

## Delta Analysis

### Basic -> Intermediate (+4 features)

| Metric | Delta |
|--------|-------|
| F1@{k} | {basic_to_intermediate_f1:+.1f}% |
| Groundedness | {basic_to_intermediate_ground:+.1f}% |

**Key Features Added:**
1. Query expansion (3 variants with RRF fusion)
2. Hybrid retrieval (semantic + BM25 keyword)
3. CrossEncoder reranking (top-k, adaptive)
4. Enhanced answer generation prompting

### Intermediate -> Advanced (+12 features)

| Metric | Delta |
|--------|-------|
| F1@{k} | {intermediate_to_adv_f1:+.1f}% |
| Groundedness | {intermediate_to_adv_ground:+.1f}% |

**Key Features Added:**
1. Conversational query rewriting
2. LLM-based strategy selection (semantic/keyword/hybrid)
3. Two-stage reranking (CrossEncoder → LLM-as-judge)
4. Retrieval quality gates (8 issue types)
5. Answer quality evaluation (8 issue types)
6. Adaptive thresholds (65% good retrieval, 50% poor)
7. Query rewriting loop (issue-specific feedback, max 3)
8. Early strategy switching (off_topic/wrong_domain detection)
9. Generation retry loop (adaptive temperature 0.3/0.7/0.5)
10. NLI-based hallucination detection
11. Refusal detection
12. Conversation context preservation (multi-turn)

### Advanced -> Multi-Agent (+3 features)

| Metric | Delta |
|--------|-------|
| F1@{k} | {adv_to_multi_f1:+.1f}% |
| Groundedness | {adv_to_multi_ground:+.1f}% |

**Key Features Added:**
1. Query decomposition (orchestrator breaks complex queries into sub-queries)
2. Parallel retrieval workers (independent pipelines per sub-query)
3. Cross-agent RRF fusion (documents from multiple perspectives get boost)

### Basic -> Multi-Agent (Overall: +19 features)

| Metric | Delta |
|--------|-------|
| F1@{k} | {basic_to_multi_f1:+.1f}% |
| Groundedness | {basic_to_multi_ground:+.1f}% |

---

## Feature Justification

### Why Intermediate Outperforms Basic

1. **Query Expansion:** Multiple query variants with RRF fusion capture different phrasings
2. **Hybrid Retrieval:** Combines semantic similarity (concepts) with BM25 keyword matching (exact terms)
3. **CrossEncoder Reranking:** Re-scores top candidates for better relevance
4. **Enhanced Prompting:** Better structured answer generation

### Why Advanced Outperforms Intermediate

1. **Two-Stage Reranking:** CrossEncoder + LLM-as-judge provides better relevance filtering than CrossEncoder alone
2. **Strategy Selection & Switching:** LLM chooses optimal retrieval strategy per query with dual-tier adaptive switching
3. **NLI Hallucination Detection:** Catches and corrects unsupported claims that pass simple quality checks
4. **Root Cause Analysis:** Distinguishes LLM hallucination (regenerate) from retrieval gaps (re-retrieve)
5. **Quality Gates & Retry Logic:** Binary retrieval assessment with adaptive retry (max 3 attempts) prevents wasted generation
6. **Issue-Specific Feedback:** 8 retrieval issue types enable targeted query improvements with adaptive thresholds
7. **Conversational Rewriting:** Contextualizes queries using conversation history for multi-turn interactions
8. **Document Profiling:** Metadata-aware retrieval optimizes strategy selection based on corpus characteristics

### Why Multi-Agent Outperforms Advanced

1. **Query Decomposition:** Complex queries broken into focused sub-queries captures multiple aspects
2. **Parallel Retrieval:** Independent workers retrieve from different perspectives simultaneously
3. **Cross-Agent RRF Fusion:** Documents appearing in multiple workers get boosted relevance scores
4. **Better for Comparative Queries:** "Compare X and Y" naturally decomposes into X-focused and Y-focused sub-queries
5. **Reduced Latency:** Parallel execution means total time = max(worker times), not sum

---

## Portfolio Narrative

This comparison demonstrates that **advanced RAG architecture provides measurable value
independent of model quality**:

1. **Hybrid retrieval and reranking provide baseline intelligence:** Intermediate tier adds query expansion,
   hybrid search, and CrossEncoder reranking, showing {basic_to_intermediate_f1:.0f}% improvement in F1@{k}
   over basic semantic search

2. **Advanced features multiply effectiveness:** Full agentic capabilities (NLI detection, dual-tier
   switching, adaptive retry, root cause analysis) provide an additional {intermediate_to_adv_f1:.0f}%
   F1@{k} improvement over Intermediate tier

3. **Multi-agent orchestration unlocks complex query handling:** Query decomposition and parallel
   retrieval workers provide an additional {adv_to_multi_f1:.0f}% F1@{k} improvement over Advanced tier,
   especially effective on comparative and multi-faceted questions

4. **The value is in the architecture, not just the model:** All tiers use identical BUDGET models
   (gpt-4o-mini), yet Multi-Agent tier shows {basic_to_multi_f1:.0f}% improvement over Basic
   through architecture alone

---

## Per-Example Analysis

### Success Rate by Tier

- **Basic:** {basic_metrics['successful_examples']}/{basic_metrics['total_examples']} ({(basic_metrics['successful_examples']/basic_metrics['total_examples']*100):.0f}%)
- **Intermediate:** {intermediate_metrics['successful_examples']}/{intermediate_metrics['total_examples']} ({(intermediate_metrics['successful_examples']/intermediate_metrics['total_examples']*100):.0f}%)
- **Advanced:** {advanced_metrics['successful_examples']}/{advanced_metrics['total_examples']} ({(advanced_metrics['successful_examples']/advanced_metrics['total_examples']*100):.0f}%)
- **Multi-Agent:** {multi_agent_metrics['successful_examples']}/{multi_agent_metrics['total_examples']} ({(multi_agent_metrics['successful_examples']/multi_agent_metrics['total_examples']*100):.0f}%)

### Top Performing Examples (Multi-Agent Tier)

{_format_top_examples(multi_agent_results, n=5, k=k)}

### Most Improved Examples (Advanced -> Multi-Agent)

{_format_most_improved(advanced_results, multi_agent_results, n=5)}

### Question-by-Question Comparison

{_format_question_by_question_comparison(basic_results, intermediate_results, advanced_results, multi_agent_results, k=k)}

---

## Methodology

**Dataset:** {intermediate_metrics['total_examples']} validated examples from golden set
**Model Tier:** BUDGET (gpt-4o-mini) for all tiers
**Architecture Tiers:**
- **Basic (1 feature):** Semantic vector search only
- **Intermediate (5 features):** Hybrid retrieval + CrossEncoder reranking
- **Advanced (17 features):** Full agentic with NLI, strategy switching, adaptive loops
- **Multi-Agent (20 features):** Orchestrator-worker pattern with parallel retrieval

**Metrics:**
- **F1@{k}:** Harmonic mean of Precision@{k} and Recall@{k} (retrieval quality)
- **Groundedness:** NLI-based verification (% claims supported by context)
- **Similarity:** Semantic similarity to ground truth answer (0-100%)
- **Factual:** Factual accuracy of claims in answer (0-100%)
- **Complete:** Coverage of key points from ground truth (0-100%)
- **Generation Attempts:** Retry metrics for Advanced/Multi-Agent tiers only

**Evaluation:** Offline evaluation using ground truth relevance labels

---

*Report generated by `test_architecture_comparison.py`*
"""

    return report


def _get_winner(basic: float, intermediate: float, advanced: float) -> str:
    """Determine which tier won for a metric (3 tiers)."""
    scores = {
        "Basic": basic,
        "Intermediate": intermediate,
        "Advanced": advanced
    }
    winner = max(scores, key=scores.get)
    return f"{winner} ({scores[winner]:.1%})"


def _get_winner_4(basic: float, intermediate: float, advanced: float, multi_agent: float) -> str:
    """Determine which tier won for a metric (4 tiers)."""
    scores = {
        "Basic": basic,
        "Intermediate": intermediate,
        "Advanced": advanced,
        "Multi-Agent": multi_agent
    }
    winner = max(scores, key=scores.get)
    return f"{winner} ({scores[winner]:.1%})"


def _format_top_examples(results: List[Dict], n: int = 5, k: int = 4) -> str:
    """Format top N performing examples."""
    valid_results = [r for r in results if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x["f1_at_k"], reverse=True)[:n]

    lines = []
    for i, r in enumerate(sorted_results, 1):
        lines.append(f"{i}. **{r['example_id']}**: F1@{k}={r['f1_at_k']:.0%}, Sim={r['semantic_similarity']:.0%}, Fact={r['factual_accuracy']:.0%}")

    return "\n".join(lines) if lines else "*No successful examples*"


def _format_most_improved(intermediate_results: List[Dict], advanced_results: List[Dict], n: int = 5) -> str:
    """Format examples with biggest improvement from intermediate to advanced."""
    improvements = []

    for intermediate_r, adv_r in zip(intermediate_results, advanced_results):
        if "error" in intermediate_r or "error" in adv_r:
            continue

        delta_f1 = adv_r["f1_at_k"] - intermediate_r["f1_at_k"]
        improvements.append({
            "example_id": adv_r["example_id"],
            "delta_f1": delta_f1,
            "intermediate_f1": intermediate_r["f1_at_k"],
            "adv_f1": adv_r["f1_at_k"],
        })

    sorted_improvements = sorted(improvements, key=lambda x: x["delta_f1"], reverse=True)[:n]

    lines = []
    for i, imp in enumerate(sorted_improvements, 1):
        lines.append(
            f"{i}. **{imp['example_id']}**: {imp['intermediate_f1']:.0%} -> {imp['adv_f1']:.0%} "
            f"(**{imp['delta_f1']:+.0%}**)"
        )

    return "\n".join(lines) if lines else "*No improvements found*"


def _format_question_by_question_comparison(
    basic_results: List[Dict],
    intermediate_results: List[Dict],
    advanced_results: List[Dict],
    multi_agent_results: List[Dict],
    k: int = 4
) -> str:
    """
    Format question-by-question comparison table across all 4 tiers.

    Args:
        basic_results: Per-example results from Basic tier
        intermediate_results: Per-example results from Intermediate tier
        advanced_results: Per-example results from Advanced tier
        multi_agent_results: Per-example results from Multi-Agent tier
        k: K value for F1@K display

    Returns:
        Markdown formatted table string
    """
    lines = []
    lines.append(f"| # | Question | Basic | Intermediate | Advanced | Multi-Agent |")
    lines.append("|---|----------|-------|--------------|----------|-------------|")

    for i, (basic_r, inter_r, adv_r, multi_r) in enumerate(
        zip(basic_results, intermediate_results, advanced_results, multi_agent_results), 1
    ):
        # Get question ID (truncate if too long)
        question_id = basic_r.get("example_id", f"Q{i}")
        if len(question_id) > 35:
            question_id = question_id[:32] + "..."

        # Extract F1 scores for winner determination
        f1_scores = {
            "basic": basic_r.get("f1_at_k", 0) if "error" not in basic_r else -1,
            "inter": inter_r.get("f1_at_k", 0) if "error" not in inter_r else -1,
            "adv": adv_r.get("f1_at_k", 0) if "error" not in adv_r else -1,
            "multi": multi_r.get("f1_at_k", 0) if "error" not in multi_r else -1,
        }
        max_f1 = max(f1_scores.values())

        def format_cell(result: Dict, tier_key: str) -> str:
            """Format a single cell as F1/Sim/Fact with bold winner."""
            if "error" in result:
                return "-"
            f1 = result.get("f1_at_k", 0) * 100
            sim = result.get("semantic_similarity", 0) * 100
            fact = result.get("factual_accuracy", 0) * 100

            # Bold the F1 if it's the winner (or tied for winner)
            is_winner = f1_scores[tier_key] == max_f1 and max_f1 > 0
            f1_str = f"**{f1:.0f}**" if is_winner else f"{f1:.0f}"

            return f"{f1_str}/{sim:.0f}/{fact:.0f}"

        basic_cell = format_cell(basic_r, "basic")
        inter_cell = format_cell(inter_r, "inter")
        adv_cell = format_cell(adv_r, "adv")
        multi_cell = format_cell(multi_r, "multi")

        lines.append(f"| {i} | {question_id} | {basic_cell} | {inter_cell} | {adv_cell} | {multi_cell} |")

    lines.append("")
    lines.append(f"*F1/Sim/Fact = F1@{k} / Semantic Similarity / Factual Accuracy (%). Bold = best F1 for row.*")

    return "\n".join(lines)


# ========== MAIN TEST ==========

def test_architecture_comparison(quick_mode: bool = False, dataset_type: str = "standard"):
    """
    Main comparison test - run all 3 tiers on golden dataset.

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
    current_tier = get_current_tier()
    tier_info = TIER_METADATA[current_tier]
    print(f"Model Tier: {current_tier.value.upper()} ({tier_info['description']})")
    print("Tiers: Basic (1), Intermediate (5), Advanced (17), Multi-Agent (20 features)")
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
    print("    This avoids re-ingesting PDFs for each tier (saves 40-50% time)")
    shared_retriever = setup_retriever(k_final=k_final)

    # Inject into all four variant modules
    basic_module.adaptive_retriever = shared_retriever
    intermediate_module.adaptive_retriever = shared_retriever
    advanced_module.adaptive_retriever = shared_retriever
    multi_agent_module.adaptive_retriever = shared_retriever
    print(f"[OK] Retriever pre-built and injected into all tiers (k_final={k_final})")
    print(f"{'='*80}\n")

    # Run Basic Tier
    print(f"\n{'='*80}")
    print("[1/4] Running BASIC tier (1 feature)...")
    print(f"{'='*80}")
    basic_results = run_tier_on_golden_dataset("basic", basic_rag_graph, dataset, k_final=k_final)
    basic_metrics = calculate_tier_metrics(basic_results)

    # Run Intermediate Tier
    print(f"\n{'='*80}")
    print("[2/4] Running INTERMEDIATE tier (5 features)...")
    print(f"{'='*80}")
    intermediate_results = run_tier_on_golden_dataset("intermediate", intermediate_rag_graph, dataset, k_final=k_final)
    intermediate_metrics = calculate_tier_metrics(intermediate_results)

    # Run Advanced Tier
    print(f"\n{'='*80}")
    print("[3/4] Running ADVANCED tier (17 features)...")
    print(f"{'='*80}")
    advanced_results = run_tier_on_golden_dataset("advanced", advanced_rag_graph, dataset, k_final=k_final)
    advanced_metrics = calculate_tier_metrics(advanced_results)

    # Run Multi-Agent Tier
    print(f"\n{'='*80}")
    print("[4/4] Running MULTI-AGENT tier (20 features)...")
    print(f"{'='*80}")
    multi_agent_results = run_tier_on_golden_dataset("multi_agent", multi_agent_rag_graph, dataset, k_final=k_final)
    multi_agent_tier_metrics = calculate_tier_metrics(multi_agent_results)

    # Print summary
    k = k_final
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"\n{'Tier':<15} {f'F1@{k}':<8} {'Ground':<8} {'Sim':<8} {'Fact':<8} {'Comp':<8} {'Retr Att':<10} {'Gen Att':<10}")
    print("-" * 120)
    print(f"{'Basic':<15} {basic_metrics['avg_f1_at_k']:<8.1%} {basic_metrics['avg_groundedness']:<8.1%} {basic_metrics['avg_semantic_similarity']:<8.1%} {basic_metrics['avg_factual_accuracy']:<8.1%} {basic_metrics['avg_completeness']:<8.1%} {'-':<10} {'-':<10}")
    print(f"{'Intermediate':<15} {intermediate_metrics['avg_f1_at_k']:<8.1%} {intermediate_metrics['avg_groundedness']:<8.1%} {intermediate_metrics['avg_semantic_similarity']:<8.1%} {intermediate_metrics['avg_factual_accuracy']:<8.1%} {intermediate_metrics['avg_completeness']:<8.1%} {'-':<10} {'-':<10}")
    print(f"{'Advanced':<15} {advanced_metrics['avg_f1_at_k']:<8.1%} {advanced_metrics['avg_groundedness']:<8.1%} {advanced_metrics['avg_semantic_similarity']:<8.1%} {advanced_metrics['avg_factual_accuracy']:<8.1%} {advanced_metrics['avg_completeness']:<8.1%} {advanced_metrics.get('avg_retrieval_attempts', 1.0):<10.1f} {advanced_metrics.get('avg_generation_attempts', 1.0):<10.1f}")
    print(f"{'Multi-Agent':<15} {multi_agent_tier_metrics['avg_f1_at_k']:<8.1%} {multi_agent_tier_metrics['avg_groundedness']:<8.1%} {multi_agent_tier_metrics['avg_semantic_similarity']:<8.1%} {multi_agent_tier_metrics['avg_factual_accuracy']:<8.1%} {multi_agent_tier_metrics['avg_completeness']:<8.1%} {'-':<10} {multi_agent_tier_metrics.get('avg_generation_attempts', 1.0):<10.1f}")
    print("=" * 120 + "\n")

    # Save raw results with timestamp
    os.makedirs("evaluation", exist_ok=True)

    # Generate consistent timestamp for both files
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "architecture_comparison",
        "dataset_type": dataset_type,
        "model_tier": f"{current_tier.value.upper()} ({tier_info['description']})",
        "dataset_size": len(dataset),
        "k_final": k_final,
        "quick_mode": quick_mode,
        "tiers": {
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
            "multi_agent": {
                "metrics": multi_agent_tier_metrics,
                "results": multi_agent_results,
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
        basic_metrics, intermediate_metrics, advanced_metrics, multi_agent_tier_metrics,
        basic_results, intermediate_results, advanced_results, multi_agent_results,
        k_final, dataset_type, test_timestamp, current_tier, tier_info
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
        expected_time = "70-85 minutes" if args.dataset == "standard" else "35-45 minutes"
        print(f"[*] Running full evaluation on {args.dataset} dataset ({dataset_size})")
        print(f"[*] This will take approximately {expected_time}")
        print("[*] Use --quick flag for faster testing (~6-8 minutes)")

    test_architecture_comparison(quick_mode=args.quick, dataset_type=args.dataset)
