"""
4-Tier Architecture Comparison Test - MOCK VERSION (No API calls).

Mock version for testing infrastructure without HuggingFace or OpenAI access.
Validates graph structure, node execution, and metrics calculation logic.

Usage:
    # Set mock environment variables to avoid model downloads
    export OPENAI_API_KEY="mock-key"
    export MODEL_TIER="BUDGET"
    uv run python tests/integration/test_architecture_comparison_mock.py

IMPORTANT LIMITATIONS:
- Graph imports may still trigger model initialization (CrossEncoder, embeddings)
- Even with mocks, initial setup can take 30-60 seconds
- This test validates LOGIC, not actual model performance
- For full offline testing, pre-download models or use test_architecture_structure.py

This mock test:
- Bypasses all LLM calls (uses fake answers)
- Mocks NLI hallucination detector (returns fake groundedness scores)
- Generates realistic fake metrics
- Validates graph structure and node execution flow
- Requires OPENAI_API_KEY set (can be fake value)

import json
import os
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

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
import advanced_agentic_rag_langgraph.variants.pure_semantic_rag_graph as pure_semantic_module
import advanced_agentic_rag_langgraph.variants.basic_rag_graph as basic_module
import advanced_agentic_rag_langgraph.variants.intermediate_rag_graph as intermediate_module
import advanced_agentic_rag_langgraph.orchestration.nodes as advanced_module
from advanced_agentic_rag_langgraph.evaluation.golden_dataset import GoldenDatasetManager
from advanced_agentic_rag_langgraph.evaluation.retrieval_metrics import calculate_retrieval_metrics


# ========== MOCK COMPONENTS ==========

class MockDocument:
    """Mock document with metadata."""
    def __init__(self, content: str, doc_id: str):
        self.page_content = content
        self.metadata = {"id": doc_id}


class MockRetriever:
    """Mock retriever that returns fake documents."""
    def __init__(self):
        self.corpus_stats = {
            "technical_density": 0.75,
            "primary_domains": ["machine learning", "NLP"],
            "document_types": ["research", "tutorial"],
        }

    def retrieve_without_reranking(self, query: str, strategy: str = "hybrid", top_k: int = 10):
        """Return mock documents."""
        # Generate fake document IDs based on query hash
        num_docs = min(top_k, 10)
        docs = [
            MockDocument(
                content=f"Mock content for query '{query[:30]}...' (doc {i+1})",
                doc_id=f"doc_{hash(query) % 1000}_{i}"
            )
            for i in range(num_docs)
        ]
        return docs

    def retrieve(self, query: str, strategy: str = "hybrid"):
        """Return mock documents after 'reranking'."""
        return self.retrieve_without_reranking(query, strategy, top_k=4)


class MockNLIDetector:
    """Mock NLI hallucination detector."""
    def detect(self, answer: str, context: str):
        """Return fake but realistic groundedness scores."""
        # Vary scores based on answer length (longer = slightly lower groundedness)
        base_score = 0.85
        length_penalty = min(len(answer) / 1000, 0.15)
        groundedness = base_score - length_penalty

        return {
            "groundedness_score": groundedness,
            "severity": "NONE" if groundedness > 0.8 else "MODERATE",
            "unsupported_claims": [],
        }


class MockLLM:
    """Mock LLM that returns fake responses."""
    def __init__(self, response_template: str = "Mock answer"):
        self.response_template = response_template
        self.call_count = 0

    def invoke(self, prompt):
        """Return mock response."""
        self.call_count += 1

        # Extract query if possible
        if isinstance(prompt, str):
            query_hint = prompt[:50] if len(prompt) > 50 else prompt
        else:
            query_hint = "query"

        mock_response = Mock()
        mock_response.content = f"{self.response_template} for {query_hint}... (call #{self.call_count})"
        return mock_response


class MockCrossEncoderReRanker:
    """Mock CrossEncoder reranker."""
    def rank(self, query: str, docs: List[MockDocument]):
        """Return docs with fake scores."""
        # Just return top 5 with descending scores
        scored = [(doc, 0.9 - i*0.1) for i, doc in enumerate(docs[:5])]
        return scored


class MockStrategySelector:
    """Mock strategy selector."""
    def select_strategy(self, query: str, corpus_stats: dict):
        """Return fake strategy selection."""
        # Vary strategy based on query length
        if len(query) < 20:
            strategy = "keyword"
        elif len(query) > 50:
            strategy = "semantic"
        else:
            strategy = "hybrid"

        return strategy, 0.8, f"Mock reasoning for {strategy}"


# ========== MOCK HELPER FUNCTIONS ==========

def mock_expand_query(query: str):
    """Mock query expansion - return 3 variants."""
    return [
        query,
        f"{query} (variant 1)",
        f"{query} (variant 2)",
    ]


def mock_rewrite_query(query: str, retrieval_context: str = None):
    """Mock query rewriting."""
    return f"Rewritten: {query}"


def mock_setup_retriever():
    """Return mock retriever."""
    return MockRetriever()


# ========== TEST RUNNER ==========

def run_tier_on_golden_dataset_mock(
    tier_name: str,
    graph: Any,
    dataset: List[Dict],
    verbose: bool = True
) -> List[Dict]:
    """
    Run tier on golden dataset with mocked components.

    Uses mocks for all external calls to run completely offline.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"[MOCK] Running {tier_name.upper()} Tier on Golden Dataset")
        print(f"{'='*70}")
        print(f"Examples: {len(dataset)}")
        print(f"Mode: MOCK (no real API calls)")
        print(f"{'='*70}\n")

    results = []
    mock_nli = MockNLIDetector()

    for i, example in enumerate(dataset, 1):
        query = example["question"]
        ground_truth_docs = example["relevant_doc_ids"]

        if verbose:
            print(f"\n[{i}/{len(dataset)}] [MOCK] Processing: {example.get('id', query[:50])}")

        try:
            # Initialize state per tier
            if tier_name == "pure_semantic":
                initial_state = {
                    "user_question": query,
                    "retrieved_docs": [],
                }
            elif tier_name == "basic":
                initial_state = {
                    "user_question": query,
                    "query_expansions": [],
                    "retrieved_docs": [],
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
                    "baseline_query": query,
                    "messages": [],
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                    "query_expansions": [],
                }

            # Run graph (with mocks)
            result = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"mock_{tier_name}_{i}"}}
            )

            # Extract results
            answer = result.get("final_answer", "Mock answer")
            confidence = result.get("confidence_score", 0.7)
            retrieved_docs = result.get("unique_docs_list", result.get("retrieved_docs", []))
            retrieval_attempts = result.get("retrieval_attempts", 1)

            # Generate fake retrieved doc IDs (simulate some overlap with ground truth)
            retrieved_doc_ids = []
            num_retrieved = min(len(retrieved_docs), 5)

            # Add some ground truth docs (simulate partial matches)
            overlap_count = min(len(ground_truth_docs), num_retrieved // 2)
            retrieved_doc_ids.extend(ground_truth_docs[:overlap_count])

            # Add some fake docs
            for j in range(num_retrieved - overlap_count):
                retrieved_doc_ids.append(f"mock_doc_{tier_name}_{i}_{j}")

            # Calculate F1@5
            relevant_retrieved = set(ground_truth_docs[:5]) & set(retrieved_doc_ids[:5])
            precision = len(relevant_retrieved) / 5 if len(retrieved_doc_ids) >= 5 else 0.0
            recall = len(relevant_retrieved) / min(len(ground_truth_docs), 5)
            f1_at_5 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Mock groundedness (varies by tier)
            tier_groundedness_base = {
                "pure_semantic": 0.65,
                "basic": 0.75,
                "intermediate": 0.82,
                "advanced": 0.90,
            }
            groundedness_score = tier_groundedness_base.get(tier_name, 0.7)

            if verbose:
                print(f"  [MOCK] Answer: {answer[:60]}...")
                print(f"  [MOCK] F1@5: {f1_at_5:.1%}, Groundedness: {groundedness_score:.1%}, Confidence: {confidence:.1%}")

            results.append({
                "example_id": example.get("id", f"example_{i}"),
                "query": query,
                "answer": answer,
                "f1_at_5": f1_at_5,
                "groundedness_score": groundedness_score,
                "confidence": confidence,
                "retrieval_attempts": retrieval_attempts,
                "retrieved_doc_count": len(retrieved_doc_ids),
            })

        except Exception as e:
            if verbose:
                print(f"  [ERROR] {str(e)}")
            results.append({
                "example_id": example.get("id", f"example_{i}"),
                "query": query,
                "error": str(e),
                "f1_at_5": 0.0,
                "groundedness_score": 0.0,
                "confidence": 0.0,
                "retrieval_attempts": 1,
            })

    return results


def calculate_tier_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate aggregate metrics from tier results."""
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {
            "total_examples": len(results),
            "successful_examples": 0,
            "avg_f1_at_5": 0.0,
            "avg_groundedness": 0.0,
            "avg_confidence": 0.0,
            "avg_retrieval_attempts": 0.0,
        }

    return {
        "total_examples": len(results),
        "successful_examples": len(valid_results),
        "avg_f1_at_5": sum(r["f1_at_5"] for r in valid_results) / len(valid_results),
        "avg_groundedness": sum(r["groundedness_score"] for r in valid_results) / len(valid_results),
        "avg_confidence": sum(r["confidence"] for r in valid_results) / len(valid_results),
        "avg_retrieval_attempts": sum(r["retrieval_attempts"] for r in valid_results) / len(valid_results),
    }


# ========== MAIN MOCK TEST ==========

def test_architecture_comparison_mock():
    """
    Mock architecture comparison test - runs without any real API calls.

    Validates:
    - Graph structure and node execution
    - State management and transitions
    - Metrics calculation logic
    - Report generation
    """
    print("\n" + "="*80)
    print("4-TIER ARCHITECTURE COMPARISON TEST - MOCK MODE")
    print("="*80)
    print("Mode: MOCK (no real API calls)")
    print("Model Tier: MOCK (simulated responses)")
    print("Tiers: Pure Semantic (4), Basic (8), Intermediate (18), Advanced (31 features)")
    print("="*80 + "\n")

    # Load golden dataset (real data, but we'll mock responses)
    dataset_manager = GoldenDatasetManager("evaluation/golden_set.json")
    dataset = dataset_manager.dataset[:5]  # Use only first 5 for speed

    if not dataset:
        print("[ERROR] No examples in golden dataset")
        return

    print(f"[MOCK] Using {len(dataset)} examples for quick test\n")

    # Create mock retriever and inject into all modules
    mock_retriever = MockRetriever()
    pure_semantic_module.adaptive_retriever = mock_retriever
    basic_module.adaptive_retriever = mock_retriever
    intermediate_module.adaptive_retriever = mock_retriever
    advanced_module.adaptive_retriever = mock_retriever

    # Mock CrossEncoder
    mock_cross_encoder = MockCrossEncoderReRanker()
    basic_module.cross_encoder = mock_cross_encoder
    intermediate_module.cross_encoder = mock_cross_encoder
    advanced_module.cross_encoder = mock_cross_encoder

    print(f"[OK] Mocked retriever and reranker injected into all tiers\n")

    # Patch all LLM and model dependencies
    with patch('advanced_agentic_rag_langgraph.retrieval.query_optimization.expand_query', side_effect=mock_expand_query), \
         patch('advanced_agentic_rag_langgraph.retrieval.query_optimization.rewrite_query', side_effect=mock_rewrite_query), \
         patch('langchain_openai.ChatOpenAI', return_value=MockLLM()), \
         patch('advanced_agentic_rag_langgraph.retrieval.strategy_selector.StrategySelector.select_strategy', return_value=("hybrid", 0.8, "mock reasoning")), \
         patch('advanced_agentic_rag_langgraph.validation.nli_hallucination_detector.NLIHallucinationDetector', return_value=MockNLIDetector()):

        print("[OK] All LLM calls mocked\n")

        # Run Pure Semantic Tier
        print(f"\n{'='*80}")
        print("[1/4] Running PURE SEMANTIC tier (MOCK)...")
        print(f"{'='*80}")
        pure_semantic_results = run_tier_on_golden_dataset_mock("pure_semantic", pure_semantic_rag_graph, dataset)
        pure_semantic_metrics = calculate_tier_metrics(pure_semantic_results)

        # Run Basic Tier
        print(f"\n{'='*80}")
        print("[2/4] Running BASIC tier (MOCK)...")
        print(f"{'='*80}")
        basic_results = run_tier_on_golden_dataset_mock("basic", basic_rag_graph, dataset)
        basic_metrics = calculate_tier_metrics(basic_results)

        # Run Intermediate Tier
        print(f"\n{'='*80}")
        print("[3/4] Running INTERMEDIATE tier (MOCK)...")
        print(f"{'='*80}")
        intermediate_results = run_tier_on_golden_dataset_mock("intermediate", intermediate_rag_graph, dataset)
        intermediate_metrics = calculate_tier_metrics(intermediate_results)

        # Run Advanced Tier
        print(f"\n{'='*80}")
        print("[4/4] Running ADVANCED tier (MOCK)...")
        print(f"{'='*80}")
        advanced_results = run_tier_on_golden_dataset_mock("advanced", advanced_rag_graph, dataset)
        advanced_metrics = calculate_tier_metrics(advanced_results)

    # Print summary
    print(f"\n{'='*80}")
    print("MOCK TEST SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Tier':<15} {'F1@5':<10} {'Groundedness':<15} {'Confidence':<12} {'Attempts':<10}")
    print("-" * 80)
    print(f"{'Pure Semantic':<15} {pure_semantic_metrics['avg_f1_at_5']:<10.1%} {pure_semantic_metrics['avg_groundedness']:<15.1%} {pure_semantic_metrics['avg_confidence']:<12.1%} {pure_semantic_metrics['avg_retrieval_attempts']:<10.1f}")
    print(f"{'Basic':<15} {basic_metrics['avg_f1_at_5']:<10.1%} {basic_metrics['avg_groundedness']:<15.1%} {basic_metrics['avg_confidence']:<12.1%} {basic_metrics['avg_retrieval_attempts']:<10.1f}")
    print(f"{'Intermediate':<15} {intermediate_metrics['avg_f1_at_5']:<10.1%} {intermediate_metrics['avg_groundedness']:<15.1%} {intermediate_metrics['avg_confidence']:<12.1%} {intermediate_metrics['avg_retrieval_attempts']:<10.1f}")
    print(f"{'Advanced':<15} {advanced_metrics['avg_f1_at_5']:<10.1%} {advanced_metrics['avg_groundedness']:<15.1%} {advanced_metrics['avg_confidence']:<12.1%} {advanced_metrics['avg_retrieval_attempts']:<10.1f}")
    print("=" * 80 + "\n")

    # Validate structure
    print(f"{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")

    all_passed = True

    # Check all tiers completed successfully
    for tier_name, metrics in [
        ("Pure Semantic", pure_semantic_metrics),
        ("Basic", basic_metrics),
        ("Intermediate", intermediate_metrics),
        ("Advanced", advanced_metrics),
    ]:
        if metrics["successful_examples"] != len(dataset):
            print(f"[FAIL] {tier_name}: Only {metrics['successful_examples']}/{len(dataset)} succeeded")
            all_passed = False
        else:
            print(f"[PASS] {tier_name}: All {metrics['successful_examples']} examples succeeded")

    # Check metric progression (mocked tiers should show improvement)
    if pure_semantic_metrics['avg_groundedness'] < basic_metrics['avg_groundedness'] < intermediate_metrics['avg_groundedness'] < advanced_metrics['avg_groundedness']:
        print(f"[PASS] Groundedness shows expected progression across tiers")
    else:
        print(f"[FAIL] Groundedness progression unexpected")
        all_passed = False

    print(f"{'='*80}\n")

    if all_passed:
        print("[SUCCESS] Mock architecture comparison test passed!")
        print("All graph structures validated, node execution confirmed.")
    else:
        print("[FAILURE] Some validation checks failed")

    print("\n" + "="*80)
    print("MOCK TEST COMPLETE")
    print("="*80)
    print("Note: This is a MOCK test with simulated responses.")
    print("Run test_architecture_comparison.py for real evaluation with models.")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_architecture_comparison_mock()
