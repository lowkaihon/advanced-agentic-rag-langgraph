"""
Simple RAGAS evaluation test to verify basic functionality.
"""

import os
import sys

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.evaluation import RAGASEvaluator


def test_ragas_simple():
    """Test RAGAS evaluator initialization and simple sample evaluation."""
    print("\n" + "="*70)
    print("SIMPLE RAGAS TEST")
    print("="*70)

    # Test 1: Initialize evaluator
    print("\n1. Testing RAGASEvaluator initialization...")
    evaluator = RAGASEvaluator()
    print(f"[OK] RAGASEvaluator initialized successfully")
    print(f"[OK] Metrics: {[m.name for m in evaluator.metrics]}")

    # Test 2: Prepare a sample
    print("\n2. Testing sample preparation...")
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

    # Test 3: Evaluate sample
    print("\n3. Testing sample evaluation...")
    print("NOTE: This will make API calls to OpenAI and may take 10-20 seconds...")

    try:
        scores = evaluator.evaluate_sample_sync(sample)
        print(f"[OK] Sample evaluated successfully")

        print("\nRAGAS Scores:")
        for metric_name, score in scores.items():
            if score is not None:
                print(f"  {metric_name:25s}: {score:.4f}")
            else:
                print(f"  {metric_name:25s}: N/A")

        print("\n" + "="*70)
        print("SIMPLE RAGAS TEST PASSED!")
        print("="*70)
        return scores

    except Exception as e:
        print(f"\n[FAIL] Sample evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_ragas_simple()
