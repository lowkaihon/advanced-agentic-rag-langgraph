"""
Test script for HHEM-based hallucination detector.

Tests the HHEM-2.1-Open detector with per-chunk verification:
- Uses vectara/hallucination_evaluation_model
- Per-chunk verification (stays under 512 token limit)
- Max aggregation across chunks for each claim
- Consistency scoring (0=hallucination, 1=consistent)

HHEM-2.1-Open outperforms zero-shot NLI models by handling paraphrases correctly.
"""

from advanced_agentic_rag_langgraph.validation import HHEMHallucinationDetector


def test_hhem_detector():
    """Test HHEM detector with sample examples"""

    print("Initializing HHEM Hallucination Detector...")
    detector = HHEMHallucinationDetector()
    print("Detector initialized successfully\n")

    # Test Case 1: Semantically similar (HHEM should handle paraphrases)
    print("="*60)
    print("TEST 1: Semantically Similar Claims (HHEM Paraphrase Handling)")
    print("="*60)
    chunks1 = [
        "BERT uses 12 transformer layers and was published in 2018.",
        "It uses bidirectional attention mechanisms."
    ]
    answer1 = "BERT has 12 layers and uses transformers."

    result1 = detector.verify_groundedness(answer1, chunks1)
    print(f"Chunks: {chunks1}")
    print(f"Answer: {answer1}")
    print(f"Claims: {result1['claims']}")
    print(f"Groundedness Score: {result1['groundedness_score']:.2f}")
    print(f"Unsupported Claims: {result1['unsupported_claims']}")
    print(f"\nEXPECTED (HHEM Behavior):")
    print(f"  - Score: HIGH (0.7-1.0) - HHEM handles paraphrases correctly")
    print(f"  - Claims: SUPPORTED (semantic equivalence recognized)")
    print()

    # Test Case 2: Factually incorrect claim (hallucination detection)
    print("="*60)
    print("TEST 2: Factually Incorrect Claim (Hallucination)")
    print("="*60)
    chunks2 = ["BERT uses 12 transformer layers and was published in 2018."]
    answer2 = "BERT has 12 layers and was published in 2020."  # Wrong year!

    result2 = detector.verify_groundedness(answer2, chunks2)
    print(f"Chunks: {chunks2}")
    print(f"Answer: {answer2}")
    print(f"Claims: {result2['claims']}")
    print(f"Groundedness Score: {result2['groundedness_score']:.2f}")
    print(f"Unsupported Claims: {result2['unsupported_claims']}")
    print(f"\nEXPECTED:")
    print(f"  - Score: MIXED - correct claim supported, wrong year unsupported")
    print(f"  - Unsupported: '2020' claim (contradicts '2018')")
    print()

    # Test Case 3: Completely hallucinated (different topic)
    print("="*60)
    print("TEST 3: Completely Hallucinated Content (Off-Topic)")
    print("="*60)
    chunks3 = ["BERT uses 12 transformer layers and was published in 2018."]
    answer3 = "GPT-3 has 96 layers and was trained on 1 trillion tokens."

    result3 = detector.verify_groundedness(answer3, chunks3)
    print(f"Chunks: {chunks3}")
    print(f"Answer: {answer3}")
    print(f"Claims: {result3['claims']}")
    print(f"Groundedness Score: {result3['groundedness_score']:.2f}")
    print(f"Unsupported Claims: {result3['unsupported_claims']}")
    print(f"\nEXPECTED:")
    print(f"  - Score: ZERO (0.0) - completely different topic")
    print(f"  - All claims: UNSUPPORTED (about GPT-3, not BERT)")
    print()

    # Test Case 4: Per-chunk verification (multiple chunks)
    print("="*60)
    print("TEST 4: Per-Chunk Verification (Multiple Chunks)")
    print("="*60)
    chunks4 = [
        "The attention mechanism in transformers allows the model to focus on different parts of the input sequence.",
        "Self-attention computes relevance scores between all token pairs.",
        "Multi-head attention uses multiple attention heads in parallel."
    ]
    answer4 = "Attention helps models focus on relevant input parts. Multi-head attention uses parallel attention heads."

    result4 = detector.verify_groundedness(answer4, chunks4)
    print(f"Number of chunks: {len(chunks4)}")
    print(f"Answer: {answer4}")
    print(f"Claims: {result4['claims']}")
    print(f"\nPer-Chunk Verification Details:")
    for detail in result4['claim_details']:
        print(f"  Claim: {detail['claim']}")
        print(f"  Best Score (max across chunks): {detail['entailment_score']:.3f}")
        print(f"  Best Chunk Index: {detail['best_chunk_idx']}")
        print(f"  All Chunk Scores: {[f'{s:.3f}' for s in detail['chunk_scores']]}")
        print(f"  Supported: {detail['supported']}")
        print()
    print(f"Groundedness Score: {result4['groundedness_score']:.2f}")
    print(f"\nPER-CHUNK VERIFICATION BENEFITS:")
    print(f"  - Each HHEM call stays under 512 tokens")
    print(f"  - Max aggregation: claim supported if ANY chunk supports it")
    print(f"  - Detailed tracking: know which chunk supports each claim")

    print("="*60)
    print("All tests completed!")
    print("="*60)
    print("\nKEY TAKEAWAYS:")
    print("1. HHEM-2.1-Open handles paraphrases correctly (unlike zero-shot NLI)")
    print("2. Per-chunk verification stays under 512 token limit")
    print("3. Max aggregation ensures claims supported by ANY chunk are marked supported")
    print("4. Consistency scoring: 0=hallucination, 1=consistent")


if __name__ == "__main__":
    test_hhem_detector()
