"""
End-to-end test of LLM-based profiling pipeline.

Tests:
1. LLM document profiling
2. Corpus-aware strategy selection
3. Metadata-aware retrieval and reranking
"""

import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.core.config import setup_retriever, reset_retriever, get_corpus_stats
from src.retrieval.strategy_selection import StrategySelector

def main():
    print("\n" + "="*70)
    print("END-TO-END TEST: LLM-Based Profiling Pipeline")
    print("="*70 + "\n")

    # Reset to ensure clean state
    reset_retriever()

    # Step 1: Load and profile document
    print("STEP 1: Loading Attention paper with LLM profiling...")
    print("-"*70)
    retriever = setup_retriever(pdfs='Attention Is All You Need.pdf', verbose=True)

    # Step 2: Get corpus statistics
    print("\n" + "="*70)
    print("STEP 2: Corpus Statistics")
    print("="*70)
    corpus_stats = get_corpus_stats()
    print(f"Documents: {corpus_stats.get('total_documents')}")
    print(f"Technical density: {corpus_stats.get('avg_technical_density', 0):.2f}")
    print(f"Document types: {corpus_stats.get('document_types')}")
    print(f"Domains: {list(corpus_stats.get('domain_distribution', {}).keys())[:5]}")
    print(f"Has math: {corpus_stats.get('pct_with_math', 0):.0f}%")
    print(f"Has code: {corpus_stats.get('pct_with_code', 0):.0f}%")

    # Step 3: Test corpus-aware strategy selection
    print("\n" + "="*70)
    print("STEP 3: Testing Corpus-Aware Strategy Selection")
    print("="*70 + "\n")

    selector = StrategySelector()

    test_queries = [
        "What is the attention mechanism?",  # Conceptual
        "transformer architecture multi-head attention",  # Technical keywords
        "How does self-attention compare to RNN?",  # Comparative
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        print("-"*70)
        explanation = selector.explain_decision(query, corpus_stats)
        print(explanation)
        print()

    # Step 4: Test end-to-end retrieval with metadata
    print("\n" + "="*70)
    print("STEP 4: Testing End-to-End Retrieval")
    print("="*70 + "\n")

    query = "What is the attention mechanism and how does it work?"
    print(f"Query: '{query}'\n")

    # Retrieve documents
    docs = retriever.retrieve(query, strategy="hybrid")

    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"{i}. Source: {meta.get('source', 'unknown')}")
        print(f"   Type: {meta.get('content_type', 'unknown')}")
        print(f"   Level: {meta.get('technical_level', 'unknown')}")
        print(f"   Domain: {meta.get('domain', 'unknown')}")
        print(f"   Best strategy: {meta.get('best_retrieval_strategy', 'unknown')}")
        print(f"   Preview: {doc.page_content[:150]}...")
        print()

    print("="*70)
    print("[SUCCESS] END-TO-END TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
