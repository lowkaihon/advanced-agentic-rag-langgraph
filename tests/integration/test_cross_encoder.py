import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.retrieval.cross_encoder_reranker import CrossEncoderReRanker
from langchain_core.documents import Document

# Create test documents
docs = [
    Document(page_content="The attention mechanism allows models to focus on relevant parts of the input.", metadata={"id": "doc_1"}),
    Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"id": "doc_2"}),
    Document(page_content="Self-attention computes attention weights between all positions in a sequence.", metadata={"id": "doc_3"}),
]

# Test reranker
print("Testing CrossEncoder reranking...")
reranker = CrossEncoderReRanker(top_k=3)
ranked = reranker.rank("What is attention mechanism?", docs)

print(f"\n[OK] CrossEncoder loaded successfully")
print(f"[OK] Ranked {len(ranked)} documents")
print("\nRanked results:")
for i, (doc, score) in enumerate(ranked, 1):
    print(f"  {i}. Score: {score:.4f} - {doc.page_content[:60]}...")
