"""
Helper script to create golden dataset with semi-automated chunk ID mapping.

This script helps create the golden dataset by:
1. Loading all PDF documents
2. For each question, finding relevant chunks using keyword search
3. Displaying candidates for manual review
4. Building the golden dataset JSON
"""

from advanced_agentic_rag_langgraph.preprocessing.pdf_loader import PDFDocumentLoader
from typing import List, Dict
import json
import os

def load_all_documents() -> Dict[str, List]:
    """Load all PDFs and return chunks by document."""
    docs_dir = "docs"
    pdf_files = [
        "Attention Is All You Need.pdf",
        "BERT - Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf",
        "Denoising Diffusion Probabilistic Models.pdf",
        # Add more as needed for cross-document examples
    ]

    loader = PDFDocumentLoader(chunk_size=1000, chunk_overlap=200)
    all_chunks_by_doc = {}

    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_dir, pdf_file)
        if os.path.exists(pdf_path):
            chunks = loader.load_pdf(pdf_path, verbose=False)
            all_chunks_by_doc[pdf_file] = chunks
            print(f"Loaded {pdf_file}: {len(chunks)} chunks")

    return all_chunks_by_doc

def search_chunks(query: str, chunks: List, top_k: int = 5):
    """Simple keyword-based search to find relevant chunks.

    Note: top_k=5 is a helper tool default for manual dataset creation,
    not used in production retrieval (which uses adaptive k_final).
    """
    query_terms = query.lower().split()

    scored_chunks = []
    for chunk in chunks:
        content_lower = chunk.page_content.lower()
        score = sum(1 for term in query_terms if term in content_lower)
        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return scored_chunks[:top_k]

def find_relevant_chunks(question: str, all_chunks_by_doc: Dict, source_doc: str) -> List[str]:
    """Find relevant chunk IDs for a question."""
    if source_doc not in all_chunks_by_doc:
        print(f"Warning: {source_doc} not loaded")
        return []

    chunks = all_chunks_by_doc[source_doc]
    relevant = search_chunks(question, chunks, top_k=5)  # Helper tool default for manual review

    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"Source: {source_doc}")
    print(f"{'='*80}")

    chunk_ids = []
    for i, (chunk, score) in enumerate(relevant, 1):
        chunk_id = chunk.metadata.get("id")
        print(f"\n{i}. Chunk ID: {chunk_id} (score: {score})")
        print(f"   Preview: {chunk.page_content[:200]}...")
        chunk_ids.append(chunk_id)

    return chunk_ids

if __name__ == "__main__":
    print("Loading documents...")
    all_chunks_by_doc = load_all_documents()

    # Example usage - finding chunks for a sample question
    sample_question = "How many attention heads are used in the base model?"
    source_doc = "Attention Is All You Need.pdf"

    chunk_ids = find_relevant_chunks(sample_question, all_chunks_by_doc, source_doc)

    print(f"\n\nRecommended chunk IDs for golden dataset:")
    print(json.dumps(chunk_ids[:3], indent=2))
