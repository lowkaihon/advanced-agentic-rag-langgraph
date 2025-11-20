import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from advanced_agentic_rag_langgraph.retrieval import AdaptiveRetriever
from advanced_agentic_rag_langgraph.preprocessing.pdf_loader import PDFDocumentLoader
from advanced_agentic_rag_langgraph.preprocessing.profiling_pipeline import DocumentLoader
from langchain_core.documents import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

if LANGSMITH_API_KEY and os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "advanced-agentic-rag"

_retriever_instance = None
_corpus_stats = None
_document_profiles = None

import advanced_agentic_rag_langgraph
PROJECT_ROOT = Path(advanced_agentic_rag_langgraph.__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


def get_all_pdf_paths_from_docs() -> List[str]:
    """Get paths to all PDF files in docs/ directory."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"docs/ directory not found at: {DOCS_DIR}\n"
            f"Please create the directory and add PDF files for RAG."
        )

    pdf_files = [
        os.path.join(DOCS_DIR, f)
        for f in os.listdir(DOCS_DIR)
        if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(DOCS_DIR, f))
    ]

    if not pdf_files:
        contents = os.listdir(DOCS_DIR) if os.path.exists(DOCS_DIR) else []
        raise FileNotFoundError(
            f"No PDF files found in docs/ directory: {DOCS_DIR}\n\n"
            f"To use this RAG system:\n"
            f"1. Add PDF files to the docs/ folder, or\n"
            f"2. Specify PDFs explicitly: setup_retriever(pdfs=['file.pdf'])\n\n"
            f"Current docs/ contents: {contents}"
        )

    return sorted(pdf_files)


def get_specific_pdf_paths(filenames: str | List[str]) -> List[str]:
    """Get absolute paths for specific PDF filenames from docs/ directory."""
    if isinstance(filenames, str):
        filenames = [filenames]

    paths = []
    missing = []

    for filename in filenames:
        pdf_path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(pdf_path):
            paths.append(pdf_path)
        else:
            missing.append(filename)

    if missing:
        available = [
            f for f in os.listdir(DOCS_DIR)
            if f.lower().endswith('.pdf')
        ] if os.path.exists(DOCS_DIR) else []

        raise FileNotFoundError(
            f"PDF file(s) not found in {DOCS_DIR}:\n"
            f"  Missing: {missing}\n\n"
            f"  Available PDFs:\n" +
            '\n'.join(f"    - {f}" for f in available)
        )

    return paths


def setup_retriever(
    pdfs: None | str | List[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True,
    k_final: int = 4
) -> AdaptiveRetriever:
    """
    Initialize hybrid retriever with PDF documents from docs/ folder.

    Process: Load full PDFs → Profile with LLM → Chunk → Attach metadata → Create retriever

    Args:
        pdfs: PDF filenames or paths (None = all PDFs in docs/)
        chunk_size: Text chunk size in characters
        chunk_overlap: Overlap between chunks
        verbose: Print progress messages
        k_final: Number of final documents after two-stage reranking (default: 4)

    Returns: AdaptiveRetriever instance with profiled documents
    """
    global _retriever_instance, _corpus_stats, _document_profiles

    if _retriever_instance is not None:
        return _retriever_instance

    if pdfs is not None and not isinstance(pdfs, (str, list)):
        raise ValueError(
            f"pdfs parameter must be None, str, or List[str], got {type(pdfs).__name__}"
        )

    if pdfs is None:
        if verbose:
            print("\n" + "="*60)
            print("LOADING ALL PDFS FROM docs/")
            print("="*60 + "\n")
        pdf_paths = get_all_pdf_paths_from_docs()
    else:
        if verbose:
            print("\n" + "="*60)
            print("LOADING SPECIFIC PDF(S)")
            print("="*60 + "\n")
        pdf_paths = get_specific_pdf_paths(pdfs)

    if verbose:
        print(f"Found {len(pdf_paths)} PDF file(s):")
        for i, path in enumerate(pdf_paths, 1):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {i}. {os.path.basename(path)} ({size_mb:.1f} MB)")
        print()

    pdf_loader = PDFDocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if verbose:
        print("="*60)
        print("STEP 1: Loading full documents (no chunking)")
        print("="*60 + "\n")

    full_documents = pdf_loader.load_multiple_pdfs_full_documents(
        pdf_paths,
        verbose=verbose
    )

    if verbose:
        print("="*60)
        print("STEP 2: Profiling documents with LLM")
        print("="*60 + "\n")

    profiling_pipeline = DocumentLoader()
    profiled_docs, corpus_stats, document_profiles = profiling_pipeline.load_documents(
        full_documents,
        verbose=verbose
    )

    if verbose:
        print("="*60)
        print("STEP 3: Chunking documents")
        print("="*60 + "\n")

    all_chunks = []
    for profiled_doc in profiled_docs:
        chunks = pdf_loader.text_splitter.split_text(profiled_doc.page_content)

        source = profiled_doc.metadata.get('source', 'unknown')

        profile = profiled_doc.metadata.get('profile')

        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "id": f"{source}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "source_type": "pdf",
            }

            if profile:
                chunk_metadata.update({
                    "content_type": profile['doc_type'],
                    "technical_level": profile['reading_level'],
                    "domain": profile['domain_tags'][0] if profile['domain_tags'] else 'general',
                    "best_retrieval_strategy": profile['best_retrieval_strategy'],
                    "strategy_confidence": profile['strategy_confidence'],
                    "has_math": profile['has_math'],
                    "has_code": profile['has_code'],
                    "document_summary": profile['summary'],
                    "key_concepts": ', '.join(profile['key_concepts'][:5]),
                })

            chunk_doc = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            all_chunks.append(chunk_doc)

    if verbose:
        print(f"Created {len(all_chunks)} chunks from {len(profiled_docs)} documents\n")

    _corpus_stats = corpus_stats
    _document_profiles = document_profiles

    if verbose:
        print("="*60)
        print("CORPUS STATISTICS")
        print("="*60)
        print(f"Total documents: {corpus_stats.get('total_documents', 'unknown')}")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Avg technical density: {corpus_stats.get('avg_technical_density', 0):.2f}")
        print(f"Document types: {corpus_stats.get('document_types', {})}")
        print(f"Has code: {corpus_stats.get('pct_with_code', 0):.0f}%")
        print(f"Has math: {corpus_stats.get('pct_with_math', 0):.0f}%")
        print("="*60 + "\n")

    _retriever_instance = AdaptiveRetriever(all_chunks, k_final=k_final)

    return _retriever_instance


def reset_retriever():
    """Reset singleton retriever instance (useful for testing)."""
    global _retriever_instance, _corpus_stats, _document_profiles
    _retriever_instance = None
    _corpus_stats = None
    _document_profiles = None


def get_corpus_stats() -> Dict:
    """Get corpus-level statistics from profiled documents."""
    return _corpus_stats or {}


def get_document_profiles() -> Dict:
    """Get all document profiles."""
    return _document_profiles or {}
