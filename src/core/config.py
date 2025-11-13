import os
from typing import Dict, List
from dotenv import load_dotenv
from src.retrieval import HybridRetriever
from src.preprocessing.document_profiler_pipeline import DocumentProfilerPipeline
from src.preprocessing.pdf_loader import PDFDocumentLoader
from langchain_core.documents import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Enable LangSmith tracing (optional)
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "advanced-agentic-rag"

# Global retriever instance and corpus statistics
_retriever_instance = None
_corpus_stats = None
_document_profiles = None

# PDF directory
DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "docs"
)


def get_all_pdf_paths_from_docs() -> List[str]:
    """
    Get paths to all PDF files in docs/ directory.

    Returns:
        List of absolute paths to PDF files, sorted alphabetically

    Raises:
        FileNotFoundError: If docs/ folder doesn't exist or contains no PDFs
    """
    if not os.path.exists(DOCS_DIR):
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
    """
    Get absolute paths for specific PDF filenames from docs/ directory.

    Args:
        filenames: Single filename or list of filenames (not full paths)

    Returns:
        List of absolute paths to specified PDFs

    Raises:
        FileNotFoundError: If any specified PDF is not found
    """
    # Normalize to list
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
    verbose: bool = True
) -> HybridRetriever:
    """
    Initialize hybrid retriever with PDF documents from docs/ folder.

    This function:
    1. Loads PDF documents (all or specific ones)
    2. Profiles each document with DocumentProfiler
    3. Enriches documents with metadata
    4. Creates HybridRetriever with profiled documents
    5. Stores corpus statistics globally

    Args:
        pdfs: Which PDFs to load:
            - None (default): Load ALL PDFs from docs/ folder
            - str: Load single PDF by filename (e.g., "Attention Is All You Need.pdf")
            - List[str]: Load specific PDFs by filenames
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters overlap between chunks
        verbose: Print loading progress

    Returns:
        HybridRetriever instance with profiled documents

    Examples:
        >>> # Load all PDFs (default)
        >>> retriever = setup_retriever()

        >>> # Load single PDF
        >>> retriever = setup_retriever(pdfs="Attention Is All You Need.pdf")

        >>> # Load multiple specific PDFs
        >>> retriever = setup_retriever(pdfs=[
        ...     "Attention Is All You Need.pdf",
        ...     "BERT - Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf"
        ... ])

    Raises:
        FileNotFoundError: If docs/ folder is empty or specified PDF not found
        ValueError: If pdfs parameter has invalid type
    """
    global _retriever_instance, _corpus_stats, _document_profiles

    # Singleton pattern check
    if _retriever_instance is not None:
        return _retriever_instance

    # Validate pdfs parameter type
    if pdfs is not None and not isinstance(pdfs, (str, list)):
        raise ValueError(
            f"pdfs parameter must be None, str, or List[str], got {type(pdfs).__name__}"
        )

    # Determine which PDFs to load
    if pdfs is None:
        # Default: load all PDFs
        if verbose:
            print("\n" + "="*60)
            print("LOADING ALL PDFS FROM docs/")
            print("="*60 + "\n")
        pdf_paths = get_all_pdf_paths_from_docs()
    else:
        # Specific PDF(s)
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

    # Load PDFs using PDFDocumentLoader
    pdf_loader = PDFDocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    raw_documents = pdf_loader.load_multiple_pdfs(
        pdf_paths,
        verbose=verbose
    )

    # Profile and enrich documents
    profiler_pipeline = DocumentProfilerPipeline()

    # Profile documents and enrich with metadata
    profiled_documents, corpus_stats, doc_profiles = profiler_pipeline.process_documents(
        raw_documents,
        verbose=verbose
    )

    # Store corpus statistics globally for strategy selection
    _corpus_stats = corpus_stats
    _document_profiles = doc_profiles

    # Create retriever with profiled documents
    _retriever_instance = HybridRetriever(profiled_documents)

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
