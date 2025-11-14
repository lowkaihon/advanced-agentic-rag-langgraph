import os
from typing import Dict, List
from dotenv import load_dotenv
from src.retrieval import HybridRetriever
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
    1. Loads full PDF documents (before chunking)
    2. Profiles each FULL document with LLM
    3. Chunks documents
    4. Attaches profile metadata to each chunk
    5. Creates HybridRetriever with profiled documents
    6. Stores corpus statistics globally

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

        >>> # Load multiple PDFs
        >>> retriever = setup_retriever(pdfs=["paper1.pdf", "paper2.pdf"])

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

    # Initialize PDF loader
    pdf_loader = PDFDocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Profile full documents BEFORE chunking
    if verbose:
        print("="*60)
        print("STEP 1: Loading full documents (no chunking)")
        print("="*60 + "\n")

    # Load full documents
    full_documents = pdf_loader.load_multiple_pdfs_full_documents(
        pdf_paths,
        verbose=verbose
    )

    # Profile each full document with LLM
    if verbose:
        print("="*60)
        print("STEP 2: Profiling documents with LLM")
        print("="*60 + "\n")

    from src.preprocessing.document_profiler import DocumentProfiler

    llm_profiler = DocumentProfiler()
    document_profiles = {}

    for doc in full_documents:
        source = doc.metadata.get('source', 'unknown')
        if verbose:
            print(f"Profiling: {source}...")

        profile = llm_profiler.profile_document(
            doc.page_content,
            doc_id=source
        )
        document_profiles[source] = profile

        if verbose:
            print(f"  Type: {profile['doc_type']}")
            print(f"  Technical density: {profile['technical_density']:.2f}")
            print(f"  Best strategy: {profile['best_retrieval_strategy']} (confidence: {profile['strategy_confidence']:.2f})")
            print(f"  Domains: {', '.join(profile['domain_tags'][:3])}")
            print()

    # Now chunk the documents
    if verbose:
        print("="*60)
        print("STEP 3: Chunking documents")
        print("="*60 + "\n")

    all_chunks = []
    for full_doc in full_documents:
        # Chunk the document
        chunks = pdf_loader.text_splitter.split_text(full_doc.page_content)

        source = full_doc.metadata.get('source', 'unknown')
        profile = document_profiles.get(source)

        # Create Document objects with profile metadata
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "id": f"{source}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "source_type": "pdf",
            }

            # Attach LLM profile metadata to each chunk
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
        print(f"Created {len(all_chunks)} chunks from {len(full_documents)} documents\n")

    # Calculate corpus statistics from LLM profiles
    corpus_stats = _calculate_corpus_stats_from_llm_profiles(list(document_profiles.values()))

    # Store corpus statistics and profiles globally
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

    # Create retriever with profiled documents
    _retriever_instance = HybridRetriever(all_chunks)

    return _retriever_instance


def _calculate_corpus_stats_from_llm_profiles(profiles: List[Dict]) -> Dict:
    """
    Calculate corpus-level statistics from LLM document profiles.

    Args:
        profiles: List of DocumentProfile dictionaries

    Returns:
        Corpus statistics dictionary
    """
    if not profiles:
        return {}

    from collections import Counter

    total_docs = len(profiles)

    # Aggregate statistics
    avg_technical_density = sum(p['technical_density'] for p in profiles) / total_docs

    # Count document types
    doc_types = Counter(p['doc_type'] for p in profiles)

    # Collect all domain tags
    all_domains = []
    for p in profiles:
        all_domains.extend(p['domain_tags'])
    domain_distribution = Counter(all_domains)

    # Percentage with code/math
    pct_with_code = sum(1 for p in profiles if p['has_code']) / total_docs * 100
    pct_with_math = sum(1 for p in profiles if p['has_math']) / total_docs * 100

    # Strategy distribution
    strategy_counts = Counter(p['best_retrieval_strategy'] for p in profiles)

    return {
        "total_documents": total_docs,
        "avg_technical_density": avg_technical_density,
        "document_types": dict(doc_types),
        "domain_distribution": dict(domain_distribution.most_common(5)),
        "pct_with_code": pct_with_code,
        "pct_with_math": pct_with_math,
        "strategy_distribution": dict(strategy_counts),
    }


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
