import os
from typing import Dict, List
from dotenv import load_dotenv
from src.retrieval import HybridRetriever
from src.preprocessing.document_loader import DocumentLoader
from src.preprocessing.pdf_loader import load_pdf_for_rag
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

# PDF paths
ATTENTION_PAPER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "docs",
    "attention is all you need.pdf"
)

def get_sample_documents() -> List[Document]:
    """Return sample documents for demo (replace with your own)"""
    return [
        Document(
            page_content="Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed",
            metadata={"source": "wiki", "topic": "ML", "id": "doc_1"}
        ),
        Document(
            page_content="Deep learning uses artificial neural networks with multiple layers to process and learn from data",
            metadata={"source": "textbook", "topic": "DL", "id": "doc_2"}
        ),
        Document(
            page_content="Natural language processing allows computers to understand, interpret, and generate human language",
            metadata={"source": "research", "topic": "NLP", "id": "doc_3"}
        ),
        Document(
            page_content="Retrieval-augmented generation (RAG) combines large language models with external knowledge bases to provide more accurate and contextual responses",
            metadata={"source": "paper", "topic": "RAG", "id": "doc_4"}
        ),
        Document(
            page_content="LangGraph is a framework for building stateful, multi-actor applications with LLMs using a graph-based architecture",
            metadata={"source": "docs", "topic": "LangGraph", "id": "doc_5"}
        ),
        Document(
            page_content="Query expansion generates multiple variations of a user query to improve retrieval coverage and recall",
            metadata={"source": "paper", "topic": "RAG", "id": "doc_6"}
        ),
        Document(
            page_content="Hybrid search combines dense vector retrieval with sparse keyword matching for better overall relevance",
            metadata={"source": "research", "topic": "RAG", "id": "doc_7"}
        ),
        Document(
            page_content="Reranking uses language models to score and re-order retrieved documents based on relevance to the query",
            metadata={"source": "paper", "topic": "RAG", "id": "doc_8"}
        ),
    ]

def get_attention_paper_documents(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True
) -> List[Document]:
    """
    Load the 'Attention Is All You Need' paper from PDF.

    Args:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters overlap between chunks
        verbose: Print loading progress

    Returns:
        List of chunked Document objects from the PDF
    """
    if not os.path.exists(ATTENTION_PAPER_PATH):
        raise FileNotFoundError(
            f"Attention paper not found at: {ATTENTION_PAPER_PATH}\n"
            f"Please ensure the PDF exists in the docs/ directory"
        )

    chunks = load_pdf_for_rag(
        ATTENTION_PAPER_PATH,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=verbose
    )

    return chunks

def setup_retriever(
    use_pdf: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True
) -> HybridRetriever:
    """
    Initialize and return the hybrid retriever with all advanced features.

    This function:
    1. Loads documents (sample docs or PDF)
    2. Profiles each document with DocumentProfiler
    3. Enriches documents with metadata
    4. Creates HybridRetriever with profiled documents
    5. Stores corpus statistics globally

    Args:
        use_pdf: If True, use Attention paper PDF; if False, use sample documents
        chunk_size: Chunk size for PDF loading (only used if use_pdf=True)
        chunk_overlap: Chunk overlap for PDF loading (only used if use_pdf=True)
        verbose: Whether to print profiling progress

    Returns:
        HybridRetriever instance with profiled documents
    """
    global _retriever_instance, _corpus_stats, _document_profiles

    if _retriever_instance is None:
        # Load documents based on configuration
        if use_pdf:
            if verbose:
                print("\n" + "="*60)
                print("USING PDF: Attention Is All You Need")
                print("="*60 + "\n")
            raw_documents = get_attention_paper_documents(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                verbose=verbose
            )
        else:
            if verbose:
                print("\n" + "="*60)
                print("USING SAMPLE DOCUMENTS")
                print("="*60 + "\n")
            raw_documents = get_sample_documents()

        # Load and profile documents
        loader = DocumentLoader()

        # Profile documents and enrich with metadata
        profiled_documents, corpus_stats, doc_profiles = loader.load_documents(
            raw_documents,
            verbose=verbose
        )

        # Store corpus statistics globally for strategy selection
        _corpus_stats = corpus_stats
        _document_profiles = doc_profiles

        # Create retriever with profiled documents
        _retriever_instance = HybridRetriever(profiled_documents)

    return _retriever_instance

def get_corpus_stats() -> Dict:
    """Get corpus-level statistics from profiled documents."""
    return _corpus_stats or {}

def get_document_profiles() -> Dict:
    """Get all document profiles."""
    return _document_profiles or {}
