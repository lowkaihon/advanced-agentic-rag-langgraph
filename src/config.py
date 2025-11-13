import os
from dotenv import load_dotenv
from src.retrieval import HybridRetriever
from langchain_core.documents import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Enable LangSmith tracing (optional)
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "advanced-agentic-rag"

# Global retriever instance
_retriever_instance = None

def get_sample_documents() -> list[Document]:
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

def setup_retriever() -> HybridRetriever:
    """Initialize and return the hybrid retriever with all advanced features"""
    global _retriever_instance
    if _retriever_instance is None:
        documents = get_sample_documents()
        _retriever_instance = HybridRetriever(documents)
    return _retriever_instance