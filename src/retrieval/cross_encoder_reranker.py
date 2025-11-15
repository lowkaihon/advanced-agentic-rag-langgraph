"""
CrossEncoder-based document reranking for improved retrieval quality.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for fast, accurate relevance scoring.
Processes query-document pairs jointly to capture semantic relationships.
"""

from typing import List, Tuple
from langchain_core.documents import Document


class CrossEncoderReRanker:
    """
    Rerank documents using CrossEncoder semantic similarity.

    CrossEncoders process [query, document] pairs jointly through a transformer,
    enabling richer interaction modeling than bi-encoders. The ms-marco-MiniLM-L-6-v2
    model provides strong relevance scoring with ~200-300ms latency for 10-20 documents.

    Benefits over LLM-based reranking:
    - 20-35% accuracy improvement in retrieval tasks
    - 200-500ms latency (vs 2-5s for LLM)
    - ~100x cost reduction ($0.0001/query vs $0.01-0.05/query)
    - Deterministic scoring (no temperature variance)

    Usage:
        reranker = CrossEncoderReRanker(top_k=10)
        ranked_docs = reranker.rank("What is attention?", documents)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        max_length: int = 512,
        batch_size: int = 16
    ):
        """
        Initialize CrossEncoder reranker.

        Args:
            model_name: HuggingFace model ID. Default: ms-marco-MiniLM-L-6-v2
            top_k: Number of top documents to return
            max_length: Maximum token length for [query, doc] pairs (default: 512)
                       MiniLM models are trained with 512 token limit
            batch_size: Batch size for prediction (default: 16)
                       Adjust based on available RAM
        """
        self.top_k = top_k
        self.batch_size = batch_size

        # Lazy import to avoid loading model at module import time
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReRanker.\n"
                "Install with: uv add sentence-transformers"
            )

        # Initialize CrossEncoder model
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rank(
        self,
        query: str,
        documents: List[Document],
        truncate_content: int = 400
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: User query string
            documents: List of Document objects to rerank
            truncate_content: Truncate doc content to N characters (default: 400)
                             Prevents exceeding 512 token limit when combined with query

        Returns:
            List of (document, score) tuples, sorted by relevance (highest first),
            limited to top_k documents

        Example:
            >>> reranker = CrossEncoderReRanker(top_k=5)
            >>> docs = retriever.get_relevant_documents("attention mechanism")
            >>> ranked = reranker.rank("attention mechanism", docs)
            >>> for doc, score in ranked[:3]:
            ...     print(f"Score: {score:.3f} - {doc.page_content[:50]}...")
        """
        if not documents:
            return []

        # Prepare query-document pairs for cross-encoder
        # Truncate content to stay within 512 token limit (query + doc + special tokens)
        pairs = [
            [query, doc.page_content[:truncate_content]]
            for doc in documents
        ]

        # Get relevance scores from cross-encoder
        # Scores are typically in range [-10, 10], higher = more relevant
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Pair documents with scores and sort by relevance (descending)
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k results
        return ranked[:self.top_k]

    def get_scores_only(
        self,
        query: str,
        documents: List[Document],
        truncate_content: int = 400
    ) -> List[float]:
        """
        Get relevance scores without sorting/truncating.

        Useful for analysis or combining with other ranking signals.

        Args:
            query: User query string
            documents: List of Document objects
            truncate_content: Truncate doc content to N characters

        Returns:
            List of relevance scores (same order as input documents)
        """
        if not documents:
            return []

        pairs = [
            [query, doc.page_content[:truncate_content]]
            for doc in documents
        ]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        return scores.tolist()
