"""
Retrieval evaluation metrics for RAG systems.

Implements standard information retrieval metrics:
- Binary relevance: Recall@K, Precision@K, F1@K, Hit Rate, MRR
- Graded relevance: nDCG@K (Normalized Discounted Cumulative Gain)
- Answer quality: Answer Relevance (embedding-based similarity)

These metrics enable:
- Quantitative benchmarking
- A/B testing of retrieval strategies
- Regression testing with golden datasets
- Systematic optimization
"""

from typing import List, Set, Dict, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import math
import numpy as np


def calculate_retrieval_metrics(
    retrieved_docs: List[Document],
    ground_truth_doc_ids: Set[str],
    k: int = 5
) -> Dict[str, float]:
    """
    Calculate standard retrieval metrics for binary relevance.

    Args:
        retrieved_docs: List of documents retrieved by the system
        ground_truth_doc_ids: Set of document IDs that are relevant (from golden dataset)
        k: Evaluation cutoff (default: 5, evaluates top-5 results)

    Returns:
        Dictionary with metrics:
        - recall_at_k: Fraction of relevant docs retrieved (0.0-1.0)
        - precision_at_k: Fraction of retrieved docs that are relevant (0.0-1.0)
        - f1_at_k: Harmonic mean of precision and recall (0.0-1.0)
        - hit_rate: Whether at least one relevant doc retrieved (0.0 or 1.0)
        - mrr: Mean Reciprocal Rank (0.0-1.0)

    Example:
        >>> ground_truth = {"doc_1", "doc_3", "doc_5"}
        >>> retrieved = [doc1, doc2, doc3, doc4, doc5]  # doc1 and doc3 are relevant
        >>> metrics = calculate_retrieval_metrics(retrieved, ground_truth, k=5)
        >>> print(f"Recall@5: {metrics['recall_at_k']:.2%}")  # 66.67% (2 of 3 relevant docs)
        >>> print(f"Precision@5: {metrics['precision_at_k']:.2%}")  # 40% (2 of 5 retrieved)

    Best Practices:
        - Recall@K: Critical for RAG (missing relevant docs can't be fixed downstream)
        - Precision@K: Important for quality (irrelevant docs waste LLM context)
        - F1@K: Balanced metric when both precision and recall matter
        - nDCG@K: Use when relevance is graded (not binary) via calculate_ndcg()
    """
    if not ground_truth_doc_ids:
        # No ground truth available - return neutral scores
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "f1_at_k": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }

    # Extract IDs from retrieved documents (top-k only)
    retrieved_ids = {
        doc.metadata.get("id", f"doc_{i}")
        for i, doc in enumerate(retrieved_docs[:k])
    }

    # Calculate relevant documents retrieved
    relevant_retrieved = retrieved_ids & ground_truth_doc_ids

    # Recall@K: What fraction of ALL relevant docs did we retrieve?
    # recall = |relevant ∩ retrieved| / |relevant|
    recall = len(relevant_retrieved) / len(ground_truth_doc_ids)

    # Precision@K: What fraction of retrieved docs are relevant?
    # precision = |relevant ∩ retrieved| / k
    precision = len(relevant_retrieved) / k if k > 0 else 0.0

    # F1@K: Harmonic mean of precision and recall
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Hit Rate: Did we get at least one relevant doc?
    # hit_rate = 1 if |relevant ∩ retrieved| > 0 else 0
    hit_rate = 1.0 if len(relevant_retrieved) > 0 else 0.0

    # Mean Reciprocal Rank: 1 / rank of first relevant doc
    # MRR = 1 / (position of first relevant doc)
    mrr = 0.0
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        doc_id = doc.metadata.get("id", f"doc_{i-1}")
        if doc_id in ground_truth_doc_ids:
            mrr = 1.0 / i
            break

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "f1_at_k": f1,
        "hit_rate": hit_rate,
        "mrr": mrr,
    }


def calculate_ndcg(
    retrieved_docs: List[Document],
    relevance_grades: Dict[str, int],
    k: int = 5
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K).

    nDCG measures ranking quality when relevance is graded (not binary).
    Rewards systems that:
    1. Retrieve highly relevant documents
    2. Rank them higher in the list

    Args:
        retrieved_docs: List of documents retrieved by the system
        relevance_grades: Dictionary mapping doc_id to relevance grade
                         Grades typically: 0=not relevant, 1=marginal, 2=relevant, 3=highly relevant
        k: Evaluation cutoff (default: 5)

    Returns:
        nDCG score (0.0-1.0), where 1.0 is perfect ranking

    Formula:
        DCG@K = Σ(i=1 to K) [rel(i) / log2(i+1)]
        nDCG@K = DCG@K / IDCG@K

        Where:
        - rel(i) = relevance grade of document at position i
        - IDCG@K = DCG of ideal ranking (best possible)
        - log2(i+1) = position discount (later positions penalized)

    Example:
        >>> # Golden dataset with graded relevance
        >>> relevance_grades = {
        ...     "doc_1": 3,  # Highly relevant
        ...     "doc_2": 2,  # Relevant
        ...     "doc_3": 1,  # Marginally relevant
        ...     "doc_4": 0   # Not relevant
        ... }
        >>> # System retrieved: [doc_4, doc_1, doc_2] (not ideal order)
        >>> ndcg = calculate_ndcg(retrieved_docs, relevance_grades, k=3)
        >>> # nDCG < 1.0 because highly relevant doc_1 is not ranked first

    Best Practices:
        - Use 4-level grading: 0 (not relevant), 1 (marginal), 2 (relevant), 3 (highly relevant)
        - nDCG is particularly valuable for RAG because LLM output quality
          often depends on retrieval order (critical docs lower = worse answers)
        - Combine with binary metrics (Recall@K) for comprehensive evaluation
    """
    if not relevance_grades:
        return 0.0

    # Calculate DCG@K (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        doc_id = doc.metadata.get("id")
        relevance = relevance_grades.get(doc_id, 0)

        # DCG formula: rel(i) / log2(i+1)
        # Position discount: documents later in list have less impact
        dcg += relevance / math.log2(i + 1)

    # Calculate IDCG@K (Ideal DCG - best possible ranking)
    # Sort relevance grades in descending order (ideal ranking)
    ideal_relevances = sorted(relevance_grades.values(), reverse=True)[:k]

    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances, start=1):
        idcg += relevance / math.log2(i + 1)

    # Normalize: nDCG = DCG / IDCG
    # nDCG = 1.0 means perfect ranking
    # nDCG = 0.0 means no relevant docs or worst possible ranking
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg


def format_metrics_report(metrics: Dict[str, float], k: int = 5) -> str:
    """
    Format retrieval metrics as human-readable report.

    Args:
        metrics: Dictionary from calculate_retrieval_metrics()
        k: Value of k used in evaluation

    Returns:
        Formatted string report

    Example:
        >>> metrics = calculate_retrieval_metrics(docs, ground_truth, k=5)
        >>> print(format_metrics_report(metrics, k=5))
        RETRIEVAL METRICS @ K=5
        =======================
        Recall@5:     75.00%  (Retrieved 3 of 4 relevant docs)
        Precision@5:  60.00%  (3 of 5 retrieved were relevant)
        F1@5:         66.67%  (Harmonic mean)
        Hit Rate:     100.00% (At least one relevant doc retrieved)
        MRR:          0.50    (First relevant doc at position 2)
    """
    report = f"""RETRIEVAL METRICS @ K={k}
{'='*50}
Recall@{k}:     {metrics['recall_at_k']:.2%}
Precision@{k}:  {metrics['precision_at_k']:.2%}
F1@{k}:         {metrics['f1_at_k']:.2%}
Hit Rate:       {metrics['hit_rate']:.2%}
MRR:            {metrics['mrr']:.4f}
{'='*50}"""

    return report


def calculate_answer_relevance(
    question: str,
    answer: str,
    embeddings: Optional[OpenAIEmbeddings] = None,
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Calculate answer relevance using embedding similarity.

    Measures whether the answer actually addresses the question,
    detecting off-topic responses even if factually correct.

    This metric complements RAGAS ResponseRelevancy with a custom
    embedding-based implementation that:
    - Uses cosine similarity between question and answer embeddings
    - Detects factually correct but off-topic answers
    - Provides interpretable relevance categories

    Args:
        question: User's original question
        answer: Generated answer
        embeddings: Optional embeddings model (creates default if None)
        threshold: Minimum similarity for relevance (default: 0.7)

    Returns:
        Dictionary with:
        - relevance_score: Cosine similarity (0.0-1.0)
        - is_relevant: Boolean (True if score >= threshold)
        - relevance_category: "high" (>0.85), "medium" (0.70-0.85), "low" (<0.70)

    Example:
        >>> # Relevant answer
        >>> result = calculate_answer_relevance(
        ...     question="How many attention heads does the base Transformer use?",
        ...     answer="The base Transformer model uses 8 attention heads."
        ... )
        >>> print(f"Score: {result['relevance_score']:.2f}, Relevant: {result['is_relevant']}")
        Score: 0.92, Relevant: True

        >>> # Off-topic answer (factually correct but irrelevant)
        >>> result = calculate_answer_relevance(
        ...     question="How many attention heads does the base Transformer use?",
        ...     answer="GPT-3 is a large language model with 175 billion parameters."
        ... )
        >>> print(f"Score: {result['relevance_score']:.2f}, Relevant: {result['is_relevant']}")
        Score: 0.45, Relevant: False

    Best Practices:
        - Use alongside RAGAS ResponseRelevancy for comprehensive evaluation
        - Threshold 0.7 works well for most cases (adjust based on use case)
        - Low scores (<0.6) indicate answer is off-topic or doesn't address question
        - Particularly useful for detecting query-answer misalignment in multi-turn conversations
    """
    # Create embeddings model if not provided
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Embed question and answer
    question_embedding = embeddings.embed_query(question)
    answer_embedding = embeddings.embed_query(answer)

    # Calculate cosine similarity
    # cosine_similarity = dot(A, B) / (||A|| * ||B||)
    dot_product = np.dot(question_embedding, answer_embedding)
    question_norm = np.linalg.norm(question_embedding)
    answer_norm = np.linalg.norm(answer_embedding)

    relevance_score = dot_product / (question_norm * answer_norm)

    # Determine relevance category
    if relevance_score >= 0.85:
        category = "high"
    elif relevance_score >= 0.70:
        category = "medium"
    else:
        category = "low"

    return {
        "relevance_score": float(relevance_score),
        "is_relevant": relevance_score >= threshold,
        "relevance_category": category
    }
