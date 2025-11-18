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

    Returns: recall_at_k, precision_at_k, f1_at_k, hit_rate, mrr

    Best Practices:
    - Recall@K: Critical for RAG (missing relevant docs can't be fixed downstream)
    - Precision@K: Important for quality (irrelevant docs waste LLM context)
    - nDCG@K: Use when relevance is graded (not binary) via calculate_ndcg()
    """
    if not ground_truth_doc_ids:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "f1_at_k": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }

    retrieved_ids = {
        doc.metadata.get("id", f"doc_{i}")
        for i, doc in enumerate(retrieved_docs[:k])
    }

    relevant_retrieved = retrieved_ids & ground_truth_doc_ids

    recall = len(relevant_retrieved) / len(ground_truth_doc_ids)
    precision = len(relevant_retrieved) / k if k > 0 else 0.0

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    hit_rate = 1.0 if len(relevant_retrieved) > 0 else 0.0

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

    Measures ranking quality for graded relevance (not binary).
    Returns 0.0-1.0, where 1.0 is perfect ranking.

    Formula: nDCG@K = DCG@K / IDCG@K
    Where DCG@K = Σ(i=1 to K) [rel(i) / log2(i+1)]

    Best Practices:
    - Use 4-level grading: 0 (not relevant), 1 (marginal), 2 (relevant), 3 (highly relevant)
    - nDCG particularly valuable for RAG (retrieval order affects LLM output quality)
    - Combine with binary metrics (Recall@K) for comprehensive evaluation
    """
    if not relevance_grades:
        return 0.0

    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        doc_id = doc.metadata.get("id")
        relevance = relevance_grades.get(doc_id, 0)
        dcg += relevance / math.log2(i + 1)

    ideal_relevances = sorted(relevance_grades.values(), reverse=True)[:k]

    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances, start=1):
        idcg += relevance / math.log2(i + 1)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg


def format_metrics_report(metrics: Dict[str, float], k: int = 5) -> str:
    """Format retrieval metrics as human-readable report."""
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

    Uses cosine similarity between question and answer embeddings.
    Detects off-topic responses even if factually correct.

    Returns: relevance_score (0.0-1.0), is_relevant (>= threshold), relevance_category

    Best Practices:
    - Use alongside RAGAS ResponseRelevancy for comprehensive evaluation
    - Threshold 0.7 works well for most cases
    - Particularly useful for detecting query-answer misalignment in multi-turn conversations
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    question_embedding = embeddings.embed_query(question)
    answer_embedding = embeddings.embed_query(answer)

    dot_product = np.dot(question_embedding, answer_embedding)
    question_norm = np.linalg.norm(question_embedding)
    answer_norm = np.linalg.norm(answer_embedding)

    relevance_score = dot_product / (question_norm * answer_norm)

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
