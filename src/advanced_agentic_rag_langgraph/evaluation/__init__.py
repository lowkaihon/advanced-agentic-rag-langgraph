"""
Evaluation module for RAG pipeline assessment.

Provides:
- Retrieval metrics (Recall@K, Precision@K, F1@K, nDCG, MRR)
- Answer quality metrics (Groundedness, Semantic Similarity, Factual Accuracy, Completeness)
- Golden dataset management
"""

from .retrieval_metrics import (
    calculate_retrieval_metrics,
    calculate_ndcg,
    calculate_answer_relevance,
)
from .golden_dataset import (
    GoldenDatasetManager,
    evaluate_on_golden_dataset,
    compare_answers,
)

__all__ = [
    "calculate_retrieval_metrics",
    "calculate_ndcg",
    "calculate_answer_relevance",
    "GoldenDatasetManager",
    "evaluate_on_golden_dataset",
    "compare_answers",
]
