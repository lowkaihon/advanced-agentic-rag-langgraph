"""
Evaluation module for RAG pipeline assessment.

Provides:
- Retrieval metrics (Recall@K, Precision@K, F1@K, nDCG)
- Golden dataset management
- RAGAS integration
- Offline evaluation suite
"""

from .retrieval_metrics import (
    calculate_retrieval_metrics,
    calculate_ndcg,
)
from .golden_dataset import (
    GoldenDatasetManager,
    evaluate_on_golden_dataset,
    compare_answers,
)

__all__ = [
    "calculate_retrieval_metrics",
    "calculate_ndcg",
    "GoldenDatasetManager",
    "evaluate_on_golden_dataset",
    "compare_answers",
]
